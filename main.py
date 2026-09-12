"""Pose landmark extraction + motion metric computation service.

Internal-only microservice used by the Rehabiphy Node backend.

- For a single photo (/analyze): runs MediaPipe Pose and returns the raw 33
  body landmarks plus a couple of basic quality signals. No clinical
  logic — that lives in the Node backend's postureMeasurementEngine.js.
- For a short video (/analyze-video): runs MediaPipe Pose per sampled frame
  AND (given a recognized `movementType`) computes that movement's raw
  metric values (motion_metrics.py) — e.g. trunk_flexion_rom degrees. This
  differs from the photo path: video analysis needs per-frame numeric/
  signal-processing work (peak-finding across a time series) that Python's
  ecosystem handles far better than hand-rolling the same in Node, so unlike
  the posture pipeline, computing the raw numbers happens here rather than
  in the Node backend. Severity bands, scoring, and report authoring still
  live in Node (motionMetrics.js / motionExplanationService.js) — this
  service only turns pixels into numbers, never numbers into clinical
  meaning.
"""

import base64
import os
import tempfile

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel

import mediapipe as mp

from motion_metrics import compute_motion_metrics

app = FastAPI(title="Rehabiphy Pose Service")

mp_pose = mp.solutions.pose

# Landmarks with visibility below this are excluded from the confidence/
# framing calculation — a low-visibility landmark is one MediaPipe guessed
# rather than actually saw.
VISIBILITY_THRESHOLD = 0.5

# Video frames are sampled down to roughly this rate before running Pose —
# processing every frame of a 30/60fps phone clip would be far slower than
# needed for the joint-angle-over-time metrics the Node engine computes, and
# would bloat the stored per-frame landmark payload for little benefit.
TARGET_SAMPLE_FPS = 8
# Rejects long gallery-picked clips outright rather than spending minutes
# processing something the movement engine isn't designed for anyway (each
# motion type is a single short rep/cycle, not an extended recording).
MAX_VIDEO_DURATION_SEC = 30

INTERNAL_SECRET = os.environ.get("POSE_SERVICE_SECRET")


class Landmark(BaseModel):
    name: str
    x: float
    y: float
    z: float
    visibility: float
    # Real-world coordinates in meters, relative to the hip midpoint
    # (MediaPipe's `pose_world_landmarks`) — unlike x/y/z above, these aren't
    # skewed by camera distance/perspective, so the Node engine uses them for
    # joint-angle metrics that need a true 3D angle (e.g. knee/ankle sagittal
    # flexion) rather than a flat image-plane approximation. None if
    # MediaPipe didn't return world landmarks for this frame.
    wx: float | None = None
    wy: float | None = None
    wz: float | None = None


class AnalyzeResponse(BaseModel):
    view: str
    personDetected: bool
    landmarkCount: int
    confidence: float
    visibleRatio: float
    landmarks: list[Landmark]


class FrameLandmarks(BaseModel):
    t: float  # seconds from the start of the (sampled) clip
    frameIndex: int  # index into the ORIGINAL video's frames, not the sampled sequence
    personDetected: bool
    landmarks: list[Landmark]


class MetricResult(BaseModel):
    value: float
    confidence: float


class AnalyzeVideoResponse(BaseModel):
    durationSec: float
    sourceFps: float
    sampledFps: float
    width: int
    height: int
    movementType: str | None = None
    # Raw computed values for movementType's metrics (motion_metrics.py) —
    # empty if movementType was omitted or isn't implemented yet.
    metrics: dict[str, MetricResult] = {}
    frames: list[FrameLandmarks]


class ExtractFramesResponse(BaseModel):
    # One base64 JPEG per requested frame index, same order as the request —
    # "" for any index that couldn't be read (e.g. past the end of the
    # video). Used by the Node backend's PDF report to grab a couple of
    # actual video frames as images for annotation, without needing to have
    # stored frame images at analysis time — the video itself (already in
    # S3) is the source of truth, re-decoded on demand at download time.
    images: list[str]


def _check_secret(x_internal_secret: str | None) -> None:
    if INTERNAL_SECRET and x_internal_secret != INTERNAL_SECRET:
        raise HTTPException(status_code=401, detail="Invalid internal secret")


def _landmarks_from_result(result) -> list[Landmark]:
    """Shared with /analyze-video — builds the Landmark list (with world
    coordinates when available) from one MediaPipe Pose result."""
    if not result.pose_landmarks:
        return []

    world_landmarks = result.pose_world_landmarks.landmark if result.pose_world_landmarks else None
    landmarks = []
    for idx, lm in enumerate(result.pose_landmarks.landmark):
        name = mp_pose.PoseLandmark(idx).name.lower()
        world = world_landmarks[idx] if world_landmarks else None
        landmarks.append(
            Landmark(
                name=name,
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=lm.visibility,
                wx=world.x if world else None,
                wy=world.y if world else None,
                wz=world.z if world else None,
            )
        )
    return landmarks


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    view: str = Form(...),
    x_internal_secret: str | None = Header(default=None),
):
    _check_secret(x_internal_secret)

    if view not in {"front", "back", "left", "right"}:
        raise HTTPException(status_code=400, detail="view must be one of front/back/left/right")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    image_array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5) as pose:
        result = pose.process(image_rgb)

    if not result.pose_landmarks:
        return AnalyzeResponse(
            view=view,
            personDetected=False,
            landmarkCount=0,
            confidence=0.0,
            visibleRatio=0.0,
            landmarks=[],
        )

    landmarks = _landmarks_from_result(result)
    visibilities = [lm.visibility for lm in landmarks]

    visible_count = sum(1 for v in visibilities if v >= VISIBILITY_THRESHOLD)
    visible_ratio = visible_count / len(visibilities)
    confidence = float(np.mean(visibilities))

    return AnalyzeResponse(
        view=view,
        personDetected=True,
        landmarkCount=len(landmarks),
        confidence=confidence,
        visibleRatio=visible_ratio,
        landmarks=landmarks,
    )


@app.post("/analyze-video", response_model=AnalyzeVideoResponse)
async def analyze_video(
    file: UploadFile = File(...),
    movementType: str | None = Form(default=None),
    x_internal_secret: str | None = Header(default=None),
):
    _check_secret(x_internal_secret)

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    # OpenCV's VideoCapture needs a real file path (no in-memory decode path
    # for arbitrary containers the way cv2.imdecode covers images), so the
    # upload is spooled to a temp file for the duration of processing.
    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not decode video")

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = (total_frames / source_fps) if source_fps and total_frames else 0.0

        if duration_sec > MAX_VIDEO_DURATION_SEC:
            cap.release()
            raise HTTPException(
                status_code=400,
                detail=f"Video is too long ({duration_sec:.0f}s) — please keep clips under {MAX_VIDEO_DURATION_SEC}s",
            )

        frame_interval = max(1, round(source_fps / TARGET_SAMPLE_FPS))
        sampled_fps = source_fps / frame_interval

        frames: list[FrameLandmarks] = []
        # static_image_mode=False enables MediaPipe's frame-to-frame tracking
        # (faster and more temporally consistent than re-detecting from
        # scratch on every sampled frame, which static mode would do).
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            frame_index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index % frame_interval == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = pose.process(frame_rgb)
                    landmarks = _landmarks_from_result(result)
                    frames.append(
                        FrameLandmarks(
                            t=frame_index / source_fps,
                            frameIndex=frame_index,
                            personDetected=bool(landmarks),
                            landmarks=landmarks,
                        )
                    )
                frame_index += 1
        cap.release()
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    metrics = {}
    if movementType:
        frames_data = [f.model_dump() for f in frames]
        metrics = compute_motion_metrics(movementType, frames_data)

    return AnalyzeVideoResponse(
        durationSec=duration_sec,
        sourceFps=source_fps,
        sampledFps=sampled_fps,
        width=width,
        height=height,
        movementType=movementType,
        metrics=metrics,
        frames=frames,
    )


@app.post("/extract-frames", response_model=ExtractFramesResponse)
async def extract_frames(
    file: UploadFile = File(...),
    frameIndices: str = Form(...),  # comma-separated original-video frame indices
    x_internal_secret: str | None = Header(default=None),
):
    _check_secret(x_internal_secret)

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        indices = [int(x) for x in frameIndices.split(",") if x.strip() != ""]
    except ValueError:
        raise HTTPException(status_code=400, detail="frameIndices must be a comma-separated list of integers")

    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    tmp_path = None
    images_by_index = {}
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not decode video")

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                images_by_index[idx] = ""
                continue
            success, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
            images_by_index[idx] = base64.b64encode(buf.tobytes()).decode() if success else ""
        cap.release()
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return ExtractFramesResponse(images=[images_by_index.get(i, "") for i in indices])
