"""Pose landmark extraction service.

Internal-only microservice used by the Rehabiphy Node backend. It does one
thing: given a single photo, run MediaPipe Pose and return the raw 33 body
landmarks plus a couple of basic quality signals. It intentionally contains
no clinical/measurement logic (angles, severities, scores) — that lives in
the Node backend's postureMeasurementEngine.js so all clinical rules stay in
one place.
"""

import os

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel

import mediapipe as mp

app = FastAPI(title="Rehabiphy Pose Service")

mp_pose = mp.solutions.pose

# Landmarks with visibility below this are excluded from the confidence/
# framing calculation — a low-visibility landmark is one MediaPipe guessed
# rather than actually saw.
VISIBILITY_THRESHOLD = 0.5

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


def _check_secret(x_internal_secret: str | None) -> None:
    if INTERNAL_SECRET and x_internal_secret != INTERNAL_SECRET:
        raise HTTPException(status_code=401, detail="Invalid internal secret")


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

    world_landmarks = result.pose_world_landmarks.landmark if result.pose_world_landmarks else None

    landmarks = []
    visibilities = []
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
        visibilities.append(lm.visibility)

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
