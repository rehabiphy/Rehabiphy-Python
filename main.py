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

    landmarks = []
    visibilities = []
    for idx, lm in enumerate(result.pose_landmarks.landmark):
        name = mp_pose.PoseLandmark(idx).name.lower()
        landmarks.append(Landmark(name=name, x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility))
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
