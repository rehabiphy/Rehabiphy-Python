"""Shared pure-geometry helpers for motion metrics — the frame-time-series
equivalent of the vector math in the Node backend's
postureMeasurementEngine.js (which does the same job for single-image
posture findings). No clinical/severity logic lives here; see
motion_metrics.py for the movement-specific computations built on top of
these, and the Node backend's motionMetrics.js for severity bands.
"""

import math


def angle_from_vertical(p_top, p_bottom):
    """Angle (degrees, 0-90ish) between the line p_bottom->p_top and true
    vertical (image y-axis grows downward, so "up" is flipped to be the
    zero-angle reference)."""
    dx = p_top["x"] - p_bottom["x"]
    dy = p_bottom["y"] - p_top["y"]
    angle = abs(math.degrees(math.atan2(dx, dy)))
    return 180 - angle if angle > 90 else angle


def point3(lm):
    """Real-world (meters) coordinates when the pose service supplied them
    (see main.py's wx/wy/wz), falling back to normalized image coordinates
    otherwise."""
    wx, wy, wz = lm.get("wx"), lm.get("wy"), lm.get("wz")
    return {
        "x": wx if wx is not None else lm["x"],
        "y": wy if wy is not None else lm["y"],
        "z": wz if wz is not None else lm.get("z", 0.0),
    }


def joint_angle_3d(a, b, c):
    """Angle in degrees at vertex b between rays b->a and b->c, in 3D."""
    v1 = {"x": a["x"] - b["x"], "y": a["y"] - b["y"], "z": a["z"] - b["z"]}
    v2 = {"x": c["x"] - b["x"], "y": c["y"] - b["y"], "z": c["z"] - b["z"]}
    dot = v1["x"] * v2["x"] + v1["y"] * v2["y"] + v1["z"] * v2["z"]
    mag1 = math.hypot(v1["x"], v1["y"], v1["z"]) or 1.0
    mag2 = math.hypot(v2["x"], v2["y"], v2["z"]) or 1.0
    cos_val = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_val))


def avg_visibility(*landmarks):
    vis = [lm["visibility"] for lm in landmarks if lm is not None]
    return sum(vis) / len(vis) if vis else 0.0


def distance3(a, b):
    return math.dist((a["x"], a["y"], a["z"]), (b["x"], b["y"], b["z"]))


def perpendicular_deviation_2d(a, b, c):
    """Perpendicular distance of point b from the line a-c, in image-plane
    (2D) units — a coarse frontal-plane knee valgus/varus proxy (same idea
    as postureMeasurementEngine.js's computeKneeAlignment), not a true
    angle. Caller normalizes by a body-width reference."""
    line_x = c["x"] - a["x"]
    line_y = c["y"] - a["y"]
    length = math.hypot(line_x, line_y) or 1.0
    cross = (b["x"] - a["x"]) * line_y - (b["y"] - a["y"]) * line_x
    return abs(cross / length)
