"""Movement-specific metric computation from a pose-landmark frame time
series — the video/motion-analysis equivalent of
postureMeasurementEngine.js. Deterministic geometry + light numeric
processing only (min/max/peak-finding over a short series); severity bands,
scoring, muscle-focus mapping, and report authoring all live in the Node
backend (motionMetrics.js / motionExplanationService.js) so clinical
judgment stays centralized there — this module only turns pixels-over-time
into plain numbers.

One compute_* function per movement type, registered in MOVEMENT_COMPUTE_FN
below, mirroring the Node data-registry pattern (assessmentModules.js /
postureMetrics.js) — add a movement by adding a function here plus a
matching entry in the Node backend's motionMetrics.js, no other plumbing.
"""

import numpy as np
from scipy.signal import find_peaks

import math

from pose_geometry import (
    angle_from_vertical,
    avg_visibility,
    distance3,
    joint_angle_3d,
    perpendicular_deviation_2d,
    point3,
)


def _landmark_map(frame_landmarks):
    return {lm["name"]: lm for lm in frame_landmarks}


def _dominant_side(frames, primary="shoulder", secondary="hip"):
    """Side-view movements only have ONE of left_*/right_* reliably visible
    (the far side is partially or fully occluded) — picks whichever side
    averaged higher visibility across the whole clip and uses that side
    consistently, rather than switching side frame-to-frame as visibility
    noise fluctuates."""
    totals = {"left": [], "right": []}
    for frame in frames:
        lm = _landmark_map(frame["landmarks"])
        for side in ("left", "right"):
            p = lm.get(f"{side}_{primary}")
            h = lm.get(f"{side}_{secondary}")
            if p and h:
                totals[side].append((p["visibility"] + h["visibility"]) / 2)
    left_avg = sum(totals["left"]) / len(totals["left"]) if totals["left"] else 0.0
    right_avg = sum(totals["right"]) / len(totals["right"]) if totals["right"] else 0.0
    return "left" if left_avg >= right_avg else "right"


def compute_forward_bending(frames):
    usable = [f for f in frames if f["personDetected"] and f["landmarks"]]
    if len(usable) < 2:
        return {}

    side = _dominant_side(usable)
    trunk_angles = []
    hip_flexions = []
    confidences = []

    for frame in usable:
        lm = _landmark_map(frame["landmarks"])
        shoulder = lm.get(f"{side}_shoulder")
        hip = lm.get(f"{side}_hip")
        knee = lm.get(f"{side}_knee")

        if not shoulder or not hip:
            trunk_angles.append(None)
            hip_flexions.append(None)
            continue

        trunk_angles.append(angle_from_vertical(shoulder, hip))
        confidences.append(avg_visibility(shoulder, hip, knee))

        if knee:
            hip_joint_angle = joint_angle_3d(point3(shoulder), point3(hip), point3(knee))
            hip_flexions.append(abs(180 - hip_joint_angle))
        else:
            hip_flexions.append(None)

    valid_indices = [i for i, angle in enumerate(trunk_angles) if angle is not None]
    if len(valid_indices) < 2:
        return {}

    start_idx, end_idx = valid_indices[0], valid_indices[-1]
    peak_idx = max(valid_indices, key=lambda i: trunk_angles[i])

    start_angle = trunk_angles[start_idx]
    end_angle = trunk_angles[end_idx]
    peak_angle = trunk_angles[peak_idx]
    overall_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0.0

    metrics = {}

    trunk_flexion_rom = max(peak_angle - start_angle, 0.0)
    metrics["trunk_flexion_rom"] = {"value": round(trunk_flexion_rom, 2), "confidence": overall_confidence}

    metrics["return_to_neutral"] = {"value": round(abs(end_angle - start_angle), 2), "confidence": overall_confidence}

    # Only meaningful once there's an actual bend to compare against — a
    # near-zero ROM makes the ratio blow up or divide by ~0 for no reason.
    hip_flexion_at_peak = hip_flexions[peak_idx]
    if hip_flexion_at_peak is not None and trunk_flexion_rom > 5:
        ratio = min(hip_flexion_at_peak / trunk_flexion_rom, 2.0)
        metrics["hip_hinge_ratio"] = {"value": round(ratio, 2), "confidence": overall_confidence}

    return metrics


def compute_gait(frames):
    """Walking analysis, filmed from the side. Only the metrics that a
    single side-on camera can actually support are computed here — true
    foot-progression angle (toeing in/out) needs a front/behind view and is
    intentionally NOT included; "trunk/pelvic movement" is approximated as
    sagittal trunk-lean variability (trunk_sway) rather than true frontal-
    plane pelvic drop, for the same reason.
    """
    usable = [f for f in frames if f["personDetected"] and f["landmarks"]]
    if len(usable) < 6:
        return {}

    ts, left_ankle_y, right_ankle_y = [], [], []
    trunk_angles, knee_angles, leg_lengths, confidences = [], [], [], []
    hip_mid_world = []

    for frame in usable:
        lm = _landmark_map(frame["landmarks"])
        l_hip, r_hip = lm.get("left_hip"), lm.get("right_hip")
        l_ankle, r_ankle = lm.get("left_ankle"), lm.get("right_ankle")
        if not (l_hip and r_hip and l_ankle and r_ankle):
            continue

        ts.append(frame["t"])
        left_ankle_y.append(l_ankle["y"])
        right_ankle_y.append(r_ankle["y"])
        confidences.append(avg_visibility(l_hip, r_hip, l_ankle, r_ankle))

        l_shoulder, r_shoulder = lm.get("left_shoulder"), lm.get("right_shoulder")
        if l_shoulder and r_shoulder:
            mid_shoulder = {"x": (l_shoulder["x"] + r_shoulder["x"]) / 2, "y": (l_shoulder["y"] + r_shoulder["y"]) / 2}
            mid_hip_img = {"x": (l_hip["x"] + r_hip["x"]) / 2, "y": (l_hip["y"] + r_hip["y"]) / 2}
            trunk_angles.append(angle_from_vertical(mid_shoulder, mid_hip_img))

        l_hip3, r_hip3, l_ankle3, r_ankle3 = point3(l_hip), point3(r_hip), point3(l_ankle), point3(r_ankle)
        l_knee, r_knee = lm.get("left_knee"), lm.get("right_knee")
        if l_knee:
            knee_angles.append(joint_angle_3d(l_hip3, point3(l_knee), l_ankle3))
        if r_knee:
            knee_angles.append(joint_angle_3d(r_hip3, point3(r_knee), r_ankle3))

        hip_mid_world.append({
            "x": (l_hip3["x"] + r_hip3["x"]) / 2,
            "y": (l_hip3["y"] + r_hip3["y"]) / 2,
            "z": (l_hip3["z"] + r_hip3["z"]) / 2,
        })
        leg_lengths.append((distance3(l_hip3, l_ankle3) + distance3(r_hip3, r_ankle3)) / 2)

    if len(ts) < 6:
        return {}

    ts_arr = np.array(ts)
    avg_dt = float(np.mean(np.diff(ts_arr))) if len(ts_arr) > 1 else 0.0
    duration = float(ts_arr[-1] - ts_arr[0])
    overall_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0.0
    avg_leg_length = float(np.mean(leg_lengths)) if leg_lengths else 0.0

    # Heel-strike proxy: a local peak in ankle image-y (foot at its lowest,
    # nearest-to-ground point). `distance` assumes steps land at least 0.3s
    # apart (a very fast cadence upper bound) to avoid detecting noise as
    # extra steps; `prominence` scales with the signal's own variability so
    # it adapts to how much vertical ankle motion this particular clip has.
    min_gap_samples = max(2, round(0.3 / avg_dt)) if avg_dt > 0 else 2

    def find_contacts(y_series):
        arr = np.array(y_series)
        if arr.std() < 1e-4:
            return np.array([], dtype=int)
        peaks, _ = find_peaks(arr, distance=min_gap_samples, prominence=0.3 * arr.std())
        return peaks

    left_peaks = find_contacts(left_ankle_y)
    right_peaks = find_contacts(right_ankle_y)

    metrics = {}

    total_steps = len(left_peaks) + len(right_peaks)
    if total_steps >= 2 and duration > 0:
        cadence = total_steps / (duration / 60.0)
        metrics["cadence"] = {"value": round(cadence, 1), "confidence": overall_confidence}

    def avg_interval(peaks):
        if len(peaks) < 2:
            return None
        return float(np.mean(np.diff(ts_arr[peaks])))

    left_interval, right_interval = avg_interval(left_peaks), avg_interval(right_peaks)
    if left_interval and right_interval:
        avg_interval_both = (left_interval + right_interval) / 2
        symmetry_pct = abs(left_interval - right_interval) / avg_interval_both * 100
        metrics["step_symmetry"] = {"value": round(symmetry_pct, 1), "confidence": overall_confidence}

    def avg_stride_ratio(peaks):
        if len(peaks) < 2 or avg_leg_length <= 0:
            return None
        dists = [distance3(hip_mid_world[a], hip_mid_world[b]) for a, b in zip(peaks[:-1], peaks[1:])]
        return (sum(dists) / len(dists)) / avg_leg_length

    stride_candidates = [v for v in (avg_stride_ratio(left_peaks), avg_stride_ratio(right_peaks)) if v is not None]
    if stride_candidates:
        metrics["stride_length_ratio"] = {"value": round(sum(stride_candidates) / len(stride_candidates), 2), "confidence": overall_confidence}

    if len(knee_angles) >= 4:
        metrics["knee_rom_gait"] = {"value": round(max(knee_angles) - min(knee_angles), 2), "confidence": overall_confidence}

    if len(trunk_angles) >= 4:
        metrics["trunk_sway"] = {"value": round(float(np.std(trunk_angles)), 2), "confidence": overall_confidence}

    return metrics


def compute_squat(frames):
    """Bodyweight squat, filmed from the front. Finds the deepest point of
    the squat (max average knee flexion across the clip — works for a
    single rep or picks the worst moment across multiple reps) and compares
    it against the first frame as a standing baseline. All metrics are
    frontal-plane, matching the front-on camera angle: sagittal-only signals
    like true forward trunk lean aren't attempted here.
    """
    usable = [f for f in frames if f["personDetected"] and f["landmarks"]]
    if len(usable) < 3:
        return {}

    required = ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_shoulder", "right_shoulder"]
    parsed = []
    for frame in usable:
        lm = _landmark_map(frame["landmarks"])
        if not all(lm.get(name) for name in required):
            continue

        l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle = (
            lm["left_hip"], lm["right_hip"], lm["left_knee"], lm["right_knee"], lm["left_ankle"], lm["right_ankle"],
        )
        knee_flexion_avg = (
            abs(180 - joint_angle_3d(point3(l_hip), point3(l_knee), point3(l_ankle)))
            + abs(180 - joint_angle_3d(point3(r_hip), point3(r_knee), point3(r_ankle)))
        ) / 2
        parsed.append({"lm": lm, "knee_flexion_avg": knee_flexion_avg})

    if len(parsed) < 3:
        return {}

    baseline = parsed[0]["lm"]
    deepest = max(parsed, key=lambda p: p["knee_flexion_avg"])["lm"]

    metrics = {}

    # Knee valgus/varus (also covers "hip-knee-ankle alignment" — same
    # geometry) — worse of the two legs, at the deepest point of the squat.
    hip_width = math.hypot(deepest["right_hip"]["x"] - deepest["left_hip"]["x"], deepest["right_hip"]["y"] - deepest["left_hip"]["y"]) or 1.0
    left_dev = perpendicular_deviation_2d(deepest["left_hip"], deepest["left_knee"], deepest["left_ankle"]) / hip_width
    right_dev = perpendicular_deviation_2d(deepest["right_hip"], deepest["right_knee"], deepest["right_ankle"]) / hip_width
    knee_conf = avg_visibility(
        deepest["left_hip"], deepest["right_hip"], deepest["left_knee"], deepest["right_knee"], deepest["left_ankle"], deepest["right_ankle"]
    )
    metrics["knee_valgus_varus"] = {"value": round(max(left_dev, right_dev), 3), "confidence": round(knee_conf, 2)}

    # Squat depth: hip vertical drop from standing, normalized by thigh
    # length (so it's comparable across camera distances/subject heights).
    baseline_hip_y = (baseline["left_hip"]["y"] + baseline["right_hip"]["y"]) / 2
    deepest_hip_y = (deepest["left_hip"]["y"] + deepest["right_hip"]["y"]) / 2
    thigh_length = (
        math.hypot(baseline["left_hip"]["x"] - baseline["left_knee"]["x"], baseline["left_hip"]["y"] - baseline["left_knee"]["y"])
        + math.hypot(baseline["right_hip"]["x"] - baseline["right_knee"]["x"], baseline["right_hip"]["y"] - baseline["right_knee"]["y"])
    ) / 2 or 1.0
    depth_ratio = max(0.0, (deepest_hip_y - baseline_hip_y) / thigh_length)
    metrics["squat_depth"] = {"value": round(depth_ratio, 2), "confidence": round(knee_conf, 2)}

    # Lateral trunk lean at the deepest frame — a frontal-plane compensation
    # (leaning sideways), not the sagittal forward lean a side view would show.
    mid_shoulder = {"x": (deepest["left_shoulder"]["x"] + deepest["right_shoulder"]["x"]) / 2, "y": (deepest["left_shoulder"]["y"] + deepest["right_shoulder"]["y"]) / 2}
    mid_hip = {"x": (deepest["left_hip"]["x"] + deepest["right_hip"]["x"]) / 2, "y": (deepest["left_hip"]["y"] + deepest["right_hip"]["y"]) / 2}
    trunk_conf = avg_visibility(deepest["left_shoulder"], deepest["right_shoulder"], deepest["left_hip"], deepest["right_hip"])
    metrics["lateral_trunk_lean"] = {"value": round(angle_from_vertical(mid_shoulder, mid_hip), 2), "confidence": round(trunk_conf, 2)}

    # Weight-shift asymmetry: how far the hips drift from centered over the
    # base of support (the midpoint between the two ankles), at the deepest frame.
    ankle_mid_x = (deepest["left_ankle"]["x"] + deepest["right_ankle"]["x"]) / 2
    stance_width = math.hypot(deepest["right_ankle"]["x"] - deepest["left_ankle"]["x"], deepest["right_ankle"]["y"] - deepest["left_ankle"]["y"]) or 1.0
    metrics["weight_shift_asymmetry"] = {"value": round(abs(mid_hip["x"] - ankle_mid_x) / stance_width, 3), "confidence": round(knee_conf, 2)}

    # Heel lift: needs heel + toe (foot_index) landmarks, which aren't
    # always framed in a front-on squat video — silently omitted if absent.
    heel_values, heel_conf_points = [], []
    for side in ("left", "right"):
        heel_base, toe_base, heel_deep = baseline.get(f"{side}_heel"), baseline.get(f"{side}_foot_index"), deepest.get(f"{side}_heel")
        if not (heel_base and toe_base and heel_deep):
            continue
        foot_length = math.hypot(toe_base["x"] - heel_base["x"], toe_base["y"] - heel_base["y"]) or 1.0
        heel_values.append(max(0.0, (heel_base["y"] - heel_deep["y"]) / foot_length))
        heel_conf_points.extend([heel_base, toe_base, heel_deep])
    if heel_values:
        metrics["heel_lift"] = {"value": round(max(heel_values), 3), "confidence": round(avg_visibility(*heel_conf_points), 2)}

    return metrics


def compute_lunge(frames):
    """Forward lunge, filmed from the front. A single clip only shows one
    leg stepping forward, so there's no true bilateral (left-lunge vs.
    right-lunge) comparison here — the "front" leg is auto-detected as
    whichever knee is more flexed at the deepest point of the clip, and
    pelvic_drop (hip height difference) stands in for left/right asymmetry
    since an uneven pelvis IS a left/right asymmetry signal, just not a
    two-rep comparison.
    """
    usable = [f for f in frames if f["personDetected"] and f["landmarks"]]
    if len(usable) < 3:
        return {}

    required = ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_shoulder", "right_shoulder"]
    parsed = []
    for frame in usable:
        lm = _landmark_map(frame["landmarks"])
        if not all(lm.get(name) for name in required):
            continue
        l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle = lm["left_hip"], lm["right_hip"], lm["left_knee"], lm["right_knee"], lm["left_ankle"], lm["right_ankle"]
        left_flex = abs(180 - joint_angle_3d(point3(l_hip), point3(l_knee), point3(l_ankle)))
        right_flex = abs(180 - joint_angle_3d(point3(r_hip), point3(r_knee), point3(r_ankle)))
        parsed.append({"lm": lm, "left_flex": left_flex, "right_flex": right_flex, "max_flex": max(left_flex, right_flex)})

    if len(parsed) < 3:
        return {}

    deepest = max(parsed, key=lambda p: p["max_flex"])
    lm = deepest["lm"]
    front_side = "left" if deepest["left_flex"] >= deepest["right_flex"] else "right"
    front_hip, front_knee, front_ankle = lm[f"{front_side}_hip"], lm[f"{front_side}_knee"], lm[f"{front_side}_ankle"]

    metrics = {}

    hip_width = math.hypot(lm["right_hip"]["x"] - lm["left_hip"]["x"], lm["right_hip"]["y"] - lm["left_hip"]["y"]) or 1.0

    knee_dev = perpendicular_deviation_2d(front_hip, front_knee, front_ankle) / hip_width
    metrics["front_knee_valgus_varus"] = {"value": round(knee_dev, 3), "confidence": round(avg_visibility(front_hip, front_knee, front_ankle), 2)}

    pelvic_drop = abs(lm["left_hip"]["y"] - lm["right_hip"]["y"]) / hip_width
    metrics["pelvic_drop"] = {"value": round(pelvic_drop, 3), "confidence": round(avg_visibility(lm["left_hip"], lm["right_hip"]), 2)}

    mid_shoulder = {"x": (lm["left_shoulder"]["x"] + lm["right_shoulder"]["x"]) / 2, "y": (lm["left_shoulder"]["y"] + lm["right_shoulder"]["y"]) / 2}
    mid_hip = {"x": (lm["left_hip"]["x"] + lm["right_hip"]["x"]) / 2, "y": (lm["left_hip"]["y"] + lm["right_hip"]["y"]) / 2}
    trunk_conf = avg_visibility(lm["left_shoulder"], lm["right_shoulder"], lm["left_hip"], lm["right_hip"])
    metrics["trunk_compensation"] = {"value": round(angle_from_vertical(mid_shoulder, mid_hip), 2), "confidence": round(trunk_conf, 2)}

    # How far the front hip drifts horizontally from directly over the front
    # ankle — poor hip control lets the pelvis drift medially/laterally
    # instead of staying stacked over the base of support.
    leg_length = distance3(point3(front_hip), point3(front_ankle)) or 1.0
    hip_over_ankle = abs(front_hip["x"] - front_ankle["x"]) / leg_length
    metrics["hip_control"] = {"value": round(hip_over_ankle, 3), "confidence": round(avg_visibility(front_hip, front_ankle), 2)}

    return metrics


def compute_single_leg_balance(frames):
    """Single-leg stand, filmed from the front. Unlike the rep-based
    movements above, a balance hold has no "deepest point" — every metric
    here is a variability/range measure across the whole clip instead of a
    single-frame comparison. The stance leg is auto-detected as whichever
    ankle stays most vertically still (planted); the other leg is assumed
    lifted. balance_asymmetry isn't separately computed — pelvic_drop_balance
    (an uneven pelvis during single-leg stance) already IS a left/right
    asymmetry signal, just not a two-leg-tested comparison.
    """
    usable = [f for f in frames if f["personDetected"] and f["landmarks"]]
    if len(usable) < 6:
        return {}

    required = ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_shoulder", "right_shoulder"]
    series = {name: [] for name in required}
    trunk_angles, confidences = [], []

    for frame in usable:
        lm = _landmark_map(frame["landmarks"])
        if not all(lm.get(name) for name in required):
            continue
        for name in required:
            series[name].append(lm[name])

        l_sh, r_sh, l_hip, r_hip = lm["left_shoulder"], lm["right_shoulder"], lm["left_hip"], lm["right_hip"]
        mid_shoulder = {"x": (l_sh["x"] + r_sh["x"]) / 2, "y": (l_sh["y"] + r_sh["y"]) / 2}
        mid_hip = {"x": (l_hip["x"] + r_hip["x"]) / 2, "y": (l_hip["y"] + r_hip["y"]) / 2}
        trunk_angles.append(angle_from_vertical(mid_shoulder, mid_hip))
        confidences.append(avg_visibility(l_hip, r_hip, lm["left_ankle"], lm["right_ankle"]))

    if len(series["left_hip"]) < 6:
        return {}

    left_ankle_ys = [p["y"] for p in series["left_ankle"]]
    right_ankle_ys = [p["y"] for p in series["right_ankle"]]
    stance_side = "left" if np.var(left_ankle_ys) <= np.var(right_ankle_ys) else "right"
    hip_key, knee_key, ankle_key = f"{stance_side}_hip", f"{stance_side}_knee", f"{stance_side}_ankle"

    overall_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0.0
    hip_widths = [math.hypot(r["x"] - l["x"], r["y"] - l["y"]) or 1.0 for l, r in zip(series["left_hip"], series["right_hip"])]

    metrics = {}

    pelvic_drops = [abs(l["y"] - r["y"]) / w for l, r, w in zip(series["left_hip"], series["right_hip"], hip_widths)]
    metrics["pelvic_drop_balance"] = {"value": round(max(pelvic_drops), 3), "confidence": overall_confidence}

    metrics["trunk_sway_balance"] = {"value": round(float(np.std(trunk_angles)), 2), "confidence": overall_confidence}

    knee_devs = [
        perpendicular_deviation_2d(h, k, a) / w
        for h, k, a, w in zip(series[hip_key], series[knee_key], series[ankle_key], hip_widths)
    ]
    metrics["knee_control"] = {"value": round(max(knee_devs) - min(knee_devs), 3), "confidence": overall_confidence}

    leg_lengths = [distance3(point3(h), point3(a)) or 1.0 for h, a in zip(series[hip_key], series[ankle_key])]
    hip_over_ankle = [abs(h["x"] - a["x"]) / ll for h, a, ll in zip(series[hip_key], series[ankle_key], leg_lengths)]
    metrics["hip_stability"] = {"value": round(float(np.std(hip_over_ankle)), 3), "confidence": overall_confidence}

    return metrics


def compute_sit_to_stand(frames):
    """Standing up from a chair, filmed from the side (the standard clinical
    angle for this test — e.g. the 5x sit-to-stand test). "Weight
    distribution" and "left/right loading asymmetry" genuinely need a
    front/behind view to see which leg bears more load and aren't
    attempted here — see the module docstring's note on single-camera-angle
    scoping (same reasoning as gait's dropped foot-progression metric).
    """
    usable = [f for f in frames if f["personDetected"] and f["landmarks"]]
    if len(usable) < 4:
        return {}

    side = _dominant_side(usable)
    trunk_angles, hip_knee_angles, confidences = [], [], []

    for frame in usable:
        lm = _landmark_map(frame["landmarks"])
        shoulder, hip, knee, ankle = lm.get(f"{side}_shoulder"), lm.get(f"{side}_hip"), lm.get(f"{side}_knee"), lm.get(f"{side}_ankle")

        trunk_angles.append(angle_from_vertical(shoulder, hip) if (shoulder and hip) else None)
        hip_knee_angles.append(joint_angle_3d(point3(hip), point3(knee), point3(ankle)) if (hip and knee and ankle) else None)
        confidences.append(avg_visibility(shoulder, hip, knee, ankle))

    valid_trunk = [a for a in trunk_angles if a is not None]
    valid_hip_knee = [a for a in hip_knee_angles if a is not None]
    if len(valid_trunk) < 3 or len(valid_hip_knee) < 3:
        return {}

    overall_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0.0
    metrics = {}

    # Peak forward lean during the rise — the "momentum strategy" people use
    # to stand up.
    metrics["forward_trunk_lean"] = {"value": round(max(valid_trunk), 2), "confidence": overall_confidence}

    # Hip/knee extension range across the whole clip (seated/most-flexed to
    # standing/most-extended).
    metrics["hip_knee_extension"] = {"value": round(max(valid_hip_knee) - min(valid_hip_knee), 2), "confidence": overall_confidence}

    # How close the FINAL frame's trunk angle is to true vertical — distinct
    # from forward_trunk_lean (that's the peak momentum-phase lean, this is
    # whether they end up fully upright rather than still hunched forward).
    last_trunk_angle = next(a for a in reversed(trunk_angles) if a is not None)
    metrics["return_to_upright"] = {"value": round(last_trunk_angle, 2), "confidence": overall_confidence}

    return metrics


MOVEMENT_COMPUTE_FN = {
    "forward_bending": compute_forward_bending,
    "gait": compute_gait,
    "squat": compute_squat,
    "lunge": compute_lunge,
    "single_leg_balance": compute_single_leg_balance,
    "sit_to_stand": compute_sit_to_stand,
}


def compute_motion_metrics(movement_type, frames):
    """`frames`: list of plain dicts shaped like main.py's FrameLandmarks
    (t, frameIndex, personDetected, landmarks: [{name,x,y,z,visibility,wx,wy,wz}, ...]).
    Returns {} for an unrecognized/not-yet-implemented movement type."""
    fn = MOVEMENT_COMPUTE_FN.get(movement_type)
    if not fn:
        return {}
    return fn(frames)
