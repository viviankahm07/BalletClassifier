"""
feature_groups.py
-----------------
Defines which joint angle features are relevant for each group of ballet positions.

Rationale:
- Leg-focused positions (tendu, arabesque, grand_battement, etc.) are distinguished
  primarily by leg angles. Arm positions vary with port de bras and add noise.
- Symmetric-body positions (third, fourth) use sorted arm pairs so the model
  cannot distinguish left-arm-up from right-arm-up — both count as the same position.
- Full-body positions (1st, 2nd, 5th, passe, attitude) use all directional angles
  since both arms and legs define the position.
"""

# Leg-only features — ignore arm angles entirely
LEG_FEATURES = [
    "left_knee",
    "right_knee",
    "left_hip",
    "right_hip",
    "left_ankle",
    "right_ankle",
    "left_hip_abduct",
    "right_hip_abduct",
]

# Leg angles + symmetric arm angles (elbow_high/low, shoulder_high/low).
# "high" = whichever arm is more extended; "low" = the more bent one.
# This makes mirrored arm configurations look identical to the model.
SYMMETRIC_BODY_FEATURES = [
    "left_knee",
    "right_knee",
    "left_hip",
    "right_hip",
    "left_ankle",
    "right_ankle",
    "left_hip_abduct",
    "right_hip_abduct",
    "elbow_high",
    "elbow_low",
    "shoulder_high",
    "shoulder_low",
]

# All 12 directional joint angles (preserves left/right distinction)
ALL_FEATURES = [
    "left_knee",
    "right_knee",
    "left_hip",
    "right_hip",
    "left_elbow",
    "right_elbow",
    "left_shoulder",
    "right_shoulder",
    "left_ankle",
    "right_ankle",
    "left_hip_abduct",
    "right_hip_abduct",
]

FEATURE_GROUPS = {
    "leg_focused": {
        "features": LEG_FEATURES,
        "classes": [
            "tendu_devant",
            "tendu_a_la_seconde",
            "tendu_derriere",
            "arabesque",
            "grand_battement",
            "developpe",
            "penche",
        ],
    },
    "symmetric_body": {
        "features": SYMMETRIC_BODY_FEATURES,
        "classes": [
            "third_position",
            "fourth_position",
        ],
    },
    "full_body": {
        "features": ALL_FEATURES,
        "classes": [
            "first_position",
            "second_position",
            "fifth_position",
            "passe",
            "attitude_derriere",
            "attitude_devant",
        ],
    },
}
