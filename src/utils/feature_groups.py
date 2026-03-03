"""
feature_groups.py
-----------------
Defines which joint angle features are relevant for each group of ballet positions.

Rationale:
- Leg-focused positions (tendu, arabesque, attitude, passe, etc.) are distinguished
  primarily by leg angles. Arm positions vary with port de bras and add noise.
  Symmetric leg features mean left/right versions of a pose look identical to the model.
- Symmetric-body positions (third, fourth) use sorted arm+leg pairs so neither
  which arm is up nor which leg is front matters.
- Full-body positions (1st, 2nd, 5th) use all directional angles since arm AND
  leg positions together define the position.
"""

# Symmetric leg features — sorted pairs so the model cannot distinguish which leg
# is the working leg. hip_high = more extended (working) leg, hip_low = standing leg.
# Used for one-legged poses so left arabesque == right arabesque, etc.
LEG_FEATURES = [
    "hip_high",
    "hip_low",
    "knee_high",
    "knee_low",
    "ankle_high",
    "ankle_low",
    "hip_abduct_high",
    "hip_abduct_low",
]

# Symmetric leg + symmetric arm angles.
# Both left/right arm and left/right leg distinctions are removed.
# This makes mirrored configurations look identical to the model.
SYMMETRIC_BODY_FEATURES = [
    "hip_high",
    "hip_low",
    "knee_high",
    "knee_low",
    "ankle_high",
    "ankle_low",
    "hip_abduct_high",
    "hip_abduct_low",
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
            "attitude_derriere",
            "attitude_devant",
            "passe",
            "penche",
            "demi_plie",
            "grand_plie",
            "degage_devant",
            "degage_a_la_seconde",
            "degage_derriere",
            "fondu",
            "releve",
            "saute",
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
        ],
    },
}
