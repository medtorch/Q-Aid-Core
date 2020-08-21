from hip import HealthIntelProviderLocal

full_topics = [
    "xr_elbow",
    "xr_forearm",
    "xr_hand",
    "xr_hummerus",
    "xr_shoulder",
    "xr_wrist",
    "xr_chest",
    "scan_brain",
    "scan_breast",
    "scan_eyes",
    "scan_heart",
]


def generate_mocks():
    return [
        HealthIntelProviderLocal("Gotham General Hospital", {"vqa": ["scan_brain"]}),
        HealthIntelProviderLocal(
            "Metropolis General Hospital",
            {"vqa": ["scan_brain", "xr_chest"], "segmentation": ["scan_brain"]},
        ),
        HealthIntelProviderLocal("Smallville Medical Center", {"vqa": ["xr_chest"]}),
        HealthIntelProviderLocal("Mercy General Hospital", {"vqa": ["xr_chest"]}),
        HealthIntelProviderLocal(
            "St. Mary's Hospital", {"segmentation": ["scan_brain"]}
        ),
    ]
