from hip import HealthIntelProviderLocal

full_topics = [
    "xr_elbow",
    "xr_forearm",
    "xr_hand",
    "xr_hummerus",
    "xr_shoulder",
    "xr_wrist",
    "xr_chest",
    "scan_rain",
    "scan_breast",
    "scan_eyes",
    "scan_heart",
]


def generate_mocks():
    return [
        HealthIntelProviderLocal("Gotham General Hospital", ["vqa"], full_topics),
        HealthIntelProviderLocal(
            "Metropolis General Hospital", ["vqa"], ["xr_elbow", "xr_forearm"]
        ),
        HealthIntelProviderLocal("Smallville Medical Center", ["vqa"], ["xr_chest"]),
        HealthIntelProviderLocal("Mercy General Hospital", [], []),
        HealthIntelProviderLocal("St. Mary's Hospital", [], full_topics),
    ]
