from hip import HealthIntelProviderLocal


def generate_mocks():
    return [
        HealthIntelProviderLocal("Gotham General Hospital", []),
        HealthIntelProviderLocal("Metropolis General Hospital", []),
        HealthIntelProviderLocal("Smallville Medical Center", []),
        HealthIntelProviderLocal("Mercy General Hospital", []),
        HealthIntelProviderLocal("St. Mary's Hospital", []),
    ]
