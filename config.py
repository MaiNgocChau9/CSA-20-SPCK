import os

ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, "data")
IMG_DIR = os.path.join(ROOT, "assets")

# Default dataset paths (existing files in workspace)
TRAIN_PATH = os.path.join(DATA_DIR, "GlobalWeatherRepository.csv")
HEALTH_PATH = os.path.join(DATA_DIR, "global_climate_health_impact_tracker_2015_2025.csv")
SEATTLE_PATH = os.path.join(DATA_DIR, "seattle-weather.csv")

os.makedirs(IMG_DIR, exist_ok=True)
