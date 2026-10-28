"""lb_lib shim for the skill scripts: the few names the bundled package modules read, and the readers of data.csv
(the visits given) and visits_to_predict.csv in the working directory. No project path exists here."""
from pathlib import Path
import pandas as pd

LB = Path(".")
DATA_CSV = LB / "data.csv"
TO_PREDICT_CSV = LB / "visits_to_predict.csv"
SENSORS = ("sentinel1", "sentinel2")
BANDS = {"sentinel1": ["VV", "VH", "VV_minus_VH"], "sentinel2": ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NDWI"]}
DECIMALS = {"sentinel2": 3, "sentinel1": 2}


def disp(band):
    return "VV-VH" if band == "VV_minus_VH" else band


def undisp(band):
    return "VV_minus_VH" if band == "VV-VH" else band


def sensor_of(bands):
    return "sentinel1" if "VV" in set(bands) else "sentinel2"


def given():
    """data.csv (band, period_id, date, y) -> (long frame with the package band names, sensor). Values are read
    exactly as written (round-trip float parsing)."""
    d = pd.read_csv(DATA_CSV, float_precision="round_trip")
    sensor = sensor_of(d.band.unique())
    d["band"] = [undisp(b) for b in d["band"]]
    d["date"] = pd.to_datetime(d["date"])
    return d, sensor


def to_predict():
    """visits_to_predict.csv (period_id, date) -> [(period_id, date)] in time order."""
    v = pd.read_csv(TO_PREDICT_CSV, dtype=str)
    return list(zip(v["period_id"], v["date"]))
