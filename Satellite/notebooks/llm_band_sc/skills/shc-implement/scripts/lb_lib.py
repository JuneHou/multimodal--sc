"""lb_lib shim for the skill scripts: the few names the bundled package modules read, derived from data.csv in the
working directory (nine periods; the target month is the calendar month after P09). No project path exists here."""
from pathlib import Path
import pandas as pd

LB = Path(".")
FEATURES_CSV = LB / "data.csv"
HER10_SITES = LB / "none.csv"
SENSORS = ("sentinel1", "sentinel2")
BANDS = {"sentinel1": ["VV", "VH", "VV_minus_VH"], "sentinel2": ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NDWI"]}
DECIMALS = {"sentinel2": 3, "sentinel1": 2}
PRE = list(range(1, 10))
TARGET = 10
SD_TOL = 1e-8


def disp(band):
    return "VV-VH" if band == "VV_minus_VH" else band


def undisp(band):
    return "VV_minus_VH" if band == "VV-VH" else band


def _calendar():
    d = pd.read_csv(FEATURES_CSV)
    months = dict(zip(d.period_id, d.month))
    cal = {int(p[1:]): m for p, m in months.items()}
    y, mo = map(int, cal[max(cal)].split("-"))
    mo += 1
    if mo == 13:
        y, mo = y + 1, 1
    cal[max(cal) + 1] = f"{y:04d}-{mo:02d}"
    return cal


CALENDAR = _calendar()


def sensor_of(bands):
    return "sentinel1" if "VV" in set(bands) else "sentinel2"


def wide_from_data(path=FEATURES_CSV):
    """data.csv (band, period_id, month, mean_value) -> the wide frame lb_her_panel.panel_from_wide expects."""
    d = pd.read_csv(path)
    sensor = sensor_of(d.band.unique())
    w = d.pivot(index=["period_id", "month"], columns="band", values="mean_value").reset_index()
    w.columns = [c if c in ("period_id", "month") else f"mean_{undisp(c)}" for c in w.columns]
    w["sensor"] = sensor
    w["site_id"] = "site"
    w["seq"] = w.period_id.str[1:].astype(int)
    return w.sort_values("seq").reset_index(drop=True), sensor
