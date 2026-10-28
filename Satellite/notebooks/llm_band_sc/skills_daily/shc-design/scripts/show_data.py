"""Print this parcel's visits given, every band, and the visits to predict."""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import lb_lib as lb

d, sensor = lb.given()
bands = lb.BANDS[sensor]
dec = lb.DECIMALS[sensor]
w = d.pivot(index=["period_id", "date"], columns="band", values="y").reset_index().sort_values("date")
print(f"{'Sentinel-1 radar' if sensor == 'sentinel1' else 'Sentinel-2 optical'}: mean band values of one land parcel at "
      f"{len(w)} satellite visits, {w['date'].iloc[0]:%Y-%m-%d} to {w['date'].iloc[-1]:%Y-%m-%d}. "
      f"Bands: {', '.join(lb.disp(b) for b in bands)}.")
for _, r in w.iterrows():
    print(f"{r.period_id} = {r.date:%Y-%m-%d}: " + ", ".join(f"{lb.disp(b)} {r[b]:.{dec}f}" for b in bands))
tp = lb.to_predict()
print(f"The {len(tp)} visits to predict, listed in visits_to_predict.csv, are below. Their values are withheld.")
for p, day in tp:
    print(f"{p} = {day}")
