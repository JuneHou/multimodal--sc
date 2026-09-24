"""Print this parcel's nine months, every band."""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import lb_lib as lb

w, sensor = lb.wide_from_data()
bands = lb.BANDS[sensor]
d = lb.DECIMALS[sensor]
print(f"{'Sentinel-1 radar' if sensor == 'sentinel1' else 'Sentinel-2 optical'}: mean band values of one land parcel, "
      f"{len(w)} consecutive calendar months. Bands: {', '.join(lb.disp(b) for b in bands)}.")
for _, r in w.iterrows():
    print(f"P{int(r.seq):02d} = {r.month}: " + ", ".join(f"{lb.disp(b)} {r[f'mean_{b}']:.{d}f}" for b in bands))
print(f"The month to predict is P{lb.TARGET:02d} = {lb.CALENDAR[lb.TARGET]}. Its values are withheld.")
