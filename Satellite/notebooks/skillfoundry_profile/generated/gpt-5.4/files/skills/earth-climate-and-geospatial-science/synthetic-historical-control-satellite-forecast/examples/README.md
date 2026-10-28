# Example Data

- `toy_satellite_history.csv` is a single-parcel long-format history with three bands over twelve monthly observations.
- The sequence includes a repeating seasonal pattern plus drift, so a useful forecast should generally prefer differenced matching or a mixed multiband residual model over a plain last-value carry-forward.
