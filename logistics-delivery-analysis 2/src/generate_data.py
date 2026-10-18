"""Builds a small, synthetic last-mile delivery dataset.

The numbers are simulated (no real company data) but follow patterns that
show up in real logistics work: longer distances take longer, rain and
rush hour slow things down, and heavier loads cost more.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 2000

zones = ["North", "South", "East", "West", "Central"]
zone = rng.choice(zones, N, p=[0.22, 0.2, 0.2, 0.18, 0.2])
distance_km = np.round(rng.gamma(4.0, 3.0, N) + 1, 1)
weight_kg = np.round(rng.gamma(2.0, 3.5, N) + 0.5, 1)
hour = rng.choice(range(8, 21), N)
rain = rng.random(N) < 0.18
rush = np.isin(hour, [8, 9, 17, 18, 19])

# delivery time (minutes) = base + distance effect + weather + rush + noise
time_min = (
    12 + 3.1 * distance_km + 0.6 * weight_kg
    + 9 * rain + 7 * rush + rng.normal(0, 6, N)
).clip(10)
promised_min = 20 + 3.4 * distance_km
on_time = time_min <= promised_min
cost = np.round(18 + 4.2 * distance_km + 1.1 * weight_kg + 5 * rain
                + rng.normal(0, 8, N), 2).clip(10)
stock_out = rng.random(N) < (0.04 + 0.01 * (zone == "East"))

df = pd.DataFrame({
    "order_id": np.arange(1, N + 1), "zone": zone, "distance_km": distance_km,
    "weight_kg": weight_kg, "hour": hour, "rain": rain.astype(int),
    "rush_hour": rush.astype(int), "delivery_time_min": np.round(time_min, 1),
    "promised_time_min": np.round(promised_min, 1),
    "on_time": on_time.astype(int), "cost_inr": cost,
    "stock_out": stock_out.astype(int),
})
# a few missing values on purpose, so the cleaning step has real work to do
df.loc[rng.choice(N, 25, replace=False), "weight_kg"] = np.nan
df.to_csv("data/deliveries.csv", index=False)
print("saved", df.shape)
