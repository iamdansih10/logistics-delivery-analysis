"""Strategic data exploration for the logistics scenario.

Steps: load -> clean -> KPIs -> EDA charts -> regression -> clustering
-> a simple route-ordering demo. Results go to /outputs.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/deliveries.csv")

# ---- 1. cleaning ----
df["weight_kg"] = df["weight_kg"].fillna(df["weight_kg"].median())
df = df.drop_duplicates("order_id")

# ---- 2. KPIs ----
kpi = {
    "on_time_rate_pct": round(df.on_time.mean() * 100, 1),
    "avg_delivery_time_min": round(df.delivery_time_min.mean(), 1),
    "avg_cost_per_delivery_inr": round(df.cost_inr.mean(), 1),
    "cost_per_km_inr": round(df.cost_inr.sum() / df.distance_km.sum(), 2),
    "stock_out_rate_pct": round(df.stock_out.mean() * 100, 1),
}
zone_kpi = df.groupby("zone").agg(
    orders=("order_id", "count"),
    on_time_pct=("on_time", lambda s: round(s.mean() * 100, 1)),
    avg_time=("delivery_time_min", "mean"),
    avg_cost=("cost_inr", "mean"),
).round(1)
rain_effect = df.groupby("rain").delivery_time_min.mean().round(1).to_dict()
rush_effect = df.groupby("rush_hour").on_time.mean().mul(100).round(1).to_dict()

# ---- 3. EDA charts ----
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].hist(df.delivery_time_min, bins=30, color="#2b6cb0")
ax[0].set(title="Delivery time distribution", xlabel="minutes")
ax[1].scatter(df.distance_km, df.delivery_time_min, s=6, alpha=.4)
ax[1].set(title="Distance vs delivery time", xlabel="km", ylabel="min")
zone_kpi.on_time_pct.plot.bar(ax=ax[2], color="#2f855a")
ax[2].set(title="On-time % by zone", ylabel="%")
plt.tight_layout(); plt.savefig("outputs/eda_overview.png", dpi=130)

# ---- 4. predictive model: delivery time ----
X = df[["distance_km", "weight_kg", "rain", "rush_hour"]]
y = df["delivery_time_min"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1)
model = LinearRegression().fit(Xtr, ytr)
pred = model.predict(Xte)
model_res = {
    "r2": round(r2_score(yte, pred), 3),
    "mae_min": round(mean_absolute_error(yte, pred), 2),
    "coefficients": {c: round(v, 2) for c, v in zip(X.columns, model.coef_)},
}

# ---- 5. clustering: group orders into delivery profiles ----
feats = StandardScaler().fit_transform(df[["distance_km", "weight_kg", "cost_inr"]])
df["cluster"] = KMeans(n_clusters=3, n_init=10, random_state=1).fit_predict(feats)
clusters = df.groupby("cluster").agg(
    orders=("order_id", "count"), distance=("distance_km", "mean"),
    weight=("weight_kg", "mean"), cost=("cost_inr", "mean"),
    on_time_pct=("on_time", lambda s: s.mean() * 100)).round(1)

# ---- 6. route optimisation demo (nearest neighbour heuristic) ----
rng = np.random.default_rng(7)
stops = rng.uniform(0, 20, (10, 2))
depot = np.array([10.0, 10.0])

def route_len(order):
    pts = [depot] + [stops[i] for i in order] + [depot]
    return sum(np.linalg.norm(pts[i] - pts[i + 1]) for i in range(len(pts) - 1))

naive = list(range(10))
left, cur, nn = set(range(10)), depot, []
while left:
    nxt = min(left, key=lambda i: np.linalg.norm(stops[i] - cur))
    nn.append(nxt); left.remove(nxt); cur = stops[nxt]
route = {"naive_km": round(route_len(naive), 1), "nearest_neighbour_km": round(route_len(nn), 1)}
route["saving_pct"] = round((1 - route["nearest_neighbour_km"] / route["naive_km"]) * 100, 1)

out = {"kpi": kpi, "rain_avg_time": rain_effect, "rush_on_time_pct": rush_effect,
       "model": model_res, "route_demo": route}
json.dump(out, open("outputs/results.json", "w"), indent=2)
zone_kpi.to_csv("outputs/zone_kpis.csv"); clusters.to_csv("outputs/clusters.csv")
print(json.dumps(out, indent=2)); print(zone_kpi); print(clusters)
