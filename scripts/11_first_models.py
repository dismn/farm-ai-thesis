"""
Крок 11. Перші моделі прогнозу врожайності (підрозділ 2.5) з перевіркою «залиш один рік»
(leave-one-year-out): модель вчиться на всіх роках, крім одного, і прогнозує той рік,
якого не бачила. Так імітується реальна ситуація: прогноз на новий сезон.
Ознаки: фенологія NDVI (крок 7) + NDVI у травні та липні + погода NASA POWER (крок 10).
Результат: data/processed/model_results.csv, docs/figures/models_pred_vs_true.png
Запуск:  python scripts/11_first_models.py
"""
import sys, pathlib, warnings
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = ROOT / "data/processed"
NDVI_F = ["ndvi_max", "t_max", "amp", "t_sos", "v_up", "A", "ndvi_may", "ndvi_jul"]
WEATHER_F = ["prec_sep_mar", "prec_apr_jun", "prec_jul_aug", "t_apr_jun", "t_jul_aug",
             "hot_days_jja", "gdd10_apr_sep", "rad_apr_aug"]
FEATURES = NDVI_F + WEATHER_F

# --- Дані ---------------------------------------------------------------------
train = pd.read_csv(P / "train_district.csv")
weather = pd.read_csv(P / "weather_features.csv")[["district_code", "year"] + WEATHER_F]
curves = pd.read_csv(P / "ndvi_curves.csv")
# NDVI у травні (день року 121–151) і липні (182–212): середнє згладженої кривої
month = lambda a, b: (curves[(curves["doy"] >= a) & (curves["doy"] <= b)]
                      .groupby(["district_code", "year"])["ndvi_smooth"].mean())
extra = pd.DataFrame({"ndvi_may": month(121, 151), "ndvi_jul": month(182, 212)}).reset_index()
data = train.merge(extra, on=["district_code", "year"]).merge(weather, on=["district_code", "year"])
print(f"Спостережень: {len(data)}; ознак: {len(FEATURES)}")


def models():
    return {
        "Середнє (база)": None,
        "МНК": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))),
        "Випадковий ліс": RandomForestRegressor(n_estimators=500, min_samples_leaf=3, random_state=0),
        "LightGBM": LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=7,
                                  min_child_samples=5, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, random_state=0, verbose=-1),
    }


# --- Перевірка «залиш один рік» ---------------------------------------------------
results, preds = [], []
for crop, g in data.groupby("crop"):
    g = g.reset_index(drop=True)
    years = sorted(g["year"].unique())
    for name, model in models().items():
        y_pred = np.full(len(g), np.nan)
        for test_year in years:
            tr, te = g["year"] != test_year, g["year"] == test_year
            if model is None:                       # база: середня врожайність інших років
                y_pred[te] = g.loc[tr, "yield_c_ha"].mean()
            else:
                model.fit(g.loc[tr, FEATURES], g.loc[tr, "yield_c_ha"])
                y_pred[te] = model.predict(g.loc[te, FEATURES])
        y = g["yield_c_ha"].to_numpy()
        results.append({"культура": crop, "модель": name, "n": len(g),
                        "RMSE": round(mean_squared_error(y, y_pred) ** 0.5, 2),
                        "MAE": round(mean_absolute_error(y, y_pred), 2),
                        "R2": round(r2_score(y, y_pred), 3)})
        preds.append(pd.DataFrame({"crop": crop, "model": name, "year": g["year"],
                                   "district": g["district"], "y": y, "y_pred": y_pred}))

res = pd.DataFrame(results)
res.to_csv(P / "model_results.csv", index=False)
pd.concat(preds).to_csv(P / "model_predictions.csv", index=False)
for crop, r in res.groupby("культура"):
    print(f"\n{crop} (n = {r['n'].iloc[0]}), помилка в ц/га:")
    print(r.drop(columns=["культура", "n"]).to_string(index=False))

# --- Найважливіші ознаки (LightGBM, навчений на всіх даних) --------------------
print("\nТоп-5 ознак LightGBM (скільки разів ознака використана для розбиття):")
for crop, g in data.groupby("crop"):
    m = models()["LightGBM"].fit(g[FEATURES], g["yield_c_ha"])
    imp = pd.Series(m.feature_importances_, FEATURES).sort_values(ascending=False).head(5)
    print(f"  {crop}: " + ", ".join(f"{k} ({v})" for k, v in imp.items()))

# --- Графік: прогноз проти факту для найкращої моделі кожної культури ----------
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
pr = pd.concat(preds)
crops = sorted(data["crop"].unique())
fig, axes = plt.subplots(1, len(crops), figsize=(5 * len(crops), 4.5))
for ax, crop in zip(axes, crops):
    best = res[(res["культура"] == crop) & (res["модель"] != "Середнє (база)")].sort_values("RMSE").iloc[0]
    q = pr[(pr["crop"] == crop) & (pr["model"] == best["модель"])]
    for i, (yr, qy) in enumerate(q.groupby("year")):
        ax.scatter(qy["y"], qy["y_pred"], s=22, color=f"C{i}", label=str(yr))
    lo, hi = q[["y", "y_pred"]].min().min() - 3, q[["y", "y_pred"]].max().max() + 3
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_title(f"{crop}: {best['модель']}, RMSE = {best['RMSE']:.1f} ц/га".replace(".", ","))
    ax.set_xlabel("Фактична урожайність, ц/га")
    ax.set_ylabel("Прогноз, ц/га")
    ax.grid(alpha=0.3)
    ax.legend(title="Рік", fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "docs/figures/models_pred_vs_true.png", dpi=200)
print("\nГрафік: docs/figures/models_pred_vs_true.png")