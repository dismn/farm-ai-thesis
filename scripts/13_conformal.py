"""
Крок 13. Інтервали прогнозу врожайності з гарантованим покриттям (підрозділ 2.6), α = 0,1.
Два методи:
  • розщеплене конформне прогнозування (split conformal) навколо точкового прогнозу;
  • конформна квантильна регресія CQR (LightGBM з квантильною функцією втрат).
Калібрувальна вибірка D2 формується з ЦІЛИХ років (як вимагає підрозділ 2.6): для кожного
навчального року модель навчається на інших навчальних роках і прогнозує цей рік; оцінки
невідповідності з усіх навчальних років об'єднуються (крос-конформний варіант із поділом за роками).
Підсумкове покриття PICP і ширина MPIW перевіряються «залиш один рік»: рік, для якого будується
інтервал, не бере участі ні в навчанні, ні в калібруванні.
Запуск:  python scripts/13_conformal.py
"""
import sys, pathlib, warnings
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

step11 = (ROOT / "scripts/11_first_models.py").read_text(encoding="utf-8").split("# --- Перевірка")[0]
exec(step11)

ALPHA = 0.1
BEST = {"кукурудза": "Ridge", "пшениця": "LightGBM", "соняшник": "Випадковий ліс"}  # з кроку 11


def quantile_models():
    common = dict(n_estimators=300, learning_rate=0.05, num_leaves=7, min_child_samples=5,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
    return (LGBMRegressor(objective="quantile", alpha=ALPHA / 2, **common),
            LGBMRegressor(objective="quantile", alpha=1 - ALPHA / 2, **common))


def conformal_q(scores, alpha=ALPHA):
    """Значення на місці ⌈(n+1)(1−α)⌉ у впорядкованому ряді оцінок (формули 2.32, 2.37)."""
    s = np.sort(np.asarray(scores))
    k = int(np.ceil((len(s) + 1) * (1 - alpha)))
    return np.inf if k > len(s) else s[k - 1]


def split_conformal(g, train_years, test_mask, model_name):
    """Інтервал прогноз ± Q, де Q – квантиль модулів залишків на калібрувальних роках."""
    scores = []
    for cal_year in train_years:
        fit_years = [y for y in train_years if y != cal_year]
        m = models()[model_name]
        fit, cal = g["year"].isin(fit_years).values, (g["year"] == cal_year).values
        m.fit(g.loc[fit, FEATURES], g.loc[fit, "yield_c_ha"])
        scores += list(np.abs(g.loc[cal, "yield_c_ha"] - m.predict(g.loc[cal, FEATURES])))
    q = conformal_q(scores)
    m = models()[model_name]
    tr = g["year"].isin(train_years).values
    m.fit(g.loc[tr, FEATURES], g.loc[tr, "yield_c_ha"])
    pred = m.predict(g.loc[test_mask, FEATURES])
    return pred, pred - q, pred + q


def cqr(g, train_years, test_mask):
    """Конформна квантильна регресія: [q_lo − Q, q_hi + Q], E = max(q_lo − y, y − q_hi)."""
    scores = []
    for cal_year in train_years:
        fit_years = [y for y in train_years if y != cal_year]
        lo, hi = quantile_models()
        fit, cal = g["year"].isin(fit_years).values, (g["year"] == cal_year).values
        lo.fit(g.loc[fit, FEATURES], g.loc[fit, "yield_c_ha"])
        hi.fit(g.loc[fit, FEATURES], g.loc[fit, "yield_c_ha"])
        y = g.loc[cal, "yield_c_ha"].to_numpy()
        scores += list(np.maximum(lo.predict(g.loc[cal, FEATURES]) - y, y - hi.predict(g.loc[cal, FEATURES])))
    q = conformal_q(scores)
    lo, hi = quantile_models()
    tr = g["year"].isin(train_years).values
    lo.fit(g.loc[tr, FEATURES], g.loc[tr, "yield_c_ha"])
    hi.fit(g.loc[tr, FEATURES], g.loc[tr, "yield_c_ha"])
    L, U = lo.predict(g.loc[test_mask, FEATURES]) - q, hi.predict(g.loc[test_mask, FEATURES]) + q
    return (L + U) / 2, L, U


rows, summary = [], []
for crop, g in data.groupby("crop"):
    g = g.reset_index(drop=True)
    years = sorted(g["year"].unique())
    for method in ["Split conformal", "CQR"]:
        for t in years:
            te = (g["year"] == t).values
            train_years = [y for y in years if y != t]
            if method == "Split conformal":
                p, L, U = split_conformal(g, train_years, te, BEST[crop])
            else:
                p, L, U = cqr(g, train_years, te)
            rows.append(pd.DataFrame({"crop": crop, "method": method, "year": t,
                                      "district": g.loc[te, "district"].values,
                                      "y": g.loc[te, "yield_c_ha"].values, "pred": p, "lo": L, "hi": U}))
out = pd.concat(rows, ignore_index=True)
out["inside"] = (out["y"] >= out["lo"]) & (out["y"] <= out["hi"])
out["width"] = out["hi"] - out["lo"]
out.to_csv(P / "conformal_intervals.csv", index=False)

print(f"Ціль: покриття PICP ≥ {1 - ALPHA:.0%} (α = {ALPHA})\n")
tab = (out.groupby(["crop", "method"])
       .agg(PICP=("inside", "mean"), MPIW=("width", "mean"), n=("y", "size")).round(3))
tab["PICP"] = (tab["PICP"] * 100).round(1).astype(str) + " %"
tab["MPIW"] = tab["MPIW"].round(1).astype(str) + " ц/га"
print(tab.to_string())

print("\nПокриття окремо за роками (чи не провалюється якийсь рік):")
by_year = out.groupby(["crop", "method", "year"])["inside"].mean().unstack().mul(100).round(0)
print(by_year.to_string())

# Приклад для Лубенського району
ex = out[(out["district"] == "Lubenskyi")].copy()
if len(ex):
    print("\nЛубенський район (прогноз з інтервалом, ц/га):")
    ex = ex.assign(інтервал=ex["lo"].round(1).astype(str) + " – " + ex["hi"].round(1).astype(str))
    print(ex[["crop", "method", "year", "y", "інтервал", "inside"]].to_string(index=False))

# Графік: інтервали для кукурудзи (split conformal)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, method in zip(axes, ["Split conformal", "CQR"]):
    q = out[(out["crop"] == "кукурудза") & (out["method"] == method)].sort_values("pred").reset_index(drop=True)
    ax.vlines(q.index, q["lo"], q["hi"], color="0.6", lw=2, label="інтервал 90 %")
    ax.scatter(q.index, q["y"], c=np.where(q["inside"], "C2", "C3"), s=16, zorder=3, label="факт")
    picp, mpiw = q["inside"].mean() * 100, q["width"].mean()
    ax.set_title(f"Кукурудза, {method}: PICP = {picp:.0f} %, MPIW = {mpiw:.0f} ц/га")
    ax.set_xlabel("Райони-роки (за зростанням прогнозу)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Урожайність, ц/га")
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "docs/figures/conformal_corn.png", dpi=200)
print("\nГрафік: docs/figures/conformal_corn.png")