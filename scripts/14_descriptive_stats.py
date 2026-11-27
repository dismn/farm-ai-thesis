"""
Крок 14. Статистичний аналіз вихідних даних (підрозділ 2.3):
  • описова статистика врожайності за культурами;
  • перевірка нормальності розподілу критерієм Шапіро–Вілка (2.3.1);
  • матриця кореляцій Спірмена між ознаками та врожайністю (2.3.2) – рисунок 2.2.
Запуск:  python scripts/14_descriptive_stats.py
"""
import sys, pathlib, warnings
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import shapiro, skew, kurtosis

step11 = (ROOT / "scripts/11_first_models.py").read_text(encoding="utf-8").split("# --- Перевірка")[0]
exec(step11)

# 1) Описова статистика і критерій Шапіро–Вілка
rows = []
for crop, g in data.groupby("crop"):
    y = g["yield_c_ha"]
    W, p = shapiro(y)
    rows.append({"культура": crop, "n": len(y), "середнє": y.mean(), "медіана": y.median(),
                 "ст. відхил.": y.std(), "CV, %": 100 * y.std() / y.mean(),
                 "мін": y.min(), "макс": y.max(), "асиметрія": skew(y), "ексцес": kurtosis(y),
                 "W": W, "p (Шапіро–Вілк)": p,
                 "нормальний?": "так" if p >= 0.05 else "ні"})
desc = pd.DataFrame(rows).round(3)
desc.to_csv(P / "descriptive_stats.csv", index=False)
print("Описова статистика врожайності, ц/га (2021–2024):")
print(desc.to_string(index=False))

print("\nСередня урожайність за роками, ц/га:")
print(data.pivot_table(index="year", columns="crop", values="yield_c_ha", aggfunc="mean").round(1).to_string())

# 2) Теплова карта кореляцій Спірмена (усі культури разом, ознаки + урожайність кожної культури)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
LABELS = {"ndvi_max": "NDVI max", "t_max": "t max", "amp": "Amp", "t_sos": "t sos", "v_up": "V up",
          "A": "A (інтеграл)", "ndvi_may": "NDVI трав.", "ndvi_jul": "NDVI лип.",
          "prec_sep_mar": "Опади IX–III", "prec_apr_jun": "Опади IV–VI", "prec_jul_aug": "Опади VII–VIII",
          "t_apr_jun": "T IV–VI", "t_jul_aug": "T VII–VIII", "hot_days_jja": "Дні > 30 °C",
          "gdd10_apr_sep": "Сума t > 10 °C", "rad_apr_aug": "Радіація IV–VIII"}
feat = data.drop_duplicates(["district_code", "year"])[["district_code", "year"] + FEATURES]
wide = data.pivot_table(index=["district_code", "year"], columns="crop", values="yield_c_ha").reset_index()
wide.columns = [c if c in ("district_code", "year") else f"Урож. ({c})" for c in wide.columns]
m = feat.merge(wide, on=["district_code", "year"], how="left")
cols = FEATURES + [c for c in m.columns if c.startswith("Урож.")]
corr = m[cols].corr(method="spearman")
corr.to_csv(P / "spearman_matrix.csv")
names = [LABELS.get(c, c) for c in cols]

fig, ax = plt.subplots(figsize=(11, 9.5))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(cols)), names, rotation=60, ha="right", fontsize=9)
ax.set_yticks(range(len(cols)), names, fontsize=9)
for i in range(len(cols)):
    for j in range(len(cols)):
        v = corr.values[i, j]
        ax.text(j, i, f"{v:.2f}".replace(".", ",").replace("-", "−"), ha="center", va="center",
                fontsize=6.5, color="white" if abs(v) > 0.6 else "black")
ax.axhline(len(FEATURES) - 0.5, color="k", lw=1)
ax.axvline(len(FEATURES) - 0.5, color="k", lw=1)
fig.colorbar(im, ax=ax, fraction=0.04, label="Коефіцієнт Спірмена ρ")
fig.tight_layout()
fig.savefig(ROOT / "docs/figures/Rysunok_2_2_spearman.png", dpi=200)
print(f"\nТеплова карта (рисунок 2.2): docs/figures/Rysunok_2_2_spearman.png")
print(f"Пар «район – рік» для ознак: {len(feat)}")