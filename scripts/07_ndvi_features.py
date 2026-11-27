"""
Крок 7. Очищення, згладжування кривих NDVI районів і фенологічні ознаки.
Бере всі файли data/raw/ndvi_districts_<рік>.csv (а також частини ndvi_districts_<рік>_<область>.csv),
результат:
  data/processed/ndvi_curves.csv    – згладжені криві (район × рік × день року)
  data/processed/ndvi_features.csv  – фенологічні ознаки (район × рік)
  docs/figures/ndvi_<район>.png     – графік кривих одного району за всі роки
Запуск:  python scripts/07_ndvi_features.py            (графік для Лубенського району)
         python scripts/07_ndvi_features.py Myrhorodskyi
"""
import sys, pathlib, glob
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import ndvi_processing as npr

DISTRICT = sys.argv[1] if len(sys.argv) > 1 else "Lubenskyi"

# 1) Читаємо всі файли районів
files = sorted(glob.glob(str(ROOT / "data/raw/ndvi_districts_20*.csv")))
raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
raw = raw.drop_duplicates(["district_code", "date"])
print(f"Файлів: {len(files)}, рядків: {len(raw)}, районів: {raw['district'].nunique()}, "
      f"роки: {sorted(raw['year'].unique())}")

# 2) Обробка
curves, feats = npr.process_all(raw, min_valid=0.5)
out = ROOT / "data/processed"
out.mkdir(parents=True, exist_ok=True)
curves.to_csv(out / "ndvi_curves.csv", index=False)
feats.to_csv(out / "ndvi_features.csv", index=False)
print(f"Пар «район – рік»: {len(feats)}. Збережено в data/processed/")

# 3) Ознаки обраного району
cols = ["year", "n_obs", "ndvi_max", "t_max", "t_sos", "t_eos", "L", "v_up", "A"]
sel = feats[feats["district"] == DISTRICT]
if sel.empty:
    sys.exit(f"Район {DISTRICT} не знайдено. Приклади назв: {', '.join(feats['district'].unique()[:8])}")
print(f"\nОзнаки для {DISTRICT} (t_* – день року):")
print(sel[cols].to_string(index=False))

# 4) Графік: точки – спостереження, лінії – згладжені криві
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
fig, ax = plt.subplots(figsize=(9, 5))
obs = npr.clean(raw[raw["district"] == DISTRICT])
for i, (year, c) in enumerate(curves[curves["district"] == DISTRICT].groupby("year")):
    color = f"C{i}"
    ax.plot(c["doy"], c["ndvi_smooth"], color=color, lw=2, label=str(year))
    o = obs[obs["year"] == year]
    ax.scatter(o["date"].dt.day_of_year, o["ndvi"], color=color, s=14, alpha=0.6)

months = pd.date_range("2021-03-01", "2021-11-01", freq="MS")
ax.set_xticks(months.day_of_year)
ax.set_xticklabels(["бер", "кві", "тра", "чер", "лип", "сер", "вер", "жов", "лис"])
ax.set_ylabel("NDVI")
ax.set_title(f"NDVI орних земель: {DISTRICT}")
ax.grid(alpha=0.3)
ax.legend(title="Рік")
fig.tight_layout()
fig_dir = ROOT / "docs/figures"
fig_dir.mkdir(parents=True, exist_ok=True)
path = fig_dir / f"ndvi_{DISTRICT}.png"
fig.savefig(path, dpi=200)
print(f"\nГрафік: {path}")