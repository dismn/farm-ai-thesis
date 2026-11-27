"""
Крок 9. Перша перевірка ідеї роботи: чи пов'язані ознаки NDVI з урожайністю?
З'єднує data/processed/ndvi_features.csv (крок 7) і data/processed/yields.csv (крок 8)
за кодом району та роком, рахує коефіцієнти кореляції Спірмена (підрозділ 2.3)
і малює графік «ознака NDVI – урожайність».
Запуск:  python scripts/09_first_correlations.py
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURES = ["ndvi_max", "t_max", "amp", "t_sos", "t_eos", "L", "v_up", "A"]
# Районо-роки, які не годяться для навчання: у 2022 р. значна частина цих районів
# була окупована або в зоні бойових дій, а NDVI рахувався для всього району.
EXCLUDE = [("Iziumskyi", 2022), ("Kupianskyi", 2022), ("Chuhuivskyi", 2022), ("Kharkivskyi", 2022)]

feats = pd.read_csv(ROOT / "data/processed/ndvi_features.csv")
ylds = pd.read_csv(ROOT / "data/processed/yields.csv")
data = ylds.merge(feats, on=["district_code", "year"], suffixes=("", "_f"))
for d, y in EXCLUDE:
    data = data[~((data["district"] == d) & (data["year"] == y))]
data.to_csv(ROOT / "data/processed/train_district.csv", index=False)
print(f"Пар «район – рік» з NDVI і врожайністю: {len(data)}  "
      f"(роки: {sorted(data['year'].unique())})")

# Кореляції Спірмена для кожної культури
rows = []
for crop, g in data.groupby("crop"):
    for f in FEATURES:
        rho, p = spearmanr(g[f], g["yield_c_ha"])
        rows.append({"культура": crop, "ознака": f, "rho": round(rho, 2), "p": round(p, 4), "n": len(g)})
res = pd.DataFrame(rows)
table = res.pivot(index="ознака", columns="культура", values="rho").loc[FEATURES]
print("\nКоефіцієнт Спірмена між ознакою NDVI і врожайністю:")
print(table.to_string())
print("\nКількість спостережень:", res.groupby("культура")["n"].first().to_dict())
sig = res[res["p"] < 0.05].sort_values("rho", key=abs, ascending=False)
print("\nСтатистично значущі зв'язки (p < 0,05):")
print(sig.to_string(index=False) if len(sig) else "  немає")
res.to_csv(ROOT / "data/processed/spearman_first.csv", index=False)

# Графік: найсильніша ознака для кожної культури
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
crops = sorted(data["crop"].unique())
fig, axes = plt.subplots(1, len(crops), figsize=(5 * len(crops), 4.3))
for ax, crop in zip(axes, crops):
    g = data[data["crop"] == crop]
    best = res[res["культура"] == crop].set_index("ознака")["rho"].abs().idxmax()
    for i, (yr, gy) in enumerate(g.groupby("year")):
        ax.scatter(gy[best], gy["yield_c_ha"], s=22, color=f"C{i}", label=str(yr))
    rho = res[(res["культура"] == crop) & (res["ознака"] == best)]["rho"].iloc[0]
    ax.set_title(f"{crop}: {best}, ρ = {rho:.2f}".replace(".", ","))
    ax.set_xlabel(best)
    ax.set_ylabel("Урожайність, ц/га")
    ax.grid(alpha=0.3)
    ax.legend(title="Рік", fontsize=8)
fig.tight_layout()
path = ROOT / "docs/figures/first_correlations.png"
fig.savefig(path, dpi=200)
print(f"\nГрафік: {path}")