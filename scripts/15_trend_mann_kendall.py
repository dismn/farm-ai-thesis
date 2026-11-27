"""
Крок 15. Тренд урожайності (підрозділ 2.3.3): тест Манна–Кендалла та оцінка нахилу Сена
на довгих рядах обласної врожайності Держстату 1995–2024 рр. (господарства всіх категорій).
Для районних даних 2021–2024 рр. (4 точки) тест не має сенсу, тому тренд оцінюється
на 30-річних обласних рядах, а висновок переноситься на районну вибірку.
Дані: data/yields/oblast_long_series.csv (кожен рядок – з посиланням на таблицю Держстату).
Запуск:  python scripts/15_trend_mann_kendall.py
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd
import pymannkendall as mk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv(ROOT / "data/yields/oblast_long_series.csv", encoding="utf-8-sig")
rows = []
for (ob, crop), g in df.groupby(["oblast_ua", "crop"]):
    g = g.sort_values("year")
    for label, part in [("1995–2024", g), ("2010–2024", g[g["year"] >= 2010])]:
        r = mk.original_test(part["yield_c_ha"].to_numpy())
        rows.append({"область": ob, "культура": crop, "період": label, "n": len(part),
                     "тренд": {"increasing": "зростає", "decreasing": "спадає",
                               "no trend": "немає"}[r.trend],
                     "τ Кендалла": round(r.Tau, 3), "p": round(r.p, 5),
                     "нахил Сена, ц/га за рік": round(r.slope, 2)})
res = pd.DataFrame(rows)
res.to_csv(ROOT / "data/processed/mann_kendall.csv", index=False)
print(res.to_string(index=False))

# Графік: ряди і лінія тренду Сена
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
crops = sorted(df["crop"].unique())
fig, axes = plt.subplots(1, len(crops), figsize=(12, 4.2))
for ax, crop in zip(axes, crops):
    for i, (ob, g) in enumerate(df[df["crop"] == crop].groupby("oblast_ua")):
        g = g.sort_values("year")
        r = mk.original_test(g["yield_c_ha"].to_numpy())
        ax.plot(g["year"], g["yield_c_ha"], "o-", ms=3, color=f"C{i}", label=ob)
        x = np.arange(len(g))
        ax.plot(g["year"], r.intercept + r.slope * x, "--", color=f"C{i}", lw=1)
    ax.set_title(f"{crop[:1].upper() + crop[1:]}: тренд Сена (штрих)")
    ax.set_xlabel("Рік")
    ax.set_ylabel("Урожайність, ц/га")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "docs/figures/trend_mann_kendall.png", dpi=200)
print("\nГрафік: docs/figures/trend_mann_kendall.png")