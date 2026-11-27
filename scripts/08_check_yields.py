"""
Крок 8. Перевірка таблиці врожайності Держстату (data/yields/yields_raw.csv).
Кожен рядок перевіряється трьома способами:
  1) арифметика: урожайність ≈ валовий збір / площа (ловить помилки при переписуванні цифр);
  2) сума площ районів ≈ площа області;
  3) назва району є в даних NDVI (переклад української назви латиницею).
Результат: data/processed/yields.csv – чиста таблиця, з'єднана з кодами районів GAUL.
Запуск:  python scripts/08_check_yields.py
"""
import sys, pathlib, glob
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd

TOL = 0.15       # допустима розбіжність урожайності через округлення, ц/га
TOL_SUM = 0.02   # допустима розбіжність суми площ районів і області (2 %)

# Транслітерація за Постановою КМУ № 55 від 27.01.2010 (як у назвах GAUL)
TR = {"а": "a", "б": "b", "в": "v", "г": "h", "ґ": "g", "д": "d", "е": "e", "є": "ie", "ж": "zh",
      "з": "z", "и": "y", "і": "i", "ї": "i", "й": "i", "к": "k", "л": "l", "м": "m", "н": "n",
      "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts",
      "ч": "ch", "ш": "sh", "щ": "shch", "ь": "", "ю": "iu", "я": "ia", "'": "", "’": "", "-": "-"}
# Райони, перейменовані після 2020 р. (у GAUL лишилася стара назва)
ALIASES = {"Берестинський": "Krasnohradskyi", "Самарівський": "Novomoskovskyi"}


def translit(name):
    name = name.strip().replace("район", "").strip()
    if name in ALIASES:
        return ALIASES[name]
    out = "".join(TR.get(ch, TR.get(ch.lower(), ch)) for ch in name)
    return "-".join(part[:1].upper() + part[1:] for part in out.split("-"))


df = pd.read_csv(ROOT / "data/yields/yields_raw.csv", encoding="utf-8-sig")
df["yield_c_ha"] = pd.to_numeric(df["yield_c_ha"], errors="coerce")      # "к" -> порожньо
df["area_ha"] = pd.to_numeric(df["area_ha"], errors="coerce")
df["production"] = pd.to_numeric(df["production"], errors="coerce")
print(f"Рядків: {len(df)}; конфіденційних (к) або порожніх: {df['yield_c_ha'].isna().sum()}")

problems = 0

# 1) Арифметика: т -> ×10 (бо 1 т = 10 ц), ц -> як є
factor = df["prod_unit"].map({"т": 10, "ц": 1})
if factor.isna().any():
    print("❌ Невідома одиниця в prod_unit:", df.loc[factor.isna(), "prod_unit"].unique())
    problems += 1
calc = df["production"] * factor / df["area_ha"]
bad = (calc - df["yield_c_ha"]).abs() > TOL
for _, r in df[bad].iterrows():
    problems += 1
    print(f"❌ {r.oblast_ua}, {r.district_ua}, {r.year}, {r.crop}: у таблиці {r.yield_c_ha}, "
          f"а збір/площа = {r.production * factor[_] / r.area_ha:.1f}. Перевірте цифри!")

# 2) Сума площ районів = площа області
for (ob, yr, cr), g in df.groupby(["oblast_ua", "year", "crop"]):
    total = g.loc[g["district_ua"] == "ОБЛАСТЬ", "area_ha"]
    parts = g.loc[g["district_ua"] != "ОБЛАСТЬ", "area_ha"]
    if total.empty or parts.isna().any():
        continue                                   # немає підсумку або є (к) – не перевіряємо
    if abs(parts.sum() - total.iloc[0]) / total.iloc[0] > TOL_SUM:
        problems += 1
        print(f"❌ {ob}, {yr}, {cr}: сума районів {parts.sum():.0f} ≠ область {total.iloc[0]:.0f}")

# 3) Відповідність назвам у NDVI
ndvi_files = glob.glob(str(ROOT / "data/raw/ndvi_districts_20*.csv"))
ndvi = pd.concat([pd.read_csv(f, usecols=["oblast", "district", "district_code"]) for f in ndvi_files])
names = ndvi.drop_duplicates("district_code")
dist = df[df["district_ua"] != "ОБЛАСТЬ"].copy()
dist["district"] = dist["district_ua"].map(translit)
dist = dist.merge(names, on="district", how="left")
for n in dist.loc[dist["district_code"].isna(), "district_ua"].unique():
    problems += 1
    print(f"❌ Район «{n}» ({translit(n)}) не знайдено серед районів NDVI. Додайте його в ALIASES.")

# Підсумок
out = dist.dropna(subset=["district_code", "yield_c_ha"])
out = out[["oblast", "district", "district_code", "year", "crop", "yield_c_ha", "area_ha", "data_status"]]
out["district_code"] = out["district_code"].astype(int)
(ROOT / "data/processed").mkdir(parents=True, exist_ok=True)
out.to_csv(ROOT / "data/processed/yields.csv", index=False)

print("\nСкільки є значень урожайності (район × рік):")
print(out.pivot_table(index="year", columns="crop", values="yield_c_ha", aggfunc="count")
      .fillna(0).astype(int).to_string())
print("\n✅ Помилок не знайдено." if problems == 0 else f"\n⚠️ Знайдено проблем: {problems}")