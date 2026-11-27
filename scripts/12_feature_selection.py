"""
Крок 12. Відбір ознак за коефіцієнтом збільшення дисперсії VIF (підрозділ 2.3, формула VIF_j = 1/(1 − R_j²)).
Ознаку з найбільшим VIF > 10 вилучаємо й перераховуємо, доки всі VIF ≤ 10.
VIF не використовує врожайність (лише самі ознаки), тому відбір не «підглядає» у відповідь.
Потім ті самі моделі перевіряються «залиш один рік» на відібраних ознаках
і порівнюються з кроком 11 (усі 16 ознак).
Запуск:  python scripts/12_feature_selection.py
"""
import sys, pathlib, warnings
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Беремо дані, ознаки й моделі з кроку 11, щоб усе було однаково
step11 = (ROOT / "scripts/11_first_models.py").read_text(encoding="utf-8").split("# --- Перевірка")[0]
exec(step11)

VIF_MAX = 10


def vif(X):
    """VIF для кожної ознаки: наскільки її можна передбачити з інших ознак."""
    out = {}
    for c in X.columns:
        others = X.drop(columns=c)
        r2 = LinearRegression().fit(others, X[c]).score(others, X[c])
        out[c] = np.inf if r2 >= 1 else 1 / (1 - r2)
    return pd.Series(out)


def select_by_vif(X, limit=VIF_MAX):
    cols, removed = list(X.columns), []
    while True:
        v = vif(X[cols])
        if v.max() <= limit:
            return cols, removed, v
        worst = v.idxmax()
        removed.append(f"{worst} ({v.max():.0f})")
        cols.remove(worst)


def loyo(g, feats, model):
    y_pred = np.full(len(g), np.nan)
    for t in sorted(g["year"].unique()):
        tr, te = (g["year"] != t).values, (g["year"] == t).values
        if model is None:
            y_pred[te] = g.loc[tr, "yield_c_ha"].mean()
        else:
            model.fit(g.loc[tr, feats], g.loc[tr, "yield_c_ha"])
            y_pred[te] = model.predict(g.loc[te, feats])
    return mean_squared_error(g["yield_c_ha"], y_pred) ** 0.5


rows, selected = [], {}
for crop, g in data.groupby("crop"):
    g = g.reset_index(drop=True)
    cols, removed, v = select_by_vif(g[FEATURES])
    selected[crop] = cols
    print(f"\n{crop}: вилучено за VIF > {VIF_MAX}: {', '.join(removed) if removed else 'нічого'}")
    print(f"  лишилось {len(cols)} ознак: {', '.join(cols)}")
    for name, model in models().items():
        rows.append({"культура": crop, "модель": name,
                     "RMSE, усі 16": round(loyo(g, FEATURES, model), 2),
                     f"RMSE, після VIF": round(loyo(g, cols, models()[name]), 2)})

res = pd.DataFrame(rows)
res.to_csv(P / "model_results_vif.csv", index=False)
pd.Series({k: ",".join(v) for k, v in selected.items()}).to_csv(P / "selected_features.csv", header=["features"])
for crop, r in res.groupby("культура"):
    print(f"\n{crop}, помилка RMSE (ц/га), «залиш один рік»:")
    print(r.drop(columns="культура").to_string(index=False))