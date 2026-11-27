"""
Обробка часових рядів NDVI (підрозділ 2.2 роботи):
  1) відкидання спостережень із малою часткою видимої ріллі (поріг η);
  2) перехід на рівномірну сітку з кроком Δ = 5 діб, лінійна інтерполяція, формула (2.8);
  3) згладжування фільтром Савіцкого–Голея (2.9) з наближенням до верхньої огинаючої (Chen et al., 2004);
  4) фенологічні ознаки з таблиці 2.2 та інтеграл NDVI (2.10).
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

STEP = 5          # Δ, діб
W = 4             # півширина вікна: 2w + 1 = 9 точок ≈ 45 діб
S = 2             # степінь полінома
N_ITER = 3        # кількість ітерацій наближення до верхньої огинаючої
SHARE = 0.2       # частка амплітуди для початку/кінця сезону (як у TIMESAT)
N_BASE = 3        # скільки перших вузлів сітки усереднювати для NDVI_base


def clean(df, min_valid=0.5):
    """Обмежує частку видимої ріллі до 1 і прибирає погані вікна (хмари)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["valid_share"] = df["valid_share"].clip(upper=1)
    return df[df["ndvi"].notna() & (df["valid_share"] >= min_valid)]


def to_grid(dates, values, year):
    """Рівномірна сітка з кроком STEP діб і лінійна інтерполяція (2.8).
    За межами першого/останнього спостереження беремо найближче значення."""
    doy = pd.DatetimeIndex(dates).day_of_year.to_numpy()
    order = np.argsort(doy)
    doy, values = doy[order], np.asarray(values, float)[order]
    start = pd.Timestamp(f"{year}-03-01").day_of_year
    end = pd.Timestamp(f"{year}-10-31").day_of_year
    grid = np.arange(start, end + 1, STEP)
    return grid, np.interp(grid, doy, values)


def smooth_upper(g):
    """Фільтр Савіцкого–Голея з ітеративним наближенням до верхньої огинаючої:
    значення нижче кривої (хмари занижують NDVI) замінюються кривою, і фільтр повторюється."""
    cur = g.copy()
    sm = savgol_filter(cur, 2 * W + 1, S, mode="interp")
    for _ in range(N_ITER):
        cur = np.maximum(g, sm)
        sm = savgol_filter(cur, 2 * W + 1, S, mode="interp")
    return sm


def phenology(grid, sm):
    """Фенологічні ознаки (таблиця 2.2) згладженої кривої."""
    i_max = int(np.argmax(sm))
    ndvi_max = float(sm[i_max])
    base = float(sm[:N_BASE].mean())
    amp = ndvi_max - base
    level = base + SHARE * amp
    above = np.where(sm >= level)[0]
    i_sos, i_eos = int(above[0]), int(above[-1])
    # площа над фоном; ділянки, де крива опускається нижче фону (після жнив ранніх культур), не віднімаємо
    seg = np.clip(sm[i_sos:i_eos + 1] - base, 0, None)
    area = float(np.sum((seg[:-1] + seg[1:]) / 2) * STEP)       # формула (2.10)
    v_up = float(np.max(np.diff(sm[:i_max + 1]))) if i_max > 0 else 0.0
    return {
        "ndvi_max": round(ndvi_max, 4),
        "t_max": int(grid[i_max]),
        "ndvi_base": round(base, 4),
        "amp": round(amp, 4),
        "t_sos": int(grid[i_sos]),
        "t_eos": int(grid[i_eos]),
        "L": int(grid[i_eos] - grid[i_sos]),
        "v_up": round(v_up, 4),
        "A": round(area, 2),
    }


def process_all(df, min_valid=0.5):
    """Для кожної пари «район – рік»: згладжена крива + ознаки.
    Повертає (таблиця кривих, таблиця ознак)."""
    df = clean(df, min_valid)
    curves, feats = [], []
    keys = ["oblast", "district", "district_code", "year"]
    for key, part in df.groupby(keys):
        info = dict(zip(keys, key))
        grid, g = to_grid(part["date"], part["ndvi"], info["year"])
        sm = smooth_upper(g)
        curves.append(pd.DataFrame({**info, "doy": grid,
                                    "ndvi_interp": g.round(4), "ndvi_smooth": sm.round(4)}))
        feats.append({**info, "n_obs": len(part), **phenology(grid, sm)})
    return pd.concat(curves, ignore_index=True), pd.DataFrame(feats)
