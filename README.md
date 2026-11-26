# Farm AI Thesis

**Information system for managing land and production resources of a small farm using AI**
(Master's thesis, Computer Science, 2026–2027)

Інформаційна система управління земельними та виробничими ресурсами фермерського господарства
з використанням технологій штучного інтелекту.

## Idea
A small farm usually has no historical yield records. The system:
1. ranks the farm's fields by productivity using multi-year Sentinel-2 NDVI (clustering);
2. forecasts crop yield with a model trained on open district-level data and transferred to fields,
   with **conformal prediction intervals**;
3. turns the intervals into scenarios and optimizes the crop plan with a **CVaR** risk measure;
4. converts the plan into purchase lists and daily work orders for the staff.

## Stack
Python · Google Earth Engine · pandas · scikit-learn · LightGBM · MAPIE · SHAP · PuLP · SQLAlchemy · FastAPI · Streamlit

## Structure
```
src/          source code (modules M1–M7)
notebooks/    experiments and analysis
data/raw/     downloaded source data
data/processed/ cleaned data and features
docs/         diagrams
```

## Status
🚧 Work in progress.
