"""
Перевірка стану завдань експорту в Earth Engine.
Запуск:  python scripts/05_task_status.py
"""
import sys, pathlib, warnings
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=DeprecationWarning)

import ee
from src import gee_data as g

g.init()
STATES = {"READY": "⏳ у черзі", "RUNNING": "⚙️ рахується", "COMPLETED": "✅ готово",
          "FAILED": "❌ помилка", "CANCELLED": "🚫 скасовано", "CANCEL_REQUESTED": "🚫 скасовується"}
for t in ee.batch.Task.list()[:12]:
    s = t.status()
    line = f"{s.get('description', ''):<24} {STATES.get(s['state'], s['state'])}"
    if s["state"] == "FAILED":
        line += f"   причина: {s.get('error_message', '')}"
    print(line)