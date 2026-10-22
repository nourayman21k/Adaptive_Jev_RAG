import sys
from pathlib import Path

# the app's modules (common.py) live one level up, next to streamlit_app.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
