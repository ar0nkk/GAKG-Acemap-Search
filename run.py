import sys
import os

# Ensure we run from the project root
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

try:
    from streamlit.web import cli as stcli
except ImportError:
    raise SystemExit("Streamlit is not installed. Run: pip install streamlit")

# Delegate to Streamlit CLI programmatically
sys.argv = ["streamlit", "run", "app.py"]
sys.exit(stcli.main())
