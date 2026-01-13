import sys
import os
from streamlit.web import cli as stcli

# Ensure we run from the project root
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

# Delegate to Streamlit CLI programmatically
sys.argv = ["streamlit", "run", "app.py"]
sys.exit(stcli.main())
