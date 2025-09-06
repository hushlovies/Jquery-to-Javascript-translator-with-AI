import warnings
import logging
import sys

# Suppress all Streamlit context warnings
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
warnings.filterwarnings("ignore", message=".*Session state does not function.*")

# Redirect Streamlit logger
logging.getLogger("streamlit").setLevel(logging.ERROR)