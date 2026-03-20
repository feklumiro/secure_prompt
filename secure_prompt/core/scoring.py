import os
from dotenv import load_dotenv

load_dotenv()
PIPELINE_POLICY = (float(os.getenv("THRESHOLD")), float(os.getenv("THRESHOLD_VECTOR")))
