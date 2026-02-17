import os
from dotenv import load_dotenv

load_dotenv()
ML_JAIL_SCORE = float(os.getenv("ML_JAIL_SCORE"))
VECTOR_JAIL_SCORE = float(os.getenv("VECTOR_JAIL_SCORE"))
PIPELINE_POLICY = (float(os.getenv("PP1")), float(os.getenv("PP2")))  # (~89.5%, ~97.9%)
