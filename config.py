# config.py
import os
from dotenv import load_dotenv

load_dotenv()  # Charge le .env

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Chemins et paramètres
DATA_PATH = "data/healthcare_data.csv"
NROWS = 5000
BATCH_SIZE = 500
VECTOR_INDEX_NAME = "rag2025"