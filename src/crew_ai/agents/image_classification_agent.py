# src/agents/image_classification_agent.py
from crewai import Agent
from tools.classification_tool import classify_image
from langchain_community.llms import Ollama  # Add this import
import os

os.makedirs("outputs", exist_ok=True)
# Add this: Create Ollama LLM instance
ollama_llm = Ollama(
    model="llama3.2", base_url="http://localhost:11434", temperature=0.7
)
image_classification_agent = Agent(
    role="Image Classification Specialist",
    goal="Accurately classify a chest X-ray image as 'Normal' or 'Anomaly' using the pre-trained ML model and return the result immediately.",
    backstory="""
You are a focused specialist in image classification.
Your ONLY job is to:
1. Use the 'Classify Image with ML Model' tool ONCE with the given image path.
2. Receive the prediction and confidence.
3. Immediately provide the Final Answer with the complete result in this format: 'Classification Result: [Normal/Anomaly] (Confidence: XX.X%)'

You do NOT analyze, interpret, or repeat.
You do NOT delegate to anyone.
After getting the prediction, you STOP and give the Final Answer.
""",
    tools=[classify_image],
    verbose=True,
    allow_delegation=False,
    memory=False,
    # Ce paramètre force l'agent à être plus direct
    max_iter=3,  # Limite le nombre d'itérations pour éviter les boucles
    llm=ollama_llm,
)
