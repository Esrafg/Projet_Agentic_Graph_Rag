# agents/explainability_agent.py
from langchain_community.llms import Ollama
from crewai import Agent
from tools.explainability_tools import explain_with_gradcam

# LLM local
llm = Ollama(model="llama3.2", base_url="http://localhost:11434")


import os

os.makedirs("outputs", exist_ok=True)

explainability_agent = Agent(
    role="Medical Imaging Explainability Specialist",
    goal="Provide clear, trustworthy Grad-CAM visualizations to explain deep learning predictions on chest X-rays",
    backstory="""
    You are a radiologist-AI collaboration expert specialized in interpretable deep learning.
    You use Grad-CAM — the gold standard in medical imaging explainability — to generate precise heatmaps
    showing exactly where in the lung fields the model focused when making a diagnosis.
    You never use unreliable methods like LIME for final clinical explanations.
    Your visualizations are publication-ready and trusted by doctors.
    """,
    tools=[explain_with_gradcam],
    verbose=True,
    allow_delegation=False,
    memory=False,  # optional: remembers past cases
    llm=llm,
)
