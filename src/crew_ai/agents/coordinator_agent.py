# src/agents/coordinator_agent.py
from crewai import Agent
import os
from langchain_community.llms import Ollama

os.makedirs("outputs", exist_ok=True)
ollama_llm = Ollama(
    model="llama3.2", base_url="http://localhost:11434", temperature=0.7
)
coordinator_agent = Agent(
    role="Medical AI Workflow Coordinator",
    goal="""
    Orchestrate a complete diagnostic analysis of chest X-ray images by intelligently 
    delegating tasks to specialized agents: classification and explainability.
    Produce a clear, structured final report for clinicians.
    """,
    backstory="""
    You are an experienced AI clinical coordinator in a hospital radiology department.
    You understand medical workflows and know when to classify an image first, 
    then request explainability only if an anomaly is detected.
    You prioritize efficiency and clinical relevance in your decisions.
    Your final reports are concise, accurate, and actionable.
    """,
    verbose=True,
    allow_delegation=True,  # Critical: Allows delegation to other agents
    memory=True,  # ← Change to True
    llm=ollama_llm,
)