# tasks/explanation_task.py
from crewai import Task
from agents.explainability_agent import explainability_agent

def create_explanation_task(image_path: str):
    return Task(
        description=f"""
        Analyze the chest X-ray at {image_path} using your trained deep learning model.
        Generate a high-quality Grad-CAM visualization showing which parts of the lungs influenced the prediction.
        Save the explanation image in the 'outputs/' folder with a clear title.
        Provide:
        - Final prediction (COVID / Normal) and confidence score
        - Clinical interpretation of the Grad-CAM heatmap
        - Path to the saved visualization
        """,
        expected_output="""
        A clear explanation including:
        • Predicted diagnosis and confidence
        • Description of key regions highlighted by Grad-CAM
        • Path to the saved Grad-CAM image
        • Brief medical interpretation suitable for a radiologist
        """,
        agent=explainability_agent,
        output_file="outputs/explainability_report.txt"
    )