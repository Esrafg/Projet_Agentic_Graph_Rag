# src/tasks/classification_task.py
from crewai import Task
from agents.image_classification_agent import image_classification_agent

def create_classification_task(image_path: str):
    return Task(
        description=f"""
        Load the image at {image_path}.
        Extract features using color moments, HSV histogram, GLCM texture, Hu shape moments, and SIFT+BoVW.
        Use the pre-trained .pkl classifier to predict if it's 'Anomaly' or 'Normal'.
        Save features and prediction to the 'outputs/' folder.
        Provide the final prediction and confidence (if available).
        """,
        expected_output="""
        A clear classification result including:
        • Predicted class ('Anomaly' or 'Normal')
        • Confidence score (if available)
        • Path to saved features/prediction files
        • Brief summary of the process
        """,
        agent=image_classification_agent,
        output_file="outputs/classification_report.txt"
    )