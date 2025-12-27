# src/tasks/report_task.py
from crewai import Task
from agents.coordinator_agent import coordinator_agent


def create_report_task(image_path: str):
    return Task(
        description=f"""
        Compile a final diagnostic report for the image at {image_path}.
        Use the classification result from outputs/classification_report.txt.
        Use the explainability result from outputs/explainability_report.txt.
        Include:
        - Classification and confidence
        - Grad-CAM findings and interpretation
        - List of generated files in outputs/
        - Clinical recommendation based on combined results.
        """,
        expected_output="""
        **Patient Chest X-ray Analysis Report**

        - Image: {image_path}
        - Classification: Normal / Anomaly
        - Confidence: XX.X%
        - Explainability (Grad-CAM):
          • Key regions highlighted by the model
          • Clinical interpretation of the heatmap
        - Generated files:
          • outputs/features_*.npy
          • outputs/prediction_*.txt
          • outputs/gradcam_*.png
        - Recommendation: Based on combined classification and explainability
        """,
        agent=coordinator_agent,
        output_file="outputs/final_diagnostic_report.txt",
    )
