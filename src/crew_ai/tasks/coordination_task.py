# src/tasks/coordination_task.py
from crewai import Task
from agents.coordinator_agent import coordinator_agent

def create_coordinator_task(user_query: str, image_path: str):
    return Task(
        description=f"""
User clinical question: {user_query}

Image to analyze: {image_path}

You are the Medical AI Workflow Coordinator.

FOLLOW THIS WORKFLOW STRICTLY:

1. FIRST: Delegate classification to "Image Classification Specialist"
   Instruction: "Classify the image at {image_path} as 'Normal' or 'Anomaly' using the traditional ML model. Return the prediction and confidence."

2. AFTER RECEIVING THE CLASSIFICATION RESULT: 
   Confirm the result (e.g., parse 'Prediction: Normal (Confidence: XX%)'). 
   Do NOT re-delegate classification. Proceed immediately to explainability.

3. SECOND (ALWAYS DO THIS, REGARDLESS OF RESULT):
   After confirming the classification result (whether Normal or Anomaly),
   ALWAYS delegate to "Medical Imaging Explainability Specialist":
   "Generate a Grad-CAM visualization and clinical interpretation for the image at {image_path}"

4. AFTER RECEIVING THE EXPLAINABILITY RESULT:
   Confirm the result (e.g., parse prediction, confidence, and heatmap path).
   Do NOT re-delegate. Proceed to final report.

5. FINAL: Write a complete professional diagnostic report including:
   - Classification result and confidence
   - Grad-CAM findings and interpretation
   - List of generated files in outputs/
   - Clinical recommendation

IMPORTANT:
- You MUST generate Grad-CAM for every image, even if classified as Normal.
- Never skip the explainability step.
- Do not repeat classification.
- If a delegation returns a result, acknowledge it in your thoughts and move to the next step without looping.
- Use memory to track completed steps.
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