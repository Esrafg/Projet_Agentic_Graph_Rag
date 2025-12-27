# app.py
import os
import glob
import streamlit as st


# === DISABLE CREWAI TELEMETRY (Fixes the connection errors) ===
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["CREWAI_DISABLE_TRACING"] = "true"

# Force Ollama as LLM provider (add these)
os.environ["OPENAI_API_KEY"] = ""  # Already there, keeps OpenAI disabled
os.environ["CREWAI_LLM_PROVIDER"] = "ollama"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
# === IMPORTS (after disabling telemetry) ===
from langchain_community.llms import Ollama
from crewai import Agent, Task, Process

# Set default LLM for all Agents
llm = Ollama(model="llama3.2", base_url="http://localhost:11434", temperature=0.7)
Agent.default_llm = llm
Task.default_llm = llm
from tasks.coordination_task import create_coordinator_task
from crewai import Crew
from agents.coordinator_agent import coordinator_agent
from agents.image_classification_agent import image_classification_agent
from agents.explainability_agent import explainability_agent


# === STREAMLIT UI ===
st.set_page_config(page_title="Chest X-ray AI Assistant", layout="centered")
st.title("🫁 Chest X-ray AI Diagnostic Assistant")
st.write(
    "Upload a chest X-ray image and ask your clinical question. The multi-agent system will analyze it using traditional ML classification and deep learning explainability."
)

user_query = st.text_input(
    "Your clinical question",
    value="Please analyze this X-ray for any signs of COVID-19 or lung anomalies.",
)

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["png", "jpg", "jpeg"],
    help="Supported formats: PNG, JPG, JPEG",
)

if uploaded_file and st.button("🚀 Start Analysis", type="primary"):
    # === NETTOYAGE ET SÉCURISATION DU NOM DE FICHIER ===
    import re

    # Nom original
    original_name = uploaded_file.name
    # Extension
    extension = original_name.split(".")[-1].lower()
    if "." not in original_name:
        extension = "png"  # fallback

    # Nettoyer le nom : enlever accents, espaces, caractères spéciaux
    clean_name = re.sub(
        r"[^\w\-_.]", "_", original_name
    )  # remplace tout sauf lettres/chiffres par _
    clean_name = (
        clean_name.replace("é", "e")
        .replace("è", "e")
        .replace("à", "a")
        .replace("ô", "o")
    )

    # Nom final sûr et court
    safe_filename = f"current_xray_analysis.{extension}"

    # Sauvegarde dans un dossier dédié
    os.makedirs("uploaded_images", exist_ok=True)
    image_path = os.path.join("uploaded_images", safe_filename)

    # Écriture du fichier
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info(f"Image saved securely as: {safe_filename}")

    # Utiliser un chemin relatif simple pour éviter tout problème d'échappement
    relative_path = image_path.replace("\\", "/")  # forward slashes seulement

    with st.spinner("Multi-agent system is analyzing the X-ray... (30-90 seconds)"):
        try:
            # Moved here: Create tasks after relative_path is defined
            from tasks.classification_task import create_classification_task
            from tasks.explanation_task import create_explanation_task
            from tasks.report_task import create_report_task  # Assuming you added this

            classification_task = create_classification_task(relative_path)
            explanation_task = create_explanation_task(relative_path)
            report_task = create_report_task(relative_path)

            crew = Crew(
                agents=[
                    coordinator_agent,
                    image_classification_agent,
                    explainability_agent,
                ],
                tasks=[classification_task, explanation_task, report_task],
                process=Process.sequential,  # If using sequential as previously suggested
                verbose=2,
                memory=False,
            )

            # Run the analysis
            result = crew.kickoff()

            # Success display
            st.success("Analysis Complete!")
            st.markdown("### 📋 Final Diagnostic Report")
            st.markdown(result)

            # Display Grad-CAM heatmap if generated
            gradcam_files = glob.glob("outputs/gradcam_*.png")
            if gradcam_files:
                latest_gradcam = max(gradcam_files, key=os.path.getctime)
                st.image(
                    latest_gradcam,
                    caption="Grad-CAM Heatmap: Regions the deep learning model focused on (red = high attention)",
                    use_column_width=True,
                )
            else:
                st.info("No Grad-CAM generated (likely classified as Normal).")

        except Exception as e:
            st.error("An error occurred during analysis.")
            st.exception(e)
        finally:
            # Clean up temporary file
            if os.path.exists(image_path):
                try:
                    os.remove(image_path)
                    st.info("Temporary image file cleaned up.")
                except:
                    pass

    st.balloons()
# Footer
st.markdown("---")
st.caption(
    "Multi-Agent Medical AI System • Traditional ML Classification + Deep Learning Explainability (Grad-CAM)"
)
