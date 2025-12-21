# main.py → VERSION URGENCE PRÉSENTATION (décembre 2025)
from data_loader import load_and_clean_data
from graph_builder import build_graph_from_df
from rag_system import ask
import gradio as gr

# Chargement rapide
df = load_and_clean_data()
build_graph_from_df(df)

print("Lancement rapide pour la présentation...\n")

# Interface ultra simple et 100% fonctionnelle
with gr.Blocks() as demo:
    gr.Markdown("# Graph RAG Santé – 5000 patients")
    gr.Markdown("### Pose ta question en français → réponse instantanée !")

    question = gr.Textbox(label="Ta question", placeholder="Ex: Combien de patients diabétiques ?", lines=2)
    btn = gr.Button("Envoyer", variant="primary")
    
    reponse = gr.Textbox(label="Réponse", lines=20, interactive=False)

    # Bouton copier rapide
    gr.HTML("""
    <button onclick="navigator.clipboard.writeText(document.querySelector('textarea[aria-label=\"Réponse\"]').value); 
    alert('Copié !')">Copier la réponse</button>
    """)

    # Exemples
    gr.Examples([
        "Combien de patients au total ?",
        "Top 5 des départements",
        "Quel médecin a le plus de patients ?",
        "Durée moyenne de séjour",
        "Combien de diabétiques ?",
        "Répartition hommes/femmes",
        "Coût moyen",
        "Département le plus cher"
    ], question)

    btn.click(ask, question, reponse)
    question.submit(ask, question, reponse)

# LANCEMENT FORCE SANS PORT FIXE → ÇA MARCHE TOUJOURS
demo.launch(share=True, inbrowser=True)   # ← ouvre direct dans ton navigateur + lien public