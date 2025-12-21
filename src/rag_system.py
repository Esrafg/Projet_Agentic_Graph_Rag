# rag_system.py
from langchain_groq import ChatGroq
from langchain_neo4j import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from graph_builder import get_graph
from config import GROQ_API_KEY

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=GROQ_API_KEY)
graph = get_graph()

cypher_prompt = PromptTemplate.from_template("""Tu es un expert Cypher. Donne seulement la requête Cypher.

Question : {question}
Cypher :""")

qa_prompt = PromptTemplate.from_template("""Réponds en français clair à partir des données :

{context}

Question : {question}
Réponse :""")

# rag_system.py  ← remplace toute la partie chain =

chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    cypher_prompt=cypher_prompt,
    qa_prompt=qa_prompt,
    verbose=False,
    top_k=50,
    allow_dangerous_requests=True   # ← ligne magique à ajouter
)

def ask(question):
    try:
        result = chain.invoke({"query": question})
        answer = result["result"].strip()
        if len(answer) > 10 and "aucun" not in answer.lower():
            return answer
    except:
        pass
    return "Je n'ai pas trouvé de réponse précise. Essayez : 'Top 5 départements', 'médecin le plus occupé', 'durée moyenne séjour'..."