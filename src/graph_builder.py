# graph_builder.py
from langchain_neo4j import Neo4jGraph
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Neo4jVector
from config import *
import time

_graph = None

def get_graph():
    global _graph
    if _graph is None:
        print("Connexion Neo4j...")
        _graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, timeout=60)
        _graph.query("RETURN 1")
        print("Connecté à Neo4j")
    return _graph

def build_graph_from_df(df):
    g = get_graph()
    
    # Si déjà plein → skip
    count = g.query("MATCH (p:Patient) RETURN count(p) as c")[0]["c"]
    if count > 1000:
        print(f"Graphe déjà présent ({count} patients) → on passe directement à l'interface")
        return

    print("Création du graphe (peut prendre 2-3 minutes)...")
    
    for i in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[i:i+BATCH_SIZE]
        print(f"Batch {i//BATCH_SIZE + 1}...", end=" ")
        for _ in range(3):
            try:
                g.query("""
                UNWIND $rows AS row
                MERGE (p:Patient {id: row.patientid})
                SET p += {
                  age: toInteger(row.parsed_age),
                  gender: row.gender,
                  insurance: row.insurance,
                  stay_days: toInteger(row.stay_days),
                  admission_type: row.admission_type,
                  billing_amount: toFloat(row.parsed_deposit),
                  ward_code: row.ward_code,
                  severity: row.severity,
                  visitor_count: toInteger(row.visitor_count)
                }
                MERGE (doc:Doctor {name: coalesce(row.doctor_name, 'Dr. Inconnu')})
                MERGE (dept:Department {name: coalesce(row.department, 'Service Général')})
                MERGE (p)-[:TREATED_BY]->(doc)
                MERGE (p)-[:ADMITTED_TO]->(dept)
                MERGE (doc)-[:WORKS_IN]->(dept)
                """, {"rows": batch.to_dict("records")})
                print("OK")
                break
            except:
                time.sleep(2)

    # Maladies + stats + descriptions + index vectoriel (version simplifiée qui marche à 100%)
    print("Finalisation (maladies, stats, descriptions, index)...")
    g.query("""
    MATCH (p:Patient) WHERE p.health_conditions IS NOT NULL AND p.health_conditions <> '' AND p.health_conditions <> 'Other'
    WITH p, trim(p.health_conditions) AS cond
    MERGE (d:Disease {name: cond}) MERGE (p)-[:HAS_CONDITION]->(d)
    """)
    
    g.query("""
    MATCH (dept:Department)<-[:ADMITTED_TO]-(p) 
    WITH dept, count(p) as cnt, avg(p.stay_days) as stay, avg(p.billing_amount) as bill
    SET dept.patient_count = cnt, dept.avg_stay_days = stay, dept.avg_billing = bill
    """)
    
    g.query("""
    MATCH (p:Patient)
    OPTIONAL MATCH (p)-[:TREATED_BY]->(doc)
    OPTIONAL MATCH (p)-[:ADMITTED_TO]->(dept)
    OPTIONAL MATCH (p)-[:HAS_CONDITION]->(d)
    WITH p, doc, dept, collect(d.name) as diseases
    SET p.description = "Patient " + p.id + ", " + coalesce(p.gender,"?") + 
      ", âge " + coalesce(toString(p.age),"?") + 
      ", séjour " + coalesce(toString(p.stay_days),"?") + " jours, " +
      coalesce(doc.name,"?") + " - " + coalesce(dept.name,"?")
    """)

    # Index vectoriel
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Neo4jVector.from_existing_graph(
        embeddings, url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
        node_label="Patient", text_node_properties=["description"],
        embedding_node_property="embedding", index_name=VECTOR_INDEX_NAME
    )
    print("Graphe 100% prêt !")