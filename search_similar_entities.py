from rdflib import Graph
from sentence_transformers import SentenceTransformer
import psycopg2
import numpy as np

# Connect to PostgreSQL
conn = psycopg2.connect("dbname=mydb user=myuser password=mypass host=localhost port=5432")
cur = conn.cursor()

# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to search for similar entities
def search_similar_entities(query_text, top_k=5):
    query_embedding = model.encode(query_text).tolist()
    cur.execute(
        "SELECT entity_id, label, 1 - (embedding <=> %s::vector) AS similarity "
        "FROM entity_embeddings ORDER BY similarity DESC LIMIT %s",
        (query_embedding, top_k)
    )

    results = cur.fetchall()
    print("\nTop Similar Entities:")
    for entity_id, label, similarity in results:
        print(f"{label} (ID: {entity_id}) - Similarity: {similarity:.4f}")

# Example search
search_query = "cooking"
search_similar_entities(search_query)

cur.close()
conn.close()