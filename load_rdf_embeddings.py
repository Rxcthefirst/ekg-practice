from rdflib import Graph
from sentence_transformers import SentenceTransformer
import psycopg2
import numpy as np

# Connect to PostgreSQL
conn = psycopg2.connect("dbname=mydb user=myuser password=mypass host=localhost port=5432")
cur = conn.cursor()

# Ensure pgvector table exists
cur.execute("""
    CREATE TABLE IF NOT EXISTS entity_embeddings (
        entity_id TEXT PRIMARY KEY,
        label TEXT,
        embedding VECTOR(384)  -- Adjust dimension based on your model
    )
""")
conn.commit()

# Load RDF data
g = Graph()
g.parse("sample_data.ttl", format="turtle")  # Ensure your RDF file is named 'data.ttl'

# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Insert entity embeddings into PostgreSQL
for s, p, o in g:
    if "label" in p:  # Extract labeled entities
        entity_id = str(s)
        label = str(o)
        embedding = model.encode(label).tolist()

        cur.execute(
            "INSERT INTO entity_embeddings (entity_id, label, embedding) VALUES (%s, %s, %s) "
            "ON CONFLICT (entity_id) DO NOTHING",
            (entity_id, label, embedding)
        )

conn.commit()
print("Embeddings inserted successfully!")


cur.close()
conn.close()