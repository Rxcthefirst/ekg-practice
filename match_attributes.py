from rdflib import Graph, Namespace
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
import time

# Start time
start_time = time.time()

# Define namespaces
EX = Namespace("http://example.org/")
RDF = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")

# Load RDF data
g = Graph()
g.parse("finance_loans.ttl", format="turtle")

# SPARQL query to extract attributes and entities
query = """
SELECT ?attribute_name ?attribute_desc ?entity_name ?entity_desc
WHERE {
    ?attribute rdf:type ex:Attribute ;
               rdfs:label ?attribute_name ;
               ex:hasDescription ?attribute_desc .
    ?entity rdf:type <http://example.org/Entity> ;
            rdfs:label ?entity_name ;
            <http://example.org/hasDescription> ?entity_desc .
}
"""

results = g.query(query, initNs={'rdf': RDF, 'rdfs': RDFS, 'ex': EX})

# Check if results are returned
if not results:
    print("No results found in the SPARQL query.")
else:
    print("Results found in the SPARQL query.")

# Store extracted attributes and entities
attributes = []
entities = []

for row in results:
    print(row)
    attr_name = str(row.attribute_name)
    attr_desc = str(row.attribute_desc)
    entity_name = str(row.entity_name)
    entity_desc = str(row.entity_desc)

    attributes.append((attr_name, attr_desc))
    entities.append((entity_name, entity_desc))

# Check if attributes and entities are populated
if not attributes:
    print("No attributes found.")
else:
    print(f"Found {len(attributes)} attributes.")

if not entities:
    print("No entities found.")
else:
    print(f"Found {len(entities)} entities.")

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_embedding(text):
    """Convert text into a dense vector using BERT."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Vectorize attributes and entities
attribute_vectors = {name: (get_embedding(name), get_embedding(desc)) for name, desc in attributes}
entity_vectors = {name: (get_embedding(name), get_embedding(desc)) for name, desc in entities}

# Function to compute cosine similarity
def cosine_sim(a, b):
    return cosine_similarity([a], [b])[0][0]

# Match attributes to best entities
matches = []
for attr_name, (attr_vec_name, attr_vec_desc) in attribute_vectors.items():
    best_match = None
    best_score = -1

    for entity_name, (entity_vec_name, entity_vec_desc) in entity_vectors.items():
        sim_name = cosine_sim(attr_vec_name, entity_vec_name)
        sim_desc = cosine_sim(attr_vec_desc, entity_vec_desc)
        avg_sim = (sim_name + sim_desc) / 2  # Weight both equally

        if avg_sim > best_score:
            best_score = avg_sim
            best_match = entity_name

    matches.append((attr_name, best_match, best_score))

# Print results
print("\nAttribute Matching Results:")
for attr, entity, score in matches:
    print(f"Attribute: {attr} → Best Match: {entity} (Similarity: {score:.4f})")

# End time
end_time = time.time()
print(f"\nTime taken: {end_time - start_time:.2f} seconds")