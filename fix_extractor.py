# fix_extractor.py
# Run once to pull actual property values from Neo4j and save to graph_values.json

import os, json
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

def get_values(query):
    with driver.session() as session:
        return [list(r.values())[0] for r in session.run(query).data()]

graph_values = {
    "Customer.state":    get_values("MATCH (c:Customer) RETURN DISTINCT c.state AS v ORDER BY v"),
    "Customer.city":     get_values("MATCH (c:Customer) RETURN DISTINCT c.city AS v ORDER BY v LIMIT 50"),
    "Order.status":      get_values("MATCH (o:Order) RETURN DISTINCT o.status AS v"),
    "Product.category":  get_values("MATCH (p:Product) RETURN DISTINCT p.category AS v ORDER BY v"),
    "Seller.state":      get_values("MATCH (s:Seller) RETURN DISTINCT s.state AS v ORDER BY v"),
    "Seller.city":       get_values("MATCH (s:Seller) RETURN DISTINCT s.city AS v ORDER BY v LIMIT 30"),
    "Review.score":      get_values("MATCH (r:Review) RETURN DISTINCT r.score AS v ORDER BY v"),
}

with open("graph_values.json", "w") as f:
    json.dump(graph_values, f, indent=2)

print("Saved graph_values.json")
for k, v in graph_values.items():
    print(f"  {k}: {v[:5]}{'...' if len(v) > 5 else ''}")

driver.close()