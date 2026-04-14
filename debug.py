# debug.py
import os, json
from neo4j import GraphDatabase
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SCHEMA = """Node labels and properties:
- Customer: {customer_id, city, state}
- Order: {order_id, status, purchase_date}
- Product: {product_id, category}
- Seller: {seller_id, city, state}
- Review: {review_id, score}

Relationships:
- (:Customer)-[:PLACED]->(:Order)
- (:Order)-[:CONTAINS]->(:Product)
- (:Product)-[:SOLD_BY]->(:Seller)
- (:Order)-[:HAS_REVIEW]->(:Review)"""

def call_groq(model, messages):
    response = client.chat.completions.create(
        model=model, messages=messages, max_tokens=300, temperature=0.0
    )
    return response.choices[0].message.content.strip()

question = "How many customers are there in total?"
gold_cypher = "MATCH (c:Customer) RETURN count(c) AS total_customers"

# 1. Check gold result
with driver.session() as session:
    gold = session.run(gold_cypher).data()
print(f"Gold result: {gold}")

# 2. Check what baseline generates
messages = [
    {"role": "system", "content": """You are a Cypher query generator for Neo4j. Return ONLY the raw Cypher query. 
Do NOT use markdown code blocks.
Do NOT use backticks.
Do NOT add any explanation.
Just the Cypher query itself, nothing else. no explanation."""},
    {"role": "user", "content": f"Generate a Cypher query to answer:\n{question}"}
]
baseline_raw = call_groq("llama-3.3-70b-versatile", messages)
print(f"\nBaseline raw output:\n{baseline_raw}")

# 3. Execute baseline
with driver.session() as session:
    try:
        baseline_result = session.run(baseline_raw).data()
        print(f"Baseline result: {baseline_result}")
    except Exception as e:
        print(f"Baseline execution error: {e}")

# 4. Check schema-aware
messages2 = [
    {"role": "system", "content": """You are a Cypher query generator for Neo4j. Return ONLY the raw Cypher query. 
Do NOT use markdown code blocks.
Do NOT use backticks.
Do NOT add any explanation.
Just the Cypher query itself, nothing else. no explanation."""},
    {"role": "user", "content": f"Generate a Cypher statement to query a graph database.\nUse ONLY the provided relationship types and properties.\n\nSchema:\n{SCHEMA}\n\nQuestion: {question}"}
]
schema_raw = call_groq("llama-3.3-70b-versatile", messages2)
print(f"\nSchema-aware raw output:\n{schema_raw}")

with driver.session() as session:
    try:
        schema_result = session.run(schema_raw).data()
        print(f"Schema-aware result: {schema_result}")
    except Exception as e:
        print(f"Schema-aware execution error: {e}")

driver.close()