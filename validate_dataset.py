# validate_dataset.py
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()
driver = GraphDatabase.driver(os.getenv("NEO4J_URI"),
                               auth=(os.getenv("NEO4J_USERNAME"),
                                     os.getenv("NEO4J_PASSWORD")))

with open("test_dataset.json") as f:
    dataset = json.load(f)

passed, failed = [], []
with driver.session() as session:
    for item in dataset:
        try:
            result = session.run(item["gold_cypher"]).data()
            passed.append(item["id"])
        except Exception as e:
            failed.append((item["id"], str(e)))

print(f"✓ Passed: {len(passed)}/50")
if failed:
    print("✗ Failed:")
    for id_, err in failed:
        print(f"  Q{id_}: {err}")

driver.close()