import os
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
DATA_PATH = os.getenv("DATA_PATH")

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

def run_query(query, params=None):
    with driver.session() as session:
        session.run(query, params or {})

# ── Constraints (run once) ──────────────────────────────────────────
def create_constraints():
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Customer) REQUIRE c.customer_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (o:Order) REQUIRE o.order_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Product) REQUIRE p.product_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Seller) REQUIRE s.seller_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Review) REQUIRE r.review_id IS UNIQUE",
    ]
    for c in constraints:
        run_query(c)
    print("✓ Constraints created")

# Add this after loading CSVs — run once to get consistent sample
def get_sample_order_ids():
    orders_df = pd.read_csv(f"{DATA_PATH}olist_orders_dataset.csv")
    sampled = orders_df.sample(n=5000, random_state=42)
    return set(sampled["order_id"].tolist())

SAMPLE_ORDER_IDS = get_sample_order_ids()

# ── Loaders ─────────────────────────────────────────────────────────
def load_customers():
    df = pd.read_csv(f"{DATA_PATH}olist_customers_dataset.csv")
    records = df[["customer_id", "customer_city", "customer_state"]].to_dict("records")
    query = """
    UNWIND $rows AS row
    MERGE (c:Customer {customer_id: row.customer_id})
    SET c.city = row.customer_city,
        c.state = row.customer_state
    """
    with driver.session() as session:
        session.run(query, rows=records)
    print(f"✓ Loaded {len(records)} customers")

def load_products():
    df = pd.read_csv(f"{DATA_PATH}olist_products_dataset.csv")
    df["product_category_name"] = df["product_category_name"].fillna("unknown")
    records = df[["product_id", "product_category_name"]].to_dict("records")
    query = """
    UNWIND $rows AS row
    MERGE (p:Product {product_id: row.product_id})
    SET p.category = row.product_category_name
    """
    with driver.session() as session:
        session.run(query, rows=records)
    print(f"✓ Loaded {len(records)} products")

def load_sellers():
    df = pd.read_csv(f"{DATA_PATH}olist_sellers_dataset.csv")
    records = df[["seller_id", "seller_city", "seller_state"]].to_dict("records")
    query = """
    UNWIND $rows AS row
    MERGE (s:Seller {seller_id: row.seller_id})
    SET s.city = row.seller_city,
        s.state = row.seller_state
    """
    with driver.session() as session:
        session.run(query, rows=records)
    print(f"✓ Loaded {len(records)} sellers")

def load_orders():
    df = pd.read_csv(f"{DATA_PATH}olist_orders_dataset.csv")
    df = df[df["order_id"].isin(SAMPLE_ORDER_IDS)]
    df["order_purchase_timestamp"] = df["order_purchase_timestamp"].fillna("")
    records = df[["order_id", "customer_id", "order_status",
                  "order_purchase_timestamp"]].to_dict("records")
    query = """
    UNWIND $rows AS row
    MATCH (c:Customer {customer_id: row.customer_id})
    MERGE (o:Order {order_id: row.order_id})
    SET o.status = row.order_status,
        o.purchase_date = row.order_purchase_timestamp
    MERGE (c)-[:PLACED]->(o)
    """
    batch_size = 500
    with driver.session() as session:
        for i in range(0, len(records), batch_size):
            session.run(query, rows=records[i:i+batch_size])
    print(f"✓ Loaded {len(records)} orders + PLACED relationships")

def load_order_items():
    df = pd.read_csv(f"{DATA_PATH}olist_order_items_dataset.csv")
    df = df[df["order_id"].isin(SAMPLE_ORDER_IDS)]
    records = df[["order_id", "product_id", "seller_id", "price"]].to_dict("records")
    query = """
    UNWIND $rows AS row
    MATCH (o:Order {order_id: row.order_id})
    MATCH (p:Product {product_id: row.product_id})
    MATCH (s:Seller {seller_id: row.seller_id})
    MERGE (o)-[:CONTAINS]->(p)
    MERGE (p)-[:SOLD_BY]->(s)
    """
    batch_size = 500
    with driver.session() as session:
        for i in range(0, len(records), batch_size):
            session.run(query, rows=records[i:i+batch_size])
    print(f"✓ Loaded {len(records)} order items + CONTAINS + SOLD_BY relationships")

def load_reviews():
    df = pd.read_csv(f"{DATA_PATH}olist_order_reviews_dataset.csv")
    df = df[df["order_id"].isin(SAMPLE_ORDER_IDS)]
    df = df.drop_duplicates(subset=["review_id"])
    records = df[["review_id", "order_id", "review_score"]].to_dict("records")
    query = """
    UNWIND $rows AS row
    MATCH (o:Order {order_id: row.order_id})
    MERGE (r:Review {review_id: row.review_id})
    SET r.score = toInteger(row.review_score)
    MERGE (o)-[:HAS_REVIEW]->(r)
    """
    batch_size = 500
    with driver.session() as session:
        for i in range(0, len(records), batch_size):
            session.run(query, rows=records[i:i+batch_size])
    print(f"✓ Loaded {len(records)} reviews + HAS_REVIEW relationships")

def print_summary():
    counts = {
        "Customer": "MATCH (n:Customer) RETURN count(n) as c",
        "Order":    "MATCH (n:Order) RETURN count(n) as c",
        "Product":  "MATCH (n:Product) RETURN count(n) as c",
        "Seller":   "MATCH (n:Seller) RETURN count(n) as c",
        "Review":   "MATCH (n:Review) RETURN count(n) as c",
        "PLACED":   "MATCH ()-[r:PLACED]->() RETURN count(r) as c",
        "CONTAINS": "MATCH ()-[r:CONTAINS]->() RETURN count(r) as c",
        "SOLD_BY":  "MATCH ()-[r:SOLD_BY]->() RETURN count(r) as c",
        "HAS_REVIEW":"MATCH ()-[r:HAS_REVIEW]->() RETURN count(r) as c",
    }
    print("\n── KG Summary ──────────────────────────")
    with driver.session() as session:
        for label, q in counts.items():
            result = session.run(q).single()
            print(f"  {label:12s}: {result['c']:,}")
    print("────────────────────────────────────────")

if __name__ == "__main__":
    print("Starting KG ingestion...")
    create_constraints()
    load_customers()
    load_products()
    load_sellers()
    load_orders()
    load_order_items()
    load_reviews()
    print_summary()
    driver.close()
    print("\nDone! KG is ready in Neo4j AuraDB.")