import os
import json
import time
from neo4j import GraphDatabase
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ── Connections ──────────────────────────────────────────────────────
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

TEXT2CYPHER_MODEL = "llama-3.3-70b-versatile"
EXTRACTOR_MODEL   = "llama-3.1-8b-instant"

# ── Schema ───────────────────────────────────────────────────────────
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


SYSTEM_MSG = """You are a Cypher query generator for Neo4j.
Return ONLY the raw Cypher query.
Do NOT use markdown code blocks.
Do NOT use backticks.
Do NOT add any explanation.
Just the Cypher query itself, nothing else."""


# ── Groq call helper ─────────────────────────────────────────────────
def call_groq(model, messages, max_tokens=300):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0
    )
    return response.choices[0].message.content.strip()

# ── System 1: Baseline ───────────────────────────────────────────────
def generate_cypher_baseline(question):
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": f"Generate a Cypher query to answer:\n{question}"}
    ]
    return postprocess(call_groq(TEXT2CYPHER_MODEL, messages))

# ── System 2: Schema-aware ───────────────────────────────────────────
def generate_cypher_schema_aware(question):
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": (
            f"Generate a Cypher statement to query a graph database.\n"
            f"Use ONLY the provided relationship types and properties.\n\n"
            f"Schema:\n{SCHEMA}\n\n"
            f"Question: {question}"
        )}
    ]
    return postprocess(call_groq(TEXT2CYPHER_MODEL, messages))

# ── System 3: Entity extraction + schema ────────────────────────────
# load once at top of evaluate.py
with open("graph_values.json") as f:
    GRAPH_VALUES = json.load(f)

def extract_entities(question):
    # build value hints from actual graph
    value_hints = "\n".join([
        f"- {prop}: {values}"
        for prop, values in GRAPH_VALUES.items()
    ])

    prompt = f"""You are an entity extractor for a Neo4j e-commerce knowledge graph.

Below are the EXACT values stored in the graph for each property.
You MUST only use values from this list — do not invent or paraphrase values.

{value_hints}

Extract entities from the question and map them to the exact property and value shown above.
Return ONLY a valid JSON array. If no entities found, return [].

Examples:
Question: "How many customers are from SP?"
Output: [{{"property": "Customer.state", "value": "SP"}}]

Question: "Which cancelled orders got a 1 star review?"
Output: [{{"property": "Order.status", "value": "canceled"}}, {{"property": "Review.score", "value": 1}}]

Question: "{question}"
Output:"""

    messages = [{"role": "user", "content": prompt}]
    raw = call_groq(EXTRACTOR_MODEL, messages, max_tokens=150)
    try:
        start = raw.find("[")
        end   = raw.rfind("]") + 1
        if start != -1 and end > 0:
            return json.loads(raw[start:end])
    except Exception:
        pass
    return []
    
def generate_cypher_with_entities(question):
    entities = extract_entities(question)

    entity_str = ""
    if entities:
        entity_str = "\nExtracted entities grounded to graph properties:\n"
        for e in entities:
            entity_str += f"  - {e['property']} = {repr(e['value'])}\n"

    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": (
            f"Generate a Cypher statement to query a graph database.\n"
            f"Use ONLY the provided relationship types and properties.\n\n"
            f"Schema:\n{SCHEMA}\n"
            f"{entity_str}\n"
            f"Question: {question}"
        )}
    ]
    return postprocess(call_groq(TEXT2CYPHER_MODEL, messages))

# ── Postprocess ──────────────────────────────────────────────────────
def postprocess(text):
    text = text.strip()
    # handle ```cypher ... ``` blocks
    if "```" in text:
        # extract content between first ``` and last ```
        parts = text.split("```")
        # parts[1] is the content inside the backticks
        if len(parts) >= 3:
            inner = parts[1]
            # strip the language identifier (e.g. "cypher\n")
            if inner.startswith("cypher"):
                inner = inner[len("cypher"):]
            text = inner.strip()
        else:
            # fallback: just remove all ``` lines
            lines = text.split("\n")
            clean = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(clean).strip()
    # cut off at explanation markers
    for marker in ["**Explanation", "Explanation:", "Note:", "This query"]:
        if marker in text:
            text = text[:text.index(marker)].strip()
    return text.strip()

# ── Execute + compare ────────────────────────────────────────────────
def execute_cypher(cypher):
    try:
        with driver.session() as session:
            return session.run(cypher).data()
    except Exception:
        return None

def results_match(gold, pred):
    if pred is None:
        return False
    try:
        def normalize(r):
            # extract just values, ignore column names
            return set(
                frozenset(str(v) for v in row.values())
                for row in r
            )
        return normalize(gold) == normalize(pred)
    except Exception:
        return False

# ── Main evaluation loop ─────────────────────────────────────────────
def evaluate():
    with open("test_dataset.json") as f:
        dataset = json.load(f)

    systems = {
        "baseline":       generate_cypher_baseline,
        "schema_aware":   generate_cypher_schema_aware,
        "with_entities":  generate_cypher_with_entities,
    }

    results = {name: [] for name in systems}

    for i, item in enumerate(dataset):
        print(f"\n[{i+1}/50] Q{item['id']} ({item['complexity']}): {item['question'][:60]}...")
        gold_result = execute_cypher(item["gold_cypher"])

        for sys_name, gen_fn in systems.items():
            try:
                generated   = gen_fn(item["question"])
                pred_result = execute_cypher(generated)
                match       = results_match(gold_result, pred_result)

                results[sys_name].append({
                    "id":               item["id"],
                    "complexity":       item["complexity"],
                    "question":         item["question"],
                    "generated_cypher": generated,
                    "executable":       pred_result is not None,
                    "correct":          match
                })
                status = "✓" if match else ("✗ not executable" if pred_result is None else "✗ wrong result")
                print(f"  {sys_name:<20}: {status}")

            except Exception as e:
                print(f"  {sys_name:<20}: ERROR — {e}")
                results[sys_name].append({
                    "id": item["id"], "complexity": item["complexity"],
                    "question": item["question"], "generated_cypher": "",
                    "executable": False, "correct": False
                })

        time.sleep(1)  # rate limit buffer

    # ── Save raw results ─────────────────────────────────────────────
    with open("results_raw.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Print summary ────────────────────────────────────────────────
    print("\n\n══════════════════════════════════════════════════")
    print("  EVALUATION RESULTS — Execution Accuracy (EX%)")
    print("══════════════════════════════════════════════════")
    print(f"  {'System':<22} {'EX%':>6}  {'Exec%':>7}")
    print("  " + "─" * 40)

    for sys_name, res in results.items():
        ex   = sum(r["correct"]    for r in res) / len(res) * 100
        exe  = sum(r["executable"] for r in res) / len(res) * 100
        print(f"  {sys_name:<22} {ex:>5.1f}%  {exe:>6.1f}%")

    print("\n  Breakdown by complexity:")
    for complexity in ["simple", "filter", "aggregation", "multihop"]:
        print(f"\n  [{complexity}]")
        for sys_name, res in results.items():
            subset = [r for r in res if r["complexity"] == complexity]
            if subset:
                ex = sum(r["correct"] for r in subset) / len(subset) * 100
                print(f"    {sys_name:<22}: {ex:.1f}%  ({sum(r['correct'] for r in subset)}/{len(subset)})")

    print("\n══════════════════════════════════════════════════")

if __name__ == "__main__":
    evaluate()
    driver.close()