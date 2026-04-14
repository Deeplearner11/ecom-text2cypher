# ecom-text2cypher

A research project investigating inference-time strategies to improve NL-to-Cypher query generation on a domain-specific e-commerce Knowledge Graph, evaluated against Neo4j's official Text2Cypher Gemma-3-4B model.

---

## Overview

This project builds a Knowledge Graph from the [Brazilian Olist e-commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), constructs a 50-query evaluation benchmark stratified by complexity, and systematically evaluates two inference-time improvement strategies — **graph-grounded entity extraction** and **LLM-based schema pruning** — using Execution Accuracy (EX%) and Google-BLEU as metrics.

The work is motivated by two published papers:

- [Text2Cypher: Bridging Natural Language and Graph Databases](https://aclanthology.org/2025.genaik-1.11.pdf) (Ozsoy, 2025)
- [Enhancing Text2Cypher with Schema Filtering](https://arxiv.org/html/2505.05118) (Ozsoy, 2025)

---

## Repository Structure

```
ecom-text2cypher/
├── data/                        # Olist CSV files (not committed, download separately)
├── ingest.py                    # Builds the Knowledge Graph in Neo4j AuraDB
├── validate_dataset.py          # Validates all gold Cypher pairs execute correctly
├── fix_extractor.py             # Pulls actual graph property values for entity grounding
├── debug.py                     # Debugging utilities for single query testing
├── schema_prune.ipynb           # Main Kaggle notebook — KG creation, benchmark, full evaluation
│                                #   ├── Knowledge Graph ingestion
│                                #   ├── Test dataset construction (50 gold pairs)
│                                #   ├── Graph value extraction for entity grounding
│                                #   ├── Schema-aware evaluation (Neo4j Gemma-3-4B)
│                                #   └── LLM schema pruning evaluation
├── test_dataset.json            # 50 gold (question, Cypher) pairs
├── graph_values.json            # Actual property values from Neo4j for entity grounding
├── results_final.json           # Final evaluation results
```

---

## Knowledge Graph

### Dataset
[Brazilian Olist E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — a real Brazilian marketplace dataset with ~100K orders across 2016–2018.

### Graph Schema

```
(Customer)-[:PLACED]->(Order)-[:CONTAINS]->(Product)-[:SOLD_BY]->(Seller)
                          |
                    [:HAS_REVIEW]
                          |
                       (Review)
```

**Node properties:**

| Node | Properties |
|---|---|
| Customer | customer_id, city, state |
| Order | order_id, status, purchase_date |
| Product | product_id, category |
| Seller | seller_id, city, state |
| Review | review_id, score (1–5) |

### Graph Statistics

| Entity | Count |
|---|---|
| Customer nodes | 99,441 |
| Order nodes | 5,000 (sampled) |
| Product nodes | 32,951 |
| Seller nodes | 3,095 |
| Review nodes | 4,979 |
| PLACED relationships | 5,000 |
| CONTAINS relationships | 5,138 |
| SOLD_BY relationships | 3,911 |
| HAS_REVIEW relationships | 4,979 |
| **Total nodes** | **~145K** |
| **Total relationships** | **~19K** |

Orders were sampled to 5,000 due to Neo4j AuraDB free tier limits (200K node cap). All other node types are fully loaded.

### Ingestion

```bash
pip install neo4j pandas python-dotenv
python ingest.py
```

Set credentials in `.env`:
```
NEO4J_URI=your_uri
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
DATA_PATH=path/to/olist/csvs/
```

---

## Evaluation Benchmark

### Design
50 hand-crafted (question, gold Cypher) pairs, stratified by query complexity:

| Complexity | Count | Description |
|---|---|---|
| Simple | 10 | Single node count/list queries |
| Filter | 10 | Queries with WHERE conditions on specific entity values |
| Aggregation | 10 | GROUP BY, ORDER BY, avg/count across relationships |
| Multihop | 20 | 3–5 hop traversals across multiple node types |

### Gold Query Design Principles
- Gold queries return only what the question asks — no extra columns
- `count(DISTINCT ...)` used where duplicate relationships could inflate counts
- Validated: all 50 gold queries execute without error on the live graph (`validate_dataset.py`)

### Validate
```bash
python validate_dataset.py
# Expected: ✓ Passed: 50/50
```

---

## Model

**Neo4j Text2Cypher Gemma-3-4B** — `neo4j/text-to-cypher-Gemma-3-4B-Instruct-2025.04.0`

An official Neo4j model fine-tuned on 35,000 (question, schema, Cypher) pairs from the [Neo4j Text2Cypher 2025 dataset](https://huggingface.co/datasets/neo4j/text2cypher-2025v1). Key features:

- Schema-aware training: each example includes node labels, relationship types, and property examples
- Native schema format: `- NodeName\n  - property: TYPE Example: "value"`
- 4B parameters, runs on Colab/Kaggle T4 GPU with 4-bit quantization

**Critical finding:** the model's native schema format (with inline property examples) must be matched exactly in the prompt. Using a generic schema string caused entity hallucination (e.g. generating `state = 'Sao Paulo'` instead of `state = 'SP'`). Matching the training format resolved this entirely.

---

## Inference-Time Strategies

### Strategy 1 — Graph-Grounded Entity Extraction

A pre-processing step that extracts entity mentions from the question and maps them to exact property values stored in the graph before passing to the Text2Cypher model.

**Pipeline:**
```
Question
  → LLM extractor (llama-3.1-8b-instant via Groq)
  → extracted entities grounded to actual graph values
  → enriched prompt → Text2Cypher model → Cypher
```

**Extractor prompt design:** The extractor is given the actual property values from the graph (`graph_values.json`) so it can map "Sao Paulo" → `'SP'`, "cancelled" → `'canceled'`, etc.

**Key finding:** Entity extraction only provides meaningful improvement when the extractor has access to actual graph property values. A generic LLM extractor without grounded values provides no improvement — it hallucinated values that don't exist in the graph.

### Strategy 2 — LLM-Based Schema Pruning

Instead of providing the full schema for every query, a lightweight LLM call selects only the nodes and relationships relevant to the specific question.

**Pipeline:**
```
Question
  → Schema pruner (llama-3.1-8b-instant via Groq)
  → pruned schema (only relevant nodes + relationships)
  → Text2Cypher model → Cypher
```

**Pruner prompt:**
```
Given the question, return ONLY the nodes and relationships needed.
Return JSON: {"nodes": [...], "relationships": [...]}
```

The pruner also enforces consistency — if a relationship is selected, both endpoint nodes are added to the node list even if not returned by the LLM.

**Motivation:** The [Schema Filtering paper](https://arxiv.org/html/2505.05118) shows that longer prompts hurt small model performance, and exact-match pruning achieves the best Google-BLEU on Llama-3.1-8B — the model class closest to Gemma-3-4B.

---

## Evaluation Setup

### Systems Evaluated

| System | Schema | Entity Extraction |
|---|---|---|
| `schema_aware` | Full schema | No |
| `with_pruning` | LLM-pruned schema | No |

### Metrics

**Execution Accuracy (EX%)** — primary metric. Both the generated and gold Cypher are executed on the live Neo4j graph. Results are compared as sets of values (column names ignored) to handle equivalent but differently aliased queries.

**Google-BLEU** — secondary metric. Sentence-level 4-gram BLEU between generated and gold Cypher. Measures structural similarity independent of execution. Directly comparable to the Schema Filtering paper's reported results.

**Average Token Count** — prompt length per query. Measures the cost/noise reduction from schema pruning.

### Hardware
- Neo4j AuraDB Free (cloud, shared)
- Kaggle T4 GPU (16GB VRAM)
- Model loaded with 4-bit NF4 quantization (~3.5GB VRAM)
- Entity extractor + schema pruner: Groq API (free tier)

---

## Results

### Overall

| System | EX% | Google-BLEU | Avg Tokens |
|---|---|---|---|
| Schema-aware | **64%** | **0.670** | 260 |
| With pruning | 58% | 0.649 | **129** |

### By Complexity

| Complexity | Schema-aware EX% | Pruning EX% | Schema-aware BLEU | Pruning BLEU |
|---|---|---|---|---|
| Simple | **100%** | **100%** | 0.777 | 0.603 |
| Filter | **100%** | **100%** | 0.602 | 0.566 |
| Aggregation | 50% | 50% | 0.598 | **0.632** |
| Multihop | 35% | 30% | 0.659 | **0.726** |

### Token Reduction

| Complexity | Schema-aware Tokens | Pruning Tokens | Reduction |
|---|---|---|---|
| Simple | 257 | 82 | **68%** |
| Filter | 260 | 85 | **67%** |
| Aggregation | 258 | 115 | **55%** |
| Multihop | 263 | 181 | **31%** |

---

## Key Findings

**1. Native schema format is critical**

Matching the model's training schema format (with inline property type examples) completely resolved entity hallucination on filter queries — achieving 100% filter EX without any entity extraction. A generic schema string caused the model to generate `state = 'Sao Paulo'` instead of `state = 'SP'`.

**2. Schema pruning achieves 50% token reduction with minimal EX impact**

LLM-based schema pruning reduces average prompt length from 260 to 129 tokens (50% reduction) with only a 6% EX drop overall. For simple and filter queries, EX is identical at 100% while tokens drop by 67–68%. This is consistent with the Schema Filtering paper's finding that smaller models benefit from reduced schema noise.

**3. Pruning improves BLEU on complex queries**

On aggregation and multihop queries, pruned schema generates Cypher that is structurally more similar to gold (BLEU 0.598→0.632 and 0.659→0.726 respectively). The model generates cleaner query structure when irrelevant schema elements are removed.


**4. Aggregation failures are structural, not entity-related**

Aggregation failures stem from incorrect WITH/RETURN ordering, node-relationship alias confusion, and missing GROUP BY patterns — not from entity grounding errors. Neither entity extraction nor schema pruning addresses these structural failures, suggesting they require continued finetuning or chain-of-thought prompting.

---



## Limitations and Future Work

**Limitations:**
- 5,000 order sample limits generalizability to full-scale graph
- 50-query benchmark is domain-specific and not directly comparable to general benchmarks like CypherBench
- SOLD_BY direction bias is a model artifact that no inference-time strategy resolves

**Future work:**
- DPO with execution-derived preference pairs to correct systematic direction errors
- Extending entity extraction to also extract relationship triples (motivated by [Refining Text2Cypher with RL](https://www.mdpi.com/2076-3417/15/15/8206))
- Expanding benchmark to 200+ queries using auto-generation pipeline
- Evaluating schema pruning strategies (exact-match, NER-masked, similarity-based) in controlled ablation

---

## References

- Ozsoy, M.G. (2025). Text2Cypher: Bridging Natural Language and Graph Databases. *GenAIK @ ACL 2025*
- Ozsoy, M.G. (2025). Enhancing Text2Cypher with Schema Filtering. *LLM-TEXT2KG @ ESWC 2025*
- Chauhan et al. (2025). Mind the Query: A Benchmark Dataset towards Text2Cypher Task. *EMNLP Industry 2025*
- Feng et al. (2025). CypherBench: Towards Precise Retrieval over Full-scale Modern Knowledge Graphs. *ACL 2025*
- Neo4j Text2Cypher Gemma-3-4B: https://huggingface.co/neo4j/text-to-cypher-Gemma-3-4B-Instruct-2025.04.0
- Brazilian Olist Dataset: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

---

## Setup

```bash
# clone
git clone https://github.com/yourusername/ecom-text2cypher
cd ecom-text2cypher

# install dependencies
pip install neo4j pandas python-dotenv groq sentence-transformers

# configure credentials
cp .env.example .env
# fill in NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, GROQ_API_KEY, DATA_PATH

# step 1 — build knowledge graph
python ingest.py

# step 2 — validate benchmark queries
python validate_dataset.py

# step 3 — extract graph property values for entity grounding
python fix_extractor.py
```

**Full evaluation** is in `schema_prune.ipynb` — run on Kaggle with a T4 GPU runtime:
1. Upload `test_dataset.json` and `graph_values.json` to the Kaggle dataset
2. Set Neo4j and Groq credentials in the notebook
3. Add your HuggingFace token via Kaggle Secrets (`HF_TOKEN`)
4. Run all cells — evaluation takes ~25 minutes on T4

---

## Citation

If you use this benchmark or findings in your work:

```bibtex
@misc{ecom-text2cypher-2025,
  title={Inference-Time Strategies for Domain-Specific NL-to-Cypher Generation},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/ecom-text2cypher}
}
```
