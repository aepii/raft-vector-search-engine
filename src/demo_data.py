"""
Demo dataset for the video recording.
50 documents across 3 categories; 4 demo queries (one per category + one negative).

Run from src/:
    python -m scratch.video.demo_data
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from client.vector_store_client import VectorStoreClient

# ── Documents ─────────────────────────────────────────────────────────────────

# Category A: Distributed Systems
DISTRIBUTED_SYSTEMS = [
    "Consensus algorithms ensure that all nodes in a cluster agree on the same value, even in the presence of failures.",
    "In a replicated database, writes are propagated to multiple nodes to ensure data survives hardware failures.",
    "Consistent hashing distributes keys across nodes so that adding or removing a node only requires remapping a fraction of the data.",
    "Leader election protocols determine which node is responsible for coordinating writes during normal operation.",
    "Eventual consistency allows replicas to diverge temporarily, with the guarantee that they will converge given enough time.",
    "A partition in a distributed system occurs when a network failure prevents nodes from communicating with each other.",
    "The CAP theorem states that a distributed system can guarantee at most two of: consistency, availability, and partition tolerance.",
    "Sharding splits a dataset across multiple machines so that each node stores and searches only a fraction of the total data.",
    "Heartbeat messages allow nodes to detect when a peer has become unresponsive and trigger failover procedures.",
    "Replication factor determines how many copies of each piece of data are stored across the cluster.",
    "A coordinator node routes client requests to the appropriate shards based on a partitioning strategy.",
    "Read repair detects inconsistencies between replicas and corrects them during read operations.",
    "Vector clocks track the causal ordering of events across distributed nodes without requiring synchronized clocks.",
    "A gossip protocol disseminates information through a cluster by having each node periodically exchange state with a random peer.",
    "Idempotent operations can be safely retried without changing the result, which is critical for fault-tolerant systems.",
    "Log-structured storage appends writes sequentially to disk, enabling high throughput at the cost of periodic compaction.",
    "Service discovery allows nodes to find each other dynamically, without hardcoding addresses into the application.",
]

# Category B: Machine Learning
MACHINE_LEARNING = [
    "A neural network learns to transform inputs into outputs by adjusting millions of parameters through gradient descent.",
    "Overfitting occurs when a model memorizes the training data and fails to generalize to unseen examples.",
    "Embeddings are dense vector representations that place semantically similar items close together in a high-dimensional space.",
    "The attention mechanism allows a model to focus on the most relevant parts of an input when producing each output token.",
    "Transfer learning reuses features learned on one task to improve performance on a different but related task.",
    "Batch normalization stabilizes training by normalizing activations within each mini-batch to reduce internal covariate shift.",
    "A loss function measures how far the model's predictions are from the ground truth, guiding the direction of parameter updates.",
    "Dropout randomly deactivates neurons during training to prevent the model from relying on any single feature.",
    "Cross-validation estimates a model's generalization ability by training and evaluating on multiple splits of the dataset.",
    "Cosine similarity measures the angle between two vectors, making it useful for comparing embeddings independent of their magnitude.",
    "Fine-tuning adapts a pretrained model to a specific domain by continuing training on a smaller, task-specific dataset.",
    "A transformer encoder converts a sequence of tokens into contextualized representations by applying self-attention layers.",
    "Nearest-neighbor search finds the most similar vectors in a dataset to a given query vector.",
    "Approximate nearest neighbor algorithms trade a small loss in recall for a large gain in search speed at scale.",
    "Sentence transformers encode entire sentences into fixed-length vectors, enabling semantic similarity comparisons.",
    "Data augmentation artificially expands the training set by applying transformations that preserve the label but change the input.",
    "The softmax function converts a vector of raw scores into a probability distribution over a set of classes.",
]

# Category C: Ecology
ECOLOGY = [
    "A food web describes the network of predator-prey relationships that determine how energy flows through an ecosystem.",
    "Keystone species have a disproportionately large effect on their environment relative to their abundance.",
    "Mutualism is a relationship between two species in which both partners benefit from the interaction.",
    "Carrying capacity is the maximum population size that an environment can sustain given its available resources.",
    "Succession describes the process by which an ecological community changes in composition over time after a disturbance.",
    "Nutrient cycling moves essential elements like carbon and nitrogen through the biotic and abiotic components of an ecosystem.",
    "Biodiversity increases an ecosystem's resilience to environmental stress by providing functional redundancy among species.",
    "Migration allows populations to exploit seasonal resources across large geographic ranges.",
    "Competition for limited resources shapes the distribution and abundance of species within a community.",
    "Apex predators regulate the populations of prey species, which in turn affects vegetation and habitat structure.",
    "A niche defines the set of environmental conditions and resources that a species requires to survive and reproduce.",
    "Invasive species disrupt established ecosystems by outcompeting native species for resources.",
    "Decomposers break down dead organic matter, releasing nutrients back into the soil and completing the nutrient cycle.",
    "Symbiosis describes any long-term interaction between two species, including mutualism, commensalism, and parasitism.",
    "Population dynamics are driven by birth rates, death rates, immigration, and emigration within a habitat.",
    "Habitat fragmentation isolates subpopulations, reducing genetic diversity and increasing extinction risk.",
]

DOCUMENTS: list[tuple[int, str]] = [
    (i, text)
    for i, text in enumerate(DISTRIBUTED_SYSTEMS + MACHINE_LEARNING + ECOLOGY)
]

# ── Demo queries ───────────────────────────────────────────────────────────────
# One per category, deliberately using different words than any stored document.
# Plus one out-of-domain negative example.

DEMO_QUERIES = [
    {
        "label": "Distributed Systems",
        "query": "How do you keep a cluster running when some machines go offline?",
        "expect": "matches on replication, heartbeats, failover, eventual consistency",
    },
    {
        "label": "Machine Learning",
        "query": "How does a computer learn to understand the meaning of words?",
        "expect": "matches on embeddings, sentence transformers, attention",
    },
    {
        "label": "Ecology",
        "query": "What keeps a natural community from collapsing when one species disappears?",
        "expect": "matches on biodiversity, keystone species, functional redundancy",
    },
    {
        "label": "Out-of-domain (negative)",
        "query": "What is the best way to proof sourdough bread at home?",
        "expect": "low scores across the board — nothing relevant in the dataset",
    },
]

# ── Ingest ────────────────────────────────────────────────────────────────────

def ingest():
    print(f"Ingesting {len(DOCUMENTS)} documents...")
    with VectorStoreClient() as client:
        statuses = client.upsert_batch(DOCUMENTS)
    print(f"Done. {len(statuses)} acks received.")


# ── Query demo ─────────────────────────────────────────────────────────────────

def query_demo(top_k: int = 3):
    all_results = []
    with VectorStoreClient() as client:
        for q in DEMO_QUERIES:
            all_results.append((q, client.search(q["query"], top_k=top_k)))

    for q, results in all_results:
        print(f"\n{'─' * 60}")
        print(f"  [{q['label']}]")
        print(f"  Query: {q['query']}")
        print(f"{'─' * 60}")
        if results:
            for rank, (text, score) in enumerate(results, 1):
                print(f"  {rank}. (score={score:.4f})  {text}")
        else:
            print("  (no results)")
    print(f"\n{'─' * 60}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["ingest", "query"], nargs="?", default="query")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    if args.mode == "ingest":
        ingest()
    else:
        query_demo(top_k=args.top_k)
