import time
import sys
import os
from client.vector_store_client import VectorStoreClient
from concurrent import futures
import hashlib
import itertools
from datasets import load_dataset
import threading

NUM_ITEMS = 80_000
NUM_QUERIES = 16_000
BATCH_SIZE = 250


def make_id(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**63 - 1)


def load_data():
    dataset = load_dataset("ms_marco", "v1.1", split="train")

    items = []
    queries = []

    # Extract passages
    for i, sample in enumerate(dataset):
        if len(items) >= NUM_ITEMS:
            break

        passages = sample["passages"]["passage_text"]
        for p in passages:
            if p.strip():
                items.append((make_id(p), p))
                break  # take first valid passage

    # Extract queries
    for i, sample in enumerate(dataset):
        if len(queries) >= NUM_QUERIES:
            break

        q = sample["query"]
        if q.strip():
            queries.append(q)

    return items, queries


def seed_data(client: VectorStoreClient, items):
    counter = itertools.count(1)
    lock = threading.Lock()
    latencies = []

    start_total = time.perf_counter()

    def upsert_batch(batch):
        start = time.perf_counter()
        statuses = client.upsert_batch(batch)
        elapsed_ms = (time.perf_counter() - start) * 1000

        n = next(counter)
        with lock:
            print(f"Progress: {n*BATCH_SIZE}/{len(items)}")
        return elapsed_ms

    batches = [items[i : i + BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]

    with futures.ThreadPoolExecutor(max_workers=4) as executor:
        latencies = list(executor.map(upsert_batch, batches))

    total_time_ms = (time.perf_counter() - start_total) * 1000

    print(f"\nUpsert Stats")
    print(f"Count: {len(items)}")
    print(f"Total time (parallel): {total_time_ms:.2f}ms")
    print(f"Avg: {sum(latencies)/len(latencies):.2f}ms per batch")
    print(f"Min: {min(latencies):.2f}ms per batch")
    print(f"Max: {max(latencies):.2f}ms per batch")


def run_queries(client: VectorStoreClient, queries):
    latencies = []

    start_total = time.perf_counter()

    def run_query(query):
        start = time.perf_counter()
        client.search(query, top_k=3)
        return (time.perf_counter() - start) * 1000

    with futures.ThreadPoolExecutor(max_workers=8) as executor:
        latencies = list(executor.map(run_query, queries))

    total_time_ms = (time.perf_counter() - start_total) * 1000

    print(f"\nSearch Stats")
    print(f"Count: {len(queries)}")
    print(f"Total time (parallel): {total_time_ms:.2f}ms")
    print(f"Avg: {sum(latencies)/len(latencies):.2f}ms")
    print(f"Min: {min(latencies):.2f}ms")
    print(f"Max: {max(latencies):.2f}ms")


if __name__ == "__main__":
    print("Loading MS MARCO...")
    ITEMS, QUERIES = load_data()

    with VectorStoreClient() as client:
        print("Seeding data...")
        seed_data(client, ITEMS)

        print("\nRunning queries...")
        run_queries(client, QUERIES)
