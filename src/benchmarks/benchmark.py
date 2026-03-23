import time
import sys
import os
from client.vector_store_client import VectorStoreClient

ITEMS = [
    # Animals
    (1, "I love dogs."),
    (2, "Dogs are very loyal animals."),
    (3, "My puppy enjoys running in the park."),
    (4, "Cats are very independent."),
    (5, "I adopted a kitten yesterday."),
    (6, "The dog likes to jump."),
    (7, "The cat likes to jump."),
    (8, "A wolf is similar to a wild dog."),
    (9, "Lions are big cats."),
    (10, "Birds can fly high in the sky."),
    # Emotions / sentiment
    (20, "I am feeling very happy today."),
    (21, "This is the best day ever!"),
    (22, "I feel sad and tired."),
    (23, "Today has been a terrible day."),
    (24, "I am extremely excited about the future."),
    (25, "I feel depressed and lonely."),
    # Tech
    (30, "Python is a great programming language."),
    (31, "I enjoy writing backend services."),
    (32, "Distributed systems are fascinating."),
    (33, "Vector databases are useful for AI."),
    (34, "Machine learning powers modern applications."),
    (35, "I like building web apps with TypeScript."),
    # Actions
    (40, "He runs every morning."),
    (41, "She enjoys jogging at sunrise."),
    (42, "They sprinted across the field."),
    (43, "He walks slowly in the evening."),
    (44, "She strolled through the park."),
    (45, "They marched forward together."),
    # Random noise (realistic data)
    (60, "The weather is nice today."),
    (61, "I had pizza for lunch."),
    (62, "The movie was surprisingly good."),
    (63, "Coffee keeps me awake."),
    (64, "Music helps me focus."),
]

QUERIES = [
    # Animal semantics
    "A dog jumping over a fence",
    "A playful puppy running outside",
    "Big wild cats in nature",
    "A kitten jumping in the house",
    # Sentiment
    "I feel extremely happy",
    "This is the worst day of my life",
    "I am excited and joyful",
    "I feel very depressed",
    # Tech semantics
    "AI embeddings and vector databases",
    "Backend programming and APIs",
    "Machine learning systems",
    "Building apps with Python",
    # Action similarity
    "Running very fast in the morning",
    "Walking slowly through a park",
    "People sprinting together",
    # Noise / mixed intent
    "Good weather and coffee",
    "Watching a great movie",
    "Relaxing with music",
]


def seed_data(client: VectorStoreClient):
    latencies = []
    for item_id, text in ITEMS:
        start = time.perf_counter()
        status = client.upsert(item_id, text)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
        print(f"Upsert ID: {item_id} | {elapsed_ms:.2f}ms | {status}")

    print(f"\nUpsert Stats")
    print(f"Avg: {sum(latencies)/len(latencies):.2f}ms")
    print(f"Min: {min(latencies):.2f}ms")
    print(f"Max: {max(latencies):.2f}ms")


def run_queries(client: VectorStoreClient):
    latencies = []
    for query in QUERIES:
        start = time.perf_counter()
        results = client.search(query, top_k=3)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
        print(f"Query: {query} | {elapsed_ms:.2f}ms | {results}")
        for text, score in results:
            print(f"  [{score:.4f}] {text}")

    print(f"\nSearch Stats")
    print(f"Avg: {sum(latencies)/len(latencies):.2f}ms")
    print(f"Min: {min(latencies):.2f}ms")
    print(f"Max: {max(latencies):.2f}ms")


if __name__ == "__main__":
    with VectorStoreClient() as client:
        print("Seeding data...")
        seed_data(client)
        print("\nRunning queries...")
        run_queries(client)
