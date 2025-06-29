import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import VectorParams, Distance

# Load environment variables
load_dotenv(override=True)


# configurations for the collections
def create_collections():
    """Create the 'bandi' and 'fornitori' collections in Qdrant with the necessary configurations."""
    for COLLECTION_NAME in ["bandi", "fornitori"]:
        # initialize Qdrant client
        client = QdrantClient(
            url=os.getenv("QDRANT_URL")
        )

        # check if the collection already exists
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        if COLLECTION_NAME in collection_names:
            print(f"Collection '{COLLECTION_NAME}' already exists. Deleting it...")
            client.delete_collection(COLLECTION_NAME)

        # Create the collection with configuration for hybrid search
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                # Dense vector (semantic)
                "dense": VectorParams(size=768, distance=Distance.COSINE),
                # Late interaction vector (ColBERT)
                "colbertv2.0": VectorParams(
                    size=128,
                    distance=Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM,
                    ),
                ),
            },
            sparse_vectors_config={
                # Sparse vector for BM25 (bag-of-words)
                "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF),
            },
        )

        print(f"Collection '{COLLECTION_NAME}' created successfully!")

        # show collection information
        collection_info = client.get_collection(COLLECTION_NAME)
        print(f"Status: {collection_info.status}")
        print(
            f"Configured vectors: {list(collection_info.config.params.vectors.keys())}"
        )
        print(
            f"Sparse vectors: {list(collection_info.config.params.sparse_vectors.keys() if collection_info.config.params.sparse_vectors else [])}"
        )
        print(f"Points: {collection_info.points_count}")
        print("\n" + "=" * 50 + "\n")
