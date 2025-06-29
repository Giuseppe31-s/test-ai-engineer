# %%
import os
import uuid
from tqdm.auto import tqdm
from typing import List
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from fastembed.sparse.bm25 import Bm25
from fastembed.late_interaction import LateInteractionTextEmbedding
from fastembed import TextEmbedding
from fastembed.sparse.bm25 import Bm25

import json
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


# %%
# dense_embedding_model = TextEmbedding(
#     "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# )
# bm25_embedding_model = Bm25("Qdrant/bm25")

# %%
import os
import uuid
from tqdm.auto import tqdm
from typing import List
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from fastembed.sparse.bm25 import Bm25
from fastembed.late_interaction import LateInteractionTextEmbedding
from fastembed import TextEmbedding

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter


# Constants
PDF_PATH = "./D23569.pdf"
EMBED_MODEL_ID = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
MAX_TOKENS = 750
NER_MODEL_NAME = "nickprock/bert-italian-finetuned-ner"  # Italian  NER model
MIN_ENTITY_CONFIDENCE = 0.80  # Minimum confidence threshold for entity extraction


def convert_pdf_to_document(pdf_path):
    """
    Convert a PDF file to a structured document format.
    """
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    return result.document


def create_document_chunks(document, embed_model_id, max_tokens):
    """
    Split the document into manageable chunks for processing.
    """
    tokenizer = AutoTokenizer.from_pretrained(embed_model_id)

    chunker = HybridChunker(
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        merge_peers=True,
    )

    chunk_iter = chunker.chunk(dl_doc=document)
    return list(chunk_iter)


def setup_ner_pipeline(model_name):
    """
    Set up a Named Entity Recognition pipeline.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    return pipeline(
        "ner", model=model, tokenizer=tokenizer, aggregation_strategy="first"
    )


def extract_entities_from_chunk(chunk_text, ner_pipeline, min_confidence=0.75):
    """
    Extract named entities from a text chunk with quality filtering.

    Args:
        chunk_text (str): The text to extract entities from
        ner_pipeline: The NER pipeline to use
        min_confidence (float): Minimum confidence threshold

    Returns:
        dict: Dictionary of entity types and their values
    """
    # Limit text size to prevent performance issues

    # Extract entities using the NER pipeline
    raw_entities = ner_pipeline(chunk_text)

    # Filter entities by confidence and length
    filtered_entities = {}

    for entity in raw_entities:
        # Skip very short tokens (unless they're acronyms)
        if len(entity["word"]) < 2 and not entity["word"].isupper():
            continue

        # Filter by confidence score
        if entity.get("score", 0) < min_confidence:
            continue

        entity_type = entity["entity_group"]

        if entity_type not in filtered_entities:
            filtered_entities[entity_type] = set()

        filtered_entities[entity_type].add(entity["word"])

    # Convert sets to lists for better serialization
    for entity_type in filtered_entities:
        filtered_entities[entity_type] = list(filtered_entities[entity_type])

    return filtered_entities


def enrich_document_with_metadata(text, ner_pipeline, name=None):
    """
    Enrich a documents text with metadata such as page numbers and headings,
    and extract named entities using a NER pipeline.
    """

    # Basic metadata
    metadata = {
        "page_numbers": getattr(text.meta, "page_numbers", None)
        if hasattr(text, "meta")
        else None,
    }
    metadata["name"] = name if name else None

    # Add headings if available
    if hasattr(text, "meta") and hasattr(text.meta, "headings"):
        metadata["headings"] = text.meta.headings

    try:
        entities = extract_entities_from_chunk(
            text, ner_pipeline, MIN_ENTITY_CONFIDENCE
        )
        if entities:
            metadata["entities"] = entities
    except Exception as e:
        metadata["entities_error"] = str(e)

    return {"text": text, "metadata": metadata}


def initialize_embedding_models():
    """
    Initialize the three embedding models needed for hybrid search:
    """
    dense_embedding_model = TextEmbedding(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    bm25_embedding_model = Bm25("Qdrant/bm25")

    colbert_embedding_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")

    return dense_embedding_model, bm25_embedding_model, colbert_embedding_model


def create_embeddings(chunk_text, dense_model, bm25_model, colbert_model):
    """
    Create the three types of embeddings for a text chunk.
    """
    # Generate embeddings for each model
    dense_embedding = list(dense_model.passage_embed([chunk_text]))[0].tolist()
    sparse_embedding = list(bm25_model.passage_embed([chunk_text]))[0].as_object()
    colbert_embedding = list(colbert_model.passage_embed([chunk_text]))[0].tolist()

    return {
        "dense": dense_embedding,
        "sparse": sparse_embedding,
        "colbertv2.0": colbert_embedding,
    }


def prepare_point(data_text: dict, embedding_models):
    """
    Prepare a single data point for Qdrant ingestion.
    """
    dense_model, bm25_model, colbert_model = embedding_models

    # Extract text from chunk based on your structure
    text = data_text.get("text", "")

    # Create embeddings
    embeddings = create_embeddings(text, dense_model, bm25_model, colbert_model)

    # Prepare payload with metadata from chunk
    payload = {"text": text, "metadata": data_text.get("metadata", {})}

    # Create and return the point
    return PointStruct(
        id=str(uuid.uuid4()),
        vector={
            "dense": embeddings["dense"],
            "sparse": embeddings["sparse"],
            "colbertv2.0": embeddings["colbertv2.0"],
        },
        payload=payload,
    )


def upload_in_batches(
    client: QdrantClient,
    collection_name: str,
    points: List[PointStruct],
    batch_size: int = 10,
):
    """
    Upload points to Qdrant in batches with progress tracking.
    If only one point is provided, upload it directly.
    """
    print(f"Uploading {len(points)} point(s) to collection '{collection_name}'")

    if len(points) == 1:
        client.upload_points(collection_name=collection_name, points=points)
    else:
        for i in tqdm(range(0, len(points), batch_size), desc="Uploading batches"):
            batch = points[i : i + batch_size]
            client.upload_points(collection_name=collection_name, points=batch)

    print(
        f"Successfully uploaded {len(points)} point(s) to collection '{collection_name}'"
    )


def process_and_upload_chunks(text, collection_name):
    """
    Process document chunks and upload them to Qdrant.
    """
    # Load environment variables
    load_dotenv()

    # Initialize client
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
    )

    # Initialize embedding models
    embedding_models = initialize_embedding_models()

    # Prepare points
    print("Preparing points with embeddings...")

    point = prepare_point(text, embedding_models)



    # Upload points in batches
    upload_in_batches(
        client=client,
        collection_name=collection_name,
        points=[point],  # Wrap in a list for single point upload
        batch_size=1,  # Adjust based on your document size and memory constraints
    )

    # Print confirmation with collection info
    collection_info = client.get_collection(collection_name)
    print(
        f"Collection '{collection_name}' now has {collection_info.points_count} points"
    )


# Running
# Convert PDF to document
# document = convert_pdf_to_document(PDF_PATH)

# # Create document chunks
# chunks = create_document_chunks(document, EMBED_MODEL_ID, MAX_TOKENS)
# print(f"Document split into {len(chunks)} chunks")

# Set up NER pipeline
# ner_pipeline = setup_ner_pipeline(NER_MODEL_NAME)

# # Process chunks and extract metadata
# enriched_chunks = enrich_document_with_metadata(chunks, ner_pipeline)

# Display results
# for i, chunk in enumerate(enriched_chunks):
#     print(chunk)

# # Send data to Qdrant
# COLLECTION_NAME = os.getenv("COLLECTION_NAME")
# process_and_upload_chunks(enriched_chunks, COLLECTION_NAME)

# %%
with open("../data/gold/technic_specification.json", "r", encoding="utf-8") as f:
    information = json.load(f)

# %%
information

# %%
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import VectorParams, Distance

# Load environment variables
load_dotenv(override=True)


# configurations for the collections

for COLLECTION_NAME in ["bandi", "fornitori"]:
    # initialize Qdrant client
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
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
                )
            )
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
    print(f"Configured vectors: {list(collection_info.config.params.vectors.keys())}")
    print(
        f"Sparse vectors: {list(collection_info.config.params.sparse_vectors.keys() if collection_info.config.params.sparse_vectors else [])}"
    )
    print(f"Points: {collection_info.points_count}")
    print("\n" + "=" * 50 + "\n")


# %%
# Set up NER pipeline
ner_pipeline = setup_ner_pipeline(NER_MODEL_NAME)

for key in information:
    print(f"Processing {key}...")

    # Process chunks and extract metadata
    enriched_chunks = enrich_document_with_metadata(
        information[key], ner_pipeline, name=key
    )

    # Send data to Qdrant
    test = process_and_upload_chunks(enriched_chunks, "bandi")

# %%
client = QdrantClient(
    url=os.getenv("QDRANT_URL")
)

# %%
all_points = []

# Use scroll to paginate through the collection
scroll_offset = None
while True:
    points, scroll_offset = client.scroll(
        collection_name="bandi",
        limit=100,  # Adjust batch size as needed
        offset=scroll_offset,
        with_payload=True,
        with_vectors=True,
    )

    all_points.extend(points)

    if scroll_offset is None:
        break  # No more points

# %%
collection_name = "bandi"

# %%
all_points[0].vector.keys()

# %%
search_result = client.query_points(
    collection_name=collection_name,
    # First stage: Get candidates using dense and sparse search
    prefetch=[
        {
            "query": all_points[0].vector["dense"],
            "using": "dense",
            "limit": 5,
        },
        {
            "query": all_points[0].vector["sparse"],
            "using": "sparse",
            "limit": 5,
        },
    ],
    # Second stage: Rerank using late interaction
    query=all_points[0].vector["colbertv2.0"],
    using="colbertv2.0",
    with_payload=True,
    with_vectors=True,
    limit=5,
)

# %%
print(search_result.model_dump())


