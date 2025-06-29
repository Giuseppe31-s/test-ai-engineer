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
from docling.document_converter import DocumentConverter


class EmbeddingProcessor:
    """A class to handle the embedding and processing of documents for Qdrant ingestion."""

    EMBED_MODEL_ID = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    MAX_TOKENS = 750
    NER_MODEL_NAME = "nickprock/bert-italian-finetuned-ner"  # Italian  NER model
    MIN_ENTITY_CONFIDENCE = 0.80  # Minimum confidence threshold for entity extraction

    def __init__(self):
        """
        Initialize the EmbeddingProcessor with necessary configurations.
        """
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
        )
        self.ner_pipeline = self.setup_ner_pipeline(self.NER_MODEL_NAME)
        self.converter = DocumentConverter()

    def convert_pdf_to_document(self, pdf_path):
        """
        Convert a PDF file to a structured document format.
        """

        result = self.converter.convert(pdf_path)
        return result.document

    def setup_ner_pipeline(self, model_name):
        """
        Set up a Named Entity Recognition pipeline.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)

        return pipeline(
            "ner", model=model, tokenizer=tokenizer, aggregation_strategy="first"
        )

    def extract_entities_from_chunk(self, chunk_text, min_confidence=0.75):
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
        raw_entities = self.ner_pipeline(chunk_text)

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

    def enrich_document_with_metadata(self, text, name=None):
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
            entities = self.extract_entities_from_chunk(
                text, self.MIN_ENTITY_CONFIDENCE
            )
            if entities:
                metadata["entities"] = entities
        except Exception as e:
            metadata["entities_error"] = str(e)

        return {"text": text, "metadata": metadata}

    def initialize_embedding_models(self):
        """
        Initialize the three embedding models needed for hybrid search:
        """
        dense_embedding_model = TextEmbedding(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        bm25_embedding_model = Bm25("Qdrant/bm25")

        colbert_embedding_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")

        return dense_embedding_model, bm25_embedding_model, colbert_embedding_model

    def create_embeddings(self, chunk_text, dense_model, bm25_model, colbert_model):
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

    def prepare_point(self, data_text: dict, embedding_models):
        """
        Prepare a single data point for Qdrant ingestion.
        """
        dense_model, bm25_model, colbert_model = embedding_models

        # Extract text from chunk based on your structure
        text = data_text.get("text", "")

        # Create embeddings
        embeddings = self.create_embeddings(
            text, dense_model, bm25_model, colbert_model
        )

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
        self,
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

    def process_and_upload_documents(self, text, collection_name):
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
        embedding_models = self.initialize_embedding_models()

        # Prepare points
        print("Preparing points with embeddings...")

        point = self.prepare_point(text, embedding_models)

        # Upload points in batches
        self.upload_in_batches(
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
