from src.embedding.embedding import EmbeddingProcessor
import json


# Read PDF from disk and load into BytesIO
def insert_gold_data():
    print("Inserting gold data into Qdrant...")
    try:
        embedder = EmbeddingProcessor()
        with open(
            "./data/gold/bandi/technic_specification.json", "r", encoding="utf-8"
        ) as file:
            bandi = json.load(file)

        with open(
            "./data/gold/fornitori/fornitori_description.json", "r", encoding="utf-8"
        ) as file:
            fornitori = json.load(file)

        for key in bandi:
            print(f"Processing {key}...")
            document_enriched = embedder.enrich_document_with_metadata(
                bandi[key], name=key
            )
            embedder.process_and_upload_documents(document_enriched, "bandi")

        for key in fornitori:
            print(f"Processing {key}...")
            document_enriched = embedder.enrich_document_with_metadata(
                fornitori[key], name=key
            )
            embedder.process_and_upload_documents(document_enriched, "fornitori")
        print("Gold data inserted successfully.")
    except Exception as e:
        print(f"Error inserting gold data: {e}")
