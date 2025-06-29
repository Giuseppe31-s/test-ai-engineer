from fastapi import FastAPI, APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from src.embedding.embedding import EmbeddingProcessor
from src.data.preprocess import extract_technic_specification
from src.model.requests import DocumentRequest
from dotenv import load_dotenv
from loguru import logger
from src.retrivel.retrivel import Retrivel
from fastapi.responses import RedirectResponse
import uvicorn
from insert_informations import insert_gold_data
from create_colletions import create_collections
from io import BytesIO
from typing import Literal


logger.add(
    "test.log",
    rotation="50 MB",
    compression="zip",
    level="TRACE",
)


class API:
    load_dotenv(override=True)

    def __init__(self):
        self.app = FastAPI()
        self.router = APIRouter()
        self.embedder = EmbeddingProcessor()
        self.retrivel = Retrivel()

        # Include the router in the FastAPI app
        self.app.include_router(self.router)

        # Add a simple health check endpoint
        self.router.add_api_route("/v1/health", self.health_check, methods=["GET"])
        # Add the document insertion endpoint
        self.router.add_api_route(
            "/v1/insert_documents", self.insert_document, methods=["POST"]
        )
        # Add the score matrix retrieval endpoint
        self.router.add_api_route(
            "/v1/get_score_matrix", self.get_score_matrix, methods=["GET"]
        )
        # Add a redirect for the docs endpoint
        self.router.add_api_route("/", self.docs_redirect, methods=["GET"])

    async def health_check(self):
        """
        Performs a health check of the service.

        Returns:
            JSONResponse: A response object containing the health status of the service.
        """
        return JSONResponse(content={"status": "ok"})

    async def insert_document(
        self,
        type: Literal["fornitori", "bandi"] = Form(...),
        name: str = Form(...),
        stream: UploadFile = File(...),
    ) -> JSONResponse:
        """
        Asynchronously processes and inserts a PDF document.

        This method performs the following steps:
        1. Validates the incoming request to ensure a file stream and a valid document type are provided.
        2. Extracts technical specifications from the provided PDF document.
        3. Enriches the extracted specifications with additional metadata.
        4. Processes and uploads the enriched document in chunks.
        5. Returns a JSON response indicating success and containing the enriched document data, or an error message if processing fails.

        Args:
            request (DocumentRequest): The request object containing the PDF file stream, document type, and document name.

        Returns:
            JSONResponse: A response object with the status and either the enriched document data or an error message.
        """
        logger.info("Processing document insertion...")

        binary_stream = BytesIO(await stream.read())

        logger.info(
            f"Received file: {name}, type: {type} with size: {len(binary_stream.getvalue())} bytes"
        )

        if not binary_stream:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "No file provided."},
            )
        if type not in ["fornitori", "bandi"]:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Invalid type provided."},
            )
        dict_request = {
            "type": type,
            "name": name,
            "stream": binary_stream,
        }

        try:
            logger.info("Extracting technical specifications from the PDF...")

            if type == "bandi":
                technic_specification = extract_technic_specification(dict_request)
            else:
                technic_specification = self.embedder.convert_pdf_to_document(
                    dict_request
                ).export_to_text()

            logger.info(
                f"Extracted technical specifications: {technic_specification[:100]}..."
            )

            logger.info("Enriching document with metadata...")
            technic_specification_enriched = (
                self.embedder.enrich_document_with_metadata(
                    technic_specification, dict_request["name"]
                )
            )
            logger.info("Enriched document with metadata successfully.")

            logger.info("Processing and uploading documents...")
            self.embedder.process_and_upload_documents(
                technic_specification_enriched, dict_request["type"]
            )
            logger.success("Document processed and uploaded successfully.")

            return JSONResponse(
                content={
                    "status": "success",
                    "data": technic_specification_enriched,
                }
            )
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": str(e)},
            )

    async def get_score_matrix(self) -> JSONResponse:
        """
        Asynchronously retrieves the score matrix and returns it as a JSON response.

        Returns:
            JSONResponse: A response object containing the status and the score matrix data if successful,
                          or an error message with status code 500 if an exception occurs.

        Raises:
            Exception: Logs and returns an error response if any exception is raised during retrieval.
        """
        try:
            logger.info("Retrieving score matrix...")
            score_matrix = self.retrivel.get_scores_matrix().to_dict()
            return JSONResponse(
                content={
                    "status": "success",
                    "data": score_matrix,
                }
            )
        except Exception as e:
            logger.error(f"Error retrieving score matrix: {e}")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": str(e)},
            )

    async def docs_redirect(self):
        """
        Asynchronously redirects the client to the API documentation page located at '/docs'.

        Returns:
            RedirectResponse: A response object that redirects the client to the '/docs' URL.
        """
        return RedirectResponse(url="/docs")


app = FastAPI(
    version="1.0",
    title="Test AI Engineer API",
    description="API for processing and retrieving documents related to 'fornitori' and 'bandi'.",
)


if __name__ == "__main__":
    # Create collections in Qdrant
    create_collections()
    # # Insert gold data into Qdrant
    insert_gold_data()
    api = API()
    app.include_router(api.router)
    uvicorn.run(app, host="0.0.0.0", port=8000)
