from typing import Literal
from io import BytesIO
from pydantic import BaseModel, ConfigDict
from fastapi import File, UploadFile


class DocumentRequest(BaseModel):
    """
    Request model for document processing.

    Attributes:
        type (Literal["fornitori", "bandi"]): The type of document being processed, either "fornitori" or "bandi".
        stream (BytesIO): The binary stream of the PDF file to be processed.
        name (str): The name of the document file.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["fornitori", "bandi"]
    name: str
    stream: UploadFile = File(..., description="The PDF file to be processed")
