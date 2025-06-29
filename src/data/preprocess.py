from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.io import DocumentStream
from io import StringIO
import pandas as pd
from openai import OpenAI
import os


def extract_html_from_pdf(file_pdf: str | DocumentStream) -> pd.DataFrame:
    """
    Extracts tables from HTML content and returns them as a DataFrame.
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    pipeline_options.ocr_options.lang = ["it"]
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    doc = doc_converter.convert(file_pdf).document
    html_content = doc.export_to_html()
    return html_content


def tables_from_html(html_content: str) -> pd.DataFrame:
    """
    Extracts tables from HTML content and returns them as a DataFrame.
    """
    df_technic_specification = pd.DataFrame()
    for table in pd.read_html(StringIO(html_content)):
        if table.shape[1] > 2:
            # Filter out tables with more than 2 columns
            continue
        df_technic_specification = pd.concat(
            [df_technic_specification, table], ignore_index=True
        )
    df_technic_specification.fillna("", inplace=True)
    df_technic_specification = df_technic_specification.T
    df_technic_specification.columns = df_technic_specification.iloc[0]
    df_technic_specification = df_technic_specification[1:]
    df_technic_specification.columns = df_technic_specification.columns.str.lower()
    return df_technic_specification


def get_description_from_df(df: pd.DataFrame) -> str:
    """
    Extracts the description from the DataFrame.
    """

    columns_with_service = df.columns[df.columns.str.contains("servizio")]

    description = ""
    for text in df[columns_with_service].values:
        full_text = " ".join(str(item) for item in text if item)
        # Append the text to the description
        description += full_text + "\n"

    return description.strip()


def assistant_extractor(html_content: str, provider: str = "openai") -> str:
    """Extracts the description of a service from HTML content using an AI assistant."""
    system_prompt = (
        "You are an expert in extracting information from technical specifications. "
        "Your task is to extract the description of the service from the provided HTML content."
        "Don't invent information, just extract the relevant details."
        "Use the following HTML content to extract the description of the service:"
        ""
    )
    messages = [{"role": "system", "content": system_prompt}] + [
        {"role": "user", "content": html_content}
    ]

    if provider == "openai":
        openai = OpenAI()
        model = "gpt-4o"
    elif provider == "together":
        openai = OpenAI(
            api_key=os.getenv("TOGETHER_API_KEY"),
            base_url=os.getenv("TOGETHER_BASE_URL"),
        )
        model = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    else:
        raise ValueError("Provider must be 'openai' or 'together'.")

    response = openai.chat.completions.create(
        model=model,
        messages=messages,
    )
    answer = response.choices[0].message.content
    return answer


def extract_technic_specification(file_pdf: str | DocumentStream) -> pd.DataFrame:
    """
    Extracts technical specifications from a PDF file and returns them as a DataFrame.
    """
    html_content = extract_html_from_pdf(file_pdf)
    df_technic_specification = tables_from_html(html_content)

    if sum(df_technic_specification.columns.str.contains("servizio")) <= 2:
        return assistant_extractor(html_content, os.getenv("PROVIDER", "openai"))

    return get_description_from_df(df_technic_specification)
