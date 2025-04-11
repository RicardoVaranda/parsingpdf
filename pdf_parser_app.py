# Standard Library Imports
import os
import json
import logging
import base64
from datetime import datetime
from typing import List
import uuid
import io
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Environment Variables
from dotenv import load_dotenv

# Streamlit
import streamlit as st

# AI/LLM Clients
import ollama
import anthropic
from openai import OpenAI
from google import genai
from google.genai import types
from mistralai import Mistral

# LangChain Core
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# LangChain Integrations
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# LangChain Chains & Vector Stores
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Document Loaders
from langchain_community.document_loaders import (
    PDFMinerLoader,
    PDFPlumberLoader,
    PyMuPDFLoader,
    PyPDFLoader,
)
import camelot
from docling.document_converter import DocumentConverter
from markitdown import MarkItDown

# AWS Services
import boto3

# Utilities
from utils.pdf_to_image import PDFToJPGConverter
from PIL import Image, ImageDraw

torch.classes.__path__ = []

from torchvision import transforms

# Reader for OCR
import numpy as np
import csv
import easyocr
from tqdm.auto import tqdm

# For Displaying Table
import pandas as pd

from transformers import AutoModelForObjectDetection, TableTransformerForObjectDetection


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOGGING_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# LLM Provider Configurations
LLM_CONFIGS = {
    "Groq": {
        "models": [
            # "llama3-8b-8192",
            # "llama3-70b-8192",
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            # "gemma2-9b-it",
            # "mixtral-8x7b-32768",
        ],
        "requires_key": "GROQ_API_KEY",
    },
    "OpenAI": {
        "models": ["gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18"],
        "requires_key": "OPENAI_API_KEY",
    },
    "Anthropic": {
        "models": [
            "claude-3-7-sonnet-latest",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
        ],
        "requires_key": "ANTHROPIC_API_KEY",
    },
    # "Gemini": {
    #     "models": [
    #         "gemini-2.0-flash-exp",
    #         "gemini-1.5-flash",
    #         "gemini-1.5-flash-8b",
    #         "gemini-1.5-pro",
    #     ],
    #     "requires_key": "GOOGLE_API_KEY",
    # },
}

# Table Image Detection Configurations
TABLE_IMAGE_DETECTION_CONFIGS = {
    "microsoft/table-transformer-detection": {
        "description": "Table detection model from Microsoft",
        "requires_key": None,
    }
}

# Parser Configurations
PARSER_CONFIGS = {
    "Docling": {
        "description": "Advanced document understanding",
        "requires_api_key": None,
    },
    "MarkItDown": {
        "description": "Converts PDFs to markdown format",
        "requires_api_key": None,
    },
    # "Gemini 2.0": {
    #     "description": "Uses Gemini's native PDF processing capabilities",
    #     "requires_api_key": "GOOGLE_API_KEY",
    # },
    # "Claude": {
    #     "description": "Uses Claude's native PDF processing capabilities",
    #     "requires_api_key": "ANTHROPIC_API_KEY",
    # },
    # "OpenAI": {
    #     "description": "Uses GPT-4o for processing PDFs",
    #     "requires_api_key": "OPENAI_API_KEY",
    # },
    # "Mistral-OCR": {
    #     "description": "Uses Mistral-OCR for processing PDFs",
    #     "requires_api_key": "MISTRAL_API_KEY",
    # },
    "Camelot": {
        "description": "Specialized in table extraction",
        "requires_api_key": None,
    },
    "PyPDF": {"description": "Simple text extraction", "requires_api_key": None},
    "PDFPlumber": {
        "description": "Good for text and simple tables",
        "requires_api_key": None,
    },
    "PDFMiner": {
        "description": "Basic text extraction with layout preservation",
        "requires_api_key": None,
    },
    "PyMuPDF": {
        "description": "Fast processing with good layout preservation",
        "requires_api_key": None,
    },
    # "Amazon Textract": {
    #     "description": "AWS service for document processing",
    #     "requires_api_key": "AWS_ACCESS_KEY_ID",
    # },
    "Llama Vision": {
        "description": "Uses Llama 3.2 Vision model",
        "requires_api_key": None,
    },
}


class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize(
            (int(round(scale * width)), int(round(scale * height)))
        )

        return resized_image


class RAGSystem:
    """Handles RAG functionality with different LLM providers"""

    def __init__(self, provider: str, model: str, temperature: float = 0.7):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L12-v2"
        )
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the appropriate LLM based on provider"""
        try:
            if self.provider == "Groq":
                os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
                return ChatGroq(temperature=self.temperature, model_name=self.model)
            elif self.provider == "OpenAI":
                os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
                return ChatOpenAI(model_name=self.model, temperature=self.temperature)
            elif self.provider == "Anthropic":
                os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
                return ChatAnthropic(
                    model_name=self.model, temperature=self.temperature
                )
            elif self.provider == "Gemini":
                os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
                return ChatGoogleGenerativeAI(
                    model=self.model, temperature=self.temperature
                )
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            logger.error(
                f"Error initializing LLM for {self.provider}: {str(e)}", exc_info=True
            )
            raise

    def create_vector_store(self, texts: List[str]) -> FAISS:
        return FAISS.from_texts(texts, self.embeddings)

    def setup_qa_chain(self, vector_store):

        system_prompt = """Use the following pieces of context to answer the question at the end. 
            Check context very carefully and reference and try to make sense of that before responding.
            If you don't know the answer, just say you don't know. 
            Don't try to make up an answer.
            Answer must be to the point.
            Think step-by-step.
            Context: {context}"""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        # Create retriever
        retriever = vector_store.as_retriever()

        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        qa_chain = create_retrieval_chain(retriever, question_answer_chain)

        return qa_chain


class MultiParser:
    """Handles multiple PDF parsing methods"""

    def __init__(self, parser_name: str):
        logger.debug("new parser" + parser_name)
        self.parser_name = parser_name
        self.image_converter = PDFToJPGConverter()

    def parse_pdf(self, uploaded_file) -> str:
        pdf_content = uploaded_file.read()
        logger.debug(f"Parsing PDF with {self.parser_name}")
        try:
            if self.parser_name == "Docling":
                return self._parse_with_docling(pdf_content)
            elif self.parser_name == "MarkItDown":
                return self._parse_with_markitdown(pdf_content)
            elif self.parser_name == "Gemini 2.0":
                return self._parse_with_gemini(pdf_content)
            elif self.parser_name == "Claude":
                return self._parse_with_claude(pdf_content)
            elif self.parser_name == "OpenAI":
                return self._parse_with_openai_vision(pdf_content)
            elif self.parser_name == "Mistral-OCR":
                return self._parse_with_mistral_ocr(uploaded_file)
            elif self.parser_name == "Camelot":
                return self._parse_with_camelot(pdf_content)
            elif self.parser_name == "PyPDF":
                return self._parse_with_pypdf(pdf_content)
            elif self.parser_name == "PDFPlumber":
                return self._parse_with_pdfplumber(pdf_content)
            elif self.parser_name == "PDFMiner":
                return self._parse_with_pdfminer(pdf_content)
            elif self.parser_name == "PyMuPDF":
                return self._parse_with_pymupdf(pdf_content)
            elif self.parser_name == "Amazon Textract":
                return self._parse_with_textract(pdf_content)
            elif self.parser_name == "Llama Vision":
                return self._parse_with_llama_vision(pdf_content)
            else:
                raise ValueError(f"Unsupported parser: {self.parser_name}")
        except Exception as e:
            logger.error(
                f"Error parsing PDF with {self.parser_name}: {str(e)}", exc_info=True
            )
            raise

    def _parse_with_gemini(self, pdf_content: bytes) -> str:
        api_key = os.getenv("GOOGLE_API_KEY")
        client = genai.Client(api_key=api_key)

        prompt = """Extract all the text content, including both plain text and tables, from the 
                provided document or image. Maintain the original structure, including headers, 
                paragraphs, and any content preceding or following the table. Format the table in 
                Markdown format, preserving numerical data and relationships. Ensure no text is excluded, 
                including any introductory or explanatory text before or after the table."""

        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=pdf_content,
                    mime_type="application/pdf",
                ),
                prompt,
            ],
        )
        return response.text

    def _parse_with_claude(self, pdf_content: bytes) -> str:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        client = anthropic.Client(
            api_key=api_key, default_headers={"anthropic-beta": "pdfs-2024-09-25"}
        )

        base64_pdf = base64.b64encode(pdf_content).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": base64_pdf,
                        },
                    },
                    {
                        "type": "text",
                        "text": """Extract all the text content, including both plain text and tables, from the 
                            provided document or image. Maintain the original structure, including headers, 
                            paragraphs, and any content preceding or following the table. Format the table in 
                            Markdown format, preserving numerical data and relationships. Ensure no text is excluded, 
                            including any introductory or explanatory text before or after the table.""",
                    },
                ],
            }
        ]

        response = client.messages.create(
            model="claude-3-7-sonnet-20250219", max_tokens=1500, messages=messages
        )
        return response.content[0].text

    def _parse_with_openai_vision(self, pdf_content: bytes) -> str:
        """
        Parse PDF using OpenAI's Vision model, handling the content as bytes.

        Args:
            pdf_content (bytes): The PDF content as bytes

        Returns:
            str: Extracted text from the PDF
        """
        client = OpenAI()
        output_path = f"converted_images/ui/{str(uuid.uuid4())}"

        # Convert PDF pages to images (keeping in memory)
        images = self.image_converter.convert_pdf(
            pdf_input=pdf_content, output_dir=output_path, save_to_disk=False
        )

        full_text = ""

        for img in images:
            # Convert PIL Image to base64
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="JPEG")
            img_byte_arr = img_byte_arr.getvalue()
            base64_img = base64.b64encode(img_byte_arr).decode("utf-8")

            # Process with OpenAI Vision
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Extract all the text content, including both plain text and tables, from the 
                                        provided document or image. Maintain the original structure, including headers, 
                                        paragraphs, and any content preceding or following the table. Format the table in 
                                        Markdown format, preserving numerical data and relationships. Ensure no text is excluded, 
                                        including any introductory or explanatory text before or after the table.""",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_img}"
                                },
                            },
                        ],
                    }
                ],
            )
            full_text += response.choices[0].message.content + "\n\n"

        return full_text

    def _parse_with_mistral_ocr(self, uploaded_file) -> str:
        api_key = os.getenv("MISTRAL_API_KEY")
        client = Mistral(api_key=api_key)
        uploaded_pdf = client.files.upload(
            file={
                "file_name": uploaded_file.name,
                "content": uploaded_file.getvalue(),
            },
            purpose="ocr",
        )
        signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id).url

        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": signed_url,
            },
            include_image_base64=True,
        )
        final_response = ""

        for page in ocr_response.pages:
            final_response += page.markdown + "\n"

        return final_response

    def _parse_with_camelot(self, pdf_content: bytes) -> str:
        # Save PDF content temporarily
        with open("temp.pdf", "wb") as f:
            f.write(pdf_content)

        # Extract tables
        tables = camelot.read_pdf("temp.pdf")

        # Convert tables to markdown format
        text = ""
        for i, table in enumerate(tables):
            text += f"\nTable {i+1}:\n"
            text += table.df.to_markdown()
            text += "\n"

        # Cleanup
        os.remove("temp.pdf")
        return text

    def _parse_with_pdfminer(self, pdf_content: bytes) -> str:
        with open("temp.pdf", "wb") as f:
            f.write(pdf_content)

        loader = PDFMinerLoader("temp.pdf")
        documents = loader.load()

        os.remove("temp.pdf")
        return "\n\n".join(doc.page_content for doc in documents)

    def _parse_with_pdfplumber(self, pdf_content: bytes) -> str:
        with open("temp.pdf", "wb") as f:
            f.write(pdf_content)

        loader = PDFPlumberLoader("temp.pdf")
        documents = loader.load()

        os.remove("temp.pdf")
        return "\n\n".join(doc.page_content for doc in documents)

    def _parse_with_pymupdf(self, pdf_content: bytes) -> str:
        with open("temp.pdf", "wb") as f:
            f.write(pdf_content)

        loader = PyMuPDFLoader("temp.pdf")
        documents = loader.load()

        os.remove("temp.pdf")
        return "\n\n".join(doc.page_content for doc in documents)

    def _parse_with_pypdf(self, pdf_content: bytes) -> str:
        with open("temp.pdf", "wb") as f:
            f.write(pdf_content)

        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        os.remove("temp.pdf")
        return "\n\n".join(doc.page_content for doc in documents)

    def _parse_with_docling(self, pdf_content: bytes) -> str:
        with open("temp.pdf", "wb") as f:
            f.write(pdf_content)

        converter = DocumentConverter()
        result = converter.convert("temp.pdf")

        os.remove("temp.pdf")
        return result.document.export_to_markdown()

    def _parse_with_markitdown(self, pdf_content: bytes) -> str:
        with open("temp.pdf", "wb") as f:
            f.write(pdf_content)

        md = MarkItDown()
        result = md.convert("temp.pdf")

        os.remove("temp.pdf")
        return result.text_content

    def _parse_with_textract(self, pdf_content: bytes) -> str:
        textract = boto3.client(
            "textract",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name="us-east-1",
        )

        response = textract.detect_document_text(Document={"Bytes": pdf_content})

        return "\n".join(
            item["Text"] for item in response["Blocks"] if item["BlockType"] == "LINE"
        )

    def _parse_with_llama_vision(self, pdf_content: bytes) -> str:
        output_path = f"converted_images/ui/{str(uuid.uuid4())}"
        images = self.image_converter.convert_pdf(pdf_content, output_path)
        full_text = ""

        for img in images:
            # Convert PIL Image to base64
            import io

            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="JPEG")
            img_byte_arr = img_byte_arr.getvalue()
            base64_img = base64.b64encode(img_byte_arr).decode("utf-8")

            response = ollama.chat(
                model="llama3.2-vision",
                messages=[
                    {
                        "role": "user",
                        "content": """Extract all the text content, including both plain text and tables, from the 
                                        provided document or image. Maintain the original structure, including headers, 
                                        paragraphs, and any content preceding or following the table. Format the table in 
                                        Markdown format, preserving numerical data and relationships. Ensure no text is excluded, 
                                        including any introductory or explanatory text before or after the table.
                                        If the image contains no text, explain the image in detail.""",
                        "images": [base64_img],
                    }
                ],
            )
            full_text += response.message.content + "\n\n"

        return full_text


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


# for rescaling bounding boxes to original image size
def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


# for converting model outputs to objects
def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs["pred_boxes"].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == "no object":
            objects.append(
                {
                    "label": class_label,
                    "score": float(score),
                    "bbox": [float(elem) for elem in bbox],
                }
            )

    return objects


# for converting Matplotlib figure to PIL Image
def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def objects_to_crops(img, tokens, objects, class_thresholds, padding=10):
    """
    Process the bounding boxes produced by the table detection model into
    cropped table images and cropped tokens.
    """

    table_crops = []
    for obj in objects:
        if obj["score"] < class_thresholds[obj["label"]]:
            continue

        cropped_table = {}

        bbox = obj["bbox"]
        bbox = [
            bbox[0] - padding,
            bbox[1] - padding,
            bbox[2] + padding,
            bbox[3] + padding,
        ]

        cropped_img = img.crop(bbox)

        table_tokens = [token for token in tokens if iob(token["bbox"], bbox) >= 0.5]
        for token in table_tokens:
            token["bbox"] = [
                token["bbox"][0] - bbox[0],
                token["bbox"][1] - bbox[1],
                token["bbox"][2] - bbox[0],
                token["bbox"][3] - bbox[1],
            ]

        # If table is predicted to be rotated, rotate cropped image and tokens/words:
        if obj["label"] == "table rotated":
            cropped_img = cropped_img.rotate(270, expand=True)
            for token in table_tokens:
                bbox = token["bbox"]
                bbox = [
                    cropped_img.size[0] - bbox[3] - 1,
                    bbox[0],
                    cropped_img.size[0] - bbox[1] - 1,
                    bbox[2],
                ]
                token["bbox"] = bbox

        cropped_table["image"] = cropped_img
        cropped_table["tokens"] = table_tokens

        table_crops.append(cropped_table)

    return table_crops


def visualize_detected_tables(img, det_tables, out_path=None):
    plt.imshow(img, interpolation="lanczos")
    fig = plt.gcf()
    fig.set_size_inches(20, 20)
    ax = plt.gca()

    for det_table in det_tables:
        bbox = det_table["bbox"]

        if det_table["label"] == "table":
            facecolor = (1, 0, 0.45)
            edgecolor = (1, 0, 0.45)
            alpha = 0.3
            linewidth = 2
            hatch = "//////"
        elif det_table["label"] == "table rotated":
            facecolor = (0.95, 0.6, 0.1)
            edgecolor = (0.95, 0.6, 0.1)
            alpha = 0.3
            linewidth = 2
            hatch = "//////"
        else:
            continue

        rect = patches.Rectangle(
            bbox[:2],
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=linewidth,
            edgecolor="none",
            facecolor=facecolor,
            alpha=0.1,
        )
        ax.add_patch(rect)
        rect = patches.Rectangle(
            bbox[:2],
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=linewidth,
            edgecolor=edgecolor,
            facecolor="none",
            linestyle="-",
            alpha=alpha,
        )
        ax.add_patch(rect)
        rect = patches.Rectangle(
            bbox[:2],
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=0,
            edgecolor=edgecolor,
            facecolor="none",
            linestyle="-",
            hatch=hatch,
            alpha=0.2,
        )
        ax.add_patch(rect)

    plt.xticks([], [])
    plt.yticks([], [])

    legend_elements = [
        Patch(
            facecolor=(1, 0, 0.45),
            edgecolor=(1, 0, 0.45),
            label="Table",
            hatch="//////",
            alpha=0.3,
        ),
        Patch(
            facecolor=(0.95, 0.6, 0.1),
            edgecolor=(0.95, 0.6, 0.1),
            label="Table (rotated)",
            hatch="//////",
            alpha=0.3,
        ),
    ]
    plt.legend(
        handles=legend_elements,
        bbox_to_anchor=(0.5, -0.02),
        loc="upper center",
        borderaxespad=0,
        fontsize=10,
        ncol=2,
    )
    plt.gcf().set_size_inches(10, 10)
    plt.axis("off")

    if out_path is not None:
        plt.savefig(out_path, bbox_inches="tight", dpi=150)

    return fig


def get_cell_coordinates_by_row(table_data):
    # Extract rows and columns
    rows = [entry for entry in table_data if entry["label"] == "table row"]
    columns = [entry for entry in table_data if entry["label"] == "table column"]

    # Sort rows and columns by their Y and X coordinates, respectively
    rows.sort(key=lambda x: x["bbox"][1])
    columns.sort(key=lambda x: x["bbox"][0])

    # Function to find cell coordinates
    def find_cell_coordinates(row, column):
        cell_bbox = [
            column["bbox"][0],
            row["bbox"][1],
            column["bbox"][2],
            row["bbox"][3],
        ]
        return cell_bbox

    # Generate cell coordinates and count cells in each row
    cell_coordinates = []

    for row in rows:
        row_cells = []
        for column in columns:
            cell_bbox = find_cell_coordinates(row, column)
            row_cells.append({"column": column["bbox"], "cell": cell_bbox})

        # Sort cells in the row by X coordinate
        row_cells.sort(key=lambda x: x["column"][0])

        # Append row information to cell_coordinates
        cell_coordinates.append(
            {"row": row["bbox"], "cells": row_cells, "cell_count": len(row_cells)}
        )

    # Sort rows from top to bottom
    cell_coordinates.sort(key=lambda x: x["row"][1])

    return cell_coordinates


def apply_ocr(table, cell_coordinates):
    reader = easyocr.Reader(["en"])
    # let's OCR row by row
    data = dict()
    max_num_columns = 0
    for idx, row in enumerate(tqdm(cell_coordinates)):
        row_text = []
        for cell in row["cells"]:
            # crop cell out of image
            cell_image = np.array(table.crop(cell["cell"]))
            # apply OCR
            result = reader.readtext(np.array(cell_image))
            if len(result) > 0:
                # print([x[1] for x in list(result)])
                text = " ".join([x[1] for x in result])
                row_text.append(text)

        if len(row_text) > max_num_columns:
            max_num_columns = len(row_text)

        data[idx] = row_text

    print("Max number of columns:", max_num_columns)

    # pad rows which don't have max_num_columns elements
    # to make sure all rows have the same number of columns
    for row, row_data in data.copy().items():
        if len(row_data) != max_num_columns:
            row_data = row_data + ["" for _ in range(max_num_columns - len(row_data))]
        data[row] = row_data

    return data


class TableDetector:
    """Handles table detection in images"""

    def __init__(self, parser_name: str):
        self.parser_name = parser_name
        self.image_converter = PDFToJPGConverter()

    def parse_pdf(self, uploaded_file) -> str:
        pdf_content = uploaded_file.read()
        logger.debug(f"Parsing PDF with {self.parser_name}")
        try:
            if self.parser_name == "microsoft/table-transformer-detection":
                return self._parse_with_ms_table_transformer_detection(pdf_content)
            else:
                raise ValueError(f"Unsupported parser: {self.parser_name}")
        except Exception as e:
            logger.error(
                f"Error parsing PDF with {self.parser_name}: {str(e)}", exc_info=True
            )
            raise

    def structure_recognition(self, table) -> str:
        logger.debug(f"Parsing PDF with {self.parser_name}")
        try:
            if self.parser_name == "microsoft/table-transformer-detection":
                return self._structure_recognition_with_ms_transformer_detection(table)
            else:
                raise ValueError(f"Unsupported parser: {self.parser_name}")
        except Exception as e:
            logger.error(
                f"Error Recognizing structure with {self.parser_name}: {str(e)}",
                exc_info=True,
            )
            raise

    def _parse_with_ms_table_transformer_detection(self, pdf_content: bytes) -> str:
        model = AutoModelForObjectDetection.from_pretrained(
            "microsoft/table-transformer-detection", revision="no_timm"
        )
        # Convert PDF pages to images
        model.to(device)

        output_path = f"converted_images/ui/{str(uuid.uuid4())}"
        images = self.image_converter.convert_pdf(pdf_content, output_path)
        # Load the table detection model

        pages = []

        for img in images:
            # Convert PIL Image to tensor
            image = img.convert("RGB")

            detection_transform = transforms.Compose(
                [
                    MaxResize(800),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

            pixel_values = detection_transform(image).unsqueeze(0)
            pixel_values = pixel_values.to(device)
            print(pixel_values.shape)

            with torch.no_grad():
                outputs = model(pixel_values)

            # update id2label to include "no object"
            id2label = model.config.id2label
            id2label[len(model.config.id2label)] = "no object"

            objects = outputs_to_objects(outputs, image.size, id2label)

            pages.append((image, objects))

        return pages

    def _structure_recognition_with_ms_transformer_detection(self, table):
        """
        Perform structure recognition on the image to extract tables and their contents.
        """

        model = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-structure-recognition-v1.1-all"
        )
        model.to(device)

        structure_transform = transforms.Compose(
            [
                MaxResize(1000),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        pixel_values = structure_transform(table).unsqueeze(0)
        pixel_values = pixel_values.to(device)

        # forward pass
        with torch.no_grad():
            outputs = model(pixel_values)

        structure_id2label = model.config.id2label
        structure_id2label[len(structure_id2label)] = "no object"

        cells = outputs_to_objects(outputs, table.size, structure_id2label)

        # Return cells
        return cells


def main():
    st.set_page_config(page_title="Pdf Parsing", layout="wide")

    # Initialize session state
    if "processed_chunks" not in st.session_state:
        st.session_state.processed_chunks = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7

    # Sidebar configuration
    with st.sidebar:
        st.header("Config")

        llm_provider = st.selectbox("LLM Provider", options=list(LLM_CONFIGS.keys()))

        # Check if API key is set for selected provider
        if LLM_CONFIGS[llm_provider].get("requires_key"):
            key_name = LLM_CONFIGS[llm_provider]["requires_key"]
            if not os.getenv(key_name):
                st.warning(f"⚠️ {key_name} not set")

        model_name = st.selectbox("Model", options=LLM_CONFIGS[llm_provider]["models"])

        temperature = st.slider(
            "Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1
        )
        st.session_state.temperature = temperature
        st.markdown("---")
        # Parser Configuration
        st.subheader("Parser Config")
        parser_name = st.selectbox(
            "Select Parser",
            options=list(PARSER_CONFIGS.keys()),
            help="Choose the method to extract text from your PDF",
        )

        # Show parser description
        st.info(PARSER_CONFIGS[parser_name]["description"])

        # Check if required API key is set
        required_key = PARSER_CONFIGS[parser_name]["requires_api_key"]
        if required_key and not os.getenv(required_key):
            st.warning(f"⚠️ {required_key} not set. This parser may not work.")

        st.markdown("---")
        # Table Image Detection Configuration
        st.subheader("Table Detector Config")
        table_detector = st.selectbox(
            "Select Parser",
            options=list(TABLE_IMAGE_DETECTION_CONFIGS.keys()),
            help="Choose the method to extract table from your PDF",
        )

        st.markdown("---")

        # Text Chunking Configuration
        st.subheader("Chunking Config")
        chunk_size = st.slider(
            "Chunk Size", min_value=500, max_value=4000, value=2000, step=100
        )

        chunk_overlap = st.slider(
            "Chunk Overlap", min_value=0, max_value=500, value=100, step=50
        )

        st.markdown("---")

        # Debug Options
        st.subheader("Debug")
        show_debug = st.checkbox("Enable Debugging", value=False)

    pdfParser, tableExtract = st.tabs(["Pdf Parser", "Table Extractor"])

    with pdfParser:
        st.subheader("PDF Q&A")
        st.caption(
            "Upload a PDF file to extract text and ask questions about the document."
        )
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        if uploaded_file:
            if st.button("Ingest PDF"):
                try:
                    with st.spinner(f"Ingesting PDF using {parser_name}..."):
                        # Create progress tracking
                        progress_text = st.empty()
                        progress_bar = st.progress(0)

                        # Initialize parser and RAG system
                        progress_text.text("Initializing systems...")
                        progress_bar.progress(0.1)

                        parser = MultiParser(parser_name)
                        rag_system = RAGSystem(llm_provider, model_name, temperature)
                        # Parse PDF
                        progress_text.text("Extracting text from PDF...")
                        progress_bar.progress(0.3)
                        extracted_text = parser.parse_pdf(uploaded_file)

                        if show_debug:
                            st.text("Extracted text sample:")
                            st.text(extracted_text[:500] + "...")

                        # Split into chunks
                        progress_text.text("Splitting text into chunks...")
                        progress_bar.progress(0.5)
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size, chunk_overlap=chunk_overlap
                        )
                        chunks = text_splitter.split_text(extracted_text)
                        st.session_state.processed_chunks = chunks

                        # Create vector store
                        progress_text.text("Creating vector store...")
                        progress_bar.progress(0.7)
                        st.session_state.vector_store = rag_system.create_vector_store(
                            chunks
                        )

                        # Setup QA chain
                        progress_text.text("Setting up question-answering system...")
                        progress_bar.progress(0.9)
                        st.session_state.qa_chain = rag_system.setup_qa_chain(
                            st.session_state.vector_store
                        )

                        progress_text.text("Processing complete!")
                        progress_bar.progress(1.0)

                        st.success(
                            f"✅ PDF processed successfully into {len(chunks)} chunks"
                        )

                        # Display chunks preview
                        with st.expander("📄 View Processed Chunks"):
                            num_preview = min(3, len(chunks))
                            for i in range(num_preview):
                                st.text_area(
                                    f"Chunk {i+1}/{len(chunks)}", chunks[i], height=150
                                )
                            if len(chunks) > num_preview:
                                st.info(
                                    f"... and {len(chunks) - num_preview} more chunks"
                                )

                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    if show_debug:
                        st.exception(e)
                    return

            # Question answering interface
            if st.session_state.qa_chain:
                st.subheader("Ask Questions")
                question = st.text_input("Enter your question about the document")

                if st.button("Send") and question:
                    try:
                        with st.spinner("Finding answer..."):
                            response = st.session_state.qa_chain.invoke(
                                {"input": question}
                            )

                            # Display answer
                            st.markdown("### 💡 Answer")
                            st.write(response["answer"])

                            # Show sources
                            with st.expander("🔍 View Source Chunks"):
                                for i, doc in enumerate(response["context"]):
                                    st.markdown(f"**Source {i+1}:**")
                                    st.text(doc.page_content)
                                    st.markdown("---")

                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
                        if show_debug:
                            st.exception(e)

                # Download processed text
                if st.button("📥 Download Processed Text"):
                    try:
                        combined_text = "\n\n".join(st.session_state.processed_chunks)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                        # Create a JSON with metadata
                        download_data = {
                            "metadata": {
                                "timestamp": timestamp,
                                "parser": parser_name,
                                "llm_provider": llm_provider,
                                "model": model_name,
                                "chunk_size": chunk_size,
                                "chunk_overlap": chunk_overlap,
                            },
                            "processed_text": combined_text,
                        }

                        download_json = json.dumps(download_data, indent=2)

                        st.download_button(
                            "Click to Download",
                            download_json,
                            file_name=f"processed_text_{timestamp}.json",
                            mime="application/json",
                        )
                    except Exception as e:
                        st.error(f"Error preparing download: {str(e)}")
                        if show_debug:
                            st.exception(e)
            # Main content area
    with tableExtract:
        st.subheader("Table Image Extractor")
        st.caption("Extracts table images from PDF using table detection models.")
        upload_file = st.file_uploader("Upload PDF with Table", type="pdf")
        if upload_file:
            if st.button("Ingest Table PDF"):
                try:
                    with st.spinner(f"Ingesting PDF using {table_detector}..."):
                        # Create progress tracking
                        progress_text = st.empty()
                        progress_bar = st.progress(0)

                        # Initialize parser and RAG system
                        progress_text.text("Initializing systems...")
                        progress_bar.progress(0.1)

                        # Initialize table parser
                        table_parser = TableDetector(table_detector)

                        # Update progress
                        progress_text.text("Processing PDF...")
                        progress_bar.progress(0.3)

                        # Extract table images
                        images = table_parser.parse_pdf(upload_file)

                        # Update progress
                        progress_text.text("Displaying results...")
                        progress_bar.progress(0.8)

                        # Display images in container
                        with st.container():
                            for i, (img, objects) in enumerate(images):
                                col1, col2, col3 = st.columns(3)
                                cropped_table = None

                                with col1:
                                    st.header("Orginal Document")
                                    st.image(
                                        img,
                                        caption=f"Page {i+1}",
                                        # width=int(img.width * 0.6),
                                        use_container_width=True,
                                    )

                                with col2:
                                    st.header("Detected Table")
                                    st.pyplot(visualize_detected_tables(img, objects))

                                with col3:
                                    st.header("Cropped Table")
                                    tokens = []
                                    detection_class_thresholds = {
                                        "table": 0.5,
                                        "table rotated": 0.5,
                                        "no object": 10,
                                    }
                                    crop_padding = 10

                                    tables_crops = objects_to_crops(
                                        img,
                                        tokens,
                                        objects,
                                        detection_class_thresholds,
                                        padding=crop_padding,
                                    )
                                    cropped_table = tables_crops[0]["image"].convert(
                                        "RGB"
                                    )
                                    st.image(cropped_table, caption=f"Table {i+1}")

                                st.divider()

                                col1, col2 = st.columns(2)
                                cells = None

                                with col1:
                                    st.header("Visualize cell")
                                    cells = table_parser.structure_recognition(
                                        cropped_table
                                    )
                                    table_visualized = cropped_table.copy()
                                    # Convert to PIL Image
                                    draw = ImageDraw.Draw(table_visualized)

                                    # Draw bounding boxes around detected cells
                                    for cell in cells:
                                        draw.rectangle(cell["bbox"], outline="red")

                                    st.json(cells)

                                    st.image(
                                        table_visualized,
                                        use_container_width=True,
                                    )

                                with col2:
                                    st.header("Visualize Rows")
                                    cell_coordinates = get_cell_coordinates_by_row(
                                        cells
                                    )

                                    data = apply_ocr(cropped_table, cell_coordinates)

                                    data = list(data.values())

                                    st.text(data)
                                    # Convert to DataFrame
                                    df = pd.DataFrame(data[1:], columns=data[0])
                                    st.table(df)

                        progress_text.text("Processing complete!")
                        progress_bar.progress(1.0)

                        st.success(f"✅ PDF processed successfully")

                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    if show_debug:
                        st.exception(e)
                    return


if __name__ == "__main__":
    main()
