



import os
import uuid
import base64
import json
import re
import asyncio
from io import BytesIO
from typing import List

from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pdf2image
import pandas as pd
from openai import OpenAI

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY_TWO")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in environment.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

def encode_image(img: Image.Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def convert_pdf_to_images(pdf_bytes: bytes):
    try:
        images = pdf2image.convert_from_bytes(pdf_bytes)
        return images
    except Exception as e:
        raise ValueError(f"Error converting PDF: {e}")

def clean_response(response_text: str) -> str:
    """
    Strips out triple backticks if they exist and returns the raw text.
    """
    if response_text.startswith("```"):
        lines = response_text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        response_text = "\n".join(lines)
    return response_text.strip()

def extract_allocations(text: str) -> dict:
    """
    Fallback function using regex to extract county: allocation pairs.
    Example pattern: "CountyName": 12.34
    """
    pattern = r'"?([^"]+)"?\s*:\s*([0-9]+(?:\.[0-9]+)?)'
    matches = re.findall(pattern, text)
    allocations = {}
    for county, value in matches:
        try:
            allocations[county.strip()] = float(value)
        except ValueError:
            continue
    return allocations

async def call_llm_with_images(image_base64_list: List[str], prompt_text: str) -> str:
    """
    Calls the LLM with a set of images (encoded to base64)
    and a given prompt text. Returns the cleaned response text.
    
    We do robust error handling here:
      - If the LLM returns an error or unexpected data,
        we raise an HTTPException with a descriptive message.
    """
    content_items = [{"type": "text", "text": prompt_text}]
    for img_b64 in image_base64_list:
        content_items.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
        })

    messages = [{"role": "user", "content": content_items}]

    try:
        completion = client.chat.completions.create(
            extra_body={},
            model="google/gemini-2.0-pro-exp-02-05:free",
            messages=messages,
            max_tokens=2048,
            temperature=0.1,
            top_p=0.9,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calling LLM: {str(e)}"
        ) from e
    
    if not completion or not getattr(completion, "choices", None):
        error_msg = getattr(completion, "error", "Unknown LLM error")
        raise HTTPException(
            status_code=500, 
            detail=f"LLM returned no valid choices. Error detail: {error_msg}"
        )

    try:
        raw_response = completion.choices[0].message.content
    except (IndexError, AttributeError) as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM response format unexpected. Error: {str(e)}"
        ) from e

    return clean_response(raw_response)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "table_data": None,
            "excel_data_url": None,
            "error_message": None
        }
    )

@app.post("/upload_pdfs", response_class=HTMLResponse)
async def process_pdfs(request: Request, files: List[UploadFile] = File(...)):
    """
    Accept up to 10 PDFs at once, process each, and store the raw page-level
    JSON in a nested list. Then call the LLM again to merge them *per PDF*,
    returning a list of JSON objects (one per PDF). Finally, write everything
    to an Excel workbook with multiple sheets, returning a data URL.
    
    Now includes robust error handling:
      - If PDF conversion fails, we raise a 400 error.
      - If LLM calls fail, we raise a 500 error (or 400 if user input is invalid).
      - If final JSON merge fails, we attempt a fallback or show an error result.
    """
    error_message = None
    df = pd.DataFrame()  # fallback for final return

    if len(files) > 10:
        error_message = "Maximum of 10 PDF files allowed at once."
        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request,
                "table_data": None,
                "excel_data_url": None,
                "error_message": error_message
            }
        )

    prompt_text = (
        "Extract county allocation data from these images. Some images may or may not contain only county names with allocation data — ignore those that do not include allocation data and their allocation percentages. "
        "Return only valid JSON with keys as county names and values as their allocation percentages, with no additional text. The sum of all percentages should be close or around to 100. "
        "Please ensure you understand the full context of the images. Focus only on extracting county allocation statistics and ignore any unrelated data. "
        "Do not include any extra text — only return the JSON with county names and their allocation percentages. "
        "Only include items that are officially recognized subnational geographic units—such as counties, provinces, districts, or municipalities. Terms like 'others' or 'internationals' may also be included, provided they refer to legitimate territorial, administrative, or geopolitical divisions recognized within a country or internationally. "
        "FROM THE IMAGES BE ADVISED. The sum of all percentages should be close or around to 100. "
        "Return only valid JSON with no extra text."
    )

    # all_raw_responses[pdf_index] = list of JSON strings from that PDF's pages
    all_raw_responses = []

    for file in files:
        pdf_responses = []
        try:
            pdf_data = await file.read()
            page_images = convert_pdf_to_images(pdf_data)
        except Exception as e:
            error_message = f"Error with {file.filename}: {str(e)}"
            return templates.TemplateResponse(
                "index.html", 
                {
                    "request": request,
                    "table_data": None,
                    "excel_data_url": None,
                    "error_message": error_message
                }
            )

        # For each page in the PDF, call the LLM
        for img in page_images:
            try:
                encoded_image = encode_image(img)
                response_text = await call_llm_with_images([encoded_image], prompt_text)
                pdf_responses.append(response_text)
            except HTTPException as e:
                # LLM or code error, bubble up to the user via template
                error_message = f"Error processing page in {file.filename}: {e.detail}"
                return templates.TemplateResponse(
                    "index.html", 
                    {
                        "request": request,
                        "table_data": None,
                        "excel_data_url": None,
                        "error_message": error_message
                    }
                )
           

        all_raw_responses.append(pdf_responses)

    consolidation_prompt = (
        "You are given a list of lists of JSON responses, where each sub-list represents the pages of a single PDF. "
        "For each sub-list, merge the JSON responses into one JSON object, and return a list of these merged objects. "
        "The final output must be a valid JSON array (e.g. [ {...}, {...}, ... ])."
        "please return like this always (e.g. [ {...}, {...}, ... ])"
    )

    consolidation_input = json.dumps(all_raw_responses)
    consolidation_message = [
        {"type": "text", "text": consolidation_prompt},
        {"type": "text", "text": f"JSON responses: {consolidation_input}"}
    ]
    consolidation_payload = [{"role": "user", "content": consolidation_message}]

    try:
        final_completion = client.chat.completions.create(
            extra_body={},
            model="google/gemini-2.0-pro-exp-02-05:free",
            messages=consolidation_payload,
            max_tokens=2048,
            temperature=0.1,
            top_p=0.9,
        )
        final_response_text = clean_response(final_completion.choices[0].message.content)
    except Exception as e:
        error_message = f"Error consolidating responses: {str(e)}"
        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request,
                "table_data": None,
                "excel_data_url": None,
                "error_message": error_message
            }
        )
    print("Consolidated final_response_text:", final_response_text)

    try:
        merged_pdf_list = json.loads(final_response_text)
    except Exception:
        try:
            fallback_completion = client.chat.completions.create(
                extra_body={},
                model="google/gemini-2.0-pro-exp-02-05:free",
                messages=consolidation_payload,
                max_tokens=2048,
                temperature=0.1,
                top_p=0.9,
            )
            final_response_text = clean_response(fallback_completion.choices[0].message.content)
            merged_pdf_list = json.loads(final_response_text)
        except Exception:
            error_message = (
                "Failed to parse final JSON array from the LLM. "
                f"Raw response: {final_response_text}"
            )
            merged_pdf_list = [{
                "error": "Failed to parse final JSON array",
                "raw_response": final_response_text
            }]

    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook  = writer.book
        worksheet = workbook.add_worksheet("Allocations")
        writer.sheets["Allocations"] = worksheet

        title_format = workbook.add_format({
            'bold': True, 'font_size': 14, 'align': 'left'
        })

        start_row = 0
        for i, (pdf_result, file) in enumerate(zip(merged_pdf_list, files)):
            filename = file.filename.rsplit('.', 1)[0] 

            if isinstance(pdf_result, dict):
                if "error" in pdf_result:
                    df = pd.DataFrame([pdf_result])
                else:
                    df = pd.DataFrame(list(pdf_result.items()), columns=["County", "Allocation (%)"])
            else:
                df = pd.DataFrame([{
                    "Error": "Invalid JSON for this PDF",
                    "raw": str(pdf_result)
                }])

            worksheet.write(start_row, 0, f"Results from: {filename}", title_format)
            start_row += 1

            df.to_excel(
                writer, 
                sheet_name="Allocations", 
                startrow=start_row, 
                index=False, 
                header=True
            )
            start_row += len(df) + 3 

    output.seek(0)
    excel_bytes = output.getvalue()
    excel_base64 = base64.b64encode(excel_bytes).decode("utf-8")
    data_url = f"data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{excel_base64}"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
             "table_data": df.to_dict(orient='records'), 
            "excel_data_url": data_url,
            "error_message": error_message
        }
    )
