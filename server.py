import os
import uuid
import logging
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from fpdf import FPDF
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load env variables
load_dotenv()

# Import pipeline components
from audio_pipeline import AudioProcessingPipeline
from nlp_pipeline import MedicalNLPPipeline
from llm import MedicalLLM
from main import transcribe_audio

app = FastAPI(title="Carelinq AI Scribe Backend API", version="1.0.0")

# Setup CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Persistent History helper
HISTORY_FILE = Path("./reports/history.json")

def save_history_record(item):
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {"chats": [], "scribes": [], "reports": []}
        if HISTORY_FILE.exists():
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load existing history: {e}")
        
        # Insert to the beginning of scribes list
        data.setdefault("scribes", []).insert(0, item)
        
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save history record: {e}")

# Request/Response Schemas
class AnalyzeRequest(BaseModel):
    transcript: str

class EntityItem(BaseModel):
    text: str
    label: str

class GeneratePDFRequest(BaseModel):
    transcript: str
    entities: List[EntityItem]
    summary_text: str

@app.post("/api/v1/scribe/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...)):
    logger.info(f"Received transcription request for file: {file.filename}")
    
    # Setup temporary directories
    upload_dir = Path("./processed/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Save upload file
    temp_suffix = Path(file.filename).suffix
    file_id = str(uuid.uuid4())
    input_path = upload_dir / f"upload_{file_id}{temp_suffix}"
    output_path = Path("./processed") / f"processed_{file_id}.wav"
    
    try:
        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        transcribe_path = input_path
        
        # Attempt to run audio pipeline
        try:
            logger.info("Running 8-stage clinical audio processing pipeline...")
            pipeline = AudioProcessingPipeline()
            pipeline.process_file(str(input_path), str(output_path))
            if output_path.exists():
                transcribe_path = output_path
                logger.info("Audio processing completed successfully.")
            else:
                logger.warning("Audio processing output file was not created. Falling back to raw audio.")
        except Exception as e:
            logger.error(f"Audio processing pipeline failed: {e}. Falling back to raw audio transcription.")
            
        # Transcribe
        api_key = os.getenv("GROQ_API_KEY")
        logger.info("Starting transcription...")
        try:
            transcript, segments = transcribe_audio(transcribe_path, api_key=api_key)
        except Exception as e:
            logger.warning(f"Transcription failed with default key. Retrying with fallback key. Error: {e}")
            fallback_key = os.getenv("FALLBACK_GROQ_API_KEY")
            if not fallback_key:
                raise ValueError("No fallback API key configured in FALLBACK_GROQ_API_KEY env variable.") from e
            transcript, segments = transcribe_audio(transcribe_path, api_key=fallback_key)
        logger.info("Transcription completed.")
        
        return {"transcript": transcript}
        
    except Exception as e:
        logger.error(f"Error during transcription endpoint execution: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
        
    finally:
        # Cleanup files
        try:
            if input_path.exists():
                input_path.unlink()
            if output_path.exists():
                output_path.unlink()
        except Exception as e:
            logger.warning(f"Error during cleanup of temporary files: {e}")

@app.post("/api/v1/scribe/analyze")
async def analyze_endpoint(req: AnalyzeRequest):
    logger.info("Received analysis request")
    try:
        api_key = os.getenv("GROQ_API_KEY")
        
        # Initialize NLP pipeline
        nlp_pipeline = MedicalNLPPipeline(config={"groq_api_key": api_key})
        try:
            nlp_report = nlp_pipeline.process_transcript(req.transcript)
            if not nlp_report or not nlp_report.get("corrected_transcript") or "warning" in nlp_report or "invalid_api_key" in str(nlp_report):
                raise ValueError("NLP pipeline returned fallback or invalid key warning")
        except Exception as e:
            logger.warning(f"NLP pipeline failed with default key. Retrying with fallback key. Error: {e}")
            fallback_key = os.getenv("FALLBACK_GROQ_API_KEY")
            if not fallback_key:
                raise ValueError("No fallback API key configured in FALLBACK_GROQ_API_KEY env variable.") from e
            nlp_pipeline = MedicalNLPPipeline(config={"groq_api_key": fallback_key})
            nlp_report = nlp_pipeline.process_transcript(req.transcript)
        
        # Extract fields
        corrected_transcript = nlp_report.get("corrected_transcript", req.transcript) or req.transcript
        entities = nlp_report.get("entities", [])
        
        # Generate Clinical SOAP Care Plan Summary
        logger.info("Generating clinical SOAP summary via LLM...")
        
        extracted_details = {
            "patient_details": nlp_report.get("patient_details", {}),
            "medical_details": nlp_report.get("medical_details", {}),
            "medicines": nlp_report.get("medicines", [])
        }
        
        try:
            llm_client = MedicalLLM(api_key=api_key)
            summary = llm_client.generate_doctor_summary(corrected_transcript, extracted_details)
            if not summary or "Error generating summary" in summary or "Invalid API Key" in summary or "Failed to analyze" in summary or "invalid_api_key" in summary:
                raise ValueError("LLM generation returned an error string: " + str(summary))
        except Exception as e:
            logger.warning(f"LLM summary generation failed. Retrying with fallback key. Error: {e}")
            fallback_key = os.getenv("FALLBACK_GROQ_API_KEY")
            if not fallback_key:
                raise ValueError("No fallback API key configured in FALLBACK_GROQ_API_KEY env variable.") from e
            llm_client = MedicalLLM(api_key=fallback_key)
            summary = llm_client.generate_doctor_summary(corrected_transcript, extracted_details)
        
        # Create history record
        history_item = {
            "id": int(time.time() * 1000),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "summary": summary,
            "entities": [{"text": ent.get("text", ""), "label": ent.get("label", "")} for ent in entities],
            "transcript": corrected_transcript
        }
        save_history_record(history_item)
        
        return {
            "transcript": corrected_transcript,
            "summary": summary,
            "entities": entities
        }
    except Exception as e:
        logger.error(f"Error during clinical analysis endpoint execution: {e}")
        return {
            "transcript": req.transcript,
            "summary": f"Failed to analyze the consultation text. Error detail: {str(e)}",
            "entities": []
        }

@app.post("/api/v1/scribe/generate-pdf")
async def generate_pdf_endpoint(req: GeneratePDFRequest):
    logger.info("Received PDF generation request")
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # 1. Document Title Header
        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(pdf.epw, 10, "CARELINQ CLINICAL CARE PLAN")
        pdf.ln(8)
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(pdf.epw, 10, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        pdf.ln(12)
        
        # 2. SOAP Notes / Clinical Summary Section
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(pdf.epw, 10, "1. CLINICAL SOAP PLAN")
        pdf.ln(8)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        
        pdf.set_font("Helvetica", "", 10)
        for line in req.summary_text.split("\n"):
            pdf.multi_cell(pdf.epw, 6, line)
        pdf.ln(10)
        
        # 3. Medical Entities Section
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(pdf.epw, 10, "2. IDENTIFIED MEDICAL ENTITIES")
        pdf.ln(8)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        
        pdf.set_font("Helvetica", "", 10)
        if req.entities:
            for ent in req.entities:
                text = ent.text
                label = ent.label.replace("_", " ").upper()
                pdf.multi_cell(pdf.epw, 6, f"- {text} [{label}]")
        else:
            pdf.cell(pdf.epw, 10, "No medical entities identified.")
            pdf.ln(6)
        pdf.ln(10)
        
        # 4. Consultation Transcript Section
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(pdf.epw, 10, "3. CONSULTATION TRANSCRIPT")
        pdf.ln(8)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        
        pdf.set_font("Helvetica", "I", 9)
        for line in req.transcript.split("\n"):
            pdf.multi_cell(pdf.epw, 5, line)
            
        # Output bytearray
        pdf_data = bytes(pdf.output())
        
        return Response(
            content=pdf_data,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=CarePlan.pdf",
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
    except Exception as e:
        logger.exception("Error during PDF generation endpoint execution")
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")

@app.get("/api/v1/history/all")
async def get_history_endpoint():
    logger.info("Received history retrieve request")
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading history file: {e}")
    return {"chats": [], "scribes": [], "reports": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
