"""
llm.py

Groq LLM Client for medical transcript post-processing, structured entity extraction,
and generating doctor/patient clinical summaries.
"""

import os
import re
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

from prompt import (
    MEDICAL_CORRECTION_PROMPT,
    MEDICAL_NER_EXTRACTION_PROMPT,
    DOCTOR_SUMMARY_PROMPT,
    PATIENT_SUMMARY_PROMPT,
    COMBINED_MEDICAL_SUMMARY_PROMPT
)

logger = logging.getLogger(__name__)

class MedicalLLM:
    """
    Client for Groq LLM medical tasks using the prompts defined in prompt.py.
    """
    
    def __init__(self, api_key=None, model="llama-3.1-8b-instant"):
        """
        Initialize the MedicalLLM client.
        
        Args:
            api_key (str): Optional Groq API Key. If None, loads from environment.
            model (str): The LLM model to use (default: llama-3.1-8b-instant).
        """
        load_dotenv()
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model = model
        
        if not self.api_key:
            logger.warning("Warning: GROQ_API_KEY is not set.")
            
        self.client = Groq(api_key=self.api_key) if self.api_key else None

    def _call_llm(self, system_prompt, user_content, temperature=0.0):
        """Helper to call the Groq completions endpoint."""
        if not self.client:
            raise ValueError("Groq client not initialized due to missing API Key.")
            
        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            model=self.model,
            temperature=temperature
        )
        return chat_completion.choices[0].message.content.strip()

    def _parse_and_repair_json(self, raw_text):
        """Attempts to parse JSON, cleaning markdown code fences and repairing once if needed."""
        def clean(text):
            if text.startswith("```"):
                lines = text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\n".join(lines).strip()
            return text

        cleaned = clean(raw_text)
        
        # Try standard parse
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        # Try repairing by isolating the first { to the last }
        try:
            match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group(1))
        except Exception as e:
            logger.error(f"Failed to parse and repair JSON: {e}")
            raise ValueError(f"JSON validation failed on text: {raw_text}")

    def correct_transcript(self, raw_transcript):
        """
        Corrects spelling errors, restores punctuation and structure.
        Uses MEDICAL_CORRECTION_PROMPT.
        """
        return self._call_llm(MEDICAL_CORRECTION_PROMPT, raw_transcript)

    def extract_ner(self, corrected_transcript):
        """
        Extracts structured medical and patient details in JSON format.
        Uses MEDICAL_NER_EXTRACTION_PROMPT.
        """
        raw_response = self._call_llm(MEDICAL_NER_EXTRACTION_PROMPT, corrected_transcript)
        return self._parse_and_repair_json(raw_response)

    def generate_doctor_summary(self, transcript, extracted_details=None):
        """
        Generates a professional doctor-facing clinical summary.
        Uses DOCTOR_SUMMARY_PROMPT.
        """
        user_content = f"Transcript:\n{transcript}"
        if extracted_details:
            user_content += f"\n\nExtracted Details:\n{json.dumps(extracted_details, indent=2)}"
            
        return self._call_llm(DOCTOR_SUMMARY_PROMPT, user_content)

    def generate_patient_summary(self, transcript, extracted_details=None):
        """
        Generates a simple, patient-friendly medical summary.
        Uses PATIENT_SUMMARY_PROMPT.
        """
        user_content = f"Transcript:\n{transcript}"
        if extracted_details:
            user_content += f"\n\nExtracted Details:\n{json.dumps(extracted_details, indent=2)}"
            
        return self._call_llm(PATIENT_SUMMARY_PROMPT, user_content)

    def generate_combined_summary(self, transcript):
        """
        Generates both Doctor and Patient summaries returned as a single JSON object.
        Uses COMBINED_MEDICAL_SUMMARY_PROMPT.
        """
        raw_response = self._call_llm(COMBINED_MEDICAL_SUMMARY_PROMPT, transcript)
        return self._parse_and_repair_json(raw_response)

if __name__ == "__main__":
    llm = MedicalLLM()
    print("MedicalLLM class compiled and initialized successfully!")
