"""
NLP Processing Pipeline
Implements the NLP stages for clinical transcription:
3. Speaker Diarization Alignment
4. Medical Speech Recognition Orchestrator
5. Medical Error Correction (SLM-based)
6. Punctuation Restoration (SLM-based)
7. Medical Named Entity Recognition (NER) (SLM-based & Keyword Fallback)
"""

import os
import re
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Common medical term correction dictionary for local/deterministic fallback
MEDICAL_CORRECTION_DICT = {
    r"\bmet\s+forming\b": "metformin",
    r"\blisten\s+april\b": "lisinopril",
    r"\blereopenum\b": "meropenem",
    r"\bneurofenem\b": "meropenem",
    r"\blereopenam\b": "meropenem",
    r"\bvacomysin\b": "vancomycin",
    r"\bvacomisin\b": "vancomycin",
    r"\be-cop\b": "ECOG",
    r"\bg\s+says\b": "GCS",
    r"\bg\s+size\b": "GCS",
    r"\bgcase\b": "GCS",
    r"\bsources\s+of\s+the\b": "shortness of breath",
    r"\bsources\s+of\s+blood\b": "shortness of breath",
    r"\bnon-spauls\b": "non-small",
    r"\blung\s+tartsinova\b": "lung carcinoma",
    r"\bdirage\b": "deranged",
    r"\bina\b": "INR",
    r"\bimr\b": "INR",
    r"\bhat\s+protocols\b": "HAP protocols",
    r"\bngi-2\b": "mg IV",
    r"\bngi\s+2\b": "mg IV",
    r"\bdexa\s+6\b": "Dexa 6"
}

# Medical Named Entity Recognition dictionaries for fallback
CLINICAL_KEYWORDS = {
    "Disease/Diagnosis": [
        "lung carcinoma", "carcinoma", "pneumonia", "sepsis", "respiratory failure",
        "metastasis", "progressive disease", "cancer", "failure"
    ],
    "Medicine/Treatment": [
        "meropenem", "vancomycin", "vitamin k", "albumin", "dexa", "dexamethasone",
        "chemotherapy", "antibiotics"
    ],
    "Symptom": [
        "shortness of breath", "mmrc", "mmrc grade", "breathlessness", "dyspnea",
        "fever", "cough"
    ],
    "Dosage/Frequency": [
        "2 gram iv tds", "1 gram iv bd", "10 mg iv", "6 mg iv", "3 liters",
        "iv tds", "iv bd", "iv", "tds", "bd"
    ],
    "Body Part": [
        "lung", "blood", "chest", "heart", "respiratory"
    ],
    "Procedure/Test": [
        "pt", "inr", "gcs", "ecog", "hydration", "investigations", "rounds",
        "blood pressure", "respiratory rate"
    ]
}

class MedicalNLPPipeline:
    """
    NLP Pipeline for medical transcription processing.
    Supports SLM-based extraction, validation, repair, and deterministic fallback.
    """
    
    def __init__(self, config=None):
        """
        Initialize the NLP Pipeline.
        
        Args:
            config (dict): Configuration overrides.
        """
        self.config = config or {}
        load_dotenv()
        self.api_key = self.config.get("groq_api_key", os.getenv("GROQ_API_KEY", ""))

    def align_diarization(self, whisper_segments, diarization_segments):
        """
        Stage 3: Speaker Diarization Alignment
        Aligns timestamped transcription segments with speaker turns.
        """
        aligned_turns = []
        
        if not diarization_segments:
            # Default to single speaker if no diarization info
            full_text = " ".join([seg.get("text", "") for seg in whisper_segments])
            return [{"speaker": "Speaker 1", "text": full_text.strip(), "start": 0.0, "end": 10.0}]
            
        for trans_seg in whisper_segments:
            t_start = trans_seg.get("start", 0.0)
            t_end = trans_seg.get("end", 0.0)
            t_text = trans_seg.get("text", "").strip()
            
            if not t_text:
                continue
                
            # Find the speaker diarization segment with the maximum overlap
            best_speaker = "Speaker 1"
            max_overlap = -1.0
            
            for diar_seg in diarization_segments:
                d_start = diar_seg.get("start", 0.0)
                d_end = diar_seg.get("end", 0.0)
                d_speaker = diar_seg.get("speaker", "Speaker")
                
                # Calculate overlap duration
                overlap_start = max(t_start, d_start)
                overlap_end = min(t_end, d_end)
                overlap = max(0.0, overlap_end - overlap_start)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = d_speaker
                    
            aligned_turns.append({
                "speaker": best_speaker,
                "text": t_text,
                "start": t_start,
                "end": t_end
            })
            
        # Group consecutive turns by the same speaker
        grouped_turns = []
        if aligned_turns:
            current_turn = aligned_turns[0].copy()
            for next_turn in aligned_turns[1:]:
                if next_turn["speaker"] == current_turn["speaker"]:
                    # Merge text and update end time
                    current_turn["text"] += " " + next_turn["text"]
                    current_turn["end"] = next_turn["end"]
                else:
                    grouped_turns.append(current_turn)
                    current_turn = next_turn.copy()
            grouped_turns.append(current_turn)
            
        return grouped_turns

    def extract_with_slm(self, raw_text):
        """
        Extracts structured medical details, patient information, and corrects text
        using an SLM/LLM via the Groq API.
        """
        if not self.api_key:
            raise ValueError("Groq API Key is not configured.")
            
        from groq import Groq
        client = Groq(api_key=self.api_key)
        
        system_prompt = (
            "You are an expert clinical NLP AI. Your task is to analyze a raw clinical transcription, "
            "correct obvious transcription spelling errors (especially medical term misrecognitions like "
            "'met forming' -> 'metformin', 'vacomysin' -> 'vancomycin', 'lereopenum' -> 'meropenem', "
            "'e-cop' -> 'ECOG', 'g says' -> 'GCS'), and extract structured clinical and demographic facts.\n\n"
            "Constraints:\n"
            "1. Return ONLY valid JSON matching the exact schema below.\n"
            "2. Do not include markdown wrappers (e.g. ```json) or preamble/explanations.\n"
            "3. Do not invent missing details. Leave fields empty if not present in the text.\n"
            "4. Preserve all numbers, dosages, routes, frequencies, and clinical facts exactly.\n"
            "5. Correct only obvious transcription typos.\n\n"
            "JSON Schema:\n"
            "{\n"
            '  "corrected_transcript": "<fully corrected, punctuated transcript>",\n'
            '  "patient_details": {\n'
            '    "name": "<patient name or empty>",\n'
            '    "age": "<age or empty>",\n'
            '    "gender": "<gender or empty>",\n'
            '    "phone": "<phone or empty>",\n'
            '    "patient_id": "<patient ID or empty>",\n'
            '    "doctor_name": "<doctor name or empty>"\n'
            "  },\n"
            '  "medical_details": {\n'
            '    "chief_complaint": "<chief complaint or empty>",\n'
            '    "symptoms": ["<symptom 1>", ...],\n'
            '    "diagnosis": ["<diagnosis 1>", ...],\n'
            '    "past_medical_history": ["<past medical history item 1>", ...],\n'
            '    "tests": ["<test/investigation 1>", ...],\n'
            '    "vitals": ["<vital sign 1>", ...],\n'
            '    "procedures": ["<procedure 1>", ...],\n'
            '    "doctor_advice": ["<doctor advice/instructions 1>", ...],\n'
            '    "follow_up": "<follow up instructions or empty>"\n'
            "  },\n"
            '  "medicines": [\n'
            "    {\n"
            '      "name": "<medicine name>",\n'
            '      "dosage": "<dosage or empty>",\n'
            '      "route": "<route/formulation or empty>",\n'
            '      "frequency": "<frequency or empty>",\n'
            '      "duration": "<duration or empty>",\n'
            '      "instruction": "<special instruction or empty>"\n'
            "    }\n"
            "  ]\n"
            "}"
        )
        
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_text}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.0
        )
        return chat_completion.choices[0].message.content.strip()

    def validate_slm_json(self, raw_response):
        """
        Validates the JSON response from the SLM.
        If invalid, tries to repair it once. If still invalid, returns None.
        """
        def clean_text(text):
            # Remove markdown backticks if present
            if text.startswith("```"):
                lines = text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\n".join(lines).strip()
            return text

        # Attempt 1: Standard load
        cleaned = clean_text(raw_response)
        try:
            parsed = json.loads(cleaned)
            if self._verify_keys(parsed):
                return parsed
        except Exception:
            pass

        # Attempt 2: Repair once (extract first { to last })
        try:
            match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
            if match:
                repaired_text = match.group(1)
                parsed = json.loads(repaired_text)
                if self._verify_keys(parsed):
                    return parsed
        except Exception:
            pass

        return None

    def _verify_keys(self, parsed):
        """Helper to verify schema keys exist."""
        required = ["corrected_transcript", "patient_details", "medical_details", "medicines"]
        return all(k in parsed for k in required)

    def correct_medical_errors(self, text):
        """
        Fallback spelling correction using local dictionary.
        """
        corrected = text
        for pattern, replacement in MEDICAL_CORRECTION_DICT.items():
            corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
            
        # Basic punctuation restoration
        corrected = corrected.strip()
        if corrected and not corrected[0].isupper():
            corrected = corrected[0].upper() + corrected[1:]
        if corrected and corrected[-1] not in ['.', '!', '?']:
            corrected += '.'
        return corrected

    def extract_medical_entities(self, text):
        """
        Fallback keyword-based Named Entity Recognition (NER).
        """
        entities = []
        text_lower = text.lower()
        
        for label, keywords in CLINICAL_KEYWORDS.items():
            for kw in keywords:
                for match in re.finditer(r'\b' + re.escape(kw) + r'\b', text_lower):
                    start_char = match.start()
                    end_char = match.end()
                    original_val = text[start_char:end_char]
                    
                    if not any(e["start"] == start_char and e["end"] == end_char for e in entities):
                        entities.append({
                            "text": original_val,
                            "label": label,
                            "start": start_char,
                            "end": end_char
                        })
        entities.sort(key=lambda x: x["start"])
        return entities

    def _run_local_fallback(self, raw_text):
        """
        Runs local keyword/regex parser to fill the structured clinical schema.
        """
        corrected = self.correct_medical_errors(raw_text)
        
        # Patient Details Fallback Extraction
        patient_details = {
            "name": "", "age": "", "gender": "", "phone": "", "patient_id": "", "doctor_name": ""
        }
        
        # Heuristics for name
        name_match = re.search(r"(?:named|patient)\s+([A-Z][a-zA-Z]+)", corrected)
        if name_match:
            patient_details["name"] = name_match.group(1)
            
        # Heuristics for age/gender
        age_gender_match = re.search(r"(\b\d+\b)?\s*(year[s]?[- ]old)?\s*(adult)?\s*(male|female|man|woman)\b", corrected, re.IGNORECASE)
        if age_gender_match:
            patient_details["age"] = age_gender_match.group(1) or ("adult" if age_gender_match.group(3) else "")
            patient_details["gender"] = age_gender_match.group(4)
            
        # Vitals/Details mapping
        medical_details = {
            "chief_complaint": "", "symptoms": [], "diagnosis": [], "past_medical_history": [], "tests": [], "vitals": [], "procedures": [], "doctor_advice": [], "follow_up": ""
        }
        
        # Fill categories using keyword lists
        keywords_mapping = {
            "symptoms": "Symptom",
            "diagnosis": "Disease/Diagnosis",
            "tests": "Procedure/Test",
            "procedures": "Procedure/Test"
        }
        
        for schema_key, clinical_label in keywords_mapping.items():
            for kw in CLINICAL_KEYWORDS[clinical_label]:
                if re.search(r'\b' + re.escape(kw) + r'\b', corrected, re.IGNORECASE):
                    medical_details[schema_key].append(kw)
                    
        # Extract vitals specifically (e.g. rate, blood pressure)
        vitals_patterns = [
            r"\b(?:respiratory rate|rr)\s+is\s+\d+\b",
            r"\b(?:blood pressure|bp)\s+is\s+[\d/]+\b",
            r"\bsystolic blood pressure\s+is\s+\S+\s+\d+\b"
        ]
        for pat in vitals_patterns:
            matches = re.findall(pat, corrected, re.IGNORECASE)
            for m in matches:
                medical_details["vitals"].append(m)
                
        # Extract medicines fallback
        medicines = []
        for kw in CLINICAL_KEYWORDS["Medicine/Treatment"]:
            if re.search(r'\b' + re.escape(kw) + r'\b', corrected, re.IGNORECASE):
                # Simple medicine object
                med_obj = {
                    "name": kw, "dosage": "", "route": "", "frequency": "", "duration": "", "instruction": ""
                }
                
                # Check for nearby dosages/routes/frequencies in window of 40 characters
                match = re.search(r'\b' + re.escape(kw) + r'\b', corrected, re.IGNORECASE)
                if match:
                    window = corrected[max(0, match.start() - 20): min(len(corrected), match.end() + 40)]
                    # Dosage regex (e.g. 1 gram, 10 mg)
                    dos_m = re.search(r"(\d+\s*(?:gram|mg|ml))", window, re.IGNORECASE)
                    if dos_m: med_obj["dosage"] = dos_m.group(1)
                    # Route regex (IV, PO, IM)
                    route_m = re.search(r"\b(iv|po|im)\b", window, re.IGNORECASE)
                    if route_m: med_obj["route"] = route_m.group(1).upper()
                    # Frequency (TDS, BD, QD)
                    freq_m = re.search(r"\b(tds|bd|qd|prn)\b", window, re.IGNORECASE)
                    if freq_m: med_obj["frequency"] = freq_m.group(1).upper()
                    # Duration (3 days, 1 week)
                    dur_m = re.search(r"(\d+\s*(?:day|week|month)[s]?)", window, re.IGNORECASE)
                    if dur_m: med_obj["duration"] = dur_m.group(1)
                    
                medicines.append(med_obj)
                
        return {
            "corrected_transcript": corrected,
            "patient_details": patient_details,
            "medical_details": medical_details,
            "medicines": medicines,
            "warning": "SLM processing failed, using fallback keyword-based extraction."
        }

    def process_transcript(self, raw_text, whisper_segments=None, diarization_segments=None):
        """
        Orchestrates the entire NLP processing pipeline.
        Tries SLM extraction first; falls back to local dictionary/keyword parsing on failure.
        """
        report = {
            "raw_transcript": raw_text,
            "corrected_transcript": "",
            "patient_details": {
                "name": "", "age": "", "gender": "", "phone": "", "patient_id": "", "doctor_name": ""
            },
            "medical_details": {
                "chief_complaint": "", "symptoms": [], "diagnosis": [], "past_medical_history": [], "tests": [], "vitals": [], "procedures": [], "doctor_advice": [], "follow_up": ""
            },
            "medicines": [],
            "dialogue": [],
            "entities": []
        }
        
        # 1. Aligned Dialogue (Stage 3)
        if whisper_segments and diarization_segments:
            aligned_dialogue = self.align_diarization(whisper_segments, diarization_segments)
            # Apply error correction to each dialogue turn individually
            for turn in aligned_dialogue:
                turn["text"] = self.correct_medical_errors(turn["text"])
            report["dialogue"] = aligned_dialogue
            
        # 2. SLM-based Extraction with validation and repair
        slm_success = False
        if self.api_key:
            try:
                raw_response = self.extract_with_slm(raw_text)
                parsed_slm = self.validate_slm_json(raw_response)
                
                if parsed_slm:
                    report.update(parsed_slm)
                    slm_success = True
            except Exception as e:
                logger.error(f"SLM extraction or validation failed: {e}")
                
        # 3. Fallback System (runs if SLM fails)
        if not slm_success:
            fallback_res = self._run_local_fallback(raw_text)
            report.update(fallback_res)
            
        # 4. Fallback/Keyword-based NER for rendering highlights
        report["entities"] = self.extract_medical_entities(report["corrected_transcript"])
        
        return report

if __name__ == "__main__":
    pipeline = MedicalNLPPipeline()
    print("MedicalNLPPipeline class compiled and loaded successfully!")
