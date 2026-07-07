import os
import json
import logging
from llm import MedicalLLM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Load transcript from the completed test transcription
    transcript_file = "audio file text/WhatsApp Audio 2026-07-07 at 2.54.36 PM.txt"
    raw_text = ""
    
    if os.path.exists(transcript_file):
        with open(transcript_file, 'r', encoding='utf-8') as f:
            raw_text = " ".join([line.split("]")[-1].strip() for line in f if "]" in line])
    else:
        # Fallback raw clinical text
        raw_text = (
            "So, the bed number 3 is named Virender, he is an adult male, he is a case of non-spauls "
            "and lung tartsinova, with extensive metastasis, where the progressive degrees pose two lines "
            "of chemotherapy, and it presented to us with sources of the MMRC grade 3 core. On present "
            "investigations, it was found out that he had type of respiratory failure. Currently, "
            "his respiratory rate is 27 per minute. His systolic blood pressure is on an average "
            "is more than 1,200 mmHg. His GCase is 15 by 15, however, his ECoG is poor 3 or 4. So, "
            "on daily rounds, it was discussed to elevate escalate his antibiotics to injection meropenem, "
            "2 gram IV TDS, and injection vancomycin 1 gram IV BD, as per HAP protocols."
        )
        
    print("\n" + "="*50)
    print("RAW TRANSCRIPT INPUT")
    print("="*50)
    print(raw_text)
    
    # Initialize MedicalLLM
    llm = MedicalLLM()
    
    # 1. Correct Transcript
    print("\n" + "="*50)
    print("1. RUNNING MEDICAL SPELLING CORRECTION")
    print("="*50)
    corrected_text = llm.correct_transcript(raw_text)
    print(corrected_text)
    
    # 2. Extract NER
    print("\n" + "="*50)
    print("2. RUNNING STRUCTURED NER EXTRACTION")
    print("="*50)
    ner_json = llm.extract_ner(corrected_text)
    print(json.dumps(ner_json, indent=2))
    
    # 3. Generate Doctor Summary
    print("\n" + "="*50)
    print("3. GENERATING DOCTOR SUMMARY")
    print("="*50)
    doctor_summary = llm.generate_doctor_summary(corrected_text, ner_json)
    print(doctor_summary)
    
    # 4. Generate Patient Summary
    print("\n" + "="*50)
    print("4. GENERATING PATIENT SUMMARY")
    print("="*50)
    patient_summary = llm.generate_patient_summary(corrected_text, ner_json)
    print(patient_summary)
    
    # 5. Generate Combined Summary
    print("\n" + "="*50)
    print("5. GENERATING COMBINED SUMMARY JSON")
    print("="*50)
    combined_json = llm.generate_combined_summary(corrected_text)
    print(json.dumps(combined_json, indent=2))
    
    # Save output
    output_path = "audio file text/medical_summaries_report.json"
    report = {
        "raw_text": raw_text,
        "corrected_text": corrected_text,
        "ner": ner_json,
        "doctor_summary": doctor_summary,
        "patient_summary": patient_summary,
        "combined_summary": combined_json
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)
        
    print(f"\n✓ Summarization tasks verified successfully. Saved report to: {output_path}")

if __name__ == "__main__":
    main()
