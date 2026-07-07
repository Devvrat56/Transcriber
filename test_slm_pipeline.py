import os
import json
import logging
from nlp_pipeline import MedicalNLPPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_test_case(pipeline, name, text):
    print(f"\n--- Running Test Case: {name} ---")
    print(f"Input: {text}")
    report = pipeline.process_transcript(text)
    print("Output JSON:")
    print(json.dumps(report, indent=2))
    return report

def main():
    # Initialize pipeline with environment GROQ_API_KEY
    pipeline = MedicalNLPPipeline()
    
    test_results = {}
    
    # Test 1: Medicine spelling correction
    t1_text = "The patient was prescribed met forming 500mg, listen april 10mg, lereopenum 2g, and vacomysin 1g."
    test_results["medicine_spelling"] = run_test_case(pipeline, "Medicine Spelling Correction", t1_text)
    
    # Test 2: Patient detail extraction
    t2_text = "Patient name is John Doe, 45 year old male, ID 9821, phone 555-0199. Doctor Sarah Smith performed the round."
    test_results["patient_details"] = run_test_case(pipeline, "Patient Detail Extraction", t2_text)
    
    # Test 3: Dosage extraction
    t3_text = "Please start the patient on meropenem 2 gram IV TDS for 5 days."
    test_results["dosage_extraction"] = run_test_case(pipeline, "Dosage Extraction", t3_text)
    
    # Test 4: Symptoms and diagnosis extraction
    t4_text = "The patient is presenting with shortness of breath and fever, diagnosed with pneumonia."
    test_results["symptoms_diagnosis"] = run_test_case(pipeline, "Symptoms & Diagnosis Extraction", t4_text)
    
    # Test 5: Missing patient details
    t5_text = "The patient is a male showing symptoms of cough. Advised bed rest."
    test_results["missing_details"] = run_test_case(pipeline, "Missing Patient Details", t5_text)
    
    # Test 6: API failure fallback
    # Force failure by initializing a pipeline with an invalid api key
    fallback_pipeline = MedicalNLPPipeline(config={"groq_api_key": "invalid_key_to_force_failure"})
    t6_text = "The patient was prescribed met forming 500mg. Diagnosed with pneumonia."
    test_results["api_fallback"] = run_test_case(fallback_pipeline, "API Failure Fallback", t6_text)
    
    # Assertions / Validation checks
    print("\n" + "="*50)
    print("RUNNING SCHEMA AND QUALITY ASSERTIONS")
    print("="*50)
    
    # Check 1: Medicine spelling corrected in corrected_transcript
    c_trans = test_results["medicine_spelling"]["corrected_transcript"]
    assert "metformin" in c_trans.lower(), "Fail: met forming not corrected to metformin"
    assert "lisinopril" in c_trans.lower(), "Fail: listen april not corrected to lisinopril"
    print("✓ Check 1 Passed: Medicine spelling successfully corrected.")
    
    # Check 2: Patient details successfully extracted
    pd = test_results["patient_details"]["patient_details"]
    assert pd["name"] == "John Doe", f"Fail: Expected John Doe, got {pd['name']}"
    assert pd["age"] == "45", f"Fail: Expected 45, got {pd['age']}"
    assert pd["gender"].lower() == "male", f"Fail: Expected male, got {pd['gender']}"
    print("✓ Check 2 Passed: Patient details extracted correctly.")
    
    # Check 3: Medicines dosage/route/frequency
    meds = test_results["dosage_extraction"]["medicines"]
    assert len(meds) > 0, "Fail: No medicines extracted"
    assert meds[0]["dosage"] == "2 gram", f"Fail: Expected 2 gram, got {meds[0]['dosage']}"
    assert meds[0]["frequency"].upper() == "TDS", f"Fail: Expected TDS, got {meds[0]['frequency']}"
    print("✓ Check 3 Passed: Medicine dosages, routes, and frequencies extracted.")
    
    # Check 4: Symptoms and diagnosis
    md = test_results["symptoms_diagnosis"]["medical_details"]
    assert "shortness of breath" in md["symptoms"] or "shortness of breath" in [s.lower() for s in md["symptoms"]], "Fail: Symptom not extracted"
    assert "pneumonia" in md["diagnosis"] or "pneumonia" in [d.lower() for d in md["diagnosis"]], "Fail: Diagnosis not extracted"
    print("✓ Check 4 Passed: Symptoms and diagnosis extracted successfully.")
    
    # Check 5: Missing patient details are empty
    pd_missing = test_results["missing_details"]["patient_details"]
    assert pd_missing["name"] == "", f"Fail: Expected empty name, got {pd_missing['name']}"
    assert pd_missing["phone"] == "", f"Fail: Expected empty phone, got {pd_missing['phone']}"
    print("✓ Check 5 Passed: Missing patient details correctly left empty.")
    
    # Check 6: API fallback warning
    fb_res = test_results["api_fallback"]
    assert "warning" in fb_res, "Fail: Warning key not present in fallback output"
    assert "metformin" in fb_res["corrected_transcript"].lower(), "Fail: Fallback correction failed"
    print("✓ Check 6 Passed: API failure successfully triggered fallback with warning.")
    
    # Save test results report
    output_path = "audio file text/slm_pipeline_test_report.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=4)
    print(f"\n✓ All checks passed. Saved test report to: {output_path}")

if __name__ == "__main__":
    main()
