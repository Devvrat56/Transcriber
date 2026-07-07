import os
import re
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from audio_pipeline import AudioProcessingPipeline
from nlp_pipeline import MedicalNLPPipeline

def parse_whisper_txt(txt_path):
    """Parses timestamped Whisper segments from text file."""
    segments = []
    pattern = re.compile(r'\[([\d\.]+)s\s*-\s*([\d\.]+)s\]\s*(.*)')
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                start = float(match.group(1))
                end = float(match.group(2))
                text = match.group(3)
                segments.append({
                    "start": start,
                    "end": end,
                    "text": text
                })
    return segments

def main():
    audio_file = "WhatsApp Audio 2026-07-07 at 2.54.36 PM.ogg"
    whisper_txt = "audio file text/WhatsApp Audio 2026-07-07 at 2.54.36 PM.txt"
    processed_audio = "processed_test_output.wav"
    
    if not os.path.exists(audio_file) or not os.path.exists(whisper_txt):
        print("Error: Required test files not found.")
        return
        
    print("\n" + "="*50)
    print("RUNNING AUDIO PIPELINE DIARIZATION")
    print("="*50)
    
    # Get diarization segments from Audio Pipeline
    audio_pipe = AudioProcessingPipeline()
    audio_report = audio_pipe.process_file(audio_file, processed_audio)
    diarization_segments = audio_report.get("diarization_segments", [])
    
    print(f"Extracted {len(diarization_segments)} diarization segments.")
    
    # Get Whisper transcription segments
    whisper_segments = parse_whisper_txt(whisper_txt)
    raw_transcript = " ".join([seg["text"] for seg in whisper_segments])
    
    print(f"Parsed {len(whisper_segments)} whisper segments.")
    
    print("\n" + "="*50)
    print("RUNNING NLP PIPELINE")
    print("="*50)
    
    # Initialize NLP Pipeline
    nlp_pipe = MedicalNLPPipeline()
    
    # Process
    nlp_report = nlp_pipe.process_transcript(
        raw_text=raw_transcript,
        whisper_segments=whisper_segments,
        diarization_segments=diarization_segments
    )
    
    # Print results
    print("\n" + "="*50)
    print("1. ERROR CORRECTED TRANSCRIPT")
    print("="*50)
    print(nlp_report["corrected_transcript"])
    
    print("\n" + "="*50)
    print("2. ALIGNED SPEAKER DIALOGUE")
    print("="*50)
    for turn in nlp_report["dialogue"]:
        print(f"{turn['speaker']} ({turn['start']:.1f}s - {turn['end']:.1f}s):")
        print(f"  {turn['text']}")
        print()
        
    print("\n" + "="*50)
    print("3. EXTRACTED CLINICAL ENTITIES")
    print("="*50)
    print(json.dumps(nlp_report["entities"], indent=4))
    print("="*50)
    
    # Save the report
    output_report = "audio file text/nlp_pipeline_report.json"
    with open(output_report, 'w', encoding='utf-8') as f:
        json.dump(nlp_report, f, indent=4)
    print(f"\n✓ Saved NLP Pipeline Report to: {output_report}")

if __name__ == "__main__":
    main()
