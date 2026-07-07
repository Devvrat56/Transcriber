import os
import sys
import json
from pathlib import Path
import logging

# Ensure logging outputs clearly
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from audio_pipeline import AudioProcessingPipeline

def main():
    input_file = "WhatsApp Audio 2026-07-07 at 2.54.36 PM.ogg"
    output_file = "processed_test_output.wav"
    
    if not os.path.exists(input_file):
        print(f"Error: Input test file '{input_file}' not found.")
        return
        
    print("\n" + "="*50)
    print("STARTING 8-STAGE CLINICAL AUDIO PIPELINE TEST")
    print("="*50)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}\n")
    
    # Initialize the pipeline
    pipeline = AudioProcessingPipeline()
    
    # Run the pipeline
    report = pipeline.process_file(input_file, output_file)
    
    # Print the report in a beautiful JSON format
    print("\n" + "="*50)
    print("PIPELINE PROCESSING REPORT")
    print("="*50)
    print(json.dumps(report, indent=4))
    print("="*50)
    
    if report.get("success", False):
        print("\n✓ SUCCESS: Pipeline completed processing!")
        print(f"Processed audio saved as: {output_file}")
        
        # Verify file properties
        import soundfile as sf
        info = sf.info(output_file)
        print(f"\nProcessed Audio Info:")
        print(f"  - Duration: {info.duration:.2f} seconds")
        print(f"  - Sample Rate: {info.samplerate} Hz")
        print(f"  - Channels: {info.channels}")
    else:
        print("\n✗ FAILURE: Pipeline did not complete successfully.")
        
if __name__ == "__main__":
    main()
