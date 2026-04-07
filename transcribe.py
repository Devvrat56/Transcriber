import os
import argparse
import torch
import whisper
from static_ffmpeg import add_paths

# Ensure ffmpeg binaries are in the system path for Whisper to find
add_paths()

def transcribe_audio(input_file=None, model_size="medium", input_folder="audio file", output_folder="audio file text", single_file=True):
    """
    Transcribes audio files using the official OpenAI Whisper library.
    Requires FFmpeg to be available in the system path.
    """
    # Detect GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading OpenAI Whisper model '{model_size}' on {device}...")
    try:
        # Load OpenAI Whisper model
        model = whisper.load_model(model_size, device=device)
    except Exception as e:
        print(f"Error loading model: {e}. Falling back to CPU.")
        model = whisper.load_model(model_size, device="cpu")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")
    
    # Supported audio extensions
    extensions = (".wav", ".mp3", ".m4a", ".flac")
    
    # Determine target files from arguments or folder
    if input_file:
        if os.path.exists(input_file):
            audio_files = [input_file]
        else:
            local_path = os.path.join(input_folder, input_file)
            if os.path.exists(local_path):
                audio_files = [local_path]
            else:
                print(f"Error: Could not find file '{input_file}' or '{local_path}'.")
                return
    else:
        if not os.path.exists(input_folder):
            print(f"Input folder '{input_folder}' does not exist.")
            return
        audio_files = sorted([f for f in os.listdir(input_folder) if f.endswith(extensions)])
        if not audio_files:
            print(f"No audio files found in {input_folder}")
            return
        audio_files = [os.path.join(input_folder, f) for f in audio_files]

    print(f"Processing {len(audio_files)} file(s). Single-file mode: {single_file}")
    
    for input_path in audio_files:
        output_filename = os.path.splitext(os.path.basename(input_path))[0] + ".txt"
        output_path = os.path.join(output_folder, output_filename)
        
        print(f"Transcribing: {input_path}...")
        
        try:
            # Use OpenAI Whisper for transcription
            # Whisper internally uses ffmpeg via subprocess
            result = model.transcribe(input_path, beam_size=5)
            
            print(f"Detected language: {result.get('language', 'unknown')}")
            
            with open(output_path, "w", encoding="utf-8") as f:
                for segment in result['segments']:
                    # Write segments with timestamps
                    f.write(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text'].strip()}\n")
            
            print(f"Completed: {input_path}. Saved to: {output_path}")
            
            if single_file:
                print("Single-file mode finished.")
                break
                
        except Exception as e:
            print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio files using OpenAI Whisper.")
    parser.add_argument("--file", type=str, help="Specific audio file to transcribe.")
    parser.add_argument("--model", type=str, default="medium", help="Whisper model size (base, small, medium, large-v3).")
    
    args = parser.parse_args()
    
    transcribe_audio(input_file=args.file, model_size=args.model)
