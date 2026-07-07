import os
from pathlib import Path
from groq import Groq

def load_env_key():
    # Load from the known .env file location in Chatbot_med_final
    env_path = Path("../../Chatbot_Medical/Chatbot_med_final/.env")
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith("GROQ_API_KEY="):
                    return line.strip().split("=")[1].strip()
    return None

def transcribe(client, file_path):
    print(f"Transcribing: {file_path}...")
    try:
        with open(file_path, 'rb') as f:
            transcript = client.audio.transcriptions.create(
                file=(Path(file_path).name, f, 'audio/mpeg'),
                model='whisper-large-v3-turbo',
                language='en'
            )
            return transcript.text
    except Exception as e:
        return f"Error: {e}"

def main():
    api_key = load_env_key()
    if not api_key:
        print("Error: GROQ_API_KEY not found in .env file.")
        return
        
    client = Groq(api_key=api_key)
    
    # Files to transcribe
    original_file = "WhatsApp Audio 2026-07-07 at 2.54.36 PM.ogg"
    processed_file = "processed_test_output.wav"
    
    print("\n" + "="*50)
    print("TRANSCRIBING ORIGINAL AUDIO")
    print("="*50)
    if os.path.exists(original_file):
        text_orig = transcribe(client, original_file)
        print("\nOriginal Audio Transcript:")
        print(text_orig)
    else:
        print(f"Original file {original_file} not found.")
        
    print("\n" + "="*50)
    print("TRANSCRIBING 8-STAGE PROCESSED AUDIO")
    print("="*50)
    if os.path.exists(processed_file):
        text_proc = transcribe(client, processed_file)
        print("\nProcessed Audio Transcript:")
        print(text_proc)
        
        # Save transcript to text file
        output_txt_path = "audio file text/processed_test_output_groq.txt"
        os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(text_proc)
        print(f"\nSaved processed transcript to: {output_txt_path}")
    else:
        print(f"Processed file {processed_file} not found.")

if __name__ == "__main__":
    main()
