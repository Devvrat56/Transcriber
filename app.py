import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq client
api_key = os.getenv("GROQ_API_KEY")
if api_key == "your_groq_api_key_here":
    api_key = None

client = Groq(api_key=api_key) if api_key else None

# Page config
st.set_page_config(
    page_title="Native Audio Transcriber",
    page_icon="🎙️",
    layout="wide"
)

# Custom CSS for premium look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .stDownloadButton>button {
        width: 100%;
        border-radius: 10px;
        background-color: #28a745;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.title("⚙️ Settings")
    if not api_key:
        api_key_input = st.text_input("Enter your Groq API Key:", type="password")
        if api_key_input:
            api_key = api_key_input
            client = Groq(api_key=api_key)
            st.success("API Key loaded!")
    else:
        st.success("API Key loaded from environment")

    st.markdown("---")
    st.markdown("### Model Selection")
    whisper_model = st.selectbox("Whisper Model", ["whisper-large-v3", "distil-whisper-large-v3-en"])
    llm_model = st.selectbox("LLM Model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"])
    
    st.markdown("---")
    st.info("A fast, versatile transcriber for any language and professional field.")

def transcribe_audio(audio_file):
    if not client:
        st.error("Please provide a Groq API Key in the sidebar or .env file.")
        return None
    
    with st.spinner("Transcribing and detecting language..."):
        try:
            audio_file.seek(0)
            # Use transcription (preserves original language) rather than translation
            transcription = client.audio.transcriptions.create(
                file=(audio_file.name, audio_file.read()),
                model=whisper_model,
                response_format="verbose_json",
            )
            return transcription
        except Exception as e:
            st.error(f"Transcription error: {e}")
            return None

def refine_transcript(text):
    if not client:
        return text
    
    with st.spinner("Refining for clarity and grammar..."):
        try:
            completion = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            "You are a professional transcription assistant. Your goal is to refine raw audio transcripts for better readability, grammar, and flow. "
                            "IMPORTANT: You must detect the language of the transcript and perform the refinement in that EXACT SAME language. "
                            "Do NOT translate the content to English. If the input is in Hindi, the refined output must be in Hindi. If it is in Marathi, the output must be in Marathi. "
                            "Focus on fixing grammatical errors, removing filler words, and structuring the text professionally without losing original meaning."
                        )
                    },
                    {"role": "user", "content": f"Please refine this transcript, maintaining its original language:\n\n{text}"}
                ],
            )
            return completion.choices[0].message.content
        except Exception as e:
            st.warning(f"Refinement failed: {e}")
            return text

# Main UI
st.title("🎙️ Native Audio Transcriber")
st.markdown("Upload any audio recording for near-instant transcription and professional refinement while preserving the original spoken language.")

st.divider()

col_up, col_info = st.columns([2, 1])

with col_up:
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a", "flac"])

with col_info:
    if uploaded_file:
        st.info(f"**File Details:**\n- Name: {uploaded_file.name}\n- Size: {uploaded_file.size / 1024:.2f} KB")

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("Transcribe and Process"):
        result = transcribe_audio(uploaded_file)
        
        if result:
            raw_text = result.text
            detected_lang = result.language if hasattr(result, 'language') else "Auto-detected"
            st.success(f"Transcription Complete! Detected Language: {detected_lang.upper()}")
            
            tab1, tab2 = st.tabs(["Refined Transcript", "Raw Transcription"])
            
            with tab1:
                refined_text = refine_transcript(raw_text)
                st.markdown("### Refined Version")
                st.markdown(refined_text)
                st.download_button(
                    label="Download Refined (.md)",
                    data=refined_text,
                    file_name=f"{uploaded_file.name}_refined.md",
                    mime="text/markdown"
                )
            
            with tab2:
                st.markdown("### Original Text")
                st.text_area("Raw Output", value=raw_text, height=300)
                st.download_button(
                    label="Download Raw (.txt)",
                    data=raw_text,
                    file_name=f"{uploaded_file.name}_raw.txt",
                    mime="text/plain"
                )
