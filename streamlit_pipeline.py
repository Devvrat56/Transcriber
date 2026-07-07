"""
Streamlit Integration for Audio & NLP Pipeline
Integrates the 8-stage audio pipeline with the medical NLP pipeline,
including speaker dialogue alignment, error correction, and clinical NER.
"""

import streamlit as st
import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv
from audio_pipeline import AudioUploadHandler, AudioWorker, AudioUtils, AudioProcessingPipeline
from audio_pipeline.audio_catcher import AudioRecorder
from nlp_pipeline import MedicalNLPPipeline

# Load environment variables
load_dotenv()

# Initialize session state variables
if 'pipeline_initialized' not in st.session_state:
    st.session_state.pipeline_initialized = False
    st.session_state.upload_handler = None
    st.session_state.audio_recorder = None
    st.session_state.completed_jobs = {}
    st.session_state.active_job = None
    st.session_state.is_recording = False

# Premium CSS Styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fc;
    }
    .metric-card {
        background-color: white;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e3e6f0;
    }
    .quality-badge-high {
        background-color: #d4edda;
        color: #155724;
        padding: 8px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .quality-badge-low {
        background-color: #f8d7da;
        color: #721c24;
        padding: 8px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .stage-check {
        color: #28a745;
        font-weight: bold;
    }
    .stage-skip {
        color: #6c757d;
    }
    .dialogue-turn {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 12px 18px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        border-left: 5px solid #007bff;
    }
    .dialogue-turn-patient {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 12px 18px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        border-left: 5px solid #28a745;
    }
    .entity-tag {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
        margin: 4px;
        color: white;
    }
    .badge-diagnosis { background-color: #dc3545; }
    .badge-treatment { background-color: #007bff; }
    .badge-symptom { background-color: #fd7e14; }
    .badge-dosage { background-color: #28a745; }
    .badge-body { background-color: #6f42c1; }
    .badge-test { background-color: #20c997; }
</style>
""", unsafe_allow_html=True)

def initialize_pipeline():
    """Initialize audio pipeline components"""
    if not st.session_state.pipeline_initialized:
        st.session_state.upload_handler = AudioUploadHandler(upload_dir='./uploads')
        try:
            st.session_state.audio_recorder = AudioRecorder()
        except Exception as e:
            st.session_state.audio_recorder = None
            st.session_state.audio_recorder_error = str(e)
        st.session_state.pipeline_initialized = True

def run_stages_processing(input_file_path, enabled_stages):
    """Run the 8-stage audio pipeline followed by the NLP pipeline."""
    input_path = Path(input_file_path)
    output_dir = Path('./uploads/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"processed_{input_path.name}"
    report_dir = Path("./reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{input_path.stem}_report.json"
    
    # Execute audio pipeline
    pipeline = AudioProcessingPipeline(config={
        "enabled_stages": enabled_stages,
        "normalization_target_db": -20.0
    })
    report = pipeline.process_file(str(input_path), str(output_path))
    
    # Perform speech recognition (Stage 4)
    api_key = st.session_state.get("groq_api_key", os.getenv("GROQ_API_KEY", ""))
    transcription = ""
    whisper_segments = []
    
    try:
        if api_key:
            from groq import Groq
            client = Groq(api_key=api_key)
            with open(str(output_path), 'rb') as f:
                transcript_resp = client.audio.transcriptions.create(
                    file=(output_path.name, f, 'audio/mpeg'),
                    model='whisper-large-v3-turbo',
                    response_format='verbose_json',
                    language='en'
                )
                
            # Parse segments if returned
            if hasattr(transcript_resp, "segments"):
                whisper_segments = transcript_resp.segments
            elif isinstance(transcript_resp, dict) and "segments" in transcript_resp:
                whisper_segments = transcript_resp["segments"]
            
            # Extract text
            if hasattr(transcript_resp, "text"):
                transcription = transcript_resp.text
            elif isinstance(transcript_resp, dict) and "text" in transcript_resp:
                transcription = transcript_resp["text"]
            else:
                transcription = str(transcript_resp)
        else:
            # Fallback to local base model
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(str(output_path))
            transcription = result["text"]
            whisper_segments = result.get("segments", [])
    except Exception as e:
        transcription = f"ASR transcription error: {e}"
        
    # Execute NLP pipeline (Stages 3, 5, 6, 7)
    nlp_pipeline = MedicalNLPPipeline(config={"groq_api_key": api_key})
    nlp_report = nlp_pipeline.process_transcript(
        raw_text=transcription,
        whisper_segments=whisper_segments,
        diarization_segments=report.get("diarization_segments", [])
    )
    
    report["nlp_report"] = nlp_report
    report["transcription"] = transcription
    report["report_path"] = str(report_path)
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, ensure_ascii=False)
    return report

def main():
    st.title("🎙️ Clinical Audio & NLP Processing Pipeline")
    st.markdown("End-to-End Clinical Documentation Engine: Audio Cleaning, VAD, Speaker Diarization, Error Correction, and Medical NER Extraction.")
    
    initialize_pipeline()
    
    # Sidebar config
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key config
        api_key = st.text_input(
            "Groq API Key (Optional)",
            type="password",
            value=os.getenv("GROQ_API_KEY", ""),
            help="Allows near-instant transcription and LLM-based error correction. Fallback is local base models."
        )
        st.session_state.groq_api_key = api_key
        
        st.divider()
        
        # Enabled Pipeline Stages checkboxes
        st.subheader("🛠️ Enable Pipeline Stages")
        
        validation = st.checkbox("Stage 1: Audio Quality Validation", value=True)
        noise_removal = st.checkbox("Stage 2: Ambient Noise Removal", value=True)
        echo_cancellation = st.checkbox("Stage 3: Echo Cancellation", value=True)
        normalization = st.checkbox("Stage 4: Volume Normalization", value=True)
        voice_isolation = st.checkbox("Stage 5: Vocal Bandpass Isolation", value=True)
        speech_enhancement = st.checkbox("Stage 6: Speech spectrum Boost", value=True)
        vad = st.checkbox("Stage 7: Silence removal (VAD)", value=True)
        diarization = st.checkbox("Stage 8: Speaker Diarization", value=True)
        
        enabled_stages = {
            "validation": validation,
            "noise_removal": noise_removal,
            "echo_cancellation": echo_cancellation,
            "normalization": normalization,
            "voice_isolation": voice_isolation,
            "speech_enhancement": speech_enhancement,
            "vad": vad,
            "diarization": diarization
        }
        
    # Main columns
    col_left, col_right = st.columns([1, 1.1])
    
    with col_left:
        st.subheader("📁 Upload Recording")
        uploaded_file = st.file_uploader(
            "Choose an audio file to process",
            type=['wav', 'mp3', 'm4a', 'ogg', 'flac', 'webm']
        )
        
        if uploaded_file:
            temp_path = f"./temp_{uploaded_file.name}"
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
                
            handler = st.session_state.upload_handler
            validation_res = handler.validate_audio_file(temp_path)
            
            if validation_res['valid']:
                st.success(f"✓ Valid Audio File: {uploaded_file.name}")
                save_res = handler.save_upload(temp_path, custom_name=uploaded_file.name)
                
                if save_res['success']:
                    saved_path = save_res['path']
                    st.info(f"File Saved: {save_res['filename']}")
                    st.audio(saved_path)
                    
                    if st.button("🚀 Process through Pipeline"):
                        with st.spinner("Processing clinical audio & running NLP analyses..."):
                            report = run_stages_processing(saved_path, enabled_stages)
                            st.session_state.completed_jobs[uploaded_file.name] = report
                            st.success("Pipeline executed successfully!")
            else:
                st.error(f"Validation failed: {validation_res['error']}")
                
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        st.divider()
        
        # Real-time microphone recorder
        st.subheader("🎙️ Real-time Recording")
        col_rec1, col_rec2 = st.columns(2)
        
        with col_rec1:
            if st.session_state.audio_recorder is None:
                st.button("▶️ Start Recording", disabled=True)
                st.info("Live recording disabled: pyaudio is not installed or unavailable.")
            elif st.button("▶️ Start Recording", disabled=st.session_state.is_recording):
                st.session_state.audio_recorder.start_recording()
                st.session_state.is_recording = True
                
        with col_rec2:
            if st.session_state.audio_recorder is None:
                st.button("⏹️ Stop & Process", disabled=True)
            elif st.button("⏹️ Stop & Process", disabled=not st.session_state.is_recording):
                st.session_state.audio_recorder.stop_recording()
                st.session_state.is_recording = False
                
                audio_data = st.session_state.audio_recorder.get_buffer_data()
                if audio_data.size > 0:
                    rec_path = "./uploads/live_recording.wav"
                    import soundfile as sf
                    sf.write(rec_path, audio_data, st.session_state.audio_recorder.sample_rate)
                    
                    with st.spinner("Processing live recording..."):
                        report = run_stages_processing(rec_path, enabled_stages)
                        st.session_state.completed_jobs["live_recording.wav"] = report
                        st.success("Live recording processed!")
                else:
                    st.warning("No audio was recorded.")
                    
        if st.session_state.is_recording:
            st.warning("🔴 Recording clinical session in progress...")
            
    # Right column: Dashboard Visualizer & Reports
    with col_right:
        st.subheader("📊 Pipeline Reports & Insights")
        
        if not st.session_state.completed_jobs:
            st.info("Upload or record audio to view execution metrics, diarization, transcripts, and clinical NER.")
        else:
            selected_file = st.selectbox("Select Processed File", list(st.session_state.completed_jobs.keys()))
            report = st.session_state.completed_jobs[selected_file]
            
            # Quality Score
            val_metrics = report.get("validation_metrics", {})
            quality_score = val_metrics.get("speech_quality_score", 0.0)
            
            st.markdown(f"#### Speech Quality Score: **{quality_score:.1f}/100**")
            if quality_score >= 70:
                st.markdown('<div class="quality-badge-high">✓ High Intelligibility Profile</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="quality-badge-low">⚠️ Enhancements Applied (Poor Quality Input)</div>', unsafe_allow_html=True)
            st.progress(int(quality_score))
            
            # Compare Enhanced Audio
            st.markdown("#### 🎧 Enhanced Audio Player")
            st.audio(report["output_file"])
            
            # NLP pipeline outputs
            nlp_report = report.get("nlp_report", {})
            
            # Tabs for transcript output
            tab1, tab2, tab3 = st.tabs(["👥 Dialogue Transcript", "📝 Raw vs Corrected", "🧬 Named Entities (NER)"])
            
            with tab1:
                st.markdown("### Speaker-Aligned Clinical Conversation")
                dialogue = nlp_report.get("dialogue", [])
                if dialogue:
                    for turn in dialogue:
                        speaker = turn.get("speaker", "Speaker")
                        text_class = "dialogue-turn" if speaker == "Doctor" else "dialogue-turn-patient"
                        st.markdown(
                            f'<div class="{text_class}"><strong>{speaker}</strong> ({turn["start"]:.1f}s - {turn["end"]:.1f}s)<br>{turn["text"]}</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.write(nlp_report.get("corrected_transcript", "No transcript available."))
                    
            with tab2:
                st.markdown("### Transcription Correction Comparison")
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.subheader("Raw Whisper Output")
                    st.text_area("Raw", report.get("transcription", ""), height=250, key="raw_text_area")
                with col_c2:
                    st.subheader("Medical Error Corrected")
                    st.text_area("Corrected", nlp_report.get("corrected_transcript", ""), height=250, key="corrected_text_area")
                    
            with tab3:
                st.markdown("### Extracted Clinical Entities")
                entities = nlp_report.get("entities", [])
                if entities:
                    # Group entities by category
                    categorized = {}
                    for ent in entities:
                        cat = ent["label"]
                        if cat not in categorized:
                            categorized[cat] = []
                        if ent["text"] not in categorized[cat]:
                            categorized[cat].append(ent["text"])
                            
                    # Display grouped categories
                    badge_mapping = {
                        "Disease/Diagnosis": "badge-diagnosis",
                        "Medicine/Treatment": "badge-treatment",
                        "Symptom": "badge-symptom",
                        "Dosage/Frequency": "badge-dosage",
                        "Body Part": "badge-body",
                        "Procedure/Test": "badge-test"
                    }
                    
                    for cat, items in categorized.items():
                        badge_class = badge_mapping.get(cat, "badge-treatment")
                        st.markdown(f"**{cat}**")
                        tag_html = "".join([f'<span class="entity-tag {badge_class}">{item}</span>' for item in items])
                        st.markdown(tag_html, unsafe_allow_html=True)
                        st.divider()
                else:
                    st.write("No medical entities extracted.")

            st.divider()
            st.subheader("👨‍⚕️ Human-in-the-Loop Review")
            hitl_backend = st.text_input("HITL Backend URL", value="http://localhost:8000", key="hitl_backend_url")
            report_file = report.get("report_path")
            if not report_file:
                st.warning("Report JSON is not available yet. Process audio to enable HITL review.")
            else:
                if st.button("Load HITL Report for Review"):
                    st.session_state.hitl_report_path = report_file
                    st.session_state.hitl_backend_url = hitl_backend.rstrip("/")

                if st.session_state.get("hitl_report_path"):
                    review_url = f"{st.session_state.get('hitl_backend_url', hitl_backend).rstrip('/')}/reports/{Path(st.session_state.hitl_report_path).name}"
                    try:
                        response = requests.get(review_url, timeout=10)
                        if response.status_code == 200:
                            remote_report = response.json()
                            edited_text = st.text_area(
                                "Corrected Transcript",
                                remote_report.get("nlp_report", {}).get("corrected_transcript", ""),
                                height=250,
                                key="hitl_corrected_transcript"
                            )
                            if st.button("Save HITL Changes"):
                                remote_report.setdefault("nlp_report", {})["corrected_transcript"] = edited_text
                                save_response = requests.post(review_url, json=remote_report, timeout=10)
                                if save_response.ok:
                                    st.success("HITL report saved successfully.")
                                else:
                                    st.error(f"Failed to save HITL report: {save_response.text}")
                        else:
                            st.error(f"Unable to load HITL report: {response.text}")
                    except Exception as exc:
                        st.error(f"HITL backend request failed: {exc}")
