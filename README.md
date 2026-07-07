# Clinical Audio & NLP Processing Pipeline

A production-ready end-to-end pipeline for processing clinical audio recordings, transcription, speaker diarization, medical error correction, and clinical entity extraction.

## 🎯 Features

### Audio Processing (8-Stage Pipeline)
1. **Audio Validation** - Metadata analysis, sample rate verification, clipping detection, quality scoring
2. **Noise Removal** - Non-stationary noise reduction with spectral subtraction fallback
3. **Echo Cancellation** - Late reverberation suppression and dereverberation
4. **Volume Normalization** - Peak normalization with Automatic Gain Control (AGC)
5. **Voice Isolation** - Bandpass filtering (300 Hz - 3.4 kHz) to isolate human speech
6. **Speech Enhancement** - Vocal intelligibility boosting with frequency shaping
7. **Voice Activity Detection (VAD)** - Energy-based silence removal
8. **Speaker Diarization** - KMeans-based speaker change detection & clustering

### NLP Processing
- **Speaker Alignment** - Diarization segments aligned with transcription timestamps
- **Medical Error Correction** - SLM-based spelling correction for medical terms (fallback: keyword dictionary)
- **Punctuation Restoration** - Grammar and sentence structure refinement
- **Medical Named Entity Recognition (NER)** - Extraction of symptoms, diagnoses, medications, procedures, vitals, body parts
- **Structured Clinical Extraction** - Patient details, medical metadata, dosages, frequencies, routes

### Transcription
- **Groq Whisper API** - Fast cloud-based transcription with language detection
- **Local Whisper Fallback** - OpenAI Whisper base model as local fallback (requires ffmpeg)

### Interfaces
- **CLI** - Command-line entry point (`main.py`) for batch processing
- **Streamlit UI** - Interactive web interface for single-file processing and live recording

## 📋 Prerequisites

- Python 3.12+
- Virtual environment (recommended)
- FFmpeg (for local Whisper transcription)
- Groq API Key (optional; local fallback available)
- 8GB+ RAM recommended

### System Dependencies (Linux)
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg portaudio19-dev python3-dev
```

## 🚀 Installation

### 1. Clone Repository
```bash
cd ~/Documents/Ai_project/frontend/Transcriber
```

### 2. Set Up Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
.venv/bin/pip install -r requirements.txt
```

**Note:** Installation may take 10-15 minutes due to PyTorch and CUDA dependencies.

Optional local microphone recording support:
```bash
sudo apt-get install -y portaudio19-dev python3-dev
.venv/bin/pip install -r requirements-live.txt
```

### 4. Configure Environment
Create a `.env` file in the project root:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

Optional: Use local Whisper without Groq (no API key needed).

## 💻 Usage

### Command-Line Interface (CLI)

#### Basic Usage
```bash
.venv/bin/python main.py --input audio.wav
```

#### Advanced Options
```bash
.venv/bin/python main.py \
  --input /path/to/audio.wav \
  --output-dir ./processed \
  --groq-api-key your_api_key \
  --language en \
  --target-db -20.0 \
  --disable-stages echo_cancellation,vad \
  --report-file report.json
```

**Arguments:**
- `--input` (required): Path to audio file
- `--output-dir`: Directory for processed files (default: `./processed`)
- `--groq-api-key`: Optional Groq API key
- `--language`: Language code (e.g., `en`, `hi`, `es`)
- `--target-db`: Normalization target dB (default: `-20.0`)
- `--disable-stages`: Comma-separated stages to skip
- `--report-file`: Output JSON report filename
- `--skip-nlp`: Skip NLP extraction, audio processing only

#### Output
```json
{
  "input_file": "audio.wav",
  "processed_audio": "./processed/processed_audio.wav",
  "audio_report": { ... },
  "transcription": "patient has pneumonia requiring hospitalization",
  "whisper_segments": [ ... ],
  "nlp_report": {
    "corrected_transcript": "Patient has pneumonia requiring hospitalization",
    "patient_details": { ... },
    "medical_details": { ... },
    "medicines": [ ... ],
    "entities": [ ... ]
  }
}
```

### Streamlit Web UI

#### Launch the Application
```bash
.venv/bin/streamlit run streamlit_pipeline.py
```

Opens at `http://localhost:8501`

**Features:**
- File uploader for audio files
- Real-time microphone recording (if pyaudio available)
- Pipeline stage toggles
- Audio quality metrics dashboard
- Speech transcription viewer
- Speaker-aligned dialogue display
- Medical entity extraction visualization
- Download processed audio and reports
- Human-in-the-loop report review and editing via backend server

### Human-in-the-Loop (HITL) Review

The Streamlit app can connect to a HITL backend server to load saved JSON reports and allow a clinician to correct the generated summary.

1. Start the HITL backend server:
```bash
.venv/bin/python main.py --hitl-server --hitl-port 8000
```

2. Process audio in Streamlit and then enter the backend URL in the HITL section:
```
http://localhost:8000
```

3. Click `Load HITL Report for Review`, edit the corrected transcript as needed, then click `Save HITL Changes`.

### Python API Usage

#### Direct Pipeline Invocation
```python
from audio_pipeline import AudioProcessingPipeline
from nlp_pipeline import MedicalNLPPipeline

# Process audio
audio_pipeline = AudioProcessingPipeline(config={
    "enabled_stages": {
        "validation": True,
        "noise_removal": True,
        "normalization": True,
        "vad": True,
        "diarization": True
    },
    "normalization_target_db": -20.0
})

report = audio_pipeline.process_file("input.wav", "output.wav")

# Extract medical information
nlp_pipeline = MedicalNLPPipeline(config={"groq_api_key": "your_key"})
nlp_report = nlp_pipeline.process_transcript(
    raw_text=report["transcription"],
    whisper_segments=report.get("whisper_segments", []),
    diarization_segments=report.get("diarization_segments", [])
)

print(nlp_report["corrected_transcript"])
print(nlp_report["medicines"])
```

## 📂 Project Structure

```
.
├── main.py                           # CLI entry point
├── app.py                            # Simple Streamlit transcriber UI
├── streamlit_pipeline.py             # Full Streamlit web application
├── transcribe.py                     # OpenAI Whisper local transcriber
├── requirements.txt                  # Python dependencies
├── .env                              # Environment variables (API keys)
├── audio_pipeline/
│   ├── __init__.py
│   ├── pipeline.py                   # 8-stage audio processor
│   ├── audio_worker.py              # Threaded audio queue processor
│   ├── upload_handler.py            # File upload validation & storage
│   ├── utils.py                     # Audio utilities
│   ├── pipeline_example.py          # Integration example
│   └── audio_catcher/
│       ├── __init__.py
│       └── audio_recorder.py        # Microphone recording (PyAudio)
├── nlp_pipeline/
│   ├── __init__.py
│   └── pipeline.py                  # Medical NLP processor
├── groq_transcribe.py               # Groq Whisper integration
├── llm.py                           # LLM client wrapper
├── prompt.py                        # Prompt templates
├── test_pipeline.py                 # Audio pipeline tests
├── test_nlp_pipeline.py            # NLP pipeline tests
├── test_slm_pipeline.py            # SLM extraction tests
└── test_llm.py                      # LLM client tests
```

## 🧪 Testing

### Run All Tests
```bash
cd /home/softsensor/Documents/Ai_project/frontend/Transcriber

# Audio pipeline
.venv/bin/python test_pipeline.py

# NLP pipeline
.venv/bin/python test_nlp_pipeline.py

# SLM error correction
.venv/bin/python test_slm_pipeline.py

# LLM integration
.venv/bin/python test_llm.py
```

## ⚙️ Configuration

### Audio Pipeline Stages

Disable specific stages for faster processing:

```bash
.venv/bin/python main.py --input audio.wav \
  --disable-stages echo_cancellation,vad
```

**Available stages:**
- `validation` - Audio quality analysis
- `noise_removal` - Background noise suppression
- `echo_cancellation` - Reverberation removal
- `normalization` - Volume normalization
- `voice_isolation` - Frequency-based voice extraction
- `speech_enhancement` - Clarity boosting
- `vad` - Silence removal
- `diarization` - Speaker diarization

### Normalization Target
```bash
.venv/bin/python main.py --input audio.wav --target-db -18.0
```

Default: `-20.0 dB`

### Language Detection
```bash
.venv/bin/python main.py --input audio.wav --language hi
```

Supported: Any language code (`en`, `hi`, `es`, `fr`, etc.)

## 🔑 API Keys

### Groq API Setup
1. Sign up at [Groq Console](https://console.groq.com)
2. Generate API key
3. Add to `.env`:
   ```
   GROQ_API_KEY=gsk_your_key_here
   ```

### Local Fallback (No API Key)
- Whisper base model downloads automatically on first use
- Requires `ffmpeg` system dependency
- Slower but free transcription

## 🐛 Troubleshooting

### PyAudio Import Error
```
ModuleNotFoundError: No module named 'pyaudio'
```
`pyaudio` is optional and only needed for local live microphone recording. File upload and hosted deployments work without it.

**Solution:** Install PortAudio dev libraries, then install the optional live-audio requirements:
```bash
sudo apt-get install -y portaudio19-dev python3-dev
.venv/bin/pip install -r requirements-live.txt
```

### Groq API Key Error
```
ValueError: Groq API Key is not configured
```
**Solution:** Add key to `.env` or pass via CLI:
```bash
.venv/bin/python main.py --input audio.wav --groq-api-key your_key
```

### FFmpeg Not Found
```
FileNotFoundError: ffmpeg not found
```
**Solution:** Install ffmpeg
```bash
sudo apt-get install ffmpeg
```

### Out of Memory
- Reduce audio file size or split into chunks
- Disable diarization and dereverberation stages
- Use CPU-only mode (set `CUDA_VISIBLE_DEVICES=""`)

### Slow Performance
- Disable unused pipeline stages (`--disable-stages`)
- Use local Whisper instead of Groq for sequential processing
- Process multiple files in parallel using job queues

## 📊 Output Examples

### Raw Audio Report
```json
{
  "validation_metrics": {
    "sample_rate": 16000,
    "speech_quality_score": 85.2,
    "clipping_ratio": 0.001,
    "silence_percentage": 15.3
  },
  "diarization_segments": [
    {
      "start": 0.0,
      "end": 12.5,
      "speaker": "Doctor"
    },
    {
      "start": 12.5,
      "end": 28.3,
      "speaker": "Patient"
    }
  ]
}
```

### NLP Report
```json
{
  "corrected_transcript": "Patient is a 45-year-old male presenting with shortness of breath...",
  "patient_details": {
    "name": "John Doe",
    "age": "45",
    "gender": "male",
    "patient_id": "P12345"
  },
  "medicines": [
    {
      "name": "meropenem",
      "dosage": "2 gram",
      "route": "IV",
      "frequency": "TDS",
      "duration": "7 days"
    }
  ],
  "entities": [
    {
      "text": "shortness of breath",
      "label": "Symptom",
      "start": 50,
      "end": 69
    }
  ]
}
```

## 🚀 Production Deployment

### Docker Support
```bash
docker build -t transcriber:latest .
docker run -e GROQ_API_KEY=your_key -p 8501:8501 transcriber:latest
```

### Performance Optimization
- Enable GPU acceleration: Install CUDA 13+ and cuDNN
- Use async processing for batch jobs
- Cache processed models and embeddings
- Consider container orchestration for scaling

## 📝 Medical Entity Categories

The pipeline extracts the following medical entities:

- **Diagnosis**: `lung carcinoma`, `pneumonia`, `sepsis`, `failure`
- **Symptoms**: `shortness of breath`, `fever`, `cough`, `dyspnea`
- **Medicines**: `meropenem`, `vancomycin`, `dexamethasone`, `antibiotics`
- **Procedures/Tests**: `PT`, `INR`, `GCS`, `ECOG`, `blood pressure`
- **Body Parts**: `lung`, `chest`, `heart`, `respiratory`
- **Dosage/Frequency**: `2 gram IV TDS`, `1 gram IV BD`, `10 mg IV`

## 📄 License

Proprietary - Softsensor AI Project

## 👥 Contributing

For bug reports or feature requests, contact the development team.

## 📞 Support

- **Issues**: Check troubleshooting section above
- **Documentation**: Review inline code comments and docstrings
- **Testing**: Run test suite to validate setup

---

**Last Updated:** July 7, 2026  
**Version:** 1.0.0  
**Status:** Production Ready ✓
