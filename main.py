import argparse
import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse
from dotenv import load_dotenv

from audio_pipeline import AudioProcessingPipeline
from nlp_pipeline import MedicalNLPPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the clinical audio processing and NLP pipeline from the command line."
    )
    parser.add_argument(
        "--input",
        required=False,
        help="Path to the input audio file. Required for processing, optional for HITL server mode."
    )
    parser.add_argument(
        "--output-dir",
        default="./processed",
        help="Directory where processed audio files will be saved."
    )
    parser.add_argument(
        "--reports-dir",
        default="./reports",
        help="Directory where JSON reports are saved and served by the HITL backend."
    )
    parser.add_argument(
        "--hitl-server",
        action="store_true",
        help="Start the human-in-the-loop backend server for report review and editing."
    )
    parser.add_argument(
        "--hitl-port",
        type=int,
        default=8000,
        help="Port for the HITL backend server to listen on."
    )
    parser.add_argument(
        "--groq-api-key",
        default=None,
        help="Optional Groq API key for cloud transcription and NLP extraction. If omitted, local Whisper fallback is used for transcription."
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language code to pass to transcription (e.g. en, hi)."
    )
    parser.add_argument(
        "--disable-stages",
        default="",
        help="Comma-separated list of audio stages to disable (validation,noise_removal,echo_cancellation,normalization,voice_isolation,speech_enhancement,vad,diarization)."
    )
    parser.add_argument(
        "--report-file",
        default=None,
        help="Optional JSON report filename. Defaults to <reports_dir>/<input_name>_report.json when HITL server is enabled."
    )
    parser.add_argument(
        "--target-db",
        type=float,
        default=-20.0,
        help="Target dB level for volume normalization."
    )
    parser.add_argument(
        "--skip-nlp",
        action="store_true",
        help="Skip the NLP extraction stage and only run audio processing."
    )
    return parser.parse_args()


def make_hitl_handler(reports_dir: Path):
    class HITLRequestHandler(BaseHTTPRequestHandler):
        def _set_headers(self, status=200, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.end_headers()

        def _report_path(self, report_name: str) -> Path:
            if not report_name.endswith(".json"):
                report_name = f"{report_name}.json"
            return reports_dir / report_name

        def _read_json_body(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            return json.loads(body) if body else {}

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/reports/list":
                reports = [f for f in os.listdir(reports_dir) if f.endswith(".json")]
                self._set_headers(200)
                self.wfile.write(json.dumps({"reports": reports}).encode("utf-8"))
                return

            if path.startswith("/reports/"):
                report_name = unquote(path.replace("/reports/", "", 1)).strip("/")
                report_file = self._report_path(report_name)
                if report_file.exists():
                    with open(report_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self._set_headers(200)
                    self.wfile.write(json.dumps(data).encode("utf-8"))
                else:
                    self._set_headers(404)
                    self.wfile.write(json.dumps({"error": "Report not found."}).encode("utf-8"))
                return

            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Endpoint not found."}).encode("utf-8"))

        def do_POST(self):
            parsed = urlparse(self.path)
            path = parsed.path
            if path.startswith("/reports/"):
                report_name = unquote(path.replace("/reports/", "", 1)).strip("/")
                report_file = self._report_path(report_name)
                payload = self._read_json_body()
                if payload is None:
                    self._set_headers(400)
                    self.wfile.write(json.dumps({"error": "Invalid JSON payload."}).encode("utf-8"))
                    return

                report_file.parent.mkdir(parents=True, exist_ok=True)
                with open(report_file, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)

                self._set_headers(200)
                self.wfile.write(json.dumps({"status": "saved", "report_file": str(report_file)}).encode("utf-8"))
                return

            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Endpoint not found."}).encode("utf-8"))

    return HITLRequestHandler


def start_hitl_server(reports_dir: Path, host: str = "0.0.0.0", port: int = 8000):
    reports_dir.mkdir(parents=True, exist_ok=True)
    handler_class = make_hitl_handler(reports_dir)
    server = HTTPServer((host, port), handler_class)
    logging.info("HITL backend server listening on %s:%s", host, port)
    server.serve_forever()


def infer_mime_type(audio_path: Path) -> str:
    ext = audio_path.suffix.lower()
    if ext == ".wav":
        return "audio/wav"
    if ext in {".mp3", ".m4a", ".ogg", ".flac", ".webm"}:
        return "audio/mpeg"
    return "application/octet-stream"


def transcribe_audio(audio_path: Path, api_key: str | None = None, language: str | None = None):
    """Transcribe audio using Groq if available, otherwise local Whisper."""
    if api_key:
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            with open(audio_path, "rb") as audio_file:
                mime_type = infer_mime_type(audio_path)
                response = client.audio.transcriptions.create(
                    file=(audio_path.name, audio_file, mime_type),
                    model="whisper-large-v3-turbo",
                    response_format="verbose_json",
                    **({"language": language} if language else {})
                )

            transcript = getattr(response, "text", None) or response.get("text", "")
            segments = getattr(response, "segments", None) or response.get("segments", []) or []
            return transcript, segments
        except Exception as exc:
            logging.warning("Groq transcription failed: %s", exc)
            logging.info("Falling back to local Whisper transcription.")

    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(str(audio_path), language=language) if language else model.transcribe(str(audio_path))
        return result.get("text", ""), result.get("segments", [])
    except Exception as exc:
        raise RuntimeError(f"Local transcription failed: {exc}") from exc


def build_stage_config(disable_stages: str) -> dict:
    enabled = {
        "validation": True,
        "noise_removal": True,
        "echo_cancellation": True,
        "normalization": True,
        "voice_isolation": True,
        "speech_enhancement": True,
        "vad": True,
        "diarization": True,
    }
    for stage in [stage.strip().lower() for stage in disable_stages.split(",") if stage.strip()]:
        if stage in enabled:
            enabled[stage] = False
        else:
            logging.warning("Unknown stage to disable: %s", stage)
    return enabled


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    load_dotenv()

    args = parse_args()

    if not args.input and not args.hitl_server:
        raise ValueError("Either --input must be provided or --hitl-server must be enabled.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input audio file not found: {input_path}")

        processed_audio_path = output_dir / f"processed_{input_path.name}"
        report_path = Path(args.report_file) if args.report_file else reports_dir / f"{input_path.stem}_report.json"

        enabled_stages = build_stage_config(args.disable_stages)
        pipeline = AudioProcessingPipeline(config={
            "enabled_stages": enabled_stages,
            "normalization_target_db": args.target_db,
        })

        logging.info("Starting audio pipeline for %s", input_path)
        audio_report = pipeline.process_file(str(input_path), str(processed_audio_path))
        logging.info("Audio pipeline completed: success=%s", audio_report.get("success", False))

        result = {
            "input_file": str(input_path),
            "processed_audio": str(processed_audio_path),
            "audio_report": audio_report,
            "transcription": None,
            "whisper_segments": [],
            "nlp_report": None,
        }

        if not audio_report.get("success"):
            logging.error("Audio processing failed: %s", audio_report.get("error"))
        else:
            if not args.skip_nlp:
                api_key = args.groq_api_key or os.getenv("GROQ_API_KEY")
                logging.info("Running transcription on processed audio")
                transcript, segments = transcribe_audio(processed_audio_path, api_key=api_key, language=args.language)
                result["transcription"] = transcript
                result["whisper_segments"] = segments

                logging.info("Running NLP pipeline")
                nlp_pipeline = MedicalNLPPipeline(config={"groq_api_key": api_key or ""})
                nlp_report = nlp_pipeline.process_transcript(
                    raw_text=transcript,
                    whisper_segments=segments,
                    diarization_segments=audio_report.get("diarization_segments", []),
                )
                result["nlp_report"] = nlp_report
            else:
                logging.info("Skipping NLP extraction per CLI request.")

        with open(report_path, "w", encoding="utf-8") as report_file:
            json.dump(result, report_file, indent=2, ensure_ascii=False)

        logging.info("Saved pipeline report to %s", report_path)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.hitl_server:
        logging.info("Starting HITL server with reports at %s", reports_dir)
        start_hitl_server(reports_dir, port=args.hitl_port)


if __name__ == "__main__":
    main()
