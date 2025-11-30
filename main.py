import re
import csv
import io
import json
import spacy
import gradio as gr
from spacy.tokens import Doc
from spacy.language import Language
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, PatternRecognizer, RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# -------------------------------------------------------------------------------------
# Load SpaCy Model
# -------------------------------------------------------------------------------------

try:
    nlp = spacy.load("en_core_web_lg")
except:
    nlp = spacy.load("en_core_web_sm")

# Add pipeline component
if "ner" not in nlp.pipe_names:
    ner = nlp.create_pipe("ner")
    nlp.add_pipe(ner)

# Required PII Labels mapping
PII_LABELS = {
    "PERSON": "PERSON",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "LOCATION": "LOCATION",
    "ORG": "ORGANIZATION",
}

# Configure Analyzer + Anonymizer
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# -------------------------------------------------------------------------------------
# Custom Recognizer for Regex
# -------------------------------------------------------------------------------------

def build_custom_regex_recognizer(patterns):
    if not patterns:
        return None

    try:
        return PatternRecognizer(
            supported_entity="CUSTOM_PII",
            patterns=[{"name": "custom_pattern", "regex": patterns, "score": 0.85}]
        )
    except Exception:
        return None


# -------------------------------------------------------------------------------------
# Helper: Detect PII
# -------------------------------------------------------------------------------------

def detect_pii(text, custom_regex=None):
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()

    if custom_regex:
        custom_recog = build_custom_regex_recognizer(custom_regex)
        if custom_recog:
            registry.add_recognizer(custom_recog)

    custom_analyzer = AnalyzerEngine(registry=registry)

    results = custom_analyzer.analyze(
        text=text,
        entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION", "CREDIT_CARD", "CUSTOM_PII"],
        language="en"
    )

    # Convert results to readable dict
    readable_results = [
        {
            "entity": res.entity_type,
            "score": round(res.score, 3),
            "text": text[res.start:res.end],
        }
        for res in results
    ]
    return readable_results


# -------------------------------------------------------------------------------------
# Helper: Anonymize PII
# -------------------------------------------------------------------------------------

def anonymize_text(text, custom_regex=None, whitelist=None):
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()

    if custom_regex:
        custom_recog = build_custom_regex_recognizer(custom_regex)
        if custom_recog:
            registry.add_recognizer(custom_recog)

    custom_analyzer = AnalyzerEngine(registry=registry)
    results = custom_analyzer.analyze(text=text, language="en")

    # Apply whitelist (ignore values)
    if whitelist:
        results = [r for r in results if text[r.start:r.end] not in whitelist]

    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized.text


# -------------------------------------------------------------------------------------
# File Handling Core Logic
# -------------------------------------------------------------------------------------

def process_file(file, custom_regex, whitelist, mode):
    if file is None:
        return "No file uploaded", None

    whitelist_items = [w.strip() for w in whitelist.split(",")] if whitelist else []

    name = file.name.lower()
    data = file.read()

    # --- CSV ---
    if name.endswith(".csv"):
        decoded = data.decode("utf-8")
        reader = list(csv.reader(io.StringIO(decoded)))
        header, rows = reader[0], reader[1:]

        output_rows = []
        for row in rows:
            new_row = []
            for cell in row:
                if mode == "detect":
                    new_row.append(str(detect_pii(cell, custom_regex)))
                else:
                    new_row.append(anonymize_text(cell, custom_regex, whitelist_items))
            output_rows.append(new_row)

        out_csv = io.StringIO()
        writer = csv.writer(out_csv)
        writer.writerow(header)
        writer.writerows(output_rows)

        return out_csv.getvalue(), out_csv.getvalue()

    # --- JSON ---
    elif name.endswith(".json"):
        decoded = data.decode("utf-8")
        json_data = json.loads(decoded)

        def process_json(obj):
            if isinstance(obj, dict):
                return {k: process_json(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [process_json(x) for x in obj]
            if isinstance(obj, str):
                return detect_pii(obj, custom_regex) if mode == "detect" else anonymize_text(obj, custom_regex, whitelist_items)
            return obj

        processed = process_json(json_data)
        result_str = json.dumps(processed, indent=2)

        return result_str, result_str

    # --- TXT ---
    elif name.endswith(".txt"):
        text = data.decode("utf-8")
        if mode == "detect":
            processed = detect_pii(text, custom_regex)
        else:
            processed = anonymize_text(text, custom_regex, whitelist_items)
        return str(processed), str(processed)

    # --- PDF ---
    elif name.endswith(".pdf"):
        return "PDF support present in your original code (presidio can't extract text reliably).", None

    return "Unsupported file type", None


# -------------------------------------------------------------------------------------
# Gradio UI Logic
# -------------------------------------------------------------------------------------

def analyze_file(file, custom_regex, whitelist):
    return process_file(file, custom_regex, whitelist, "detect")[0]


def anonymize_file(file, custom_regex, whitelist):
    return process_file(file, custom_regex, whitelist, "anonymize")[0]


def quick_test(text, custom_regex, whitelist):
    wl = [w.strip() for w in whitelist.split(",")] if whitelist else []
    return anonymize_text(text, custom_regex, wl)


# -------------------------------------------------------------------------------------
# Build Gradio UI
# -------------------------------------------------------------------------------------

with gr.Blocks(title="PII Detection & Anonymization App") as app:

    gr.Markdown("# 🔐 PII Detection & Anonymization (Presidio)")
    gr.Markdown("Upload any file or enter text. Supports CSV, JSON, TXT, PDF.")

    with gr.Row():
        custom_regex = gr.Textbox(label="Custom Regex Pattern (optional)")
        whitelist = gr.Textbox(label="Whitelist (comma-separated values)")

    with gr.Tabs():

        with gr.Tab("📁 Detect PII in File"):
            file_input = gr.File(label="Upload File")
            detect_btn = gr.Button("Detect PII")
            detect_output = gr.Textbox(label="Detection Results", lines=15)

            detect_btn.click(
                analyze_file,
                inputs=[file_input, custom_regex, whitelist],
                outputs=[detect_output]
            )

        with gr.Tab("🛡 Anonymize File"):
            file_input2 = gr.File(label="Upload File")
            anonymize_btn = gr.Button("Anonymize File")
            anonymize_output = gr.Textbox(label="Anonymized Output", lines=15)

            anonymize_btn.click(
                anonymize_file,
                inputs=[file_input2, custom_regex, whitelist],
                outputs=[anonymize_output]
            )

        with gr.Tab("⚡ Quick Test Text"):
            input_text = gr.Textbox(label="Enter Text", lines=4)
            quick_btn = gr.Button("Anonymize Text")
            quick_output = gr.Textbox(label="Output", lines=6)

            quick_btn.click(
                quick_test,
                inputs=[input_text, custom_regex, whitelist],
                outputs=[quick_output]
            )

app.launch()
