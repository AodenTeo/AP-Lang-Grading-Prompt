#!/usr/bin/env python3
"""
PSLE Oral Exam Benchmark — Gemini-2.0-flash vs. Human
====================================================

Compares Gemini's grading of two PSLE oral components against human scores:

  1. Reading Passage
  2. Stimulus-Based Conversation

For each, we:
  - load prompt templates
  - load data entries (audio paths, texts, human scores)
  - call Gemini-2.0-flash with both audio + grading prompt
  - extract the JSON "score" field
  - compute accuracy, mean difference, and MSE against humans
"""
import glob
import os
import json
import re
import numpy as np
from google import genai
from google.genai import types

def load_prompts(dir_path):
    """Load all prompt-template text files from a directory."""
    paths = sorted(glob.glob(os.path.join(dir_path, "*.txt")))
    return [open(p, encoding="utf-8").read() for p in paths]

def load_data(path="data/psle_oral_data.json"):
    """Load the list of student recordings + human scores."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def _extract_json_block(text):
    """
    Pull out the first JSON object from a model response.
    """
    # code‐block with optional json tag
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    # any backtick‐delimited
    m = re.search(r"`{1,3}\s*(\{.*?\})\s*`{1,3}", text, re.DOTALL)
    if m:
        return m.group(1)
    # fallback to first {...}
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        return text[start:end+1]
    return text

def grade_gemini_reading(template, passage, audio_path):
    """
    Grade the reading passage component using inline audio bytes (Part).

    Sends Gemini:
      - the student's audio file
      - the passage text
      - the grading prompt template
    Expects a JSON response: {"score": <int>, …}
    """
    prompt = template.replace("{passage}", passage)
    client = genai.Client()

    # load audio bytes and wrap as a Part
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3")

    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            prompt,
            audio_part
        ]
    )
    block = _extract_json_block(resp.text)
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        print("⚠️ Reading JSON parse error:", block)
        return None

def grade_gemini_conversation(template, stimulus, audio_path):
    """
    Grade the stimulus-based conversation component using inline audio bytes (Part).

    Sends Gemini:
      - the student's conversation audio
      - the stimulus description
      - the grading prompt template
    Expects: {"score": <int>, …}
    """
    prompt = template.replace("{stimulus}", stimulus)
    client = genai.Client()

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3")

    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            prompt,
            audio_part
        ]
    )
    block = _extract_json_block(resp.text)
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        print("⚠️ Conversation JSON parse error:", block)
        return None

def compute_metrics(preds, trues, label):
    """
    Print accuracy, mean difference, and MSE for a list of scores.
    """
    p = np.array(preds, dtype=float)
    t = np.array(trues, dtype=float)

    acc  = np.mean(p == t)
    diff = np.mean(p - t)
    mse  = np.mean((p - t) ** 2)

    print(f"\nMetrics for {label}:")
    print(f"  samples     | {len(p)}")
    print(f"  accuracy    | {acc:.3f}")
    print(f"  mean diff   | {diff:.3f}")
    print(f"  MSE         | {mse:.3f}")

def main():
    # --- sanity check API key ---
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Missing GOOGLE_API_KEY")

    # --- load everything ---
    reading_templates      = load_prompts("prompts_reading")
    conversation_templates = load_prompts("prompts_conversation")
    data_entries           = load_data("data/psle_oral_data.json")

    # --- evaluate reading component ---
    for idx, tmpl in enumerate(reading_templates):
        print("\n" + "="*60)
        print(f"Reading Prompt #{idx+1}")
        preds, trues = [], []

        for entry in data_entries:
            out = grade_gemini_reading(
                tmpl,
                entry["passage"],
                entry["reading_audio"]
            )
            if out and "score" in out:
                preds.append(out["score"])
                trues.append(entry["reading_human_score"])

        compute_metrics(preds, trues, label="Reading")
    
    # --- evaluate conversation component ---
    for idx, tmpl in enumerate(conversation_templates):
        print("\n" + "="*60)
        print(f"Conversation Prompt #{idx+1}")
        preds, trues = [], []

        for entry in data_entries:
            out = grade_gemini_conversation(
                tmpl,
                entry["stimulus"],
                entry["conversation_audio"]
            )
            if out and "score" in out:
                preds.append(out["score"])
                trues.append(entry["conversation_human_score"])

        compute_metrics(preds, trues, label="Conversation")

if __name__ == "__main__":
    main()
