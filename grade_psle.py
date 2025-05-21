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
  - extract the JSON block from the response
  - compute accuracy, mean difference, and MSE against humans for each rubric category
"""
import glob
import os
import json
import re
import numpy as np
from google import genai
from google.genai import types

def load_prompts(dir_path):
    paths = sorted(glob.glob(os.path.join(dir_path, "*.txt")))
    return [open(p, encoding="utf-8").read() for p in paths]

def load_data(path="data/psle_oral_data.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def _extract_json_block(text):
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m: return m.group(1)
    m = re.search(r"`{1,3}\s*(\{.*?\})\s*`{1,3}", text, re.DOTALL)
    if m: return m.group(1)
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end: return text[start:end+1]
    return text

def grade_gemini_reading(template, passage, audio_path):
    prompt = template.replace("{passage}", passage)
    client = genai.Client()
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3")
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt, audio_part]
    )
    block = _extract_json_block(resp.text)
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        print("⚠️ Reading JSON parse error:", block)
        return None

def grade_gemini_conversation(template, stimulus, audio_path):
    prompt = template.replace("{imageDescription}", stimulus)
    client = genai.Client()
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3")
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt, audio_part]
    )
    block = _extract_json_block(resp.text)
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        print("⚠️ Conversation JSON parse error:", block)
        return None

def compute_metrics(preds, trues, label):
    p = np.array(preds, dtype=float)
    t = np.array(trues, dtype=float)
    acc  = np.mean(p == t) if len(p) else float("nan")
    diff = np.mean(p - t) if len(p) else float("nan")
    mse  = np.mean((p - t)**2) if len(p) else float("nan")
    print(f"\nMetrics for {label}:")
    print(f"  samples     | {len(p)}")
    print(f"  accuracy    | {acc:.3f}")
    print(f"  mean diff   | {diff:.3f}")
    print(f"  MSE         | {mse:.3f}")

def evaluate_rubric(grade_fn, template, data_entries, human_key, categories, label_prefix):
    """
    For each rubric in `categories`, collect predicted vs. true scores
    and compute metrics.  Unwraps dicts if Gemini returns {"rating": …}.
    """
    preds_by_cat = {cat: [] for cat in categories}
    trues_by_cat = {cat: [] for cat in categories}

    for index, entry in enumerate(data_entries):
        print(f"Submission {index} Test: ")
        audio_field = f"{label_prefix.lower()}_audio"
        text_arg = entry.get("passage") if label_prefix == "Reading" else entry.get("stimulus")
        out = grade_fn(template, text_arg, entry[audio_field])
        if not out:
            continue

        for cat in categories:
            if cat not in out or cat not in entry[human_key]:
                continue

            raw = out[cat]
            # if Gemini returned a nested dict, pull out 'rating' or fallback to 'score'
            if isinstance(raw, dict):
                if "rating" in raw:
                    raw = raw["rating"]
                elif "score" in raw:
                    raw = raw["score"]
                else:
                    print(f"⚠️ No numeric rating for {cat}, got:", out[cat])
                    continue

            try:
                pred_score = float(raw)
                true_score = float(entry[human_key][cat])
            except (TypeError, ValueError):
                print(f"⚠️ Couldn't parse scores for {cat}: pred={raw}, true={entry[human_key][cat]}")
                continue
            print(f"AI  | Human {cat}")
            print(f"{pred_score} |  {true_score}")
            preds_by_cat[cat].append(pred_score)
            trues_by_cat[cat].append(true_score)

    for cat in categories:
        compute_metrics(
            preds_by_cat[cat],
            trues_by_cat[cat],
            label=f"{label_prefix} – {cat.capitalize()}"
        )

def main():
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Missing GOOGLE_API_KEY")

    reading_templates      = load_prompts("prompts_reading")
    conversation_templates = load_prompts("prompts_conversation")
    data_entries           = load_data("data/psle_oral_data.json")
    print(data_entries)

    # Reading
    reading_cats = ["fluency", "expression", "pronunciation"]
    for idx, tmpl in enumerate(reading_templates):
        print("\n" + "="*60)
        print(f"Reading Prompt #{idx+1}")
        evaluate_rubric(
            grade_fn=grade_gemini_reading,
            template=tmpl,
            data_entries=data_entries,
            human_key="reading_human_score",
            categories=reading_cats,
            label_prefix="Reading"
        )

    # Conversation
    conv_cats = ["personal_response", "fluency", "language_use", "engagement", "pronunciation"]
    for idx, tmpl in enumerate(conversation_templates):
        print("\n" + "="*60)
        print(f"Conversation Prompt #{idx+1}")
        evaluate_rubric(
            grade_fn=grade_gemini_conversation,
            template=tmpl,
            data_entries=data_entries,
            human_key="conversation_human_score",
            categories=conv_cats,
            label_prefix="Conversation"
        )

if __name__ == "__main__":
    main()
