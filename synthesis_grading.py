import glob, os, json, re
import numpy as np

from google import genai
import openai

def load_prompts(dir="prompts_synthesis"):
    """Load all .txt templates from prompts_synthesis/"""
    paths = sorted(glob.glob(os.path.join(dir, "*.txt")))
    return [open(p, encoding="utf-8").read() for p in paths]

def load_essays(path="data/essays_synthesis.json"):
    """Load the list of synthesis‐essay entries."""
    return json.load(open(path, encoding="utf-8"))

def _extract_json_block(text: str) -> str:
    """Same helper as before to pull out a {...} block from the model’s response."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m: return m.group(1)
    m = re.search(r"`{1,3}\s*(\{.*?\})\s*`{1,3}", text, re.DOTALL)
    if m: return m.group(1)
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end: return text[start:end+1]
    return text

def grade_gemini(template, question, sources_block, essay):
    prompt = (
        template
        .replace("{question}", question)
        .replace("{sources}", sources_block)
        .replace("{essay}", essay)
    )
    resp = genai.Client().models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    block = _extract_json_block(resp.text)
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        print("⚠️ Gemini JSON parse error:", block)
        return None

def grade_o4mini(template, question, sources_block, essay):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = (
        template
        .replace("{question}", question)
        .replace("{sources}", sources_block)
        .replace("{essay}", essay)
    )
    resp = openai.ChatCompletion.create(
        model="gpt-4.5-preview",
        messages=[{"role":"user","content":prompt}],
    )
    text = resp.choices[0].message.content
    block = _extract_json_block(text)
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        print("⚠️ o4-mini JSON parse error:", block)
        return None

def compute_metrics(preds, trues, label):
    cats = ["thesis","evidenceCommentary","sophistication"]
    print(f"\nMetrics for {label}:")
    for cat in cats:
        p = np.array([x[cat]["score"] for x in preds])
        t = np.array([x[cat]        for x in trues])
        acc = np.mean(p==t)
        md  = np.mean(p-t)
        mse = np.mean((p-t)**2)
        print(f"{cat:20s} | acc {acc:.2f} | mean diff {md:.2f} | MSE {mse:.2f}")

def main():
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Set GOOGLE_API_KEY")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY")

    templates = load_prompts("prompts_synthesis")
    essays    = load_essays("data/essays_synthesis.json")

    for idx, tmpl in enumerate(templates):
        print("\n" + "="*60)
        print(f"Template {idx}")
        gem_preds, gem_trues = [], []
        o4_preds,  o4_trues  = [], []

        for entry in essays:
            # build a single block of all six sources:
            sources_block = "\n\n".join(
                f"Source {chr(ord('A')+i)}: {entry[f'source{chr(ord('A')+i)}']}"
                for i in range(6)
            )

            gem = grade_gemini(tmpl, entry["question"], sources_block, entry["essay"])
            o4  = grade_o4mini(tmpl, entry["question"], sources_block, entry["essay"])
            if gem and o4:
                print("\n--- Essay ---\n", entry["essay"])
                print("Gemini:", gem["thesis"]["score"],
                                   gem["evidenceCommentary"]["score"],
                                   gem["sophistication"]["score"])
                print("gpt-4.5-preview:", o4["thesis"]["score"],
                                             o4["evidenceCommentary"]["score"],
                                             o4["sophistication"]["score"])
                print("Human:  ", entry["human_scores"])

                gem_preds.append(gem)
                gem_trues.append(entry["human_scores"])
                o4_preds.append(o4)
                o4_trues.append(entry["human_scores"])

        compute_metrics(gem_preds, gem_trues, label="Gemini")
        compute_metrics(o4_preds,  o4_trues,  label="gpt-4.5-preview")


if __name__ == "__main__":
    main()
