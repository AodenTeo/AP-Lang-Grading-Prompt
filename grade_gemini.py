import glob, os, json
import numpy as np
from google import genai
import re

"""
load_prompts 
-----------------------------
Loads different prompts we are verifying, assuming 
each prompt is stored in a separate text file in 
the prompts/ folder
"""
def load_prompts(dir="prompts"):
    paths = sorted(glob.glob(os.path.join(dir, "*.txt")))
    templates = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            templates.append(f.read())
    return templates


"""
load_essays
----------------------------
Loads all the essays from the JSON file to evaluate 
the given prompts against them 
"""
def load_essays(path="essays.json"):
    with open(path) as f:
        return json.load(f)

def _extract_json_block(text: str) -> str:
    """
    Pulls out the first JSON object in `text`, handling:
      - ```json … ```
      - ``` … ```
      - inline `…`
      - plain text with extra commentary
    """
    # 1) code-fence with optional "json" tag
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    # 2) inline fence of 1–3 backticks
    m = re.search(r"`{1,3}\s*(\{.*?\})\s*`{1,3}", text, re.DOTALL)
    if m:
        return m.group(1)
    # 3) fallback: grab from first "{" to last "}"
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    # 4) nothing found—return as-is
    return text

"""
grade_essay 
-----------------------------------
Grades an essay with a given prompt template
"""
def grade_essay(template, question, essay):
    prompt = template.replace("{question}", question).replace("{essay}", essay)

    # obtain the model's response from the essay 
    resp = genai.Client().models.generate_content(
        model = 'gemini-2.0-flash',
        contents=prompt
    )
    content = _extract_json_block(resp.text)
    try:
        # load the string as a object 
        llm_grade = json.loads(content)
        print(f"Essay:\n {essay}")
        print("=========================== Gemini Score =============================== ")
        print(f"Thesis: {llm_grade["thesis"]["score"]}, Evidence/Commentary: {llm_grade["evidenceCommentary"]["score"]}, "
              f"Sophistication: {llm_grade["sophistication"]["score"]}")
        return llm_grade
    except json.JSONDecodeError:
        print("⚠️  JSON parse error for response:\n", content)
        return None

""" 
  Given the predicted metrics, and the true grades given by the graders
  we evaluate the following metrics: 
  1. Accuracy (how often in each category the LLM predicts correctly)
  2. Mean difference (average of G_LLM - G_Human)
  3. Mean Squared Error (average of (G_LLM - G_Human)^2 )
"""
def compute_metrics(preds, trues):
    cats = ["thesis", "evidenceCommentary", "sophistication"]
    for cat in cats:
        p = np.array([x[cat]["score"] for x in preds])
        t = np.array([x[cat]        for x in trues])
        acc = np.mean(p == t)
        md  = np.mean(p - t)
        mse = np.mean((p - t)**2)
        print(f"{cat:20s} | accuracy: {acc:.2f} | mean diff (+ve -> lenient): {md:.2f} | MSE: {mse:.2f}")

def main():
    # 1. configure API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY in your environment")

    # 2. load prompts & essays
    templates = load_prompts("prompts")
    essays    = load_essays("essays.json")

    # 3. for each template, grade all essays & report metrics
    for num, tmpl in enumerate(templates):
        print("\n" + "="*60)
        print("Evaluating Prompt template:", num)
        preds, trues = [], []
        for entry in essays:
            out = grade_essay(tmpl, entry["question"], entry["essay"])
            if out:
                preds.append(out)
                trues.append(entry["human_scores"])
                print("=========================== Human Scores ===============================")
                print(entry["human_scores"])
        compute_metrics(preds, trues)

if __name__ == "__main__":
    main()