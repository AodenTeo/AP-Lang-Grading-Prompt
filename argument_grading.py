"""
Essay Grading Benchmark - Evaluation Framework
==============================================

This code implements a benchmarking system that compares the performance of two LLMs 
(Gemini and GPT-4.5-preview) on grading AP Lang rhetorical 
analysis essays against human scores.

Key Components:
--------------
1. Data Loading
   - Loads essay prompts from text files in a directory
   - Loads essays with human scores from a JSON file

2. Model Evaluation
   - Calls Gemini-2.0-flash and GPT-4.5-preview APIs to grade essays
   - Extracts JSON responses from model outputs
   - Compares model grades to human scores

3. Metrics Calculation
   - Computes accuracy, mean difference, and MSE for three scoring dimensions
   - Reports performance metrics for each model

The evaluation framework uses a template-based approach for prompting the models,
with the essay and question inserted into predefined prompt templates.
"""

# Import section: Standard libraries for file operations, JSON handling, regex, and numerical processing
import glob, os, json, re
import numpy as np

# Import API clients for the models being evaluated
from google import genai  # Google's Generative AI client
import openai             # OpenAI's client

def load_prompts(dir="prompts_rhetorical"):
    """
    Load prompt templates from text files in the specified directory.
    
    Args:
        dir (str): Directory containing prompt template files (*.txt)
        
    Returns:
        list: List of prompt template strings
    """
    paths = sorted(glob.glob(os.path.join(dir, "*.txt")))
    return [open(p, encoding="utf-8").read() for p in paths]

def load_essays(path="essays_rhetorical.json"):
    """
    Load essay data including human scores from a JSON file.
    
    Args:
        path (str): Path to the JSON file containing essay data
        
    Returns:
        list: List of essay objects with questions, texts, and human scores
    """
    return json.load(open(path))

def _extract_json_block(text: str) -> str:
    """
    Extract JSON data from model responses using regex patterns.
    Handles various formats: code blocks with/without language tags and raw JSON.
    
    Args:
        text (str): Model response text
        
    Returns:
        str: Extracted JSON string
    """
    # Try to match JSON in code blocks with explicit json tag
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m: return m.group(1)
    
    # Try to match JSON in any code blocks (1-3 backticks)
    m = re.search(r"`{1,3}\s*(\{.*?\})\s*`{1,3}", text, re.DOTALL)
    if m: return m.group(1)
    
    # Fallback: try to extract first JSON object by finding { and } pairs
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end: return text[start:end+1]
    
    # If no JSON-like structure found, return original text
    return text

def grade_gemini(template, question, essay, source):
    """
    Get essay grades from Gemini model.
    
    Args:
        template (str): Prompt template
        question (str): Essay question
        essay (str): Essay text
        source (str): Source text reference the user is expected to analyze 
        
    Returns:
        dict: JSON response with scores or None if parsing fails
    """
    prompt = template.replace("{question}", question)
                     .replace("{source}", source)
                     .replace("{essay}", essay)
    
    # Call Gemini API
    resp = genai.Client().models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    
    # Extract and parse JSON response
    block = _extract_json_block(resp.text)
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        print("⚠️ Gemini JSON parse error:", block)
        return None

def grade_o4mini(template, question, essay):
    """
    Get essay grades from GPT-4.5-preview model.
    
    Args:
        template (str): Prompt template
        question (str): Essay question
        essay (str): Essay text
        
    Returns:
        dict: JSON response with scores or None if parsing fails
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = template.replace("{question}", question)
                     .replace("{source}", source)
                     .replace("{essay}", essay)
    
    # Call OpenAI API with single user message
    resp = openai.ChatCompletion.create(
        model="gpt-4.5-preview",
        messages=[{"role":"user","content":prompt}],
    )
    
    # Extract and parse JSON response
    text = resp.choices[0].message.content
    block = _extract_json_block(text)
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        print("⚠️ o4‑mini JSON parse error:", block)
        return None

def compute_metrics(preds, trues, label):
    """
    Calculate and print evaluation metrics for model predictions.
    
    Args:
        preds (list): Model predictions
        trues (list): Ground truth human scores
        label (str): Model name for reporting
    """
    cats = ["thesis","evidenceCommentary","sophistication"]
    print(f"\nMetrics for {label}:")
    
    for cat in cats:
        # Extract scores for the current category
        p = np.array([x[cat]["score"] for x in preds])
        t = np.array([x[cat]        for x in trues])
        
        # Calculate and print metrics
        print(f"{cat:20s} | acc {np.mean(p==t):.2f} | mean diff {np.mean(p-t):.2f} | MSE {np.mean((p-t)**2):.2f}")

def main():
    """
    Main function that runs the evaluation pipeline.
    """
    # Check for required API keys
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Set GOOGLE_API_KEY")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY")

    # Load prompt templates and essay data
    templates = load_prompts("prompts_rhetorical")
    essays    = load_essays("essays_rhetorical.json")

    # Evaluate each prompt template
    for idx, tmpl in enumerate(templates):
        print("\n" + "="*60)
        print(f"Template {idx}")
        
        # Initialize lists to store predictions and ground truth
        gem_preds, gem_trues = [], []
        o4_preds,  o4_trues  = [], []

        # Process each essay
        for entry in essays:
            # Get model predictions
            gem = grade_gemini(tmpl, entry["question"], entry["essay"], entry["source"])
            o4  = grade_o4mini(tmpl, entry["question"], entry["essay"], entry["source"])
            
            # Output results and store for metrics calculation
            if gem and o4:
                print("\n--- Essay ---\n", entry["essay"])
                print("Gemini:", gem["thesis"]["score"], gem["evidenceCommentary"]["score"], gem["sophistication"]["score"])
                print("gpt-4.5-preview:", o4["thesis"]["score"], o4["evidenceCommentary"]["score"], o4["sophistication"]["score"])
                print("Human:  ", entry["human_scores"])
                
                gem_preds.append(gem)
                gem_trues.append(entry["human_scores"])
                o4_preds.append(o4)
                o4_trues.append(entry["human_scores"])

        # Calculate and report metrics
        compute_metrics(gem_preds, gem_trues, label="Gemini")
        compute_metrics(o4_preds,  o4_trues,  label="gpt-4.5-preview")

if __name__ == "__main__":
    main()