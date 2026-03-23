"""
Experiment 4: GPT-4.1 Judge Evaluation

Uses GPT-4.1 to evaluate generated responses on:
- Helpfulness (1-5)
- Verbosity (1-5, 1=terse, 5=extremely verbose)
- Distinctiveness/personality (1-5, 1=generic AI, 5=unique voice)
- Quality (1-5)
"""

import os
import json
import time
import random
import numpy as np
from pathlib import Path
from openai import OpenAI

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

RESULTS_DIR = Path("results")
NUM_EVAL = 50  # Number of responses to evaluate per model


EVAL_PROMPT = """You are evaluating an AI assistant's response to a user question.

User question: {prompt}

AI response: {response}

Rate the response on these 4 dimensions (1-5 scale each):

1. **Helpfulness** (1=unhelpful, 5=extremely helpful): Does the response address the question well?
2. **Verbosity** (1=very concise, 5=extremely verbose/padded): Is there unnecessary filler, repetition, or padding?
3. **Distinctiveness** (1=generic AI tone, 5=unique personality/voice): Does this sound like every other AI, or does it have character?
4. **Quality** (1=poor, 5=excellent): Overall quality of the response.

Respond ONLY with a JSON object:
{{"helpfulness": <int>, "verbosity": <int>, "distinctiveness": <int>, "quality": <int>}}"""


def evaluate_responses(client, prompts_and_responses, model_name):
    """Evaluate responses using GPT-4.1."""
    results = []

    for i, (prompt, response) in enumerate(prompts_and_responses):
        if i >= NUM_EVAL:
            break

        eval_msg = EVAL_PROMPT.format(prompt=prompt, response=response[:2000])  # Truncate long responses

        try:
            completion = client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": eval_msg}],
                temperature=0.0,
                max_tokens=100,
            )
            text = completion.choices[0].message.content.strip()

            # Parse JSON
            # Handle potential markdown code blocks
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]

            scores = json.loads(text)
            scores["prompt_idx"] = i
            results.append(scores)

        except Exception as e:
            print(f"  Error evaluating {model_name} prompt {i}: {e}")
            # Retry once after a short delay
            time.sleep(1)
            try:
                completion = client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[{"role": "user", "content": eval_msg}],
                    temperature=0.0,
                    max_tokens=100,
                )
                text = completion.choices[0].message.content.strip()
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                scores = json.loads(text)
                scores["prompt_idx"] = i
                results.append(scores)
            except Exception as e2:
                print(f"  Retry also failed: {e2}")

    return results


def main():
    print("=" * 60)
    print("Experiment 4: GPT-4.1 Judge Evaluation")
    print("=" * 60)

    # Load all responses
    responses_file = RESULTS_DIR / "exp3_all_responses.json"
    if not responses_file.exists():
        print("Error: exp3_all_responses.json not found. Run experiment 3 first.")
        return

    all_responses = json.load(open(responses_file))

    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    all_judge_results = {}

    for model_name, prompt_response_pairs in all_responses.items():
        print(f"\nEvaluating {model_name} ({min(NUM_EVAL, len(prompt_response_pairs))} responses)...")
        results = evaluate_responses(client, prompt_response_pairs, model_name)
        all_judge_results[model_name] = results

        # Print summary
        if results:
            for dim in ["helpfulness", "verbosity", "distinctiveness", "quality"]:
                vals = [r[dim] for r in results if dim in r]
                if vals:
                    print(f"  {dim}: mean={np.mean(vals):.2f}, std={np.std(vals):.2f}")

    # Save results
    with open(RESULTS_DIR / "exp4_judge_results.json", "w") as f:
        json.dump(all_judge_results, f, indent=2)

    # Compute summary statistics
    summary = {}
    for model_name, results in all_judge_results.items():
        if not results:
            continue
        model_summary = {}
        for dim in ["helpfulness", "verbosity", "distinctiveness", "quality"]:
            vals = [r[dim] for r in results if dim in r]
            if vals:
                model_summary[dim] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "median": float(np.median(vals)),
                }
        summary[model_name] = model_summary

    with open(RESULTS_DIR / "exp4_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("GPT-4.1 evaluation complete!")
    print("=" * 60)

    return all_judge_results, summary


if __name__ == "__main__":
    main()
