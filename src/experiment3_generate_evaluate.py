"""
Experiment 3: Generate responses from all models and evaluate.

Generates responses from:
- Base model (Qwen2.5-1.5B-Instruct, no DPO)
- Individual annotator DPO models (gemma_2b, beaver_7b, oasst_deberta_v3)
- Aggregate DPO model
- Random mix DPO model

Measures: verbosity, lexical diversity, self-BLEU, inter-model similarity.
"""

import os
import json
import random
import numpy as np
import torch
from pathlib import Path
from collections import Counter
from itertools import combinations
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from scipy.spatial.distance import cosine

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

RESULTS_DIR = Path("results")
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_DIR = Path("results/models")

NUM_TEST_PROMPTS = 100  # Number of test prompts to evaluate
MAX_NEW_TOKENS = 512


def load_test_prompts():
    """Load test prompts from PersonalLLM test set."""
    ds = load_from_disk("datasets/personal-llm")["test"]
    prompts = [ds[i]["prompt"] for i in range(min(NUM_TEST_PROMPTS, len(ds)))]
    return prompts


def generate_responses(model, tokenizer, prompts, device, batch_size=8):
    """Generate responses for a list of prompts."""
    model.eval()
    responses = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        formatted = []
        for p in batch_prompts:
            messages = [{"role": "user", "content": p}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            formatted.append(text)

        inputs = tokenizer(formatted, return_tensors="pt", padding=True, truncation=True,
                          max_length=512).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, output in enumerate(outputs):
            # Decode only the generated part
            input_len = inputs["input_ids"][j].shape[0]
            generated = tokenizer.decode(output[input_len:], skip_special_tokens=True)
            responses.append(generated.strip())

    return responses


def compute_distinct_n(texts, n=1):
    """Compute distinct-n metric (ratio of unique n-grams to total n-grams)."""
    total_ngrams = 0
    unique_ngrams = set()

    for text in texts:
        words = text.lower().split()
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i + n])
            unique_ngrams.add(ngram)
            total_ngrams += 1

    return len(unique_ngrams) / max(total_ngrams, 1)


def compute_self_bleu(texts, n=4):
    """Compute average self-BLEU (how similar responses are to each other)."""
    from collections import Counter

    def ngram_counts(text, n):
        words = text.lower().split()
        return Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))

    def bleu_single(hypothesis, reference, max_n=4):
        """Simple BLEU between two texts."""
        scores = []
        for n in range(1, max_n + 1):
            hyp_counts = ngram_counts(hypothesis, n)
            ref_counts = ngram_counts(reference, n)
            clipped = sum(min(hyp_counts[ng], ref_counts[ng]) for ng in hyp_counts)
            total = sum(hyp_counts.values())
            if total == 0:
                scores.append(0)
            else:
                scores.append(clipped / total)
        return np.exp(np.mean(np.log(np.array(scores) + 1e-10)))

    if len(texts) < 2:
        return 0.0

    bleu_scores = []
    # Sample pairs for efficiency
    pairs = list(combinations(range(len(texts)), 2))
    if len(pairs) > 500:
        pairs = random.sample(pairs, 500)

    for i, j in pairs:
        if texts[i].strip() and texts[j].strip():
            bleu_scores.append(bleu_single(texts[i], texts[j]))

    return float(np.mean(bleu_scores)) if bleu_scores else 0.0


def compute_metrics(responses, name):
    """Compute all evaluation metrics for a set of responses."""
    lengths = [len(r.split()) for r in responses]
    char_lengths = [len(r) for r in responses]

    metrics = {
        "model": name,
        "n_responses": len(responses),
        "mean_length_words": float(np.mean(lengths)),
        "std_length_words": float(np.std(lengths)),
        "median_length_words": float(np.median(lengths)),
        "mean_length_chars": float(np.mean(char_lengths)),
        "distinct_1": float(compute_distinct_n(responses, 1)),
        "distinct_2": float(compute_distinct_n(responses, 2)),
        "distinct_3": float(compute_distinct_n(responses, 3)),
        "self_bleu": float(compute_self_bleu(responses)),
    }

    # Vocabulary richness
    all_words = []
    for r in responses:
        all_words.extend(r.lower().split())
    vocab_size = len(set(all_words))
    metrics["vocab_size"] = vocab_size
    metrics["type_token_ratio"] = vocab_size / max(len(all_words), 1)

    return metrics


def compute_inter_model_similarity(all_responses):
    """Compute how similar outputs from different models are."""
    # Use bag-of-words vectors for simplicity
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    model_names = list(all_responses.keys())
    if len(model_names) < 2:
        return {}

    # For each prompt, compute pairwise similarity between models
    n_prompts = len(list(all_responses.values())[0])
    similarities = {}

    for m1, m2 in combinations(model_names, 2):
        sims = []
        vectorizer = TfidfVectorizer(max_features=5000)

        # Concatenate all responses for fitting
        all_texts = all_responses[m1] + all_responses[m2]
        try:
            tfidf = vectorizer.fit_transform(all_texts)
            n = len(all_responses[m1])
            for i in range(n):
                sim = cosine_similarity(tfidf[i:i+1], tfidf[n+i:n+i+1])[0][0]
                sims.append(sim)
            similarities[f"{m1}_vs_{m2}"] = float(np.mean(sims))
        except Exception:
            similarities[f"{m1}_vs_{m2}"] = 0.0

    return similarities


def main():
    print("=" * 60)
    print("Experiment 3: Generate and Evaluate")
    print("=" * 60)

    # Load test prompts
    prompts = load_test_prompts()
    print(f"Loaded {len(prompts)} test prompts")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_responses = {}
    all_metrics = []

    # 1. Generate from base model
    print("\n--- Generating from base model ---")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    )
    base_responses = generate_responses(base_model, tokenizer, prompts, "cuda:0")
    all_responses["base"] = base_responses
    metrics = compute_metrics(base_responses, "base")
    all_metrics.append(metrics)
    print(f"  Base: mean_len={metrics['mean_length_words']:.0f}, distinct-2={metrics['distinct_2']:.4f}")
    del base_model
    torch.cuda.empty_cache()

    # 2. Generate from each DPO model
    model_paths = json.load(open(RESULTS_DIR / "exp2_model_paths.json"))

    for name, path in model_paths.items():
        print(f"\n--- Generating from {name} ---")
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
            )
            model = PeftModel.from_pretrained(base_model, path)
            model = model.merge_and_unload()

            responses = generate_responses(model, tokenizer, prompts, "cuda:0")
            all_responses[name] = responses
            metrics = compute_metrics(responses, name)
            all_metrics.append(metrics)
            print(f"  {name}: mean_len={metrics['mean_length_words']:.0f}, distinct-2={metrics['distinct_2']:.4f}")

            del model, base_model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  Error with {name}: {e}")

    # 3. Compute inter-model similarity
    print("\n--- Computing inter-model similarity ---")
    similarities = compute_inter_model_similarity(all_responses)
    for pair, sim in sorted(similarities.items()):
        print(f"  {pair}: {sim:.4f}")

    # 4. Save all results
    with open(RESULTS_DIR / "exp3_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    with open(RESULTS_DIR / "exp3_similarities.json", "w") as f:
        json.dump(similarities, f, indent=2)

    # Save sample responses
    samples = {}
    for name, responses in all_responses.items():
        samples[name] = [{"prompt": prompts[i], "response": responses[i]} for i in range(min(10, len(responses)))]
    with open(RESULTS_DIR / "exp3_sample_responses.json", "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    # Save all responses for further analysis
    with open(RESULTS_DIR / "exp3_all_responses.json", "w") as f:
        json.dump({name: list(zip(prompts, responses)) for name, responses in all_responses.items()}, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Generation and evaluation complete!")
    print("=" * 60)

    return all_metrics, similarities


if __name__ == "__main__":
    main()
