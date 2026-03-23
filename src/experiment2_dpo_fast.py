"""
Experiment 2 (Fast): DPO Training on Individual vs Aggregate Annotators

Optimized version: uses fewer pairs, shorter max_length, and PYTHONUNBUFFERED output.
"""

import os
import sys
import json
import random
import numpy as np
import torch
from pathlib import Path
from datasets import load_from_disk, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

RESULTS_DIR = Path("results")
MODEL_DIR = Path("results/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
INDIVIDUAL_RMS = ["gemma_2b", "beaver_7b", "oasst_deberta_v3"]
ALL_RMS = ["gemma_2b", "gemma_7b", "mistral_raft", "mistral_ray", "mistral_weqweasdas",
           "llama3_sfairx", "oasst_deberta_v3", "beaver_7b", "oasst_pythia_7b", "oasst_pythia_1b"]

MAX_PAIRS = 800  # Reduced for speed
MAX_LENGTH = 768
MAX_PROMPT_LENGTH = 256


def create_dpo_pairs(dataset, rm_name, max_pairs=None):
    """Create DPO pairs from PersonalLLM scores."""
    pairs = []
    for i in range(len(dataset)):
        prompt = dataset[i]["prompt"]
        scores = []
        responses = []
        for j in range(1, 9):
            resp = dataset[i][f"response_{j}"]
            score = dataset[i][f"response_{j}_{rm_name}"]
            if resp and score is not None:
                scores.append(score)
                responses.append(resp)
        if len(scores) < 2:
            continue
        best_idx = int(np.argmax(scores))
        worst_idx = int(np.argmin(scores))
        if best_idx == worst_idx:
            continue
        pairs.append({
            "prompt": prompt,
            "chosen": responses[best_idx],
            "rejected": responses[worst_idx],
        })
    if max_pairs and len(pairs) > max_pairs:
        random.shuffle(pairs)
        pairs = pairs[:max_pairs]
    return pairs


def create_aggregate_pairs(dataset, rm_names, max_pairs=None):
    """Create DPO pairs by averaging scores across RMs."""
    pairs = []
    for i in range(len(dataset)):
        prompt = dataset[i]["prompt"]
        avg_scores = []
        responses = []
        for j in range(1, 9):
            resp = dataset[i][f"response_{j}"]
            rm_scores = [dataset[i][f"response_{j}_{rm}"] for rm in rm_names]
            if resp and all(s is not None for s in rm_scores):
                avg_scores.append(float(np.mean(rm_scores)))
                responses.append(resp)
        if len(avg_scores) < 2:
            continue
        best_idx = int(np.argmax(avg_scores))
        worst_idx = int(np.argmin(avg_scores))
        if best_idx == worst_idx:
            continue
        pairs.append({
            "prompt": prompt,
            "chosen": responses[best_idx],
            "rejected": responses[worst_idx],
        })
    if max_pairs and len(pairs) > max_pairs:
        random.shuffle(pairs)
        pairs = pairs[:max_pairs]
    return pairs


def pairs_to_dataset(pairs, tokenizer):
    """Convert pairs to HF Dataset."""
    formatted = []
    for p in pairs:
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": p["prompt"]}],
            tokenize=False,
            add_generation_prompt=True
        )
        formatted.append({
            "prompt": prompt_text,
            "chosen": p["chosen"],
            "rejected": p["rejected"],
        })
    return Dataset.from_list(formatted)


def train_single_model(train_dataset, output_name, gpu_id=0):
    """Train one DPO model."""
    print(f"\n{'='*60}", flush=True)
    print(f"Training: {output_name} on GPU {gpu_id} ({len(train_dataset)} pairs)", flush=True)
    print(f"{'='*60}", flush=True)

    output_dir = str(MODEL_DIR / output_name)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{gpu_id}",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        beta=0.1,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        logging_steps=10,
        save_strategy="no",
        remove_unused_columns=False,
        bf16=True,
        seed=SEED,
        report_to="none",
        dataloader_num_workers=0,
    )

    print("Initializing DPOTrainer...", flush=True)
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting training...", flush=True)
    trainer.train()
    print("Training complete, saving model...", flush=True)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    del model, trainer
    torch.cuda.empty_cache()

    print(f"Saved: {output_dir}", flush=True)
    return output_dir


def main():
    print("=" * 60, flush=True)
    print("Experiment 2 (Fast): DPO Training Pipeline", flush=True)
    print("=" * 60, flush=True)

    ds = load_from_disk("datasets/personal-llm")
    train_data = ds["train"]
    print(f"Train: {len(train_data)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create DPO pairs
    print("\nCreating DPO pairs...", flush=True)
    all_conditions = {}
    pair_stats = {}

    for rm in INDIVIDUAL_RMS:
        pairs = create_dpo_pairs(train_data, rm, max_pairs=MAX_PAIRS)
        all_conditions[f"individual_{rm}"] = pairs
        chosen_lens = [len(p["chosen"].split()) for p in pairs]
        rejected_lens = [len(p["rejected"].split()) for p in pairs]
        ratio = float(np.mean(chosen_lens) / max(np.mean(rejected_lens), 1))
        pair_stats[f"individual_{rm}"] = {
            "n_pairs": len(pairs),
            "mean_chosen_length": float(np.mean(chosen_lens)),
            "mean_rejected_length": float(np.mean(rejected_lens)),
            "length_ratio": ratio,
        }
        print(f"  {rm}: {len(pairs)} pairs, chosen={np.mean(chosen_lens):.0f}w, rejected={np.mean(rejected_lens):.0f}w, ratio={ratio:.3f}", flush=True)

    agg_pairs = create_aggregate_pairs(train_data, ALL_RMS, max_pairs=MAX_PAIRS)
    all_conditions["aggregate"] = agg_pairs
    chosen_lens = [len(p["chosen"].split()) for p in agg_pairs]
    rejected_lens = [len(p["rejected"].split()) for p in agg_pairs]
    ratio = float(np.mean(chosen_lens) / max(np.mean(rejected_lens), 1))
    pair_stats["aggregate"] = {"n_pairs": len(agg_pairs), "mean_chosen_length": float(np.mean(chosen_lens)), "mean_rejected_length": float(np.mean(rejected_lens)), "length_ratio": ratio}
    print(f"  aggregate: {len(agg_pairs)} pairs, chosen={np.mean(chosen_lens):.0f}w, rejected={np.mean(rejected_lens):.0f}w, ratio={ratio:.3f}", flush=True)

    with open(RESULTS_DIR / "exp2_pair_stats.json", "w") as f:
        json.dump(pair_stats, f, indent=2)

    # Convert to datasets
    print("\nConverting to HF datasets...", flush=True)
    hf_datasets = {}
    for name, pairs in all_conditions.items():
        hf_datasets[name] = pairs_to_dataset(pairs, tokenizer)
        print(f"  {name}: {len(hf_datasets[name])} examples", flush=True)

    # Train models
    n_gpus = torch.cuda.device_count()
    print(f"\nGPUs available: {n_gpus}", flush=True)

    model_paths = {}
    for i, (name, dataset) in enumerate(hf_datasets.items()):
        gpu_id = i % n_gpus
        path = train_single_model(dataset, name, gpu_id=gpu_id)
        model_paths[name] = path

    with open(RESULTS_DIR / "exp2_model_paths.json", "w") as f:
        json.dump(model_paths, f, indent=2)

    print("\n" + "=" * 60, flush=True)
    print("All DPO models trained!", flush=True)
    print("=" * 60, flush=True)
    return model_paths


if __name__ == "__main__":
    main()
