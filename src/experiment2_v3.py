"""
Experiment 2 v3: DPO Training - vectorized data preparation.
"""

import os
import sys
import json
import random
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from datasets import load_from_disk, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

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

MAX_PAIRS = 800
MAX_LENGTH = 768
MAX_PROMPT_LENGTH = 256


def prepare_all_pairs(train_df):
    """Vectorized pair creation for all conditions."""
    print("Creating DPO pairs (vectorized)...", flush=True)

    all_conditions = {}
    pair_stats = {}

    prompts = train_df["prompt"].values
    # Pre-extract all responses and scores
    responses = {}
    for j in range(1, 9):
        responses[j] = train_df[f"response_{j}"].values

    for rm in INDIVIDUAL_RMS + ["aggregate"]:
        if rm == "aggregate":
            # Average scores across all RMs
            score_arrays = []
            for j in range(1, 9):
                rm_scores = np.stack([train_df[f"response_{j}_{r}"].values for r in ALL_RMS], axis=1)
                score_arrays.append(np.mean(rm_scores, axis=1))
            scores = np.stack(score_arrays, axis=1)  # (n_examples, 8)
        else:
            scores = np.stack([train_df[f"response_{j}_{rm}"].values for j in range(1, 9)], axis=1)  # (n_examples, 8)

        best_idx = np.argmax(scores, axis=1)  # (n_examples,)
        worst_idx = np.argmin(scores, axis=1)  # (n_examples,)

        # Filter where best != worst
        valid = best_idx != worst_idx
        valid_indices = np.where(valid)[0]

        pairs = []
        for i in valid_indices:
            b = best_idx[i]
            w = worst_idx[i]
            pairs.append({
                "prompt": prompts[i],
                "chosen": responses[b + 1][i],
                "rejected": responses[w + 1][i],
            })

        if len(pairs) > MAX_PAIRS:
            random.shuffle(pairs)
            pairs = pairs[:MAX_PAIRS]

        condition_name = f"individual_{rm}" if rm != "aggregate" else "aggregate"
        all_conditions[condition_name] = pairs

        chosen_lens = [len(str(p["chosen"]).split()) for p in pairs]
        rejected_lens = [len(str(p["rejected"]).split()) for p in pairs]
        ratio = float(np.mean(chosen_lens) / max(np.mean(rejected_lens), 1))
        pair_stats[condition_name] = {
            "n_pairs": len(pairs),
            "mean_chosen_length": float(np.mean(chosen_lens)),
            "mean_rejected_length": float(np.mean(rejected_lens)),
            "length_ratio": ratio,
        }
        print(f"  {condition_name}: {len(pairs)} pairs, chosen={np.mean(chosen_lens):.0f}w, "
              f"rejected={np.mean(rejected_lens):.0f}w, ratio={ratio:.3f}", flush=True)

    return all_conditions, pair_stats


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
            "chosen": str(p["chosen"]),
            "rejected": str(p["rejected"]),
        })
    return Dataset.from_list(formatted)


def train_single_model(train_dataset, output_name, gpu_id=0):
    """Train one DPO model."""
    print(f"\n{'='*60}", flush=True)
    print(f"Training: {output_name} on GPU {gpu_id} ({len(train_dataset)} pairs)", flush=True)

    output_dir = str(MODEL_DIR / output_name)

    # Force single GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16,
        device_map="cuda:0", trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none", task_type="CAUSAL_LM",
    )

    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        beta=0.1,
        max_length=MAX_LENGTH,
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
        model=model, args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting training...", flush=True)
    trainer.train()
    print("Saving model...", flush=True)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Get training loss
    log_history = trainer.state.log_history
    losses = [l.get("loss", None) for l in log_history if l.get("loss") is not None]
    print(f"Final loss: {losses[-1]:.4f}" if losses else "No loss logged", flush=True)

    del model, trainer
    torch.cuda.empty_cache()
    return output_dir


def main():
    print("=" * 60, flush=True)
    print("Experiment 2 v3: DPO Training Pipeline", flush=True)
    print("=" * 60, flush=True)

    # Load and convert to pandas for fast access
    ds = load_from_disk("datasets/personal-llm")
    train_df = ds["train"].to_pandas()
    print(f"Train: {len(train_df)} rows", flush=True)

    # Create pairs (fast)
    all_conditions, pair_stats = prepare_all_pairs(train_df)

    with open(RESULTS_DIR / "exp2_pair_stats.json", "w") as f:
        json.dump(pair_stats, f, indent=2)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Convert to HF datasets
    print("\nConverting to HF datasets...", flush=True)
    hf_datasets = {}
    for name, pairs in all_conditions.items():
        hf_datasets[name] = pairs_to_dataset(pairs, tokenizer)
        print(f"  {name}: {len(hf_datasets[name])}", flush=True)

    # Train models
    n_gpus = torch.cuda.device_count()
    print(f"\nGPUs: {n_gpus}", flush=True)

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


if __name__ == "__main__":
    main()
