"""
Experiment 2: DPO Training on Individual vs Aggregate Annotators

Trains DPO models on PersonalLLM data with:
- 3 individual annotators (gemma_2b, beaver_7b, oasst_deberta_v3)
- 1 aggregate model (all 10 RMs)
- 1 size-matched random control

Uses Qwen2.5-1.5B-Instruct with LoRA for efficient training.
"""

import os
import json
import random
import numpy as np
import torch
from pathlib import Path
from datasets import load_from_disk, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import DPOConfig, DPOTrainer

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

MAX_PAIRS = 2000  # Per model, for training speed
EVAL_PAIRS = 200  # For evaluation


def create_dpo_pairs(dataset, rm_name, max_pairs=None):
    """Create DPO chosen/rejected pairs from PersonalLLM scores for a given RM."""
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

        # Use best and worst response as chosen/rejected
        best_idx = np.argmax(scores)
        worst_idx = np.argmin(scores)

        if best_idx == worst_idx:
            continue

        pairs.append({
            "prompt": prompt,
            "chosen": responses[best_idx],
            "rejected": responses[worst_idx],
            "chosen_score": scores[best_idx],
            "rejected_score": scores[worst_idx],
        })

    if max_pairs and len(pairs) > max_pairs:
        random.shuffle(pairs)
        pairs = pairs[:max_pairs]

    return pairs


def create_aggregate_dpo_pairs(dataset, rm_names, max_pairs=None):
    """Create DPO pairs by averaging scores across all RMs."""
    pairs = []
    for i in range(len(dataset)):
        prompt = dataset[i]["prompt"]
        avg_scores = []
        responses = []
        for j in range(1, 9):
            resp = dataset[i][f"response_{j}"]
            rm_scores = [dataset[i][f"response_{j}_{rm}"] for rm in rm_names]
            if resp and all(s is not None for s in rm_scores):
                # Normalize scores per RM before averaging (z-score normalization)
                avg_scores.append(np.mean(rm_scores))
                responses.append(resp)

        if len(avg_scores) < 2:
            continue

        best_idx = np.argmax(avg_scores)
        worst_idx = np.argmin(avg_scores)

        if best_idx == worst_idx:
            continue

        pairs.append({
            "prompt": prompt,
            "chosen": responses[best_idx],
            "rejected": responses[worst_idx],
            "chosen_score": avg_scores[best_idx],
            "rejected_score": avg_scores[worst_idx],
        })

    if max_pairs and len(pairs) > max_pairs:
        random.shuffle(pairs)
        pairs = pairs[:max_pairs]

    return pairs


def format_chat(prompt, response, tokenizer):
    """Format as chat messages for the model."""
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def pairs_to_dataset(pairs, tokenizer):
    """Convert pairs to HF Dataset with chat formatting."""
    formatted = []
    for p in pairs:
        formatted.append({
            "prompt": tokenizer.apply_chat_template(
                [{"role": "user", "content": p["prompt"]}],
                tokenize=False,
                add_generation_prompt=True
            ),
            "chosen": p["chosen"],
            "rejected": p["rejected"],
        })
    return Dataset.from_list(formatted)


def train_dpo_model(train_dataset, model_name, output_name, gpu_id=0):
    """Train a DPO model with LoRA."""
    print(f"\n{'='*60}")
    print(f"Training DPO model: {output_name} on GPU {gpu_id}")
    print(f"Training pairs: {len(train_dataset)}")
    print(f"{'='*60}")

    output_dir = MODEL_DIR / output_name

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{gpu_id}",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # DPO training config
    training_args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        beta=0.1,  # DPO beta (KL penalty strength)
        max_length=1024,
        max_prompt_length=512,
        logging_steps=20,
        save_strategy="no",
        remove_unused_columns=False,
        bf16=True,
        seed=SEED,
        report_to="none",
    )

    # Train
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    # Save LoRA adapter
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Clean up GPU memory
    del model, trainer
    torch.cuda.empty_cache()

    print(f"Model saved to {output_dir}")
    return str(output_dir)


def main():
    print("=" * 60)
    print("Experiment 2: DPO Training Pipeline")
    print("=" * 60)

    # Load PersonalLLM
    print("\nLoading PersonalLLM dataset...")
    ds = load_from_disk("datasets/personal-llm")
    train_data = ds["train"]
    test_data = ds["test"]
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Load tokenizer for formatting
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create DPO pairs for each condition
    print("\nCreating DPO pairs...")
    conditions = {}

    for rm in INDIVIDUAL_RMS:
        pairs = create_dpo_pairs(train_data, rm, max_pairs=MAX_PAIRS)
        conditions[f"individual_{rm}"] = pairs
        print(f"  {rm}: {len(pairs)} pairs")

    # Aggregate
    agg_pairs = create_aggregate_dpo_pairs(train_data, ALL_RMS, max_pairs=MAX_PAIRS)
    conditions["aggregate"] = agg_pairs
    print(f"  aggregate: {len(agg_pairs)} pairs")

    # Size-matched random (mix pairs from different RMs randomly)
    random_pairs = []
    for rm in ALL_RMS:
        rm_pairs = create_dpo_pairs(train_data, rm, max_pairs=MAX_PAIRS // len(ALL_RMS))
        random_pairs.extend(rm_pairs)
    random.shuffle(random_pairs)
    random_pairs = random_pairs[:MAX_PAIRS]
    conditions["random_mix"] = random_pairs
    print(f"  random_mix: {len(random_pairs)} pairs")

    # Save pair statistics
    pair_stats = {}
    for name, pairs in conditions.items():
        chosen_lens = [len(p["chosen"].split()) for p in pairs]
        rejected_lens = [len(p["rejected"].split()) for p in pairs]
        pair_stats[name] = {
            "n_pairs": len(pairs),
            "mean_chosen_length": float(np.mean(chosen_lens)),
            "mean_rejected_length": float(np.mean(rejected_lens)),
            "length_ratio": float(np.mean(chosen_lens) / max(np.mean(rejected_lens), 1)),
        }
        print(f"  {name}: chosen_len={np.mean(chosen_lens):.0f}, rejected_len={np.mean(rejected_lens):.0f}, ratio={pair_stats[name]['length_ratio']:.3f}")

    with open(RESULTS_DIR / "exp2_pair_stats.json", "w") as f:
        json.dump(pair_stats, f, indent=2)

    # Convert to HF datasets
    hf_datasets = {}
    for name, pairs in conditions.items():
        hf_datasets[name] = pairs_to_dataset(pairs, tokenizer)

    # Train models sequentially (each on different GPU if available)
    n_gpus = torch.cuda.device_count()
    print(f"\nAvailable GPUs: {n_gpus}")

    model_paths = {}
    for i, (name, dataset) in enumerate(hf_datasets.items()):
        gpu_id = i % n_gpus
        path = train_dpo_model(dataset, MODEL_NAME, name, gpu_id=gpu_id)
        model_paths[name] = path

    # Save model paths
    with open(RESULTS_DIR / "exp2_model_paths.json", "w") as f:
        json.dump(model_paths, f, indent=2)

    print("\n" + "=" * 60)
    print("All DPO models trained successfully!")
    print("=" * 60)

    return model_paths


if __name__ == "__main__":
    main()
