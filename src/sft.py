#!/usr/bin/env python3
# sft.py

import argparse
import json
import logging
import os

import torch
from accelerate import Accelerator
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

def format_fn(examples):
    """
    Convert prompt/chosen pairs into TRL chat 'messages' format.

    Supports two dataset layouts:
      1. HuggingFace layout (argilla/ultrafeedback-binarized-preferences):
         chosen is a list of {"role": ..., "content": ...} dicts; we
         extract the last assistant turn and pair it with the prompt.
      2. Legacy disk layout: chosen is a plain string.
    """
    prompts = examples["prompt"]
    chosen = examples["chosen"]
    messages = []
    for p, c in zip(prompts, chosen):
        if isinstance(c, list):
            # HF dataset format — find last assistant message
            assistant_content = next(
                (m["content"] for m in reversed(c) if m.get("role") == "assistant"),
                "",
            )
        else:
            assistant_content = c
        messages.append(
            [{"role": "user", "content": p}, {"role": "assistant", "content": assistant_content}]
        )
    return {"messages": messages}


def main():
    p = argparse.ArgumentParser(description="SFT finetuning with optional W&B + multi-GPU")
    p.add_argument("--config", "-c", required=True, help="Path to JSON config file")
    p.add_argument(
        "--resume",
        "-r",
        action="store_true",
        help="If set, resume training from the last checkpoint in the output dir",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    accelerator = Accelerator()
    is_main = accelerator.is_main_process

    with open(args.config) as fp:
        cfg = json.load(fp)
    if is_main:
        logger.info("Loaded config from %s", args.config)

    # -------------------------
    # Optional W&B setup (rank0 only)
    # -------------------------
    wb_cfg = cfg.get("wandb", {})
    use_wandb = bool(wb_cfg.get("project"))

    wandb = None
    if use_wandb and is_main:
        import wandb as _wandb

        wandb = _wandb

        # Only set env vars that are valid + provided
        env_map = {
            "WANDB_PROJECT": wb_cfg.get("project"),
            "WANDB_ENTITY": wb_cfg.get("entity"),  # optional
            "WANDB_DIR": wb_cfg.get("dir"),
            "WANDB_CACHE_DIR": wb_cfg.get("cache_dir"),
            "WANDB_CONFIG_DIR": wb_cfg.get("config_dir"),
            "WANDB_DATA_DIR": wb_cfg.get("data_dir"),
            "WANDB_ARTIFACT_DIR": wb_cfg.get("artifact_dir"),
            "WANDB_NOTEBOOK_NAME": wb_cfg.get("notebook_name"),
        }

        # WANDB_LOG_MODEL: newer wandb is picky; accept only common values
        # Good values: "false", "end", "checkpoint"
        log_model = wb_cfg.get("log_model")
        if isinstance(log_model, str):
            lm = log_model.strip().lower()
            if lm in {"false", "end", "checkpoint"}:
                env_map["WANDB_LOG_MODEL"] = lm
            else:
                if is_main:
                    logger.warning(
                        "Ignoring wandb.log_model=%r (unsupported). Use one of: false/end/checkpoint.",
                        log_model,
                    )

        for k, v in env_map.items():
            if v:
                os.environ[k] = str(v)
                logger.info("  → set %s=%s", k, v)

        # Login
        if wb_cfg.get("api_key"):
            wandb.login(key=wb_cfg["api_key"])
            logger.info("Authenticated to W&B with API key")
        else:
            wandb.login()
            logger.info("Authenticated to W&B via default method")

    if not use_wandb and is_main:
        logger.info("W&B disabled (no wandb.project in config)")

    # -------------------------
    # Model & tokenizer
    # -------------------------
    model_id = cfg["model"]["model_id"]
    tokenizer_id = cfg["model"].get("tokenizer_id", model_id)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    # For CausalLM SFT, right padding is typically safer, but keep your original choice if desired.
    # Gemma tokenizers commonly use eos as pad.
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.bfloat16,
    )
    if is_main:
        logger.info("Loaded model & tokenizer from %s", model_id)

    # If you are doing text-only SFT, freeze vision tower to save VRAM
    if hasattr(model, "model") and hasattr(model.model, "vision_tower"):
        for p in model.model.vision_tower.parameters():
            p.requires_grad = False
    # -------------------------
    # Dataset
    # -------------------------
    dd = cfg["data"]

    hf_dataset = dd.get("hf_dataset")
    if hf_dataset:
        split = dd.get("hf_split", "train")
        revision = dd.get("hf_revision")
        if is_main:
            logger.info(
                "Loading dataset %s (split=%s) from HuggingFace Hub", hf_dataset, split
            )
        ds = load_dataset(hf_dataset, split=split, revision=revision)
    else:
        if is_main:
            logger.info("Loading dataset from disk: %s", dd["dataset_path"])
        ds = load_from_disk(dd["dataset_path"])

    ds = ds.train_test_split(test_size=dd["test_size"], seed=dd["seed"])

    # Convert to chat messages format
    remove_cols = dd.get("remove_columns", [])
    train_ds = ds["train"].map(
        format_fn,
        remove_columns=remove_cols,
        batched=True,
        desc="Formatting train dataset",
    )
    eval_ds = ds["test"].map(
        format_fn,
        remove_columns=remove_cols,
        batched=True,
        desc="Formatting eval dataset",
    )

    if is_main:
        logger.info("Prepared %d train / %d eval examples", len(train_ds), len(eval_ds))

    # -------------------------
    # Trainer config
    # -------------------------
    trainer_cfg = cfg["trainer"].copy()

    # IMPORTANT:
    # transformers integration utils no longer accept None here.
    # If wandb is disabled, set report_to to "none".
    if not use_wandb:
        trainer_cfg["report_to"] = "none"
    else:
        # ensure it's wandb (or list) so it works across versions
        rt = trainer_cfg.get("report_to", "wandb")
        if rt is None:
            trainer_cfg["report_to"] = "wandb"

    sft_args = SFTConfig(**trainer_cfg)

    # -------------------------
    # Gemma-3 fix: token_type_ids required during training
    # -------------------------
    # TRL's SFTTrainer will tokenize the 'messages' using the tokenizer / chat template.
    # We inject token_type_ids via a custom data collator hook by overriding the trainer's
    # default collator behavior: easiest is to set tokenizer.model_input_names to include it,
    # but safer is to post-process batches in a custom collator.
    #
    # We'll use the trainer's default data collator but add token_type_ids on the fly.

    base_collator = None
    try:
        # TRL sets its own collator internally if not provided; we can still provide one.
        from transformers import DataCollatorForLanguageModeling

        base_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    except Exception:
        base_collator = None

    def collate_with_token_type_ids(features):
        # Let HF collator build input_ids/attention_mask/labels
        if base_collator is not None:
            batch = base_collator(features)
        else:
            # fallback: rely on default torch stacking; should rarely happen

            batch = {k: torch.tensor([f[k] for f in features]) for k in features[0].keys()}

        # Add token_type_ids as zeros matching input_ids shape
        if "input_ids" in batch and "token_type_ids" not in batch:
            batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
        return batch

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=collate_with_token_type_ids,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)],
    )

    # -------------------------
    # Resume support
    # -------------------------
    ckpt_to_resume = None
    if args.resume:
        output_dir = trainer.args.output_dir
        last_ckpt = get_last_checkpoint(output_dir)
        if last_ckpt:
            ckpt_to_resume = last_ckpt
            if is_main:
                logger.info("Resuming from checkpoint %s", last_ckpt)
        else:
            if is_main:
                logger.warning("No checkpoint found in %s; starting from scratch", output_dir)

    if is_main:
        logger.info("Starting training for %s epochs", sft_args.num_train_epochs)
    trainer.train(resume_from_checkpoint=ckpt_to_resume)

    # -------------------------
    # Save final model/tokenizer (main process only)
    # -------------------------
    out_dir = dd["output_dir"]
    if is_main:
        os.makedirs(out_dir, exist_ok=True)
        trainer.model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        logger.info("Model & tokenizer saved to %s", out_dir)

    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()

    if is_main:
        logger.info("Done.")

    if use_wandb and wandb is not None and is_main:
        wandb.finish()


if __name__ == "__main__":
    main()
