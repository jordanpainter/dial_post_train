#!/usr/bin/env python3
# dpo.py

import argparse
import json
import logging
import os
import sys

import torch
from datasets import load_from_disk
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          EarlyStoppingCallback)
from transformers.trainer_utils import get_last_checkpoint
from trl import DPOConfig, DPOTrainer


def format_fn(examples, remove_cols):
    prompts = examples["prompt"]
    chosen = examples["chosen"]
    rejected = examples["rejected"]
    chosen_message = []
    rejected_message = []
    for p, c, r in zip(prompts, chosen, rejected):
        chosen_message.append(
            [{"role": "user", "content": p}, {"role": "assistant", "content": c}]
        )

        rejected_message.append(
            [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
        )
    return {"chosen": chosen_message, "rejected": rejected_message}


def main():
    p = argparse.ArgumentParser(
        description="DPO finetuning with optional W&B + multi-GPU"
    )
    p.add_argument("--config", "-c", required=True, help="Path to JSON config file")
    p.add_argument(
        "--resume",
        "-r",
        action="store_true",
        help="If set, resume training from the last checkpoint in the output dir",
    )
    args = p.parse_args()

    with open(args.config) as fp:
        cfg = json.load(fp)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    logger.info("Loaded config from %s", args.config)

    # Optional W&B setup
    wb_cfg = cfg.get("wandb", {})
    use_wandb = bool(wb_cfg.get("project"))
    if use_wandb:
        import wandb

        env_map = {
            "WANDB_PROJECT": wb_cfg.get("project"),
            "WANDB_DIR": wb_cfg.get("dir"),
            "WANDB_CACHE_DIR": wb_cfg.get("cache_dir"),
            "WANDB_CONFIG_DIR": wb_cfg.get("config_dir"),
            "WANDB_DATA_DIR": wb_cfg.get("data_dir"),
            "WANDB_ARTIFACT_DIR": wb_cfg.get("artifact_dir"),
            "WANDB_NOTEBOOK_NAME": wb_cfg.get("notebook_name"),
            "WANDB_LOG_MODEL": wb_cfg.get("log_model"),
        }
        for k, v in env_map.items():
            if v:
                os.environ[k] = v
                logger.info(f"  → set {k}={v}")

        if wb_cfg.get("api_key"):
            wandb.login(key=wb_cfg["api_key"])
            logger.info("Authenticated to W&B with API key")
        else:
            wandb.login()
            logger.info("Authenticated to W&B via default method")
    else:
        logger.info("W&B config not found or no project set; skipping W&B")

    model_id = cfg["model"]["model_id"]
    tokenizer_id = cfg["model"].get("tokenizer_id", model_id)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.bfloat16,
    )
    logger.info("Loaded model & tokenizer from %s", model_id)

    dd = cfg["data"]
    ds = load_from_disk(dd["dataset_path"])
    ds = ds.train_test_split(test_size=dd["test_size"], seed=dd["seed"])

    train_ds = ds["train"].map(
        lambda ex: format_fn(ex, dd["remove_columns"]),
        remove_columns=dd["remove_columns"],
        batched=True,
    )
    eval_ds = ds["test"].map(
        lambda ex: format_fn(ex, dd["remove_columns"]),
        remove_columns=dd["remove_columns"],
        batched=True,
    )
    logger.info("Prepared %d train / %d eval examples", len(train_ds), len(eval_ds))

    trainer_cfg = cfg["trainer"].copy()
    if not use_wandb:
        trainer_cfg["report_to"] = None

    dpo_args = DPOConfig(**trainer_cfg)
    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(3, 0.01)],
    )

    # Checking for checkpoint to resume from (if any)
    ckpt_to_resume = None
    if args.resume:
        output_dir = trainer.args.output_dir
        last_ckpt = get_last_checkpoint(output_dir)
        if last_ckpt:
            ckpt_to_resume = last_ckpt
            logger.info("Resuming from checkpoint %s", last_ckpt)
        else:
            logger.warning(
                "No checkpoint found in %s; starting from scratch", output_dir
            )

    logger.info("Starting training for %d epochs", dpo_args.num_train_epochs)
    trainer.train(resume_from_checkpoint=ckpt_to_resume)

    out_dir = dd["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    logger.info("Model & tokenizer saved to %s", out_dir)

    torch.cuda.empty_cache()
    logger.info("Done.")
    wandb.finish()


if __name__ == "__main__":
    main()
