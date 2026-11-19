#!/usr/bin/env python3
"""Instruction-tune Comma v0.1-2T (full fine-tuning, FSDP) on the Tulu 3 SFT mixture.

    # smoke test: a few steps, measures seconds per step
    accelerate launch --config_file sft/fsdp.yaml -m sft.train --run smoke --max-steps 10
    # the real run: N examples, one epoch, stops cleanly after --max-hours
    accelerate launch --config_file sft/fsdp.yaml -m sft.train --run comma2t-tulu3 --num-examples 400000

Data: allenai/tulu-3-sft-mixture (pinned), minus
  - its WildChat subset (tulu_v3.9_wildchat_100k): the benchmark prompts come
    from WildChat, so the model must not have seen them with GPT-4 answers;
  - every example whose first user turn matches a WildChat benchmark candidate
    (data/queries/wildchat/candidates.parquet, same normalisation and
    80-character prefix as the candidate deduplication), whatever its source;
  - tulu_hard_coded_repeated_10 (identity answers: "I am Tulu, made by Ai2").
A seeded random subset of --num-examples is trained for one epoch; the ids
of the used examples are written next to the model (data_ids.txt), so the
SFT data the queries can be attributed to is known exactly.

Format: Comma has no chat template; the Tulu format is used as plain text
(no new tokens), see CHAT_TEMPLATE. Loss only on the assistant turns.
Hyperparameters follow Tulu 3 SFT for 8B models: lr 5e-6, linear decay,
3% warmup, effective batch 128 sequences, max length 4096.

Output: data/sft/<run>/ (checkpoints, final model in HF format, data_ids.txt,
manifest.json). A killed run resumes from its last checkpoint.
"""

from __future__ import annotations

import argparse
import json
import socket
import time

from pipeline.common import DATA

MODEL = "common-pile/comma-v0.1-2t"
MODEL_REVISION = "3fba8938462ae9303c6e88770fc2d098370f0eaa"
DATASET = "allenai/tulu-3-sft-mixture"
EXCLUDE_SOURCES = {"ai2-adapt-dev/tulu_v3.9_wildchat_100k", "ai2-adapt-dev/tulu_hard_coded_repeated_10"}
CANDIDATES = DATA / "queries" / "wildchat" / "candidates.parquet"

# Tulu format as plain text. {% generation %} marks the tokens the loss is computed on.
CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for m in messages %}"
    "{% if m['role'] == 'system' %}<|system|>\n{{ m['content'] }}\n"
    "{% elif m['role'] == 'user' %}<|user|>\n{{ m['content'] }}\n"
    "{% elif m['role'] == 'assistant' %}<|assistant|>\n"
    "{% generation %}{{ m['content'] }}{{ eos_token }}{% endgeneration %}\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


def load_sft_data(args):
    """Tulu 3 minus the exclusions, shuffled, first --num-examples."""
    from datasets import load_dataset
    from huggingface_hub import HfApi

    from queries.wildchat import norm_text

    revision = HfApi().dataset_info(DATASET).sha
    ds = load_dataset(DATASET, split="train", revision=revision)
    n0 = len(ds)
    ds = ds.filter(lambda r: r["source"] not in EXCLUDE_SOURCES, num_proc=8)
    n_sources = len(ds)

    exclude_full, exclude_prefix = set(), set()
    if CANDIDATES.exists():
        import pyarrow.parquet as pq
        for p in pq.read_table(CANDIDATES, columns=["prompt"]).column("prompt").to_pylist():
            n = norm_text(p)
            exclude_full.add(n)
            exclude_prefix.add(n[:80])
    else:
        print(f"WARNING: {CANDIDATES} missing, no prompt-level WildChat decontamination")

    def keep(r):
        msgs = r["messages"]
        if not msgs or any(m["role"] not in ("system", "user", "assistant") for m in msgs):
            return False
        if not any(m["role"] == "assistant" for m in msgs):
            return False
        first_user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        n = norm_text(first_user or "")
        return n not in exclude_full and n[:80] not in exclude_prefix

    ds = ds.filter(keep, num_proc=8)
    n_clean = len(ds)
    ds = ds.shuffle(seed=args.seed)
    if args.num_examples:
        ds = ds.select(range(min(args.num_examples, len(ds))))
    stats = {"dataset": DATASET, "revision": revision, "rows": n0, "after_source_exclusion": n_sources,
             "after_prompt_decontamination": n_clean, "trained_examples": len(ds),
             "excluded_sources": sorted(EXCLUDE_SOURCES), "candidates_for_decontamination": len(exclude_full)}
    return ds.select_columns(["id", "messages"]), stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="output name: data/sft/<run>")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=MODEL_REVISION)
    ap.add_argument("--num-examples", type=int, default=0, help="subset size (0 = all)")
    ap.add_argument("--max-steps", type=int, default=-1, help="smoke test: stop after N optimizer steps")
    ap.add_argument("--max-hours", type=float, default=0, help="save and stop cleanly after this many hours")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--effective-batch", type=int, default=128, help="sequences per optimizer step")
    ap.add_argument("--per-device-batch", type=int, default=1)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    out = DATA / "sft" / args.run
    out.mkdir(parents=True, exist_ok=True)
    world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else \
        int(__import__("os").environ.get("WORLD_SIZE", 1))
    grad_accum = max(1, args.effective_batch // (args.per_device_batch * world))

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    tok.chat_template = CHAT_TEMPLATE
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train, stats = load_sft_data(args)

    class StopAfterHours(TrainerCallback):
        def __init__(self, hours):
            self.deadline = time.time() + hours * 3600 if hours else None

        def on_step_end(self, a, state, control, **kw):
            if self.deadline and time.time() > self.deadline:
                control.should_save = True
                control.should_training_stop = True

    cfg = SFTConfig(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=1,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_steps=0.03,  # float < 1 = share of total steps (transformers 5)
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=grad_accum,
        max_length=args.max_length,
        packing=False,
        assistant_only_loss=True,
        bf16=True,
        optim="adamw_torch_fused",
        weight_decay=0.0,
        logging_steps=5,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=1,  # one sharded checkpoint incl. optimizer is ~84 GB for 7B
        report_to="none",
        seed=args.seed,
        dataset_num_proc=8,
        model_init_kwargs={"dtype": torch.float32, "revision": args.revision},
    )
    trainer = SFTTrainer(model=args.model, args=cfg, train_dataset=train, processing_class=tok,
                         callbacks=[StopAfterHours(args.max_hours)])

    last = None
    ckpts = sorted((out / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if ckpts:
        last = str(ckpts[-1])
        print(f"resuming from {last}")
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=last)

    # Final model as a normal HF checkpoint (full, not sharded), with the chat template.
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    final = out / "final"
    trainer.save_model(str(final))
    if trainer.accelerator.is_main_process:
        tok.save_pretrained(str(final))
        (out / "data_ids.txt").write_text("\n".join(train["id"]) + "\n")
        manifest = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "host": socket.gethostname().strip(),
                    "base_model": args.model, "base_revision": args.revision, "data": stats,
                    "args": vars(args), "world_size": world, "gradient_accumulation_steps": grad_accum,
                    "train_seconds": round(time.time() - t0), "global_step": trainer.state.global_step,
                    "train_loss": result.training_loss, "chat_template": CHAT_TEMPLATE}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        print(f"done: {final}")


if __name__ == "__main__":
    main()
