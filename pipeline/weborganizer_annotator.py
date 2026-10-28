"""WebOrganizer topic + format classifiers (Wettig et al., 2025).

The -NoURL variants, because Comma records have no URL. Both are
gte-base-en-v1.5 fine-tunes and share a tokenizer, so each batch is tokenized
once and run through both. Settings follow WebOrganizer's own
annotate_data/domains.py: truncation at 8192 tokens, bf16.

Output per document: {topic, topic_prob, topic_probs[24], format, format_prob, format_probs[24]}
"""

from __future__ import annotations

from loguru import logger

MODELS = {"topic": "WebOrganizer/TopicClassifier-NoURL", "format": "WebOrganizer/FormatClassifier-NoURL"}


def add_args(ap) -> None:
    ap.add_argument("--batch-size", type=int, default=1024, help="docs per checkpoint")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--max-batch-tokens", type=int, default=32768,
                    help="padded tokens per forward pass; lower it on out-of-memory")
    ap.add_argument("--xformers", action="store_true",
                    help="unpadding + memory-efficient attention (faster; needs xformers installed)")


def load(args):
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # The gte remote code computes position ids / rotary tables at construction;
    # transformers 5 leaves them uninitialised (IndexError on CPU, device-side
    # assert on GPU). SGLang needs transformers 5, so WebOrganizer gets its own
    # venv: VENV=.venv-weborganizer (see slurm_startup.sh).
    if int(transformers.__version__.split(".")[0]) >= 5:
        raise RuntimeError(f"WebOrganizer needs transformers<5, found {transformers.__version__}; "
                           "use the WebOrganizer venv (VENV=.venv-weborganizer, see README)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    # trust_remote_code also here: the config references the gte custom code, and
    # without it transformers asks interactively (fatal in a batch job).
    tok = AutoTokenizer.from_pretrained(MODELS["topic"], trust_remote_code=True)
    models, labels = {}, {}
    for name, repo in MODELS.items():
        m = AutoModelForSequenceClassification.from_pretrained(
            repo, trust_remote_code=True,
            unpad_inputs=args.xformers, use_memory_efficient_attention=args.xformers)
        models[name] = m.to(device, dtype).eval()
        labels[name] = [m.config.id2label[i] for i in range(m.config.num_labels)]
    logger.info(f"loaded {list(MODELS.values())} on {device}")

    @torch.inference_mode()
    def annotate(texts: list[str]) -> list[dict]:
        ids = tok(texts, truncation=True, max_length=args.max_length)["input_ids"]
        order = sorted(range(len(texts)), key=lambda i: len(ids[i]))
        out: list[dict] = [{} for _ in texts]

        # Sub-batches of at most --max-batch-tokens padded tokens.
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and (end + 1 - start) * len(ids[order[end]]) <= args.max_batch_tokens:
                end += 1
            chunk = order[start:end]
            batch = tok.pad({"input_ids": [ids[i] for i in chunk]}, return_tensors="pt").to(device)
            for name, model in models.items():
                probs = model(**batch).logits.float().softmax(-1).cpu()
                for i, p in zip(chunk, probs):
                    j = int(p.argmax())
                    out[i][name] = labels[name][j]
                    out[i][f"{name}_prob"] = round(float(p[j]), 4)
                    out[i][f"{name}_probs"] = [round(x, 4) for x in p.tolist()]
            start = end
        return out

    return annotate
