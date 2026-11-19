#!/usr/bin/env python3
"""Select benchmark prompts from WildChat: information-seeking (factoid / analytical)
and creative requests.

    python -m queries.wildchat                       # all steps; finished steps are skipped
    python -m queries.wildchat --stop-after filter   # CPU part only
    python -m queries.wildchat --per-stratum 500 --min-prob 0.8 --min-openness-prob 0.7

Three steps, each with its own output in data/queries/wildchat/ (rerun one
with --redo <step>):

  filter    -> candidates.parquet   cheap filters + deduplication (CPU)
     1. the conversation is English, not toxic, not redacted (PII removed)
     2. the first message is from the user and English
     3. 15 <= length <= --max-chars characters (Comma's context is 4,096 tokens)
     4. no URLs; no image-prompt templates (Midjourney etc.); no references to an
        earlier conversation, an attachment or the assistant (CONTEXT_PATTERNS)
     4b. ChatGPT's first reply does not just ask for the input (ASKS_FOR_INPUT:
        a reply under 300 characters with "please provide/share ...", "I'm
        ready", "once you send ..."). This catches setup messages whose
        content only follows in later turns ("for all the following posts I
        give you, give feedback ..."), which the prompt's wording alone does
        not reveal. Tested on one WildChat file: 1.4% of candidates, almost all
        setup messages or questions ChatGPT could not answer without more
        information. The reply's first 200 characters are kept in
        candidates.parquet (chatgpt_reply_start).
     5. deduplication: exact (normalised text), then by the first 80 normalised
        characters, which collapses templates that only differ at the end;
        the oldest conversation is kept
  classify  -> classified.<name>.parquet, one per classifier (GPU if available)
     request   tceron/info-seek-classifier: information seeking, content creation,
               coding, not english, no request
     openness  tceron/open-endedness-classifier: Factoid, Analytical (defined for
               information-seeking requests; run on all candidates, used only for those)
     Both: ModernBERT-large fine-tuned on manually labelled WildSEEK data (ODC-BY;
     Ceron et al., EMNLP 2026), trained on single user turns of <= 256 tokens,
     which is what --max-length defaults to. Label counts per probability
     threshold go to each manifest.
  sample    -> ../prompts.parquet
     three strata (STRATA): info_seeking/factoid, info_seeking/analytical, creative.
     A prompt enters a stratum if its request label has probability >= --min-prob
     and, for the info-seeking strata, its openness label probability >=
     --min-openness-prob. Fixed-seed random sample of --per-stratum per stratum.

Only the first user turn is used: later turns depend on ChatGPT's answers,
which Comma never produced. The classifiers were trained on single user
messages, which is what a first turn is.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from loguru import logger
from tqdm import tqdm

from queries.store import PROMPTS, QDATA, open_append, read_jsonl, write_manifest, write_parquet

DATASET = "allenai/WildChat-4.8M"
# name -> default model; labels are compared case-insensitively
CLASSIFIERS = {"request": "tceron/info-seek-classifier", "openness": "tceron/open-endedness-classifier"}
REQUIRED_LABELS = {"request": {"information seeking", "content creation"}, "openness": {"factoid", "analytical"}}
# Both configs only carry generic names (LABEL_0, ...); the mapping is in each
# model card (huggingface.co/tceron/<model>, checked 2026-11-18). Used only when
# the config has no real names.
LABEL_NAMES = {
    "tceron/info-seek-classifier": ["information seeking", "content creation", "coding", "not english",
                                    "no request"],
    "tceron/open-endedness-classifier": ["Analytical", "Factoid"],
}
# stratum -> (request label, openness label or None)
STRATA = {
    "info_seeking/factoid": ("information seeking", "factoid"),
    "info_seeking/analytical": ("information seeking", "analytical"),
    "creative": ("content creation", None),
}
STEPS = ["filter", "classify", "sample"]

WDIR = QDATA / "wildchat"
CANDIDATES = WDIR / "candidates.parquet"


def classified_path(name: str):
    return WDIR / f"classified.{name}.parquet"

URL = re.compile(r"https?://|www\.\w", re.I)
IMAGE_TEMPLATES = re.compile(r"midjourney|stable diffusion|dall-?e|image prompt", re.I)
CONTEXT_PATTERNS = re.compile(
    r"\b(as an ai|you (said|wrote|mentioned)|your (previous|last) (answer|response|message)"
    r"|(the|this|that) (previous|above) (text|answer|response|message|conversation)"
    r"|continue (the|this|from)|rewrite (it|this|that)|the attached|this (image|picture|photo|screenshot|file|pdf))\b",
    re.I)
# ChatGPT's first reply only asks for the input (applied to replies < 300 chars)
ASKS_FOR_INPUT = re.compile(
    r"\b(please|kindly|go ahead and|feel free to)\s+(provide|share|send|paste|give|post|upload|enter|type|list)\b"
    r"|\bi('m| am) ready\b|\bwhenever you('re| are) ready\b|\bgo ahead\b"
    r"|\b(once|when|after|as soon as) you (provide|share|send|paste|give|post)\b|\bwaiting for (your|the)\b",
    re.I)


def asks_for_input(reply: str) -> bool:
    return len(reply) < 300 and bool(ASKS_FOR_INPUT.search(reply))


def norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", t.lower())).strip()


def sample_key(prompt_id: str, seed: int) -> str:
    return hashlib.sha1(f"{seed}:{prompt_id}".encode()).hexdigest()


def resolve(repo: str, revision: str, repo_type: str) -> str:
    from huggingface_hub import HfApi
    info = HfApi().dataset_info if repo_type == "dataset" else HfApi().model_info
    return info(repo, revision=revision).sha


# ---- step 1: filter ---------------------------------------------------------

def load_files(args) -> tuple[list[str], str]:
    """Local parquet files of the dataset (downloaded on first use) and the revision."""
    if args.files:
        return args.files, "local"
    from huggingface_hub import snapshot_download
    sha = resolve(args.dataset, args.revision, "dataset")
    path = snapshot_download(args.dataset, repo_type="dataset", revision=sha, allow_patterns=["*.parquet"])
    files = sorted(str(p) for p in Path(path).rglob("*.parquet"))
    logger.info(f"{args.dataset}@{sha[:8]}: {len(files)} parquet files")
    return files, sha


def first_turns(files: list[str]):
    """Per batch: the conversation-level fields plus the first message."""
    cols = ["conversation_hash", "model", "timestamp", "language", "toxic", "redacted", "turn", "conversation"]
    for f in tqdm(files, desc="wildchat files", unit="file"):
        pf = pq.ParquetFile(f)
        have = [c for c in cols if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(columns=have, batch_size=50_000):
            conv = batch.column("conversation")
            first = pc.list_element(conv, 0)
            out = {c: batch.column(c) for c in have if c != "conversation"}
            for k in ["role", "content", "language"]:
                out[f"first_{k}"] = pc.struct_field(first, k)
            out["second"] = pc.list_slice(conv, 1, 2)  # ChatGPT's reply ([] if there is none)
            yield pa.table(out)


def step_filter(args) -> None:
    files, revision = load_files(args)
    counts = Counter()
    kept = []
    for t in first_turns(files):
        counts["conversations"] += len(t)
        mask = pc.and_(pc.equal(t["language"], "English"),
                       pc.and_(pc.invert(t["toxic"]), pc.invert(t["redacted"])))
        t = t.filter(pc.fill_null(mask, False))
        counts["1_english_nontoxic_unredacted"] += len(t)
        t = t.filter(pc.fill_null(pc.and_(pc.equal(t["first_role"], "user"),
                                          pc.equal(t["first_language"], "English")), False))
        counts["2_first_turn_user_english"] += len(t)
        for r in t.select(["conversation_hash", "first_content", "model", "timestamp", "turn",
                           "second"]).to_pylist():
            text = (r["first_content"] or "").strip()
            if not args.min_chars <= len(text) <= args.max_chars:
                continue
            counts["3_length_ok"] += 1
            if URL.search(text) or IMAGE_TEMPLATES.search(text) or CONTEXT_PATTERNS.search(text):
                continue
            counts["4_self_contained"] += 1
            reply = ((r["second"] or [{}])[0].get("content") or "").strip()
            if asks_for_input(reply):
                continue
            counts["4b_reply_not_asking_for_input"] += 1
            kept.append({"prompt_id": r["conversation_hash"], "prompt": text, "chatgpt_model": r["model"],
                         "timestamp": str(r["timestamp"]), "n_turns": r["turn"], "n_chars": len(text),
                         "chatgpt_reply_start": reply[:200]})

    # Oldest conversation first, so a prompt's first occurrence is kept.
    kept.sort(key=lambda r: (r["timestamp"], r["prompt_id"]))
    seen_full, seen_prefix, unique = set(), set(), []
    for r in kept:
        n = norm_text(r["prompt"])
        if n in seen_full or n[:80] in seen_prefix:
            continue
        seen_full.add(n)
        seen_prefix.add(n[:80])
        unique.append(r)
    counts["5_deduplicated"] = len(unique)
    write_parquet(unique, CANDIDATES)
    write_manifest(CANDIDATES, dataset=args.dataset, revision=revision, min_chars=args.min_chars,
                   max_chars=args.max_chars, filter_counts=dict(counts))
    for k, v in counts.items():
        logger.info(f"{k:<32}{v:>12,}")
    logger.success(f"{len(unique):,} candidate prompts -> {CANDIDATES}")


# ---- step 2: classify -------------------------------------------------------

def load_classifier(name: str, model_id: str, revision: str | None, args):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision)
    model = model.to(device, dtype).eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    if all(re.fullmatch(r"LABEL_\d+", lab) for lab in labels) and model_id in LABEL_NAMES:
        if len(LABEL_NAMES[model_id]) != len(labels):
            raise SystemExit(f"{model_id} has {len(labels)} labels, LABEL_NAMES lists {len(LABEL_NAMES[model_id])}")
        labels = LABEL_NAMES[model_id]
        logger.info(f"{name}: config has generic label names; using the model card's: {labels}")
    missing = REQUIRED_LABELS[name] - {lab.lower() for lab in labels}
    if missing:
        raise SystemExit(f"{model_id} labels {labels} lack {sorted(missing)}: update STRATA")
    logger.info(f"{name}: loaded {model_id} on {device} ({dtype}), labels {labels}")

    @torch.inference_mode()
    def classify(texts: list[str]) -> list[dict]:
        ids = tok(texts, truncation=True, max_length=args.max_length)["input_ids"]
        order = sorted(range(len(texts)), key=lambda i: len(ids[i]))
        out: list[dict] = [{} for _ in texts]
        start = 0
        while start < len(order):  # sub-batches of at most --max-batch-tokens padded tokens
            end = start + 1
            while end < len(order) and (end + 1 - start) * len(ids[order[end]]) <= args.max_batch_tokens:
                end += 1
            chunk = order[start:end]
            batch = tok.pad({"input_ids": [ids[i] for i in chunk]}, return_tensors="pt").to(device)
            probs = model(**batch).logits.float().softmax(-1).cpu()
            for i, p in zip(chunk, probs):
                j = int(p.argmax())
                out[i] = {"label": labels[j].lower(), "prob": round(float(p[j]), 4),
                          "probs": [round(x, 4) for x in p.tolist()]}
            start = end
        return out

    return classify, labels


def run_classifier(name: str, args) -> None:
    out_path = classified_path(name)
    model_id = getattr(args, f"{name}_model")
    cands = pq.read_table(CANDIDATES, columns=["prompt_id", "prompt"]).to_pylist()
    partial = out_path.with_suffix(".partial.jsonl")
    done = {r["prompt_id"] for r in read_jsonl(partial)}
    todo = [c for c in cands if c["prompt_id"] not in done]
    logger.info(f"{name}: {len(cands):,} candidates, {len(done):,} classified, {len(todo):,} to go")
    revision = None
    try:
        revision = resolve(model_id, "main", "model")
    except Exception as e:  # offline or a local path
        logger.warning(f"{name}: could not resolve the revision of {model_id}: {e}")
    labels = None
    if todo:
        classify, labels = load_classifier(name, model_id, revision, args)
        with open_append(partial) as sink, tqdm(total=len(cands), initial=len(done), desc=name,
                                              unit="prompt") as bar:
            for b in range(0, len(todo), args.batch_size):
                batch = todo[b: b + args.batch_size]
                for c, res in zip(batch, classify([c["prompt"] for c in batch])):
                    sink.write(json.dumps({"prompt_id": c["prompt_id"], **res}) + "\n")
                sink.flush()
                bar.update(len(batch))
        del classify  # free the GPU before the next classifier

    rows = {r["prompt_id"]: r for r in read_jsonl(partial)}
    if len(rows) < len(cands):
        logger.info(f"{name}: classification stays partial")
        return
    out = [rows[c["prompt_id"]] for c in cands]
    write_parquet(out, out_path)
    # Label counts per threshold on the top-label probability, to choose the cutoffs.
    dist = {lab: {f">={th}": sum(r["label"] == lab and r["prob"] >= th for r in out)
                  for th in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95)}
            for lab in sorted({r["label"] for r in out})}
    write_manifest(out_path, classifier=name, model=model_id, revision=revision, labels=labels,
                   max_length=args.max_length, label_counts=dist)
    partial.unlink()
    for lab, d in dist.items():
        logger.info(f"{name} {lab:<20}" + "  ".join(f"{k} {v:>9,}" for k, v in d.items()))
    logger.success(f"{name}: {len(out):,} classified prompts -> {out_path}")


def step_classify(args) -> None:
    cand_ids = set(pq.read_table(CANDIDATES, columns=["prompt_id"])["prompt_id"].to_pylist())
    for name in args.classifiers:
        path = classified_path(name)
        if path.exists() and not args.redo_classify:
            ids = set(pq.read_table(path, columns=["prompt_id"])["prompt_id"].to_pylist())
            if ids == cand_ids:
                logger.info(f"{name}: {path} exists, skipped")
                continue
            logger.warning(f"{name}: {path} doesn't match the current candidates, classifying again")
            path.with_suffix(".partial.jsonl").unlink(missing_ok=True)
        run_classifier(name, args)


# ---- step 3: sample ---------------------------------------------------------

def step_sample(args) -> None:
    cands = {r["prompt_id"]: r for r in pq.read_table(CANDIDATES).to_pylist()}
    cls = {name: {r["prompt_id"]: r for r in
                  pq.read_table(classified_path(name), columns=["prompt_id", "label", "prob"]).to_pylist()}
           for name in CLASSIFIERS}
    pools: dict[str, list[dict]] = {s: [] for s in args.strata}
    cross = Counter()  # request x openness label, all candidates (for the manifest)
    for pid, c in cands.items():
        req, opn = cls["request"][pid], cls["openness"][pid]
        cross[f"{req['label']} / {opn['label']}"] += 1
        for stratum in args.strata:
            want_req, want_opn = STRATA[stratum]
            if req["label"] != want_req or req["prob"] < args.min_prob:
                continue
            if want_opn and (opn["label"] != want_opn or opn["prob"] < args.min_openness_prob):
                continue
            pools[stratum].append({**c, "stratum": stratum, "task": stratum.split("/")[0],
                                   "task_label": req["label"], "task_prob": req["prob"],
                                   "openness": opn["label"], "openness_prob": opn["prob"]})
    sample = []
    for stratum, rows in pools.items():
        rows.sort(key=lambda r: sample_key(r["prompt_id"], args.seed))
        if len(rows) < args.per_stratum:
            logger.warning(f"{stratum}: only {len(rows)} prompts available (< {args.per_stratum})")
        sample += rows[: args.per_stratum]
    sample.sort(key=lambda r: (r["stratum"], r["prompt_id"]))
    write_parquet(sample, PROMPTS)
    avail = {s: len(r) for s, r in pools.items()}
    taken = dict(Counter(r["stratum"] for r in sample))
    write_manifest(PROMPTS, candidates=str(CANDIDATES), strata={s: STRATA[s] for s in args.strata},
                   min_prob=args.min_prob, min_openness_prob=args.min_openness_prob,
                   per_stratum=args.per_stratum, seed=args.seed, available=avail, sampled=taken,
                   request_x_openness=dict(cross.most_common()))
    logger.success(f"{len(sample)} prompts {taken} (available: {avail}) -> {PROMPTS}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--revision", default="main", help="resolved to a commit sha, stored in the manifest")
    ap.add_argument("--files", nargs="*", help="local parquet files instead of the hub (tests)")
    ap.add_argument("--min-chars", type=int, default=15)
    ap.add_argument("--max-chars", type=int, default=2000, help="~500 tokens, leaves room in Comma's 4k context")
    # classify
    ap.add_argument("--classifiers", default=",".join(CLASSIFIERS), help="which classifiers to run")
    ap.add_argument("--request-model", default=CLASSIFIERS["request"])
    ap.add_argument("--openness-model", default=CLASSIFIERS["openness"])
    ap.add_argument("--batch-size", type=int, default=20_000, help="prompts per checkpoint")
    ap.add_argument("--max-length", type=int, default=256, help="classifier truncation in tokens (as in training)")
    ap.add_argument("--max-batch-tokens", type=int, default=65_536,
                    help="padded tokens per forward pass; lower it on out-of-memory")
    # sample
    ap.add_argument("--strata", default=",".join(STRATA), help=f"comma-separated subset of {list(STRATA)}")
    ap.add_argument("--min-prob", type=float, default=0.8, help="minimum probability of the request label")
    ap.add_argument("--min-openness-prob", type=float, default=0.7,
                    help="minimum probability of the openness label (info-seeking strata)")
    ap.add_argument("--per-stratum", type=int, default=500, help="prompts sampled per stratum")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--redo", choices=STEPS, help="rerun this step (and the ones after it)")
    ap.add_argument("--stop-after", choices=STEPS, default="sample")
    args = ap.parse_args()
    args.strata = args.strata.split(",")
    args.classifiers = args.classifiers.split(",")
    for opt, have, valid in [("--strata", args.strata, STRATA), ("--classifiers", args.classifiers, CLASSIFIERS)]:
        if set(have) - set(valid):
            ap.error(f"{opt}: unknown {sorted(set(have) - set(valid))}; choose from {list(valid)}")

    redo = STEPS[STEPS.index(args.redo):] if args.redo else []
    args.redo_classify = "classify" in redo
    if args.redo_classify:  # stale checkpoints would mix two candidate sets
        for name in args.classifiers:
            classified_path(name).with_suffix(".partial.jsonl").unlink(missing_ok=True)

    if "filter" in redo or not CANDIDATES.exists():
        step_filter(args)
    else:
        logger.info(f"filter: {CANDIDATES} exists, skipped (--redo filter to rerun)")
    if args.stop_after == "filter":
        return
    step_classify(args)
    if not all(classified_path(n).exists() for n in CLASSIFIERS):
        missing = [n for n in CLASSIFIERS if not classified_path(n).exists()]
        logger.info(f"classification incomplete ({missing}); sample needs both classifiers")
        return
    if args.stop_after == "classify":
        return
    step_sample(args)


if __name__ == "__main__":
    main()
