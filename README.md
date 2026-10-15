# Attribution benchmark — corpus annotation

Stage 1 of the attribution benchmark (P4 in the midway evaluation): annotate
every document of the [Comma v0.1 training dataset](https://huggingface.co/datasets/common-pile/comma_v0.1_training_dataset)
with silver labels that are later used to pre-filter candidates.

| Annotator | Model | Output per document |
|---|---|---|
| `weborganizer` | `WebOrganizer/{Topic,Format}Classifier-NoURL` (gte-base, 140M) | topic + format label, top probability, full distribution |
| `propella` | `ellamind/propella-1-4b` | 18 properties incl. one-sentence description, `ok` flag |

## Design

- **Unit of work = one dataset file ("shard")**, named by its repo path, e.g. `cccc/cccc.chunk.00.jsonl.gz`
  (~2,000 in total, mostly 64 per source). A document is identified by
  `(shard, row)`, its line number in that file. The dataset revision is pinned
  in `pipeline/common.py`, so rows never shift.
- **Order:** web-like sources first (`PRIORITY` in `pipeline/common.py`). Sources
  that are not listed are only processed when named with `--sources`.
- **One process = one GPU = one annotator.** Each process walks the shard list
  and takes the next shard that is neither annotated nor locked. Several
  processes on one machine (or on the cluster's shared filesystem) stay out of
  each other's way through lock files. A GPU can be freed at any time by
  killing its process.
- **Checkpointing:** annotations are appended to `<shard>.partial.jsonl` after
  every batch. A killed process resumes there. When every row is done, the
  file becomes `<shard>.parquet` and the shard is logged as `annotated`.
- **Progress log = `progress.tsv`** (in git). It is append-only, with one line
  per event (`downloaded`, `annotated`) and host, doc count, failures and
  duration. It doubles as the overview: logs from different machines are
  merged by concatenating them. Git does this automatically (`merge=union`),
  or you can paste the lines over by hand. Shards in the log are skipped
  everywhere.

```
progress.tsv  ──► which shards are done (all machines, once merged)
data/
  raw/<source>/<chunk>.jsonl.gz                     downloaded dataset files (kept)
  annotations/<annotator>/<source>/<chunk>.parquet  finished shards: row + labels
  annotations/<annotator>/<source>/<chunk>.partial.jsonl   in progress
  locks/<annotator>/<source>/<chunk>.lock           shard being worked on (host, pid)
  logs/                                             sglang server logs
```

## Layout

```
pipeline/
  common.py                  shard list, download, progress log, locks
  run.py                     main loop: python -m pipeline.run <annotator> [...]
  weborganizer_annotator.py  load(args) -> annotate(texts) -> list[dict]
  propella_annotator.py      same interface; transformers or sglang backend
  status.py                  overview table: python -m pipeline.status
slurm/annotate.sbatch        one GPU per array task
progress.tsv                 the log (created on first download)
```

Adding another annotator means adding a module with `add_args(ap)` and
`load(args) -> annotate(texts)`, then registering it in `run.py`'s `ANNOTATORS`.

## Setup

```bash
uv sync                      # transformers-only environment
uv pip install "sglang[all]" # GPU machines only: SGLang for fast propella (pins its own torch)
huggingface-cli login        # or export HF_TOKEN
```

Data goes to `./data` unless `ATTRIB_DATA` points elsewhere (e.g. a scratch
filesystem on the cluster).

## Running

Prototype (one small shard, a few documents, stays partial):

```bash
python -m pipeline.run weborganizer --sources news --max-shards 1 --limit-docs 100
python -m pipeline.run propella --sources news --max-shards 1 --limit-docs 20
```

The cluster and cir don't share locks, so they split the shard list with
`--part`: e.g. the cluster takes `0,1/3` (two thirds, it has more GPUs), cir
takes `2/3`. Both still
follow the priority order. When one side runs out of work, sync
`progress.tsv` and restart it without `--part` to pick up what is left.

cir (external 4x A100): four independent processes, one per GPU. A100s don't
support fp8, so `--fp8` is left off. SGLang's default flashinfer kernels fail
to build on cir, so it uses the Triton attention backend and PyTorch sampling.
Only start processes on GPUs that `nvidia-smi` shows as free.

```bash
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g nohup .venv/bin/python -m pipeline.run propella --part 2/3 --backend sglang \
    --sglang-args "--attention-backend triton --sampling-backend pytorch" \
    > logs/propella-gpu$g.log 2>&1 &
  echo $! > logs/propella-gpu$g.pid
done
kill $(cat logs/propella-gpu2.pid)   # free GPU 2; its shard resumes later
```

SLURM (cn11: 2x H100, cn19: 4x L40S, both fp8-capable; jobs share locks via
the home directory). `slurm_startup.sh` loads `GCC/13.3.0 CUDA/12.6.0`, which SGLang needs
(`CUDA_HOME`, `nvcc`, a C++20 compiler). In an interactive `srun` session, run
`source slurm_startup.sh` first (the sbatch script does this itself).

```bash
sbatch -w cn11 --array=0-1 slurm/annotate.sbatch propella --part 0,1/3 --backend sglang --fp8
sbatch -w cn19 --array=0-3 slurm/annotate.sbatch propella --part 0,1/3 --backend sglang --fp8
sbatch slurm/annotate.sbatch weborganizer   # only the cluster runs weborganizer, so no --part
```

Overview:

```bash
python -m pipeline.status
```

## Notes

- WebOrganizer settings follow its own `annotate_data/domains.py`: 8192-token
  truncation, bf16. `--xformers` enables the faster unpadded attention if
  xformers is installed.
- propella truncates at 50k characters, as recommended on the model card. Its
  prompts and schema are loaded from the model repo's `propella.py`. With the
  transformers backend, invalid JSON is stored as `ok=False`. Under SGLang the
  schema is enforced.
- Raw files are kept after annotation, so the second annotator doesn't
  download them again. Delete `data/raw/` when disk space is tight.
- Machines without a shared filesystem coordinate only through `--part` and a
  manually synced `progress.tsv` (`git pull && git add progress.tsv && git
  commit -m progress && git push`). The workers never commit themselves.
