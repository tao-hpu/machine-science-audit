# Machine Science Audit

Code and derived data for a retrieval-grounded novelty audit of machine-generated
research papers: measuring whether the contributions claimed by autonomous AI
research systems represent genuine discovery or recombination of prior literature.

**Status:** work in progress. Full methodology, corpus details, and results will be
released together with the accompanying paper.

## Repository layout

- `scripts/` — the extraction → retrieval → contribution-level judgment pipeline
- `data/extractions/` — extracted contribution records
- `data/neighbors_s2/`, `data/gold_neighbors/`, `data/judgments/` — retrieval and judgment outputs
- `data/fars-index.csv` — corpus index

## Reproduction

The scripts require API credentials (Semantic Scholar, an LLM endpoint) supplied
through a local `.env` file; copy `.env.example` and fill in your own values.
Restricted source corpora are not redistributed in this repository: the scripts
fetch them from their original venues under the appropriate licenses.

Two follow-up analyses run on top of the frozen judgments:

- `scripts/mirage_audit.py --auditor <model> [--variant harsh]` repeats the
  adversarial prior-art audit with a different auditor model. Outputs go to
  `data/mirage_audit[_harsh]_<model>.json`; the committed files cover
  `claude-sonnet-4-6` and `gemini-2.5-pro`.
- `scripts/cr_robustness.py` makes no model calls. It recomputes the
  machine-vs-human gap under claim-type stratification, stricter facet-novel
  derivation rules, and match-similarity filtering, and splits gold prior-art
  misses into search failures and judge failures. Output:
  `data/cr_robustness.json`.

## Retrieval backends: API or local snapshot

Prior-art retrieval supports two interchangeable backends. Pick whichever fits
your budget and hardware; results are equivalent by design.

**1. Live APIs (default, zero setup).** Semantic Scholar (plus OpenAlex as a
second channel) queried directly. Only needs `S2_API_KEY` in `.env`. Simple, but
rate-limited (1 req/s per key), so a full-corpus run takes hours to days.

**2. Local snapshot (faster, reproducible).** Download the official offline
dumps once, build a local BM25 index, and the whole retrieval queue runs in
about a day with no rate limits. A frozen snapshot release also pins the corpus,
which helps reproducibility.

```bash
# 1. Fetch the S2 snapshot (papers + abstracts, ~115 GB; resumable, re-run to continue)
python3 scripts/download_s2_snapshot.py            # --status to check progress

# 2. Build the local index (sqlite abstract KV + tantivy BM25, resumable)
python3 scripts/build_local_index.py

# 3. Optional second channel: OpenAlex works dump (~670 GB; resumable)
OPENALEX_DEST=/path/to/big/disk bash scripts/download_openalex_snapshot.sh
```

Storage paths are configurable in `.env` (`S2_SNAPSHOT_DIR`, `S2_LOCAL_INDEX_DIR`,
`OPENALEX_DEST`), so snapshots can live on an external drive. Query the local
index directly with `scripts/local_search.py "your query" --cutoff 2025-01-01`.

## License

Code in `scripts/` is released under the MIT License (see `LICENSE`). Derived data
under `data/` is released under CC BY 4.0. Restricted source corpora (FARS,
Agents4Science, ICLR 2025 submissions) are **not** redistributed here; the scripts
fetch them from their original venues under those venues' terms.
