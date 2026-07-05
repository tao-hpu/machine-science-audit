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
through a local `.env` file; see each script's header for the expected variables.
Restricted source corpora are not redistributed in this repository — the scripts
fetch them from their original venues under the appropriate licenses.

## License

TBD.
