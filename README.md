# ReMIND: Orchestrating Modular LLMs for Controllable Serendipity

**ReMIND (REM-inspired Modular Ideation Network for Discovery)** is a modular framework for serendipitous idea generation using Large Language Models (LLMs), inspired by the functional roles of wake-like stability and dream-like exploration observed in REM sleep.
version 2

> **Note:** The original codebase (v1) is archived in [`ReMINDv1/`](./ReMINDv1/).

## Concept Overview

Creative ideation in LLMs often faces a trade-off between **exploration** (novelty) and **stabilization** (coherence). ReMIND addresses this by separating the process into independent computational stages, each implemented as a distinct LLM module:

1. **Wake**: Produces a low-temperature, high-consistency semantic baseline.
2. **Dream**: Performs high-temperature stochastic generation to explore unconventional conceptual combinations.
3. **Judge**: An independent module that filters outputs and extracts salient novel ideas (`idea_dream`).
4. **Re-wake**: Re-articulates selected ideas into a coherent final output.

Each module can be assigned a **different LLM**, enabling role-specialised configurations (e.g., a high-entropy model for Dream, a structured model for Re-wake).

## Repository Structure

```
ReMIND/
├── remind-run/            # Core execution pipeline
│   ├── remind.py          # Core Wake–Dream–Judge–Re-wake logic
│   ├── sweep.py           # Full parameter sweep (3-letter LLM codes)
│   ├── sweep_dream.py     # Dream-only sweep for temperature sensitivity analysis
│   ├── wake.py            # Wake module
│   ├── dream.py           # Dream module
│   ├── judge.py           # Judge module (JSON-structured evaluation)
│   ├── judge_one.py       # Single-run judge utility
│   ├── prompting.py       # Prompt templates and post-processing
│   ├── config.py          # Model endpoints and LLM code mappings
│   ├── count_ideas.py     # Count successfully generated outputs
│   └── ensure_tmux.py     # Tmux session management for local LLM servers
│
├── remind-analysis/       # Post-hoc quantitative analysis
│   ├── review.py          # External LLM reviewer (wake / dream / rewake)
│   ├── review_violin.py   # Violin plots of review scores across pipeline stages
│   ├── review_similarity.py  # Merge similarity data with review scores
│   ├── similarity.py      # Cosine similarity between wake_out and dream_out
│   ├── similarity_his.py  # Similarity distribution and histogram analysis
│   ├── similarity_violin.py  # Similarity vs. temperature violin plots
│   ├── select_top.py      # Select top rewake candidates by novelty score
│   ├── evaluate.py        # Relative ranking of top-10 rewake outputs
│   ├── cliff_ranking.py   # Cliff's δ ranking across all conditions and themes
│   ├── convert_review.py  # Format conversion for review JSONL files
│   ├── count_ideas.py     # Count valid outputs per condition
│   ├── extract_low_judgedream.py  # Extract low-scored judge-dream outputs
│   ├── judge_check.py     # Validate judge output structure
│   └── monitor_sweep.py   # Monitor ongoing sweep progress
│
└── ReMINDv1/              # Archived v1 codebase
```

## Core Execution (`remind-run`)

### LLM Role Codes

LLM assignments are specified as a **3-letter code** representing Wake / Dream / Judge roles:

| Code | Model |
|------|-------|
| `o`  | GPT-OSS-120B |
| `g`  | Gemma-4-31B |
| `q`  | Qwen3.5-27B |
| `n`  | Nemo-30B |

Example: `qnq` = Qwen3.5-27B (Wake) + Nemo-30B (Dream) + Qwen3.5-27B (Judge).

Homogeneous configurations (e.g., `ooo`, `qqq`) use a single model throughout the pipeline. Heterogeneous configurations (e.g., `qnq`, `ogg`) mix models across roles.

### Running a Sweep

```bash
# Single condition
python sweep.py --llm ogo

# Batch sweep (shared timestamp)
python sweep.py --batch ooo ogo ogg qnq

# Sanity test (3 runs per condition)
python sweep.py --batch ooo ogo --test
```

Output: `logs/remind_sweep_<llmcode>_<timestamp>_<topic>.jsonl`

### Output Format (JSONL)

Each line is a JSON record:

```
run_id
sweep        – concept pair and key control parameters
meta         – run metadata and status
result
├── task     – pair, template_id, word_limit, prompt
├── params   – models, temperatures, seeds, token limits
├── wake     – wake_out, judgewake (score, idea_wake)
├── dream    – dream_out, judgedream (score, idea_dream)
└── rewake   – rewake_out, rewake_skipped_reason
```

## Analysis Pipeline (`remind-analysis`)

Scripts are designed to be run sequentially after a sweep completes.

### 1. External Review

`review.py` uses an external LLM (e.g., GPT-5.2) to score outputs on **Alignment**, **Coherence**, and **Novelty** (1–5 scale). All three pipeline stages are reviewed independently:

```bash
python review.py logs/remind_sweep_1ooo_*.jsonl --all --model gpt-5.2
```

Outputs three JSONL files per condition:
- `logs/remind_review_<cond>_core_wake_gpt5.2.jsonl`
- `logs/remind_review_<cond>_core_dream_gpt5.2.jsonl`
- `logs/remind_review_<cond>_core_rewake_gpt5.2.jsonl`

### 2. Cosine Similarity

`similarity.py` computes cosine similarity between `wake_out` and `dream_out` using sentence embeddings, quantifying semantic displacement during the Dream phase.

```bash
python similarity.py logs/remind_sweep_1ooo_*.jsonl
```

### 3. Violin Plots

`review_violin.py` compares review scores (novelty, coherence, alignment) across Wake / Dream / Re-wake for a single condition, and reports Mann–Whitney U statistics with Cliff's δ.

```bash
python review_violin.py logs/remind_review_1ooo_core_rewake_gpt5.2.jsonl
```

### 4. Top Candidate Selection and Relative Evaluation

`select_top.py` filters rewake outputs by novelty threshold per theme. `evaluate.py` then ranks the top-10 candidates using relative pairwise evaluation.

```bash
python select_top.py logs/remind_review_1ooo_core_rewake_gpt5.2.jsonl
python evaluate.py 1ooo_time_space.csv --model gpt-5.2
```

### 5. Cliff's δ Ranking (Main Analysis)

`cliff_ranking.py` computes Cliff's δ across all 19 conditions and 3 themes:

- **Wake → Dream** (Mann–Whitney U, unmatched): quantifies novelty gain during the Dream phase.
- **Dream → Re-wake** (Wilcoxon signed-rank, matched by `run_id`): quantifies coherent elaboration during the Re-wake phase.

Sign convention: δ < 0 indicates improvement (the target phase scores higher than the source phase).

```bash
python cliff_ranking.py                          # wake–rewake full analysis
python cliff_ranking.py --stage wake-dream
python cliff_ranking.py --stage dream-rewake
python cliff_ranking.py --theme time_space       # single theme
```

## Experimental Design

The paper evaluates **19 LLM configurations** across **3 prompt themes**:

| Theme | Description |
|-------|-------------|
| T&S   | Time & Space (core scientific concepts) |
| PT    | Philosophical-Technical |
| AC    | Analytical-Creative |

Configurations: 4 homogeneous (`ooo`, `qqq`, `ggg`, `nnn`) + 15 heterogeneous combinations.

## Citation

If you use this framework in your research, please cite:

Sato, M. (2026).
**ReMIND: Orchestrating Modular Large Language Models for Controllable Serendipity —
A REM-Inspired System Design for Emergent Creative Ideation**
[arXiv:2601.07121](https://arxiv.org/abs/2601.07121)
