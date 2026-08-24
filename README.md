# ReMIND: Orchestrating Modular LLMs for Controllable Serendipity

**ReMIND (REM-inspired Modular Ideation Network for Discovery)** is a modular framework for serendipitous idea generation orchestrating three functionally distinct LLM modules.
version 3

> **Note:** Earlier codebases are archived in [`ReMINDv1/`](./ReMINDv1/) (version 1) and [`ReMINDv2/`](./ReMINDv2/) (version 2).

> The manuscript currently in preparation (v4) uses this same version-3 codebase unchanged — v4 is a text-only revision of the paper, not a code update.

## Overview

ReMIND comprises three LLM modules — **Wake**, **Dream**, and **Judge** — that together constitute the following four-phase pipeline:

1. **Wake**: Produces a low-temperature baseline response to the input prompt.
2. **Dream**: Performs high-temperature stochastic generation, producing exploratory outputs that deviate from the baseline.
3. **Judge**: Evaluates the wake and dream outputs and extracts the most salient novel concept as a single sentence.
4. **Re-wake**: The wake module is re-used to re-articulate the selected idea into a coherent final output.

Each module can be assigned a **different LLM**, enabling role-specialised configurations.

## Repository Structure

```
ReMIND/
├── remind-run/               # Core execution pipeline
│   ├── remind.py             # Core Wake–Dream–Judge–Re-wake logic
│   ├── sweep.py               # Full parameter sweep (3-letter LLM codes)
│   ├── sweep_dream.py         # Dream-only sweep for temperature sensitivity analysis
│   ├── sweep_rewake_wake.py   # Augments an existing sweep log with a Wake-path Re-wake (idea_wake → rewakeout_wake), for isolating the Dream module's contribution
│   ├── wake.py                # Wake module
│   ├── dream.py               # Dream module
│   ├── judge.py               # Judge module (JSON-structured evaluation)
│   ├── judge_one.py           # Single-run judge utility
│   ├── prompting.py           # Prompt templates and post-processing
│   ├── config.py              # Model endpoints and LLM code mappings
│   ├── count_ideas.py         # Count successfully generated outputs
│   └── ensure_tmux.py         # Tmux session management for local LLM servers
│
├── remind-analysis/          # Post-hoc quantitative analysis
│   ├── review.py                        # External LLM reviewer (wake / dream / rewake / rewake_wake)
│   ├── review_violin.py                 # Violin plots of review scores across pipeline stages
│   ├── review_violin_rewake_wake.py     # Violin plots comparing Dream-path vs. Wake-path Re-wake
│   ├── review_similarity.py             # Merge similarity data with review scores
│   ├── similarity.py                    # Cosine similarity between wake_out and dream_out
│   ├── similarity_his.py                # Similarity distribution and histogram analysis
│   ├── similarity_violin.py             # Similarity vs. temperature violin plots
│   ├── select_top.py                    # Select top rewake candidates by novelty score
│   ├── evaluate.py                      # Relative ranking of top-10 rewake outputs
│   ├── cliff_ranking.py                 # Cliff's δ ranking across all conditions and themes
│   ├── convert_review.py                # Format conversion for review JSONL files
│   ├── count_ideas.py                   # Count valid outputs per condition
│   ├── extract_low_judgedream.py        # Extract low-scored judge-dream outputs
│   ├── judge_check.py                   # Validate judge output structure
│   └── monitor_sweep.py                 # Monitor ongoing sweep progress
│
├── ReMINDv1/                  # Archived v1 codebase
└── ReMINDv2/                  # Archived v2 codebase
```

## Core Execution (`remind-run`)

### LLM Role Codes

LLM assignments are specified as a **3-letter code** representing Wake / Dream / Judge roles:

| Code | Model |
|------|-------|
| `o`  | gpt-oss120b |
| `q`  | qwen3.5_27b |
| `g`  | gemma4_31b |
| `n`  | nemo3_30b |

Example: `qno` = qwen3.5_27b (Wake) + nemo3_30b (Dream) + gpt-oss120b (Judge).

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

### Wake-path Re-wake (`sweep_rewake_wake.py`)

To isolate the Dream module's contribution, `sweep_rewake_wake.py` augments an existing sweep log with a second Re-wake pass that re-articulates `idea_wake` directly (bypassing Dream/Judge). Only the Wake LLM server needs to be running:

```bash
python sweep_rewake_wake.py --in logs/remind_sweep_qoq_TIMESTAMP_core.jsonl
```

This writes a new file with `rewakeout_wake` added to each record, without modifying the input.

## Analysis Pipeline (`remind-analysis`)

### 1. External Review

`review.py` uses an external LLM (e.g., GPT-5.2) to score outputs on **Alignment**, **Coherence**, and **Novelty** (1–5 scale). All pipeline stages are reviewed independently, including the wake-path re-wake:

```bash
python review.py logs/remind_sweep_1ooo_*.jsonl --all --model gpt-5.2
python review.py logs/remind_sweep_1ooo_*_core_rewake_wake.jsonl --rewake-wake --model gpt-5.2
```

Outputs one JSONL file per stage per condition, e.g.:
- `logs/remind_review_<cond>_core_wake_gpt5.2.jsonl`
- `logs/remind_review_<cond>_core_dream_gpt5.2.jsonl`
- `logs/remind_review_<cond>_core_rewake_gpt5.2.jsonl`
- `logs/remind_review_<cond>_core_rewake_wake_gpt5.2.jsonl`

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

`review_violin_rewake_wake.py` runs the same comparison between the Dream-path Re-wake and the Wake-path Re-wake, isolating what the Dream module contributes on top of Wake alone:

```bash
python review_violin_rewake_wake.py logs/remind_review_1ooo_core_rewake_gpt5.2.jsonl
```

### 4. Top Candidate Selection and Relative Evaluation

`select_top.py` filters rewake outputs by novelty threshold per theme. `evaluate.py` then ranks the top-10 candidates using relative pairwise evaluation.

```bash
python select_top.py logs/remind_review_1ooo_core_rewake_gpt5.2.jsonl
python evaluate.py 1ooo_time_space.csv --model gpt-5.2
```

### 5. Cliff's δ Ranking (Main Analysis)

`cliff_ranking.py` computes Cliff's δ across all 19 conditions and 3 themes, for any of **novelty**, **alignment**, **coherence**, or their **sum**:

- **Wake → Dream** (Mann–Whitney U, unmatched): quantifies novelty gain during the Dream phase.
- **Dream → Re-wake** (Wilcoxon signed-rank, matched by `run_id`): quantifies coherent elaboration during the Re-wake phase.
- **Re-wake → Wake-path Re-wake** (Wilcoxon signed-rank, matched): quantifies the Dream module's net contribution.

Sign convention: δ > 0 indicates improvement (the target phase scores higher than the source phase),
matching the sign reported in the manuscript text/tables.

```bash
python cliff_ranking.py                                   # wake–rewake full analysis
python cliff_ranking.py --stage wake-dream
python cliff_ranking.py --stage dream-rewake
python cliff_ranking.py --stage rewake-rewake_wake
python cliff_ranking.py --metric alignment                # or coherence, sum_score
python cliff_ranking.py --theme time_space                # single theme
```

## Experimental Design

The paper evaluates **19 LLM configurations** across **3 prompt themes**:

| Theme | Description |
|-------|-------------|
| T&S   | Time & Space  |
| PT    | Periodic Tarot |
| AC    | Aperiodic Craft |

Configurations: 4 homogeneous (`ooo`, `qqq`, `ggg`, `nnn`) + 15 heterogeneous combinations.

## Citation

Sato, M. (2026).
**ReMIND: Orchestrating Modular Large Language Models for Controllable Serendipity —
A REM-Inspired System Design for Emergent Creative Ideation**
[arXiv:2601.07121](https://arxiv.org/abs/2601.07121)
