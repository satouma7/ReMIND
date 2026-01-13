# ReMIND: Orchestrating Modular LLMs for Controllable Serendipity

**ReMIND (REM-inspired Modular Ideation Network for Discovery)** is a modular framework for serendipitous idea generation in Large Language Models (LLMs), inspired by the functional roles of wake-like stability and dream-like exploration observed in REM sleep.
This repository provides a unified environment for both execution (`ReMIND-run`) and post-hoc quantitative analysis (`ReMIND-analysis`).
##  Concept Overview
Creative ideation in LLMs often faces a trade-off between **exploration** (novelty) and **stabilization**(coherence). ReMIND addresses this by separating the process into independent computational stages, each implemented as a distinct LLM module:
1. **Wake**: Produces a low-temperature, high-consistency semantic baseline.
2. **Dream**: Performs high-temperature stochastic generation to explore unconventional conceptual combinations.
3. **Judge**: An independent module that filters incoherent outputs and extracts salient novel ideas.
4. **Re-wake**: Re-articulates selected ideas into a coherent final output using the stable wake model.
## Repository Structure
```
ReMIND/
├── remind-run/            # Core execution logic
│   ├── remind.py          # Main orchestration logic
│   ├── sweep.py           # Parameter sweep runner
│   ├── wake.py            # Wake module implementation
│   ├── dream.py           # Dream module implementation
│   ├── judge.py           # Judge module (JSON-based evaluation)
│   ├── prompting.py       # Prompt templates and post-processing
│   ├── config.py          # Model endpoints and parameter configurations
│   └── logs/              # Generated JSONL sweep outputs
│
└── remind-analysis/       # Analysis pipeline
    ├── similarity.py      # Computes cosine similarity via sentence embeddings
    ├── similarity_his.py  # Distribution and histogram analysis
    ├── similarity_violin.py # Visualizes temperature dependency
    ├── review.py          # External LLM-based quality assessment
    ├── review_similarity.py # Merges similarity data with evaluation scores
    └── plot_scatter.py    # Generates correlation and distribution plots
```

## Core Execution (`ReMIND-run`)
ReMIND is designed to support **LLM role specialization**, for example:
- `gemma-27b`: Optimized for high-temperature exploratory generation (Dream).
- `gpt-oss-120b`: Optimized for low-temperature stabilization (Wake) and strict evaluation (Judge).
### Running a Parameter Sweep
To initiate a systematic parameter sweep across prompts, temperatures, and seeds:

```
python sweep.py
```

This produces a timestamped JSONL file in the `logs/` directory containing all generation metadata and module outputs.

### Output Format of sweep (JSONL)
```
run_id
sweep – sweep parameters (concept pair and key control parameter)
meta – run metadata and status
result
├─ task specification (ideation task definition)
│  ├─ pair
│  ├─ template_id
│  ├─ word_limit
│  └─ prompt
│
├─ params
│  └─ execution parameters (models, temperatures, seeds, token limits)
│
├─ wake phase
│  ├─ wakeout
│  └─ judgewake (score, idea)
│
├─ dream phase
│  ├─ dreamout
│  └─ judgedream (score, idea)
│
├─ extracted ideas
│  ├─ idea_wake
│  └─ idea_dream
│
└─ re-wake phase
   ├─ rewakeout
   └─ rewake_skipped_reason
   ```
   
##  Analysis Pipeline (`ReMIND-analysis`)
The analysis scripts quantify the relationship between semantic exploration and ideational quality.
### 1. Semantic Displacement
`similarity.py`  computes the **cosine similarity** between `idea_wake` and `idea_dream` using sentence embeddings. This quantifies the conceptual distance during the "dream" phase.
### 2. Controllability and Distribution
- `similarity_his.py` compares Wake–Dream similarity against a Wake–Wake negative control, `wake_out` (Fig. 2a).
- `similarity_violin.py`: Demonstrates how dream temperature shifts the semantic distribution (Fig. 2b).
### 3. External Evaluation
`review.py` utilizes an external reviewer (e.g., GPT-5.2) to score `rewakeout` on **Alignment**, **Coherence**, and **Novelty**.
#### Output Format of review (JSONL)
```
run_id
sweep – parameter sweep definition (concept pair and key control parameters)
prompt – final instantiated prompt
rewakeout – final stabilized output (ReMIND result)
meta – run metadata and status
reviews – external evaluation results (per reviewer / model)
└─ <reviewer_name> (e.g., "openai")
   ├─ model
   ├─ alignment
   ├─ coherence
   ├─ novelty
   └─ short_rationale
```
### 4. Correlation and Singularity Detection
`plot_scatter.py` visualizes the relationship between semantic distance and quality scores (Fig. 3). It calculates Spearman’s rho (correlation) and highlights that high-quality outputs appear sporadically across the similarity axis, confirming that serendipity is a rare-event process.
## Citation
If you use this framework in your research, please cite our work:

Sato, M. (2026). 
**ReMIND: Orchestrating Modular Large Language Models for Controllable Serendipity**
**A REM-Inspired System Design for Emergent Creative Ideation**
[arXiv:2601.07121](https://arxiv.org/abs/2601.07121)
