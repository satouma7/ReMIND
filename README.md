# ReMIND: Orchestrating Modular LLMs for Controllable Serendipity

**ReMIND (REM-inspired Modular Ideation Network for Discovery)** is a modular framework for serendipitous idea generation in Large Language Models (LLMs), inspired by the functional roles of wake-like stability and dream-like exploration observed in REM sleep.
This repository provides a unified environment for both execution (`ReMIND-run`) and post-hoc quantitative analysis (`ReMIND-analysis`).
##  Concept Overview
Creative ideation in LLMs often faces a trade-off between **exploration** (novelty) and **stabilization**(coherence)3333. ReMIND addresses this by separating the process into independent computational stages, each implemented as a distinct LLM module:
1. **Wake**: Produces a low-temperature, high-consistency semantic baseline.
2. **Dream**: Performs high-temperature stochastic generation to explore unconventional conceptual combinations.
3. **Judge**: An independent module that filters incoherent outputs and extracts salient novel ideas.
4. **Re-wake**: Re-articulates selected ideas into a coherent final output using the stable wake model.
By treating creative emergence as a **rare-event process** rather than a deterministic optimization task, ReMIND enables the systematic discovery of "emergent singularities"—high-value ideas that resist traditional predictive metrics.

## Repository Structure
```
ReMIND/
├── ReMIND-run/            # Core execution logic (High-performance environment)
│   ├── remind.py          # Main orchestration logic
│   ├── sweep.py           # Parameter sweep runner for large-scale generation
│   ├── wake.py            # Wake module implementation
│   ├── dream.py           # Dream module implementation
│   ├── judge.py           # Judge module (JSON-based evaluation)
│   ├── prompting.py       # Prompt templates and post-processing
│   ├── config.py          # Model endpoints and parameter configurations
│   └── logs/              # Generated JSONL sweep outputs
│
└── ReMIND-analysis/       # Analysis pipeline (Local/Research environment)
    ├── similarity.py      # Computes cosine similarity via sentence embeddings
    ├── similarity_his.py  # Distribution and histogram analysis
    ├── similarity_violin.py # Visualizes exploration controllability by temperature
    ├── review.py          # External LLM-based (GPT-5.2) quality assessment
    ├── review_similarity.py # Merges similarity data with evaluation scores
    ├── plot_scatter.py    # Generates correlation and distribution plots
    └── reports/           # Analysis results and visualizations
```

## Core Execution (`ReMIND-run`)
ReMIND is built on the **BiMoLLM (Brain-Inspired Modular LLM)** paradigm, supporting role specialization across different models10:
- `gemma-27b`: Optimized for high-temperature exploratory generation (Dream).
- `gpt-oss-120b`: Optimized for low-temperature stabilization (Wake) and strict evaluation (Judge).
### Running a Parameter Sweep
To initiate a systematic generation sweep across prompts, temperatures, and seeds:

```
python sweep.py
```

This produces a timestamped JSONL file in the `logs/` directory containing all generation metadata and module outputs.

##  Analysis Pipeline (`ReMIND-analysis`)
The analysis scripts quantify the relationship between semantic exploration and ideational quality.
### 1. Semantic Displacement
`similarity.py` (and the stabilized `similarity2.py`) computes the **cosine similarity** between `idea_wake` and `idea_dream` using sentence embeddings. This quantifies the conceptual distance traveled during the "dream" phase15.
### 2. Controllability and Distribution
- `similarity_his.py`: Compares Wake–Dream similarity against a Wake–Wake negative control.
- `similarity_violin.py`: Demonstrates how dream temperature shifts the semantic distribution.
### 3. External Evaluation
`review.py` utilizes an external reviewer (GPT-5.2) to score `rewake_out` on **Alignment**, **Coherence**, and **Novelty**.
### 4. Correlation and Singularity Detection
`plot_scatter.py` visualizes the relationship between semantic distance and quality scores (Fig. 3). It calculates Spearman’s $\rho$ (correlation) and highlights that high-quality outputs appear sporadically across the similarity axis, confirming that serendipity is a rare-event process.
## Citation
If you use this framework or the BiMoLLM paradigm in your research, please cite our work:
Sato, M. (2026). 
**ReMIND: Orchestrating Modular Large Language Models for Controllable Serendipity**
**A REM-Inspired System Design for Emergent Creative Ideation**
arXiv:2601.XXXXX. 21

