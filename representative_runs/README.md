# Representative run outputs

Full intermediate outputs (Wake / Dream / idea_dream / Re-wake) for the representative runs cited in the manuscript's Supplementary Text ("Representative ReMIND outputs", Table S2), for the runs where only the Re-wake output appears in the Supplementary Text itself.

Each run has two files:

- `<run_id>.md` — human-readable walkthrough (prompt, configuration, Wake, Dream, idea_dream, Re-wake), in the same layout as run 1 in the Supplementary Text.
- `<run_id>.json` — the raw pipeline record for that run, as written by `remind-run/sweep.py` (full `params`, `judgewake`/`judgedream` scores, all four stage outputs).

Run IDs follow the paper's `<llm_code><run_id>` convention (see [`../README.md`](../README.md) → LLM Role Codes for the letter mapping).

| Run | Concept pair | LLM code (Wake/Dream/Judge) |
|---|---|---|
| [`noo132`](noo132.md) | time / space | nemo3_30b / gpt-oss120b / gpt-oss120b |
| [`qoq531`](qoq531.md) | aperiodic tile / traditional craft | qwen3.5_27b / gpt-oss120b / qwen3.5_27b |
| [`qoq797`](qoq797.md) | periodic table / tarot divination | qwen3.5_27b / gpt-oss120b / qwen3.5_27b |
| [`qnq177`](qnq177.md) | time / space | qwen3.5_27b / nemo3_30b / qwen3.5_27b |
