# runrem.py: single run for ReMIND v1.0
from __future__ import annotations

from ensure_tmux import ensure_tmux
from remind import run_remind, DEFAULT_PAIRS, DEFAULT_LLM

def main() -> None:
    ensure_tmux()

    pair = DEFAULT_PAIRS[2]
    template_id = 2
    word_limit = 150

    llm_wake  = DEFAULT_LLM["wake"]
    llm_dream = DEFAULT_LLM["dream"]
    llm_judge = DEFAULT_LLM["judge"]

    temp_wake  = 0.6
    temp_dream = 10.0
    temp_judge = 0.0
    temp_rewake = temp_wake

    seed_wake  = 0
    seed_judge = 0
    seed_dream = 1892088617
    seed_rewake = seed_wake

    _ = run_remind(
        pair=pair,
        template_id=template_id,
        word_limit=word_limit,

        llm_wake=llm_wake,
        llm_dream=llm_dream,
        llm_judge=llm_judge,

        temp_wake=temp_wake,
        temp_dream=temp_dream,
        temp_judge=temp_judge,
        temp_rewake=temp_rewake,

        seed_wake=seed_wake,
        seed_dream=seed_dream,
        seed_judge=seed_judge,
        seed_rewake=seed_rewake,

        verbose=True,
    )

if __name__ == "__main__":
    main()