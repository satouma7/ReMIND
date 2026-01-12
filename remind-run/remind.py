# remind.py: core component of ReMIND v1.0
from __future__ import annotations
from typing import Any, Optional
from wake import wake
from dream import dream
from judge import judge

DEFAULT_PAIRS: list[tuple[str, str]] = [
    ("time", "space"),
    ("aperiodic tile", "traditional craft"),
    ("periodic table", "tarot divination"),
]

DEFAULT_LLM: dict[str, str] = {
    "wake": "oss120b",
    "dream": "gemma27b",
    "judge": "gemma27b",
}

def build_prompts(pair: tuple[str, str], word_limit: int = 150) -> list[str]:
    a, b = pair
    return [
        f'Compare the meaning of "{a}" and "{b}" within {word_limit} words.',
        f'Describe the unexpected relationship between "{a}" and "{b}" within {word_limit} words.',
        f'Propose a new idea about the relationship between "{a}" and "{b}" within {word_limit} words.',
    ]

def run_remind(
    *,
    pair: tuple[str, str],
    template_id: int = 2,
    word_limit: int = 150,

    # Automatically computed if None
    max_tokens_wake: Optional[int] = None,
    max_tokens_dream: Optional[int] = None,
    max_tokens_rewake: Optional[int] = None,
    max_tokens_judge: int = 200,  # a small fixed value is recommended for the judge

    llm_wake: str = DEFAULT_LLM["wake"],
    llm_dream: str = DEFAULT_LLM["dream"],
    llm_judge: str = DEFAULT_LLM["judge"],

    temp_wake: float = 0.6,
    temp_dream: float = 10.0,
    temp_judge: float = 0.0,
    temp_rewake: Optional[float] = None,

    seed_wake: int = 0,
    seed_dream: int = 0,
    seed_judge: int = 0,
    seed_rewake: Optional[int] = None,

    score_threshold: int = 4,
    verbose: bool = True,
) -> dict[str, Any]:

    prompts = build_prompts(pair, word_limit=word_limit)
    if not (0 <= template_id < len(prompts)):
        raise ValueError(f"template_id must be 0..{len(prompts)-1}, got {template_id}")
    prompt = prompts[template_id]

    # Automatic max_tokens calculation (only when set to None)
    if max_tokens_wake is None:
        max_tokens_wake = word_limit + 50
    if max_tokens_dream is None:
        max_tokens_dream = word_limit + 50
    if max_tokens_rewake is None:
        max_tokens_rewake = word_limit + 100

    if temp_rewake is None:
        temp_rewake = temp_wake
    if seed_rewake is None:
        seed_rewake = seed_wake

    if verbose:
        print(prompt)

    # ---- WAKE ----
    wakeout = wake(
        prompt,
        llm=llm_wake,
        max_tokens=max_tokens_wake,
        temperature=temp_wake,
        seed=seed_wake,
    )
    if verbose:
        print(f"\n=== WAKE (LLM={llm_wake}:temp={temp_wake}:seed={seed_wake}) ===")
        print(wakeout)

    # ---- JUDGE WAKE ----
    judgewake = judge(
        f"===BEGIN===\n{wakeout}\n===END===",
        llm=llm_judge,
        max_tokens=max_tokens_judge,
        temperature=temp_judge,
        seed=seed_judge,
    )
    if verbose:
        print(f"\n=== JUDGE WAKE (LLM={llm_judge}:temp={temp_judge}:seed={seed_judge}) ===")
        print(judgewake)

    # ---- DREAM ----
    dreamout = dream(
        prompt,
        llm=llm_dream,
        max_tokens=max_tokens_dream,
        temperature=temp_dream,
        seed=seed_dream,
    )
    if verbose:
        print(f"\n=== DREAM (LLM={llm_dream}:temp={temp_dream}:seed={seed_dream}) ===")
        print(dreamout)

    # ---- JUDGE DREAM ----
    judgedream = judge(
        f"===BEGIN===\n{dreamout}\n===END===",
        llm=llm_judge,
        max_tokens=max_tokens_judge,
        temperature=temp_judge,
        seed=seed_judge,
    )
    if verbose:
        print(f"\n=== JUDGE DREAM (LLM={llm_judge}:temp={temp_judge}:seed={seed_judge}) ===")
        print(judgedream)

    idea_wake = (judgewake.get("idea") or "").strip()
    score_dream = int(judgedream.get("score", 0) or 0)
    idea_dream = (judgedream.get("idea") or "").strip()

    # ---- REWAKE ----
    rewakeout = None
    rewake_skipped_reason = None

    if score_dream >= score_threshold and idea_dream:
        prompt_idea = (
            f"Propose the following idea to the user within {word_limit} words.\n"
            f"IDEA:\n{idea_dream}\n"
        )
        rewakeout = wake(
            prompt_idea,
            llm=llm_wake,
            max_tokens=max_tokens_rewake,
            temperature=temp_rewake,
            seed=seed_rewake,
        )
        if verbose:
            print(f"\n=== REWAKE (LLM={llm_wake}:temp={temp_rewake}:seed={seed_rewake}) ===")
            print(rewakeout)
    else:
        rewake_skipped_reason = f"score_dream={score_dream}, idea_dream_empty={not bool(idea_dream)}"
        if verbose:
            print(f"\n=== REWAKE skipped ({rewake_skipped_reason}) ===")

    return {
        "pair": pair,
        "template_id": template_id,
        "word_limit": word_limit,
        "prompt": prompt,
        "params": {
            "llm_wake": llm_wake,
            "llm_dream": llm_dream,
            "llm_judge": llm_judge,
            "temp_wake": temp_wake,
            "temp_dream": temp_dream,
            "temp_judge": temp_judge,
            "temp_rewake": temp_rewake,
            "seed_wake": seed_wake,
            "seed_dream": seed_dream,
            "seed_judge": seed_judge,
            "seed_rewake": seed_rewake,
            "max_tokens_wake": max_tokens_wake,
            "max_tokens_dream": max_tokens_dream,
            "max_tokens_rewake": max_tokens_rewake,
            "max_tokens_judge": max_tokens_judge,
            "score_threshold": score_threshold,
        },
        "wakeout": wakeout,
        "judgewake": judgewake,
        "dreamout": dreamout,
        "judgedream": judgedream,
        "idea_wake": idea_wake,
        "idea_dream": idea_dream,
        "rewakeout": rewakeout,
        "rewake_skipped_reason": rewake_skipped_reason,
    }