import json
import pandas as pd

rows = []

with open("logs/remind_sweep_1ooo_maxt_core.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        res = r.get("result")
        if not res:
            continue

        judgewake = res.get("judgewake") or {}
        judgedream = res.get("judgedream") or {}

        rows.append({
            # ---- idea presence ----
            "idea_wake": bool((res.get("idea_wake") or "").strip()),
            "idea_dream": bool((res.get("idea_dream") or "").strip()),

            # ---- judge scores ----
            "score_wake": judgewake.get("score"),
            "score_dream": judgedream.get("score"),
        })

df = pd.DataFrame(rows)

print("=== IDEA COUNTS ===")
print(df[["idea_wake", "idea_dream"]].sum())
print("\n=== IDEA RATIO ===")
print(df[["idea_wake", "idea_dream"]].mean())

print("\n=== SCORE SUMMARY ===")
print(df[["score_wake", "score_dream"]].describe())

print("\n=== SCORE DISTRIBUTION ===")
print("Wake score counts:")
print(df["score_wake"].value_counts().sort_index())

print("\nDream score counts:")
print(df["score_dream"].value_counts().sort_index())