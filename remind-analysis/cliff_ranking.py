#!/usr/bin/env python3
"""
cliff_ranking.py: Cliff's delta ranking for all 19 conditions × 3 themes.

Stages:
  wake-rewake  (default) — full analysis: ranking, role table, rewake mean, total ranking, etc.
  wake-dream             — unmatched by default; delta ranking only
  dream-rewake           — matched by default (Wilcoxon); delta ranking only

Convention: δ < 0 means 'to' phase > 'from' phase (improvement in novelty)
Metrics: novelty, sum_score (alignment+coherence+novelty)

Usage:
  python cliff_ranking.py
  python cliff_ranking.py --stage wake-dream
  python cliff_ranking.py --stage dream-rewake
  python cliff_ranking.py --stage dream-rewake --unmatched
  python cliff_ranking.py --metric sum_score
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy import stats

THEMES = {
    "time_space":      "time",
    "periodic_tarot":  "tarot",
    "aperiodic_craft": "aperiodic",
}

OLD_18    = "qqq qoq qqo qoo qgq qqg qnq qqn ggg ooo oqo ooq oqq nnn nqq noq nqo noo".split()
ALL_CONDS = OLD_18 + ["gqq"]  # 19 conditions

LOGS = Path("logs")


def delta_label(d: float) -> str:
    a = abs(d)
    if a < 0.147:
        return "negligible"
    elif a < 0.474:
        return "small-medium"
    return "large"


def load_review(cond: str, target: str, metric: str) -> Dict[str, List[float]]:
    """Return {theme_key: [scores]} for each theme.

    Filters: temp_dream >= 1.0 (exclude control runs), dedup by run_id.
    """
    path = LOGS / f"remind_review_{cond}_core_{target}_gpt5.2.jsonl"
    buckets: Dict[str, List[float]] = {k: [] for k in THEMES}
    if not path.exists():
        return buckets
    seen_ids: set = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if (r.get("meta") or {}).get("status") != "ok":
                continue
            td = r.get("temp_dream")
            if td is not None and float(td) < 1.0:
                continue
            run_id = r.get("run_id")
            if run_id is not None and run_id in seen_ids:
                continue
            if run_id is not None:
                seen_ids.add(run_id)
            blob = (r.get("reviews") or {}).get("openai") or {}
            nov = blob.get("novelty")
            if nov is None:
                continue
            aln = blob.get("alignment") or 0
            coh = blob.get("coherence") or 0
            score = float(nov) if metric == "novelty" else float(nov) + float(aln) + float(coh)
            pair = str(r.get("pair") or r.get("result", {}).get("pair") or "").lower()
            for theme_key, keyword in THEMES.items():
                if keyword in pair:
                    buckets[theme_key].append(score)
    return buckets


def load_review_with_ids(cond: str, target: str, metric: str) -> Dict[str, Dict[str, float]]:
    """Return {theme_key: {run_id: score}} for matched comparisons."""
    path = LOGS / f"remind_review_{cond}_core_{target}_gpt5.2.jsonl"
    buckets: Dict[str, Dict[str, float]] = {k: {} for k in THEMES}
    if not path.exists():
        return buckets
    seen_ids: set = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if (r.get("meta") or {}).get("status") != "ok":
                continue
            td = r.get("temp_dream")
            if td is not None and float(td) < 1.0:
                continue
            run_id = r.get("run_id")
            if run_id is None:
                continue
            if run_id in seen_ids:
                continue
            seen_ids.add(run_id)
            blob = (r.get("reviews") or {}).get("openai") or {}
            nov = blob.get("novelty")
            if nov is None:
                continue
            aln = blob.get("alignment") or 0
            coh = blob.get("coherence") or 0
            score = float(nov) if metric == "novelty" else float(nov) + float(aln) + float(coh)
            pair = str(r.get("pair") or r.get("result", {}).get("pair") or "").lower()
            for theme_key, keyword in THEMES.items():
                if keyword in pair:
                    buckets[theme_key][run_id] = score
    return buckets


def cliffs_delta(xa: np.ndarray, xb: np.ndarray) -> float:
    """δ = cliffs_delta(from, to): negative = to > from (improvement)."""
    xa = np.asarray(xa, dtype=float)
    xb = np.asarray(xb, dtype=float)
    if xa.size == 0 or xb.size == 0:
        return float("nan")
    diff = xa[:, None] - xb[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / (xa.size * xb.size))


def mann_whitney_p(xa: np.ndarray, xb: np.ndarray) -> float:
    if xa.size < 2 or xb.size < 2:
        return float("nan")
    _, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
    return float(p)


def wilcoxon_p(xa: np.ndarray, xb: np.ndarray) -> float:
    """Wilcoxon signed-rank test for matched pairs (xa=from, xb=to)."""
    if xa.size < 2:
        return float("nan")
    diff = xa - xb
    if np.all(diff == 0):
        return float("nan")
    try:
        _, p = stats.wilcoxon(xa, xb, alternative="two-sided")
        return float(p)
    except Exception:
        return float("nan")


def compute_cond(cond: str, metric: str) -> Dict[str, dict]:
    """Wake→Rewake unmatched (original behavior)."""
    wa_all = load_review(cond, "wake",   metric)
    ra_all = load_review(cond, "rewake", metric)

    results = {}
    for theme_key in THEMES:
        wa = np.array(wa_all[theme_key])
        ra = np.array(ra_all[theme_key])
        results[theme_key] = {
            "cond":     cond,
            "theme":    theme_key,
            "delta":    cliffs_delta(wa, ra),
            "dmean":    float(np.mean(ra) - np.mean(wa)) if ra.size > 0 and wa.size > 0 else float("nan"),
            "p":        mann_whitney_p(wa, ra),
            "n_wake":   int(wa.size),
            "n_rewake": int(ra.size),
        }

    # All themes pooled
    all_wa = np.concatenate([np.array(wa_all[tk]) for tk in THEMES])
    all_ra = np.concatenate([np.array(ra_all[tk]) for tk in THEMES])
    results["all"] = {
        "cond":     cond,
        "theme":    "all",
        "delta":    cliffs_delta(all_wa, all_ra),
        "dmean":    float(np.mean(all_ra) - np.mean(all_wa)) if all_ra.size > 0 and all_wa.size > 0 else float("nan"),
        "p":        mann_whitney_p(all_wa, all_ra),
        "n_wake":   int(all_wa.size),
        "n_rewake": int(all_ra.size),
    }
    return results


def compute_cond_stage(cond: str, metric: str, from_target: str, to_target: str,
                       matched: bool) -> Dict[str, dict]:
    """Generic stage comparison (wake→dream or dream→rewake, matched or unmatched)."""
    if matched:
        from_all = load_review_with_ids(cond, from_target, metric)
        to_all   = load_review_with_ids(cond, to_target,   metric)
        results = {}
        all_pairs: List[tuple] = []
        for theme_key in THEMES:
            common = sorted(set(from_all[theme_key]) & set(to_all[theme_key]))
            xa = np.array([from_all[theme_key][rid] for rid in common])
            xb = np.array([to_all[theme_key][rid]   for rid in common])
            results[theme_key] = {
                "cond":  cond,
                "theme": theme_key,
                "delta": cliffs_delta(xa, xb) if xa.size > 0 else float("nan"),
                "dmean": float(np.mean(xb) - np.mean(xa)) if xa.size > 0 else float("nan"),
                "p":     wilcoxon_p(xa, xb),
                "n":     int(xa.size),
            }
            all_pairs.extend(zip(xa.tolist(), xb.tolist()))
        all_xa = np.array([x for x, _ in all_pairs])
        all_xb = np.array([y for _, y in all_pairs])
        results["all"] = {
            "cond":  cond,
            "theme": "all",
            "delta": cliffs_delta(all_xa, all_xb) if all_xa.size > 0 else float("nan"),
            "dmean": float(np.mean(all_xb) - np.mean(all_xa)) if all_xa.size > 0 else float("nan"),
            "p":     wilcoxon_p(all_xa, all_xb),
            "n":     int(all_xa.size),
        }
    else:
        from_all = load_review(cond, from_target, metric)
        to_all   = load_review(cond, to_target,   metric)
        results = {}
        for theme_key in THEMES:
            xa = np.array(from_all[theme_key])
            xb = np.array(to_all[theme_key])
            results[theme_key] = {
                "cond":  cond,
                "theme": theme_key,
                "delta": cliffs_delta(xa, xb),
                "dmean": float(np.mean(xb) - np.mean(xa)) if xb.size > 0 and xa.size > 0 else float("nan"),
                "p":     mann_whitney_p(xa, xb),
                "n":     int(xb.size),
            }
        all_xa = np.concatenate([np.array(from_all[tk]) for tk in THEMES])
        all_xb = np.concatenate([np.array(to_all[tk])   for tk in THEMES])
        results["all"] = {
            "cond":  cond,
            "theme": "all",
            "delta": cliffs_delta(all_xa, all_xb),
            "dmean": float(np.mean(all_xb) - np.mean(all_xa)) if all_xb.size > 0 and all_xa.size > 0 else float("nan"),
            "p":     mann_whitney_p(all_xa, all_xb),
            "n":     int(all_xb.size),
        }
    return results


# ── role effectiveness table ──────────────────────────────────────────────────
# Single-substitution from qqq: replace one role with LLM X, keep others at q
ROLE_CONDS = {
    "Wake":  {"n": "nqq", "o": "oqq", "q": "qoo", "g": "gqq"},
    "Dream": {"n": "qnq", "o": "qoq", "q": "oqo", "g": "qgq"},
    "Judge": {"n": "qqn", "o": "qqo", "q": "ooq", "g": "qqg"},
}
BASELINES = {"n": "qqq", "o": "qqq", "q": "ooo", "g": "qqq"}
LLMS = ["o", "q", "g", "n"]


def delta_rating(dd: float, bold: bool = True) -> str:
    if np.isnan(dd):
        return "N/A"
    if dd <= -0.330:
        sym = "++"
    elif dd <= -0.147:
        sym = "+"
    elif dd < 0.147:
        sym = "+/−"
    elif dd < 0.330:
        sym = "−"
    else:
        sym = "−−"
    return f"**{sym}**" if bold and sym not in ("+/−",) else sym


def fmt_p(p: float) -> str:
    if np.isnan(p):
        return "   n/a"
    if p < 1e-10:
        return f"{p:.1e}"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


THEME_LABELS = [
    ("time_space",      "Time & Space"),
    ("periodic_tarot",  "Periodic Table × Tarot"),
    ("aperiodic_craft", "Aperiodic Tile × Craft"),
    ("all",             "All Themes"),
]
THEME_KEY_MAP = {
    "ts":  "time_space",
    "pt":  "periodic_tarot",
    "ac":  "aperiodic_craft",
    "all": "all",
}


def print_stage_ranking(df: pd.DataFrame, stage_label: str, metric: str, matched: bool,
                        theme_filter: str | None = None) -> None:
    """Print delta ranking tables for wake-dream or dream-rewake stages."""
    match_label = "matched (Wilcoxon)" if matched else "unmatched"
    for theme_key, theme_label in THEME_LABELS:
        if theme_filter and theme_key != theme_filter:
            continue
        sub = df[df["theme"] == theme_key].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("delta", ascending=True)
        print(f"\n## {theme_label}  |  {stage_label} ({match_label})  |  metric={metric}\n")
        print(f"| Rank | cond |      δ |  Δmean |          p |   n |")
        print(f"|-----:|:-----|-------:|-------:|-----------:|----:|")
        for rank, (_, row) in enumerate(sub.iterrows(), 1):
            d = row["delta"]
            sig = " *" if row["p"] < 0.05 and d < 0 else ""
            print(f"| {rank} | {row['cond']}{sig} | {d:+.3f} | {row['dmean']:+.3f} | {fmt_p(row['p']):>10} | {int(row['n'])} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric",   choices=["novelty", "sum_score"], default="novelty")
    ap.add_argument("--stage",    choices=["wake-rewake", "wake-dream", "dream-rewake"],
                    default="wake-rewake",
                    help="comparison stage (default: wake-rewake)")
    ap.add_argument("--unmatched", action="store_true",
                    help="force unmatched comparison (overrides default matched for dream-rewake)")
    ap.add_argument("--theme",    choices=["ts", "pt", "ac", "all"], default=None,
                    help="filter output to one theme: ts=Time&Space, pt=Periodic Tarot, ac=Aperiodic Craft, all=All Themes")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    theme_filter = THEME_KEY_MAP[args.theme] if args.theme else None

    # ── non-default stages: wake-dream / dream-rewake ─────────────────────────
    if args.stage != "wake-rewake":
        from_target, to_target = args.stage.split("-", 1)  # "wake","dream" or "dream","rewake"
        matched = (args.stage == "dream-rewake") and not args.unmatched
        stage_label = args.stage.replace("-", "→").replace("wake", "Wake").replace("dream", "Dream").replace("rewake", "Rewake")

        all_rows = []
        for cond in ALL_CONDS:
            try:
                res = compute_cond_stage(cond, args.metric, from_target, to_target, matched)
                for row in res.values():
                    all_rows.append(row)
            except Exception as e:
                print(f"[warn] {cond}: {e}")

        df = pd.DataFrame(all_rows)
        print_stage_ranking(df, stage_label, args.metric, matched, theme_filter)

        out = args.out or f"reports/cliff_ranking_{args.stage}_{args.metric}.csv"
        df.to_csv(out, index=False, float_format="%.4f")
        print(f"\n[saved] {out}")
        return

    # ── wake-rewake (default): full analysis ──────────────────────────────────
    all_rows = []
    for cond in ALL_CONDS:
        try:
            res = compute_cond(cond, args.metric)
            for row in res.values():
                all_rows.append(row)
        except Exception as e:
            print(f"[warn] {cond}: {e}")

    df = pd.DataFrame(all_rows)

    for theme_key, theme_label in THEME_LABELS:
        if theme_filter and theme_key != theme_filter:
            continue
        sub = df[df["theme"] == theme_key].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("delta", ascending=True)

        print(f"\n## {theme_label}  |  Wake→Rewake (unmatched)  |  metric={args.metric}\n")
        print(f"| Rank | cond |      δ |  Δmean |          p |   n |")
        print(f"|-----:|:-----|-------:|-------:|-----------:|----:|")
        for rank, (_, row) in enumerate(sub.iterrows(), 1):
            d = row["delta"]
            print(f"| {rank} | {row['cond']} | {d:+.3f} | {row['dmean']:+.3f} | {fmt_p(row['p']):>10} | {int(row['n_rewake'])} |")

    # ── all-theme combined table ──────────────────────────────────────────────
    THEME_SHORT = {
        "time_space":      "T&S",
        "periodic_tarot":  "PT",
        "aperiodic_craft": "AC",
    }
    wide = df.pivot(index="cond", columns="theme", values="delta").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns=THEME_SHORT)
    wide["mean_δ"] = wide[list(THEME_SHORT.values())].mean(axis=1)
    wide = wide.sort_values("mean_δ", ascending=True).reset_index(drop=True)

    if not theme_filter or theme_filter == "all":
        print(f"\n## All Themes (mean δ)  |  Wake→Rewake (unmatched)  |  metric={args.metric}\n")
        print(f"| Rank | cond |    T&S |     PT |     AC | mean δ |")
        print(f"|-----:|:-----|-------:|-------:|-------:|-------:|")
        for rank, row in wide.iterrows():
            ts  = row["T&S"]
            pt  = row["PT"]
            ac  = row["AC"]
            md  = row["mean_δ"]
            print(f"| {rank+1} | {row['cond']} | {ts:+.3f} | {pt:+.3f} | {ac:+.3f} | {md:+.3f} |")

    # ── role effectiveness tables ─────────────────────────────────────────────
    pivot = df.pivot(index="cond", columns="theme", values="delta")
    THEME_KEYS = ["time_space", "periodic_tarot", "aperiodic_craft"]

    for theme_key, theme_label in [
        ("time_space",      "Time & Space"),
        ("periodic_tarot",  "Periodic Table × Tarot"),
        ("aperiodic_craft", "Aperiodic Tile × Craft"),
    ]:
        if theme_filter and theme_filter not in (theme_key, "all"):
            continue
        print(f"\n## {theme_label} — LLM role effectiveness  |  metric={args.metric}\n")
        print(f"| LLM | Wake | Dream | Judge |")
        print(f"|:----|:----:|:-----:|:-----:|")
        for llm in LLMS:
            base = pivot.loc[BASELINES[llm], theme_key] if BASELINES[llm] in pivot.index else float("nan")
            cells = []
            for role in ["Wake", "Dream", "Judge"]:
                cond = ROLE_CONDS[role][llm]
                d = pivot.loc[cond, theme_key] if cond in pivot.index else float("nan")
                cells.append(delta_rating(d - base, bold=True))
            print(f"| **{llm}** | {cells[0]} | {cells[1]} | {cells[2]} |")

    # ── all-theme combined role table ─────────────────────────────────────────
    if not theme_filter or theme_filter == "all":
        print(f"\n## All Themes — LLM role effectiveness  |  metric={args.metric}\n")
        print(f"| LLM | role | T&S | PT | AC | total |")
        print(f"|:----|:-----|:---:|:--:|:--:|:-----:|")
        for llm in LLMS:
            for ri, role in enumerate(["Wake", "Dream", "Judge"]):
                cond = ROLE_CONDS[role][llm]
                dds = []
                per_theme = []
                for tk in THEME_KEYS:
                    base = pivot.loc[BASELINES[llm], tk] if BASELINES[llm] in pivot.index else float("nan")
                    d    = pivot.loc[cond, tk] if cond in pivot.index else float("nan")
                    dd   = d - base
                    dds.append(dd)
                    per_theme.append(delta_rating(dd, bold=False))
                mean_dd = float(np.nanmean(dds))
                total   = delta_rating(mean_dd, bold=True)
                llm_col = f"**{llm}**" if ri == 0 else ""
                print(f"| {llm_col} | {role} | {per_theme[0]} | {per_theme[1]} | {per_theme[2]} | {total} |")

    # ── mean rewake novelty ranking ───────────────────────────────────────────
    mean_rows = []
    for cond in ALL_CONDS:
        try:
            wa_all = load_review(cond, "wake",   args.metric)
            ra_all = load_review(cond, "rewake", args.metric)
            for theme_key in THEMES:
                wa = np.array(wa_all[theme_key])
                ra = np.array(ra_all[theme_key])
                mean_rows.append({
                    "cond":        cond,
                    "theme":       theme_key,
                    "rewake_mean": float(np.mean(ra)) if ra.size > 0 else float("nan"),
                    "wake_mean":   float(np.mean(wa)) if wa.size > 0 else float("nan"),
                    "n":           int(ra.size),
                })
            # All themes combined
            all_wa = np.concatenate([np.array(wa_all[tk]) for tk in THEMES])
            all_ra = np.concatenate([np.array(ra_all[tk]) for tk in THEMES])
            mean_rows.append({
                "cond":        cond,
                "theme":       "all",
                "rewake_mean": float(np.mean(all_ra)) if all_ra.size > 0 else float("nan"),
                "wake_mean":   float(np.mean(all_wa)) if all_wa.size > 0 else float("nan"),
                "n":           int(all_ra.size),
            })
        except Exception as e:
            print(f"[warn means] {cond}: {e}")

    mdf = pd.DataFrame(mean_rows)

    for theme_key, theme_label in THEME_LABELS:
        if theme_filter and theme_key != theme_filter:
            continue
        sub = mdf[mdf["theme"] == theme_key].dropna(subset=["rewake_mean"]).copy()
        sub = sub[sub["n"] > 0].sort_values("rewake_mean", ascending=False).reset_index(drop=True)

        print(f"\n## {theme_label} — Mean rewake {args.metric}\n")
        print(f"| Rank | cond | rewake mean | wake mean |   n |")
        print(f"|-----:|:-----|------------:|----------:|----:|")
        for i, row in sub.iterrows():
            print(f"| {i+1} | {row['cond']} | {row['rewake_mean']:.3f} | {row['wake_mean']:.3f} | {int(row['n'])} |")

    # ── total ranking (All Themes δ-rank × All Themes mean-rank) ────────────
    if not theme_filter or theme_filter == "all":
        d_sub = (df.groupby("cond")["delta"].mean()
                   .reset_index()
                   .sort_values("delta", ascending=True)
                   .reset_index(drop=True))
        d_rank = {row["cond"]: i + 1 for i, row in d_sub.iterrows()}
        m_sub = (mdf[mdf["theme"] == "all"]
                   .dropna(subset=["rewake_mean"])
                   .query("n > 0")
                   .sort_values("rewake_mean", ascending=False)
                   .reset_index(drop=True))
        m_rank = {row["cond"]: i + 1 for i, row in m_sub.iterrows()}
        conds = [c for c in ALL_CONDS if c in d_rank and c in m_rank]
        rows = sorted(conds, key=lambda c: (d_rank[c] + m_rank[c]) / 2)

        print(f"\n## All Themes — Total ranking  |  metric={args.metric}\n")
        print(f"| 条件 | δランク | rewake meanランク |")
        print(f"|:-----|-------:|-----------------:|")
        for cond in rows:
            dr = d_rank[cond]
            mr = m_rank[cond]
            dr_str = f"**{dr}位**" if dr <= 3 else f"{dr}位"
            mr_str = f"**{mr}位**" if mr <= 3 else f"{mr}位"
            cond_str = f"**{cond}**" if dr <= 3 or mr <= 3 else cond
            print(f"| {cond_str} | {dr_str} | {mr_str} |")

    # ── theme comparison (cross-theme; skip when a single theme is selected) ──
    if theme_filter and theme_filter != "all":
        out = args.out or f"reports/cliff_ranking_{args.metric}.csv"
        df.to_csv(out, index=False, float_format="%.4f")
        print(f"\n[saved] {out}")
        return

    def sig(p: float) -> str:
        if np.isnan(p): return ""
        if p < 0.001: return " ***"
        if p < 0.01:  return " **"
        if p < 0.05:  return " *"
        return ""

    THEME_ORDER = ["time_space", "periodic_tarot", "aperiodic_craft"]
    # Collect paired δ and Δmean for the 19 conditions in each theme
    t_delta = {}
    t_dmean = {}
    for tk in THEME_ORDER:
        sub = df[df["theme"] == tk].set_index("cond")
        t_delta[tk] = np.array([sub.loc[c, "delta"] for c in ALL_CONDS if c in sub.index])
        t_dmean[tk] = np.array([sub.loc[c, "dmean"]  for c in ALL_CONDS if c in sub.index])

    def fmt_p2(p: float) -> str:
        if p < 0.001: return f"{p:.1e}"
        if p < 0.1:   return f"{p:.3f}"
        return f"{p:.2f}"

    best_tk = min(THEME_ORDER, key=lambda tk: float(np.mean(t_delta[tk])))

    print(f"\n**Wilcoxon signed-rank test**（{len(ALL_CONDS)}条件の対応δ値による対比較）\n")
    print(f"| Theme | mean δ | mean Δmean | p vs periodic | p vs aperiodic |")
    print(f"|:------|-------:|-----------:|--------------:|---------------:|")
    for tk in THEME_ORDER:
        md  = float(np.mean(t_delta[tk]))
        mdm = float(np.mean(t_dmean[tk]))

        # lower-triangular: each pair shown only once
        if tk in ("periodic_tarot", "aperiodic_craft"):
            p_per = "—"
        else:
            _, pv = stats.wilcoxon(t_delta[tk], t_delta["periodic_tarot"])
            p_per = fmt_p2(pv) + sig(pv)

        if tk == "aperiodic_craft":
            p_aper = "—"
        elif tk == "periodic_tarot":
            _, pv = stats.wilcoxon(t_delta[tk], t_delta["aperiodic_craft"])
            p_aper = fmt_p2(pv) + sig(pv)
        else:
            _, pv = stats.wilcoxon(t_delta[tk], t_delta["aperiodic_craft"])
            p_aper = fmt_p2(pv) + sig(pv)

        if tk == best_tk:
            print(f"| {tk} | **{md:+.3f}** | **{mdm:+.3f}** | {p_per} | {p_aper} |")
        else:
            print(f"| {tk} | {md:+.3f} | {mdm:+.3f} | {p_per} | {p_aper} |")

    # ── wake novelty summary ──────────────────────────────────────────────────
    THEME_DISP = {
        "time_space":      "time & space",
        "periodic_tarot":  "periodic tarot",
        "aperiodic_craft": "aperiodic craft",
    }

    # Pool all wake scores across 19 conditions per theme
    wake_pool = {}
    for tk in THEME_ORDER:
        pool = []
        for cond in ALL_CONDS:
            pool.extend(load_review(cond, "wake", args.metric)[tk])
        wake_pool[tk] = np.array(pool)

    # Count conditions with significant improvement (δ<0 and p<0.05) among ALL_CONDS
    sig_cnt = {}
    for tk in THEME_ORDER:
        sub = df[(df["theme"] == tk) & (df["cond"].isin(ALL_CONDS))]
        sig_cnt[tk] = int(((sub["delta"] < 0) & (sub["p"] < 0.05)).sum())

    mean_w = {tk: float(np.mean(wake_pool[tk])) for tk in THEME_ORDER}
    med_w  = {tk: float(np.median(wake_pool[tk])) for tk in THEME_ORDER}
    min_mean_tk = min(THEME_ORDER, key=lambda tk: mean_w[tk])
    max_med_tk  = max(THEME_ORDER, key=lambda tk: med_w[tk])
    max_sig_tk  = max(THEME_ORDER, key=lambda tk: sig_cnt[tk])
    min_sig_tk  = min(THEME_ORDER, key=lambda tk: sig_cnt[tk])

    print(f"\n## Wake {args.metric} summary  |  19 conditions\n")
    print(f"| Theme | mean wake | median wake | 有意改善cond数 |")
    print(f"|:------|----------:|------------:|--------------:|")
    for tk in THEME_ORDER:
        mw  = f"**{mean_w[tk]:.3f}**" if tk == min_mean_tk else f"{mean_w[tk]:.3f}"
        mdw = f"**{med_w[tk]:.1f}**"  if tk == max_med_tk  else f"{med_w[tk]:.1f}"
        sc  = sig_cnt[tk]
        n_conds = len(ALL_CONDS)
        sig = f"**{sc} / {n_conds}**" if tk in (max_sig_tk, min_sig_tk) else f"{sc} / {n_conds}"
        print(f"| {THEME_DISP[tk]} | {mw} | {mdw} | {sig} |")

    out = args.out or f"reports/cliff_ranking_{args.metric}.csv"
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
