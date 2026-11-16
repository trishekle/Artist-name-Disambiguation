"""Robustness evaluation of the misattribution scorer across all analyzable groups.

Extends tools/injection_eval.py (track-level injection on one group) with:

* Two injection modes
    - track:   relabel N random tracks to a different same-name artist
               (same draw order and seeds as tools/injection_eval.py).
    - release: move ALL tracks of whole albums (album_id) to one different
               same-name artist. Real misattributions usually move whole
               releases, so album-sibling overlap no longer points back to the
               true artist. Only albums whose artist keeps >= 2 tracks after the
               move are eligible; trials where that is impossible are skipped
               and reported.
  Credited ids are rewritten with injection_eval._relabel in both modes.
* Several injection sizes, 20 trials each, fixed seeds.
* Metrics: ROC AUC (mean/std over trials), recall/FPR at the default rule
  (best-fit != assigned, i.e. gap > 0), recall at fixed 1% / 5% FPR (thresholds
  from pooled clean-track gap quantiles), precision@5 / @10 per trial, and
  precision at an assumed real base rate via injection_eval.precision_at_base_rate.
* Mean and std across groups for every setting.
* Leave-one-group-out threshold transfer (5% FPR on the other groups' clean tracks).
* Block ablation (drop one block / use one block alone) at track N=10 and release 1 album.
* Single-track-artist fallback audit: tracks whose assigned artist (after
  injection) has only one track are excluded from every metric and counted.

Usage:
  python tools/robust_eval.py
  python tools/robust_eval.py --groups WITCH Morphine --trials 5
  python tools/robust_eval.py --out-dir .agents/tasks/robust-eval
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis import data_source
from src.analysis.features import load_features
from src.analysis.profiles import build_artist_profiles
from src.analysis.scoring import WEIGHTS, score_tracks

# injection_eval reads sys.argv at import time; import its helpers with a clean argv.
_saved_argv = sys.argv
sys.argv = [_saved_argv[0]]
try:
    from tools.injection_eval import BASE_SEED, _relabel, precision_at_base_rate
finally:
    sys.argv = _saved_argv

BLOCKS = list(WEIGHTS)
TARGET_FPRS = (0.01, 0.05)
ABLATION_SETTINGS = {("track", 10), ("release", 1)}
LOGO_SETTINGS = [("track", 10), ("release", 1)]


def weight_sets(with_ablation: bool) -> dict[str, dict[str, float]]:
    """Default weights plus, optionally, drop-one and only-one block variants."""
    sets = {"full": dict(WEIGHTS)}
    if not with_ablation:
        return sets
    for block in BLOCKS:
        rest = sum(v for k, v in WEIGHTS.items() if k != block)
        sets[f"drop_{block}"] = {k: 0.0 if k == block else WEIGHTS[k] / rest for k in BLOCKS}
        sets[f"only_{block}"] = {k: 1.0 if k == block else 0.0 for k in BLOCKS}
    return sets


# --------------------------------------------------------------------------- injection

def inject_tracks(frame: pd.DataFrame, n: int, rng: np.random.Generator):
    """Track mode; mirrors tools/injection_eval._run_once draw order exactly."""
    artist_ids = frame["artist_id"].astype(str).unique().tolist()
    n = min(n, len(frame))
    rows = rng.choice(len(frame), size=n, replace=False)
    mask = np.zeros(len(frame), dtype=bool)
    mask[rows] = True
    for row in rows:
        true_id = str(frame.at[row, "artist_id"])
        new_id = rng.choice([a for a in artist_ids if a != true_id])
        _relabel(frame, row, new_id, true_id)
    return mask, []


def inject_releases(frame: pd.DataFrame, n_albums: int, rng: np.random.Generator, attempts: int = 50):
    """Release mode: move every track of ``n_albums`` albums to one other artist each.

    Returns (impostor_mask, moves) or (None, []) if no eligible album set was found.
    """
    artist_ids = frame["artist_id"].astype(str).unique().tolist()
    albums = frame["album_id"].astype(str)
    owners = frame.groupby(albums)["artist_id"].agg(lambda s: s.astype(str).unique().tolist())
    if (owners.str.len() > 1).any():
        raise ValueError("an album_id spans several artist ids; release mode assumes one owner")
    owner = owners.str[0]
    size = albums.value_counts()
    album_list = sorted(owner.index)

    chosen = None
    for _ in range(attempts):
        counts = frame["artist_id"].astype(str).value_counts().to_dict()
        picked = []
        for alb in rng.permutation(album_list):
            if counts[owner[alb]] - size[alb] >= 2:
                picked.append(alb)
                counts[owner[alb]] -= size[alb]
                if len(picked) == n_albums:
                    break
        if len(picked) == n_albums:
            chosen = picked
            break
    if chosen is None:
        return None, []

    mask = np.zeros(len(frame), dtype=bool)
    moves = []
    for alb in chosen:
        true_id = owner[alb]
        new_id = str(rng.choice([a for a in artist_ids if a != true_id]))
        rows = np.where(albums.to_numpy() == alb)[0]
        for row in rows:
            _relabel(frame, row, new_id, true_id)
        mask[rows] = True
        moves.append((alb, true_id, new_id, len(rows)))

    # Debug assertion: each chosen album moved as a whole, credits rewritten.
    for alb, true_id, new_id, k in moves:
        sel = albums.to_numpy() == alb
        assert sel.sum() == k and (frame.loc[sel, "artist_id"].astype(str) == new_id).all()
        for credited in frame.loc[sel, "credited_artist_ids"].astype(str):
            assert true_id not in credited.split("|") and new_id in credited.split("|")
        assert mask[sel].all()
    assert mask.sum() == sum(m[3] for m in moves)
    return mask, moves


# --------------------------------------------------------------------------- scoring

def score_frame(frame: pd.DataFrame, weights: dict[str, dict[str, float]]):
    """Rebuild features/profiles on the corrupted frame; return gaps per weight set and
    the single-track-artist mask (assigned artist has exactly one track)."""
    with tempfile.TemporaryDirectory() as td:
        csv = Path(td) / "injected.csv"
        frame.to_csv(csv, index=False)
        features = load_features(csv)
    profiles = build_artist_profiles(features)
    assigned = features.frame["artist_id"].astype(str)
    single = (assigned.map(assigned.value_counts()) == 1).to_numpy()
    gaps = {name: score_tracks(features, profiles, w, leave_one_out=True)["score_gap"].to_numpy()
            for name, w in weights.items()}
    return gaps, single


def _auc(imp: np.ndarray, gap: np.ndarray) -> float:
    if imp.sum() and (~imp).sum() and len(np.unique(gap)) > 1:
        return float(roc_auc_score(imp.astype(int), gap))
    return float("nan")


def _precision_at_k(imp: np.ndarray, gap: np.ndarray, k: int) -> float:
    order = np.argsort(-gap, kind="stable")[:k]
    return float(imp[order].mean()) if len(order) else float("nan")


def run_setting(base: pd.DataFrame, group: str, mode: str, size: int, n_trials: int,
                with_ablation: bool, log):
    weights = weight_sets(with_ablation)
    trials = []
    for t in range(n_trials):
        if mode == "track":
            rng = np.random.default_rng(BASE_SEED + t)  # same seeds as injection_eval.py
        else:
            rng = np.random.default_rng([BASE_SEED, 1, size, t])
        frame = base.copy().reset_index(drop=True)
        if mode == "track":
            imp, moves = inject_tracks(frame, size, rng)
        else:
            imp, moves = inject_releases(frame, size, rng)
            if imp is None:
                trials.append({"trial": t, "skipped": True})
                continue
            if t == 0:
                for alb, old, new, k in moves:
                    log(f"    [release check] {group} trial0 album {alb}: {k} tracks "
                        f"moved {old} -> {new} (all {k} album tracks relabeled: OK)")
        gaps, single = score_frame(frame, weights)
        keep = ~single
        imp_k = imp[keep]
        rec = {"trial": t, "skipped": False, "n_tracks": len(frame),
               "n_imp": int(imp_k.sum()), "n_clean": int((~imp_k).sum()),
               "excl_imp": int((imp & single).sum()), "excl_clean": int((~imp & single).sum()),
               "imp": imp_k, "gaps": {name: g[keep] for name, g in gaps.items()}}
        g = rec["gaps"]["full"]
        flagged = g > 0
        rec["recall_default"] = float(flagged[imp_k].mean()) if imp_k.sum() else float("nan")
        rec["fpr_default"] = float(flagged[~imp_k].mean()) if (~imp_k).sum() else float("nan")
        rec["p_at_5"] = _precision_at_k(imp_k, g, 5)
        rec["p_at_10"] = _precision_at_k(imp_k, g, 10)
        rec["auc"] = {name: _auc(imp_k, gg) for name, gg in rec["gaps"].items()}
        trials.append(rec)
    return trials


def summarize(trials, n_artists: int, base_rate: float) -> dict:
    done = [r for r in trials if not r["skipped"]]
    out = {"n_trials": len(done), "n_skipped": len(trials) - len(done),
           "random_baseline": 1 - 1 / n_artists}
    if not done:
        return out
    aucs = np.array([r["auc"]["full"] for r in done])
    aucs = aucs[~np.isnan(aucs)]
    out["auc_mean"] = float(aucs.mean()) if len(aucs) else float("nan")
    out["auc_std"] = float(aucs.std()) if len(aucs) else float("nan")
    out["recall_default"] = float(np.nanmean([r["recall_default"] for r in done]))
    out["fpr_default"] = float(np.nanmean([r["fpr_default"] for r in done]))
    out["p_at_5"] = float(np.nanmean([r["p_at_5"] for r in done]))
    out["p_at_10"] = float(np.nanmean([r["p_at_10"] for r in done]))
    out["prec_base_default"] = precision_at_base_rate(out["recall_default"], out["fpr_default"], base_rate)
    gaps = np.concatenate([r["gaps"]["full"] for r in done])
    imp = np.concatenate([r["imp"] for r in done])
    for fpr in TARGET_FPRS:
        thr = float(np.quantile(gaps[~imp], 1 - fpr))
        rec = float((gaps[imp] > thr).mean())
        got = float((gaps[~imp] > thr).mean())
        key = f"{int(fpr * 100)}"
        out[f"thr_at_{key}"] = thr
        out[f"recall_at_{key}"] = rec
        out[f"fpr_got_{key}"] = got
        out[f"prec_base_at_{key}"] = precision_at_base_rate(rec, got, base_rate)
    out["n_imp"] = int(imp.sum())
    out["n_clean"] = int((~imp).sum())
    out["excl_imp"] = sum(r["excl_imp"] for r in done)
    out["excl_clean"] = sum(r["excl_clean"] for r in done)
    names = done[0]["auc"].keys()
    out["ablation"] = {}
    for name in names:
        vals = np.array([r["auc"][name] for r in done])
        vals = vals[~np.isnan(vals)]
        out["ablation"][name] = float(vals.mean()) if len(vals) else float("nan")
    return out


# --------------------------------------------------------------------------- report

def _f(x, pct=False, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.1%}" if pct else f"{x:.{nd}f}"


def _mean_std(values):
    v = np.array([x for x in values if x is not None and not np.isnan(x)])
    return (float(v.mean()), float(v.std())) if len(v) else (float("nan"), float("nan"))


def build_report(groups, sizes_info, settings, results, logo, base_rate, n_trials, runtime, cmd):
    L = []
    L.append("# Robust evaluation results\n")
    L.append(f"Command: `{cmd}`  \nTrials per setting: {n_trials}. Runtime: {runtime / 60:.1f} min. "
             f"Base rate for precision: {base_rate:.1%}. Seeds: track mode uses "
             f"`default_rng({BASE_SEED} + trial)` (identical to tools/injection_eval.py); release mode "
             f"uses `default_rng([{BASE_SEED}, 1, n_albums, trial])`.\n")
    L.append("Flag rule 'default' = best-fit artist != assigned (score gap > 0). Tracks whose assigned "
             "artist has only one track after injection are excluded from every metric "
             "(single-track fallback audit below).\n")

    L.append("## Group sizes\n")
    L.append("| group | tracks | artists | tracks per artist | albums per artist |")
    L.append("|---|---|---|---|---|")
    for g in groups:
        s = sizes_info[g]
        L.append(f"| {g} | {s['tracks']} | {s['artists']} | {s['tpa']} | {s['apa']} |")
    L.append("")

    L.append("## Across-group summary (mean ± std over groups)\n")
    L.append("| setting | groups | AUC | recall default | FPR default | recall @1% FPR | recall @5% FPR "
             "| P@5 | P@10 | precision@1% base (5% FPR) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for mode, size in settings:
        rows = [results[(g, mode, size)] for g in groups if results[(g, mode, size)].get("n_trials")]
        def ms(key, pct=False):
            m, s = _mean_std([r.get(key, float("nan")) for r in rows])
            return f"{_f(m, pct)} ± {_f(s, pct)}"
        label = f"{mode} N={size}" if mode == "track" else f"release {size} album(s)"
        L.append(f"| {label} | {len(rows)} | {ms('auc_mean')} | {ms('recall_default', True)} | "
                 f"{ms('fpr_default', True)} | {ms('recall_at_1', True)} | {ms('recall_at_5', True)} | "
                 f"{ms('p_at_5', True)} | {ms('p_at_10', True)} | {ms('prec_base_at_5', True)} |")
    L.append("")

    L.append("## Per group and setting\n")
    L.append("| group | setting | trials (skipped) | AUC mean ± std | recall default | FPR default "
             "| random baseline | recall @1% FPR | recall @5% FPR (thr) | P@5 | P@10 "
             "| prec@base default | prec@base 5% FPR | excluded imp/clean |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for mode, size in settings:
        for g in groups:
            r = results[(g, mode, size)]
            label = f"track N={size}" if mode == "track" else f"release {size}"
            if not r.get("n_trials"):
                L.append(f"| {g} | {label} | 0 ({r['n_skipped']}) | skipped: no eligible album set "
                         "| | | | | | | | | | |")
                continue
            L.append(f"| {g} | {label} | {r['n_trials']} ({r['n_skipped']}) | "
                     f"{_f(r['auc_mean'])} ± {_f(r['auc_std'])} | {_f(r['recall_default'], True)} | "
                     f"{_f(r['fpr_default'], True)} | {_f(r['random_baseline'], True)} | "
                     f"{_f(r['recall_at_1'], True)} | {_f(r['recall_at_5'], True)} ({_f(r['thr_at_5'], nd=4)}) | "
                     f"{_f(r['p_at_5'], True)} | {_f(r['p_at_10'], True)} | "
                     f"{_f(r['prec_base_default'], True)} | {_f(r['prec_base_at_5'], True)} | "
                     f"{r['excl_imp']}/{r['excl_clean']} |")
    L.append("")
    L.append("P@k is capped at N/k when fewer than k impostors are planted (e.g. track N=2: P@5 <= 40%).\n")

    L.append("## Leave-one-group-out threshold (5% FPR on the other groups' pooled clean tracks)\n")
    for (mode, size), rows in logo.items():
        label = f"track N={size}" if mode == "track" else f"release {size} album(s)"
        L.append(f"### {label}\n")
        L.append("| held-out group | threshold | held-out FPR | held-out recall |")
        L.append("|---|---|---|---|")
        for g, thr, fpr, rec in rows:
            L.append(f"| {g} | {thr:.4f} | {_f(fpr, True)} | {_f(rec, True)} |")
        if rows:
            fm, fs = _mean_std([r[2] for r in rows])
            rm, rs = _mean_std([r[3] for r in rows])
            L.append(f"| **mean ± std** | median {np.median([r[1] for r in rows]):.4f} | "
                     f"{_f(fm, True)} ± {_f(fs, True)} | {_f(rm, True)} ± {_f(rs, True)} |")
        L.append("")

    L.append("## Block ablation (AUC, mean over trials)\n")
    for mode, size in settings:
        if (mode, size) not in ABLATION_SETTINGS:
            continue
        label = f"track N={size}" if mode == "track" else f"release {size} album(s)"
        names = list(weight_sets(True))
        L.append(f"### {label}\n")
        L.append("| group | " + " | ".join(names) + " |")
        L.append("|---|" + "---|" * len(names))
        cols = {n: [] for n in names}
        for g in groups:
            r = results[(g, mode, size)]
            if not r.get("n_trials"):
                continue
            vals = [r["ablation"].get(n, float("nan")) for n in names]
            for n, v in zip(names, vals):
                cols[n].append(v)
            L.append(f"| {g} | " + " | ".join(_f(v) for v in vals) + " |")
        L.append("| **mean** | " + " | ".join(_f(_mean_std(cols[n])[0]) for n in names) + " |")
        L.append("")

    L.append("## Single-track-artist fallback audit\n")
    L.append("Tracks excluded because their assigned artist (after injection) had exactly one track, "
             "summed over trials:\n")
    L.append("| group | setting | excluded impostors | excluded clean | scored impostors | scored clean |")
    L.append("|---|---|---|---|---|---|")
    for mode, size in settings:
        for g in groups:
            r = results[(g, mode, size)]
            if r.get("n_trials"):
                L.append(f"| {g} | {mode} {size} | {r['excl_imp']} | {r['excl_clean']} | "
                         f"{r['n_imp']} | {r['n_clean']} |")
    L.append("")
    return "\n".join(L)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Robust injection evaluation across groups.")
    parser.add_argument("--groups", nargs="*", default=None, help="Default: all analyzable groups.")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--track-sizes", nargs="*", type=int, default=[2, 5, 10])
    parser.add_argument("--release-sizes", nargs="*", type=int, default=[1, 2])
    parser.add_argument("--base-rate", type=float, default=0.01)
    parser.add_argument("--out-dir", default=str(ROOT / ".agents" / "tasks" / "robust-eval"))
    args = parser.parse_args(argv)

    start = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = args.groups or data_source.available_groups()
    settings = [("track", n) for n in args.track_sizes] + [("release", n) for n in args.release_sizes]

    def log(msg):
        print(msg, flush=True)

    bases, sizes_info = {}, {}
    for g in groups:
        base = data_source.load_group(g).frame.copy()
        bases[g] = base
        by_artist = base.groupby(base["artist_id"].astype(str))
        sizes_info[g] = {"tracks": len(base), "artists": base["artist_id"].astype(str).nunique(),
                         "tpa": ", ".join(str(v) for v in sorted(by_artist.size(), reverse=True)),
                         "apa": ", ".join(str(v) for v in sorted(by_artist["album_id"].nunique(), reverse=True))}

    results, raw_trials, per_trial_rows = {}, {}, []
    for mode, size in settings:
        for g in groups:
            t0 = time.time()
            trials = run_setting(bases[g], g, mode, size, args.trials,
                                 (mode, size) in ABLATION_SETTINGS, log)
            raw_trials[(g, mode, size)] = trials
            res = summarize(trials, sizes_info[g]["artists"], args.base_rate)
            results[(g, mode, size)] = res
            for r in trials:
                row = {"group": g, "mode": mode, "size": size, "trial": r["trial"], "skipped": r["skipped"]}
                if not r["skipped"]:
                    row.update({k: r[k] for k in ("n_imp", "n_clean", "excl_imp", "excl_clean",
                                                  "recall_default", "fpr_default", "p_at_5", "p_at_10")})
                    row.update({f"auc_{k}": v for k, v in r["auc"].items()})
                per_trial_rows.append(row)
            log(f"{g:<10} {mode:<7} {size:>2}: trials {res['n_trials']} skipped {res['n_skipped']} "
                f"AUC {_f(res.get('auc_mean'))} +/- {_f(res.get('auc_std'))} "
                f"recall@5%FPR {_f(res.get('recall_at_5'), True)} "
                f"default rec/FPR {_f(res.get('recall_default'), True)}/{_f(res.get('fpr_default'), True)} "
                f"excluded {res.get('excl_imp', 0)}/{res.get('excl_clean', 0)} ({time.time() - t0:.0f}s)")

    logo = {}
    for mode, size in LOGO_SETTINGS:
        if (mode, size) not in settings:
            continue
        pooled = {}
        for g in groups:
            done = [r for r in raw_trials[(g, mode, size)] if not r["skipped"]]
            if done:
                pooled[g] = (np.concatenate([r["gaps"]["full"] for r in done]),
                             np.concatenate([r["imp"] for r in done]))
        rows = []
        for g in pooled:
            others = [pooled[o] for o in pooled if o != g]
            if not others:
                continue
            clean_other = np.concatenate([gp[~im] for gp, im in others])
            thr = float(np.quantile(clean_other, 0.95))
            gp, im = pooled[g]
            rows.append((g, thr, float((gp[~im] > thr).mean()), float((gp[im] > thr).mean())))
            log(f"LOGO {mode} {size} held-out {g}: thr {thr:.4f} FPR {rows[-1][2]:.1%} recall {rows[-1][3]:.1%}")
        logo[(mode, size)] = rows

    runtime = time.time() - start
    cmd = "python tools/robust_eval.py " + " ".join(argv if argv is not None else sys.argv[1:])
    report = build_report(groups, sizes_info, settings, results, logo, args.base_rate,
                          args.trials, runtime, cmd.strip())
    (out_dir / "robust_results.md").write_text(report, encoding="utf-8")
    pd.DataFrame(per_trial_rows).to_csv(out_dir / "robust_per_trial.csv", index=False)
    log(f"\nWrote {out_dir / 'robust_results.md'} and robust_per_trial.csv  (runtime {runtime / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
