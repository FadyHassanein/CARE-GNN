"""Finish the interrupted GraphConsis n=20 run, then merge and test.

The default path in graphconsis_experiment.py runs both feature configs and
writes its JSON only at the end, so an interrupted run cannot be resumed with
that script without repeating work already done. This runs only the missing
(config, seed) pairs, appends each result to the partial file as it lands --
so a second interruption costs at most one run -- and emits the final table.

    python scripts/graphconsis_resume_n20.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from graphconsis_experiment import (  # noqa: E402
    build_neigh_arrays, run_once, frozen_split,
)
from utils import load_data, normalize  # noqa: E402

RES = Path("results/experiments")
PARTIAL = RES / "graphconsis_n20_partial.json"
FINAL = RES / "graphconsis_table_n20.json"
SEEDS = list(range(100, 120))
TEXT_FILE = "llm_embeddings/yelp/text_risk_scores.pt"


def main():
    state = json.loads(PARTIAL.read_text(encoding="utf-8"))
    banked = {c: {int(k): v for k, v in d.items()} for c, d in state["banked"].items()}

    adj_lists, feat_data, labels = load_data("yelp")
    feat = normalize(feat_data)
    feat = np.asarray(feat.todense() if hasattr(feat, "todense") else feat,
                      dtype=np.float32)
    y = np.asarray(labels).ravel()
    rel_arrays = [build_neigh_arrays(a, len(y)) for a in adj_lists[1:4]]
    text6 = torch.load(TEXT_FILE, weights_only=True).numpy()
    idx_tr, idx_val, idx_te, _, _, _ = frozen_split(len(y), y)

    feats = {"raw32": feat,
             "raw32+text6": np.hstack([feat, text6]).astype(np.float32)}

    todo = [(c, s) for c in feats for s in SEEDS if s not in banked.get(c, {})]
    print(f"{len(todo)} runs remaining\n", flush=True)

    for i, (cfg, seed) in enumerate(todo, 1):
        t0 = time.time()
        _, auc, ap = run_once(feats[cfg], y, rel_arrays, idx_tr, idx_val, idx_te,
                              seed, epochs=30, device="cpu")
        banked.setdefault(cfg, {})[seed] = {"auc": float(auc), "ap": float(ap)}
        print(f"[{i}/{len(todo)}] {cfg:12s} seed {seed}: AUC={auc:.4f} AP={ap:.4f}"
              f"  ({(time.time()-t0)/60:.1f} min)", flush=True)
        # checkpoint after every run, so an interruption costs one run at most
        state["banked"] = {c: {str(k): v for k, v in d.items()}
                           for c, d in banked.items()}
        state["missing"] = {c: [s for s in SEEDS if s not in banked.get(c, {})]
                            for c in feats}
        PARTIAL.write_text(json.dumps(state, indent=1), encoding="utf-8")

    # ---- final table -------------------------------------------------------
    a = np.array([banked["raw32"][s]["auc"] for s in SEEDS])
    b = np.array([banked["raw32+text6"][s]["auc"] for s in SEEDS])
    ap_a = np.array([banked["raw32"][s]["ap"] for s in SEEDS])
    ap_b = np.array([banked["raw32+text6"][s]["ap"] for s in SEEDS])
    t, p = stats.ttest_rel(b, a)
    d = 100 * (b.mean() - a.mean())
    ci = stats.t.interval(0.95, len(a) - 1, loc=(b - a).mean() * 100,
                          scale=stats.sem((b - a) * 100))

    out = {
        "protocol": "GraphConsis, frozen YelpChi 25/15/60, val-AUC checkpoint, "
                    "20 seeds (100-119). Supersedes the 5-seed table: those "
                    "seeds (72-76) proved to be a favourable draw from a "
                    "high-variance baseline.",
        "seeds": SEEDS,
        "raw32": {"auc_mean": a.mean(), "auc_std": a.std(ddof=1),
                  "ap_mean": ap_a.mean(), "per_seed_auc": a.tolist()},
        "raw32+text6": {"auc_mean": b.mean(), "auc_std": b.std(ddof=1),
                        "ap_mean": ap_b.mean(), "per_seed_auc": b.tolist()},
        "text6_delta": {"delta_pp": d, "ci95_pp": list(ci),
                        "positive": int((b > a).sum()), "n": len(a),
                        "t": float(t), "p": float(p), "sig": bool(p < 0.05)},
    }
    FINAL.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")

    print(f"\n{'':14s}{'AUC':>18}{'AP':>10}")
    print(f"{'standalone':14s}{a.mean():>10.4f} ± {a.std(ddof=1):.4f}{ap_a.mean():>10.4f}")
    print(f"{'+text6':14s}{b.mean():>10.4f} ± {b.std(ddof=1):.4f}{ap_b.mean():>10.4f}")
    print(f"\ndelta {d:+.2f} pp   95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]   "
          f"{int((b > a).sum())}/{len(a)} positive   t={t:.2f}  p={p:.4g}   "
          f"{'SIGNIFICANT' if p < 0.05 else 'n.s.'}")
    print(f"-> {FINAL}")


if __name__ == "__main__":
    main()
