"""CAPN and full-framework rows for GraphConsis at n=20.

The five-seed versions of these rows are measured against a standalone baseline
that twenty seeds showed to be optimistic by 4.4 pp, so their deltas cannot be
compared with the twenty-seed semantic rows. This re-runs both CAPN
configurations on the same seeds (100-119) and tests them against the
twenty-seed standalone.

Checkpoints after every run, so an interruption costs one run.

    python scripts/graphconsis_capn_n20.py [--device cuda|cpu]
"""
from __future__ import annotations

import argparse
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
    build_neigh_arrays, run_capn_once, precompute_selection_stats, frozen_split,
)
from utils import load_data, normalize  # noqa: E402

RES = Path("results/experiments")
PARTIAL = RES / "graphconsis_capn_n20_partial.json"
FINAL = RES / "graphconsis_capn_table_n20.json"
STANDALONE = RES / "graphconsis_table_n20.json"
SEEDS = list(range(100, 120))
CONFIGS = {"GraphConsis+CAPN": False, "GraphConsis+CAPN+LLM (framework)": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    banked = {}
    if PARTIAL.exists():
        banked = {c: {int(k): v for k, v in d.items()}
                  for c, d in json.loads(PARTIAL.read_text())["banked"].items()}

    adj, feat_data, labels = load_data("yelp")
    feat = normalize(feat_data)
    feat = np.asarray(feat.todense() if hasattr(feat, "todense") else feat,
                      dtype=np.float32)
    y = np.asarray(labels).ravel()
    rel = [build_neigh_arrays(a, len(y)) for a in adj[1:4]]
    text6 = torch.load("llm_embeddings/yelp/text_risk_scores.pt",
                       weights_only=True).numpy()
    t0 = time.time()
    sel = precompute_selection_stats(feat, rel)
    print(f"selection stats precomputed in {time.time()-t0:.0f}s", flush=True)
    itr, iva, ite, _, _, _ = frozen_split(len(y), y)

    todo = [(c, s) for c in CONFIGS for s in SEEDS
            if s not in banked.get(c, {})]
    print(f"{len(todo)} runs remaining on {args.device}\n", flush=True)

    for i, (cfg, seed) in enumerate(todo, 1):
        t0 = time.time()
        _, auc, ap = run_capn_once(feat, text6, y, rel, sel, itr, iva, ite,
                                   seed, use_llm=CONFIGS[cfg],
                                   epochs=args.epochs, device=args.device)
        mins = (time.time() - t0) / 60
        banked.setdefault(cfg, {})[seed] = {"auc": float(auc), "ap": float(ap)}
        done, left = i, len(todo) - i
        print(f"[{done}/{len(todo)}] {cfg[:26]:26s} seed {seed}: "
              f"AUC={auc:.4f} AP={ap:.4f}  ({mins:.1f} min, "
              f"~{left*mins/60:.1f} h left)", flush=True)
        PARTIAL.write_text(json.dumps(
            {"banked": {c: {str(k): v for k, v in d.items()}
                        for c, d in banked.items()},
             "missing": {c: [s for s in SEEDS if s not in banked.get(c, {})]
                         for c in CONFIGS}}, indent=1))

    # ---- final table, tested against the n=20 standalone --------------------
    stand = json.loads(STANDALONE.read_text())
    base = np.array(stand["raw32"]["per_seed_auc"])
    text = np.array(stand["raw32+text6"]["per_seed_auc"])
    out = {"protocol": "GraphConsis + CAPN at 20 seeds (100-119), frozen "
                       "25/15/60 split. Deltas are against the twenty-seed "
                       "standalone, matched seed for seed.",
           "seeds": SEEDS,
           "standalone": {"auc_mean": float(base.mean()),
                          "auc_std": float(base.std(ddof=1))},
           "+text6": {"auc_mean": float(text.mean()),
                      "auc_std": float(text.std(ddof=1))}}

    print(f"\n{'configuration':32s}{'AUC':>18}{'vs standalone':>16}{'p':>10}")
    print(f"{'standalone':32s}{base.mean():>10.4f} ± {base.std(ddof=1):.4f}"
          f"{'---':>16}{'---':>10}")
    print(f"{'+ text6':32s}{text.mean():>10.4f} ± {text.std(ddof=1):.4f}"
          f"{100*(text.mean()-base.mean()):>+15.2f}{stats.ttest_rel(text, base)[1]:>10.4f}")

    for cfg in CONFIGS:
        v = np.array([banked[cfg][s]["auc"] for s in SEEDS])
        t_b, p_b = stats.ttest_rel(v, base)
        t_t, p_t = stats.ttest_rel(v, text)
        out[cfg] = {"auc_mean": float(v.mean()), "auc_std": float(v.std(ddof=1)),
                    "per_seed_auc": v.tolist(),
                    "vs_standalone": {"delta_pp": 100*(v.mean()-base.mean()),
                                      "t": float(t_b), "p": float(p_b),
                                      "positive": int((v > base).sum())},
                    "vs_text6": {"delta_pp": 100*(v.mean()-text.mean()),
                                 "t": float(t_t), "p": float(p_t),
                                 "positive": int((v > text).sum())}}
        print(f"{cfg[:32]:32s}{v.mean():>10.4f} ± {v.std(ddof=1):.4f}"
              f"{100*(v.mean()-base.mean()):>+15.2f}{p_b:>10.4f}")

    fw = out["GraphConsis+CAPN+LLM (framework)"]["vs_text6"]
    print(f"\nadaptive pillar on top of the semantic one: "
          f"{fw['delta_pp']:+.2f} pp, {fw['positive']}/20, p={fw['p']:.4f}")
    FINAL.write_text(json.dumps(out, indent=1, default=float))
    print(f"-> {FINAL}")


if __name__ == "__main__":
    main()
