#!/usr/bin/env python3
"""
Entry-FFSZ focusing analysis
============================

Tests the *entry-side* (forward-flow stagnation zone) focusing question raised
by the co-authors, as distinct from the exit-RFSZ focusing already demonstrated
in the manuscript:

    Is the distribution of near-surface ENTRY angles, measured about the
    forward-flow stagnation point(s) of the receiving grain, TIGHTER at the
    next intercepted grain (Grain Y) than at the first intercepted grain
    (Grain X)?

If yes -> grain-to-grain focusing amplifies (entry-FFSZ focusing, the effect
Bill is looking for). If no -> memory persists on the exit side (M30) but entry
angles are reset by flow topology + inter-grain diffusion, and the inherited
state lives in event occurrence/timing instead. Either outcome is informative.

What it computes, per chemistry/condition
-----------------------------------------
  Generation 1 (Grain X): entry angle of each FIRST near-surface event,
      re-referenced to the nearest forward-flow stagnation angle of that grain.
  Generation 2 (Grain Y): entry angle of each NEXT near-surface event from the
      release-to-next-interception restart diagnostic, re-referenced to the
      nearest forward point of the receiving grain, CONDITIONED on the next
      event occurring on a distinct grain (same-grain wake re-encounters do
      not arrive through the FFSZ and are reported separately).
  Metrics: E30 = Pr(min_f |theta_entry - theta_f| <= 30 deg), cutoff-free
      curve E(alpha), median |delta_ffs|, alignment A_ffs = <cos(delta_ffs)>,
      particle-level bootstrap CIs, and the gen1 vs gen2 comparison.
  Optional: iterate the measured (gen-n entry bin) -> (gen-n+1 entry bin)
      kernel to its stationary distribution to ask whether entry angles
      contract toward a fixed point tighter than the first-pass distribution.

Data interface (ADAPT TO THE SIMULATION REPO)
---------------------------------------------
The script expects three tabular inputs (CSV or parquet). Column names can be
remapped in COLMAP below. Angles in degrees, grain-local frame, wrapped to
(-180, 180], zero at the grain's rear stagnation point (the convention of the
manuscript). If your stored angles use a different zero, set ANGLE_ZERO.

1) --stagnation stagnation_points.csv : one row per stagnation point
     grain_id, kind ('forward'|'rear'), angle_deg
   (These are the Fig. 9a points from the OpenFOAM shell diagnostic.)

2) --events events.csv : one row per near-surface/secondary-minimum event
   from the production runs (used for Generation 1 = Grain X)
     particle_id, condition, seed, grain_id, event_index (0 = first event
     of that particle), theta_rel_deg, dtheta_net_deg
   If theta_entry_deg is stored directly, provide it and it will be used;
   otherwise it is reconstructed as theta_entry = theta_rel - dtheta_net.

3) --restarts restarts.csv : one row per restarted release from the
   release-to-next-interception diagnostic (used for Generation 2 = Grain Y)
     condition, release_grain_id, theta_rel_deg, next_grain_id,
     theta_next_deg, time_to_next_s
   Rows with no next event within the horizon: next_grain_id empty/NaN.

Run `python entry_ffsz_focusing.py --selftest` to validate the machinery on
synthetic data with a known answer (no repo data needed).
"""

import argparse
import sys
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Configuration — adapt to the simulation repo's column names / conventions
# ----------------------------------------------------------------------------

COLMAP = {
    # canonical_name : name_in_your_files
    "particle_id": "particle_id",
    "condition": "condition",
    "seed": "seed",
    "grain_id": "grain_id",
    "event_index": "event_index",
    "theta_rel_deg": "theta_rel_deg",
    "dtheta_net_deg": "dtheta_net_deg",
    "theta_entry_deg": "theta_entry_deg",      # optional; reconstructed if absent
    "kind": "kind",
    "angle_deg": "angle_deg",
    "release_grain_id": "release_grain_id",
    "next_grain_id": "next_grain_id",
    "theta_next_deg": "theta_next_deg",
    "time_to_next_s": "time_to_next_s",
}

ANGLE_ZERO = "rear"     # 'rear': stored angles have 0 at rear point (manuscript
                        # convention). If stored in an absolute lab frame,
                        # set to 'absolute' and stagnation angles must be in
                        # the same absolute frame.
CONE_DEG = 30.0         # E30 half-cone, mirrors F30/M30
N_BOOT = 10_000         # particle-level bootstrap resamples
KERNEL_BINS = 24        # angular bins for the generation-iteration kernel
RNG = np.random.default_rng(20260817)


# ----------------------------------------------------------------------------
# Core geometry helpers
# ----------------------------------------------------------------------------

def wrap180(a):
    """Wrap angle(s) in degrees to (-180, 180]."""
    a = np.asarray(a, dtype=float)
    return -((-a + 180.0) % 360.0 - 180.0)


def dist_to_nearest(angles_deg, ref_angles_deg):
    """Min absolute angular distance from each angle to a set of reference
    angles (all degrees, any frame, handles wraparound)."""
    a = np.asarray(angles_deg, dtype=float)[:, None]
    r = np.asarray(ref_angles_deg, dtype=float)[None, :]
    d = np.abs(wrap180(a - r))
    return d.min(axis=1)


def entry_angles_to_ffs(df, stag, angle_col, grain_col):
    """Angular distance from each event's entry angle to the nearest
    FORWARD stagnation angle of the same grain. Returns a Series aligned
    with df.index (NaN where the grain has no forward point recorded)."""
    fwd = (stag[stag["kind"] == "forward"]
           .groupby("grain_id")["angle_deg"].apply(np.asarray).to_dict())
    out = np.full(len(df), np.nan)
    for i, (g, th) in enumerate(zip(df[grain_col].to_numpy(),
                                    df[angle_col].to_numpy())):
        refs = fwd.get(g)
        if refs is not None and len(refs) and np.isfinite(th):
            out[i] = dist_to_nearest([th], refs)[0]
    return pd.Series(out, index=df.index, name="delta_ffs_deg")


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def summarize(delta_ffs, label, particle_ids=None):
    """E30, median, alignment, with particle-level bootstrap CIs."""
    d = np.asarray(delta_ffs, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return {"label": label, "n": 0}
    res = {
        "label": label,
        "n": int(len(d)),
        "E30": float(np.mean(d <= CONE_DEG)),
        "median_deg": float(np.median(d)),
        "A_ffs": float(np.mean(np.cos(np.radians(d)))),
    }
    # bootstrap over unique particles when ids given, else over events
    if particle_ids is not None:
        ids = np.asarray(particle_ids)[np.isfinite(np.asarray(delta_ffs, float))]
        uniq = np.unique(ids)
        groups = {u: d[ids == u] for u in uniq}
        stats = np.empty((N_BOOT, 2))
        keys = np.array(list(groups))
        for b in range(N_BOOT):
            pick = RNG.choice(keys, size=len(keys), replace=True)
            sample = np.concatenate([groups[k] for k in pick])
            stats[b] = (np.mean(sample <= CONE_DEG), np.median(sample))
    else:
        idx = RNG.integers(0, len(d), size=(N_BOOT, len(d)))
        sample = d[idx]
        stats = np.column_stack([np.mean(sample <= CONE_DEG, axis=1),
                                 np.median(sample, axis=1)])
    res["E30_ci"] = [float(x) for x in np.percentile(stats[:, 0], [2.5, 97.5])]
    res["median_ci"] = [float(x) for x in np.percentile(stats[:, 1], [2.5, 97.5])]
    return res


def cumulative_curve(delta_ffs, alphas=None):
    """Cutoff-free E(alpha) curve, analog of F(alpha)."""
    d = np.asarray(delta_ffs, dtype=float)
    d = d[np.isfinite(d)]
    if alphas is None:
        alphas = np.arange(0.0, 180.5, 2.5)
    return alphas, np.array([np.mean(d <= a) for a in alphas])


def iterate_entry_kernel(gen1_delta, gen2_delta, pairs, n_iter=50):
    """Build the discrete kernel P(entry bin at n+1 | entry bin at n) from
    matched (gen1, gen2) pairs on distinct grains, then iterate the gen-1
    marginal to the stationary distribution.

    pairs : (N, 2) array of [delta_ffs at gen n, delta_ffs at gen n+1], deg.
    Returns dict with bin edges, kernel, and E30 trajectory over iterations.
    """
    edges = np.linspace(0.0, 180.0, KERNEL_BINS + 1)
    i1 = np.clip(np.digitize(pairs[:, 0], edges) - 1, 0, KERNEL_BINS - 1)
    i2 = np.clip(np.digitize(pairs[:, 1], edges) - 1, 0, KERNEL_BINS - 1)
    K = np.zeros((KERNEL_BINS, KERNEL_BINS))
    for a, b in zip(i1, i2):
        K[a, b] += 1.0
    rowsum = K.sum(axis=1, keepdims=True)
    support = rowsum[:, 0] > 0
    K[support] /= rowsum[support]
    # rows with no support: identity (particle count too small to inform)
    for r in np.where(~support)[0]:
        K[r, r] = 1.0

    d1 = np.asarray(gen1_delta, float)
    d1 = d1[np.isfinite(d1)]
    p, _ = np.histogram(d1, bins=edges, density=False)
    p = p / p.sum()
    centers = 0.5 * (edges[:-1] + edges[1:])
    e30_traj = []
    for _ in range(n_iter):
        e30_traj.append(float(p[centers <= CONE_DEG].sum()))
        p = p @ K
    return {"edges": edges, "kernel": K, "E30_trajectory": e30_traj,
            "row_support": support.tolist()}


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------

def load_table(path):
    if str(path).endswith((".parquet", ".pq")):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    inv = {v: k for k, v in COLMAP.items() if v in df.columns}
    return df.rename(columns=inv)


# ----------------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------------

def run(events, restarts, stag, make_plots=True, outstem="entry_ffsz"):
    results = []
    if ANGLE_ZERO == "rear":
        # convert rear-referenced stagnation angles: rear point itself is 0,
        # forward angles are already expressed in the same grain-local frame.
        pass  # angles are directly comparable; nothing to do

    # --- Generation 1: first interception per particle (Grain X) ---
    ev = events.copy()
    if "theta_entry_deg" not in ev.columns or ev["theta_entry_deg"].isna().all():
        ev["theta_entry_deg"] = wrap180(ev["theta_rel_deg"] - ev["dtheta_net_deg"])
    first = ev[ev["event_index"] == 0] if "event_index" in ev.columns else \
        ev.sort_values("particle_id").groupby("particle_id", as_index=False).first()

    # --- Generation 2: next interception from restarts (Grain Y) ---
    rs = restarts.copy()
    has_next = rs["next_grain_id"].notna()
    distinct = has_next & (rs["next_grain_id"] != rs["release_grain_id"])

    for cond, g1 in first.groupby("condition"):
        d1 = entry_angles_to_ffs(g1, stag, "theta_entry_deg", "grain_id")
        s1 = summarize(d1, f"{cond} | gen1 firstInterception",
                       particle_ids=g1["particle_id"].to_numpy()
                       if "particle_id" in g1.columns else None)

        g2all = rs[(rs["condition"] == cond) & has_next]
        g2 = rs[(rs["condition"] == cond) & distinct]
        d2 = entry_angles_to_ffs(g2, stag, "theta_next_deg", "next_grain_id")
        s2 = summarize(d2, f"{cond} | gen2 nextInterception distinctGrain")
        same_grain_frac = float(
            1.0 - len(g2) / len(g2all)) if len(g2all) else np.nan

        row = {"condition": cond,
               "gen1": s1, "gen2_distinct_grain": s2,
               "same_grain_next_fraction": same_grain_frac,
               "entry_tightening": (s2.get("E30", np.nan)
                                    - s1.get("E30", np.nan))}
        # kernel iteration where pairs exist
        if len(g2) >= 50:
            drel = entry_angles_to_ffs(
                g2.assign(_g=g2["release_grain_id"]),
                stag, "theta_rel_deg", "_g")  # release-side angle at grain X
            pairs = np.column_stack([
                entry_angles_to_ffs(g2, stag, "theta_rel_deg",
                                    "release_grain_id").to_numpy(),
                d2.to_numpy()])
            pairs = pairs[np.isfinite(pairs).all(axis=1)]
            if len(pairs) >= 50:
                row["kernel_iteration"] = iterate_entry_kernel(
                    d1.to_numpy(), d2.to_numpy(), pairs)
        results.append(row)

    report(results)
    if make_plots:
        plot(results, first, rs, distinct, stag, outstem)
    return results


def report(results):
    print("\n================ Entry-FFSZ focusing summary ================")
    print(f"(E30 = fraction of entries within {CONE_DEG:.0f} deg of the "
          f"nearest forward-flow stagnation point)\n")
    for r in results:
        s1, s2 = r["gen1"], r["gen2_distinct_grain"]
        print(f"condition: {r['condition']}")
        for tag, s in (("Grain X (gen 1)", s1), ("Grain Y (gen 2)", s2)):
            if s.get("n", 0):
                print(f"  {tag:18s} n={s['n']:6d}  E30={s['E30']:.3f} "
                      f"CI[{s['E30_ci'][0]:.3f},{s['E30_ci'][1]:.3f}]  "
                      f"median={s['median_deg']:.1f} deg  A_ffs={s['A_ffs']:.3f}")
            else:
                print(f"  {tag:18s} n=0 (insufficient support)")
        print(f"  same-grain next-event fraction (excluded from gen 2): "
              f"{r['same_grain_next_fraction']:.3f}")
        if np.isfinite(r.get("entry_tightening", np.nan)):
            verdict = ("TIGHTER at Grain Y -> entry-FFSZ focusing"
                       if r["entry_tightening"] > 0 else
                       "NOT tighter at Grain Y -> exit-side memory only")
            print(f"  delta E30 (gen2 - gen1) = {r['entry_tightening']:+.3f}  "
                  f"[{verdict}]")
        ki = r.get("kernel_iteration")
        if ki:
            t = ki["E30_trajectory"]
            print(f"  kernel iteration E30: gen1 {t[0]:.3f} -> "
                  f"stationary {t[-1]:.3f}")
        print()


def plot(results, first, rs, distinct, stag, outstem):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping plots")
        return
    conds = [r["condition"] for r in results]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # (a) cumulative curves gen1 vs gen2 per condition
    ax = axes[0]
    for cond in conds:
        g1 = first[first["condition"] == cond]
        d1 = entry_angles_to_ffs(g1, stag, "theta_entry_deg", "grain_id")
        a, c = cumulative_curve(d1)
        ln, = ax.plot(a, c, lw=1.8, label=f"{cond} gen1")
        g2 = rs[(rs["condition"] == cond) & distinct]
        d2 = entry_angles_to_ffs(g2, stag, "theta_next_deg", "next_grain_id")
        a, c = cumulative_curve(d2)
        ax.plot(a, c, lw=1.8, ls="--", color=ln.get_color(),
                label=f"{cond} gen2")
    ax.axvline(CONE_DEG, color="0.6", lw=0.8)
    ax.set_xlabel(r"$\alpha$ (deg from nearest forward point)")
    ax.set_ylabel(r"$E(\alpha)$")
    ax.set_title("(a) entry-angle cumulative curves")
    ax.legend(fontsize=7)

    # (b) E30 gen1 vs gen2 with CIs
    ax = axes[1]
    x = np.arange(len(conds))
    for j, key, lab in ((0, "gen1", "Grain X (first)"),
                        (1, "gen2_distinct_grain", "Grain Y (next, distinct)")):
        y = [r[key].get("E30", np.nan) for r in results]
        lo = [r[key].get("E30", np.nan) - r[key].get("E30_ci", [np.nan] * 2)[0]
              if r[key].get("n", 0) else np.nan for r in results]
        hi = [r[key].get("E30_ci", [np.nan] * 2)[1] - r[key].get("E30", np.nan)
              if r[key].get("n", 0) else np.nan for r in results]
        ax.errorbar(x + 0.12 * (j * 2 - 1), y, yerr=[lo, hi], fmt="o",
                    capsize=3, label=lab)
    ax.set_xticks(x, conds, rotation=20, fontsize=8)
    ax.set_ylabel(r"$E_{30}$")
    ax.set_title("(b) entry-FFSZ focusing, gen 1 vs gen 2")
    ax.legend(fontsize=8)

    # (c) kernel-iteration trajectories
    ax = axes[2]
    for r in results:
        ki = r.get("kernel_iteration")
        if ki:
            ax.plot(ki["E30_trajectory"], marker=".", label=r["condition"])
    ax.set_xlabel("generation")
    ax.set_ylabel(r"$E_{30}$")
    ax.set_title("(c) iterated entry-kernel trajectory")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{outstem}_summary.png", dpi=200)
    print(f"wrote {outstem}_summary.png")


# ----------------------------------------------------------------------------
# Self-test on synthetic data with a known answer
# ----------------------------------------------------------------------------

def selftest():
    """Two synthetic conditions:
    'tightening'  : gen2 entries drawn tighter around forward points -> E30 up
    'memoryless'  : gen2 entries same width as gen1 -> E30 flat
    """
    n_grain, n_part = 12, 4000
    stag = pd.DataFrame({
        "grain_id": np.repeat(np.arange(n_grain), 2),
        "kind": ["forward", "rear"] * n_grain,
        # rear at 0 (grain-local frame), forward near 180
        "angle_deg": np.tile([172.0, 0.0], n_grain)
        + RNG.normal(0, 4, 2 * n_grain)})
    stag.loc[stag["kind"] == "rear", "angle_deg"] = 0.0

    def make(cond, width1, width2):
        gid = RNG.integers(0, n_grain, n_part)
        fwd = stag[stag["kind"] == "forward"].set_index("grain_id")["angle_deg"]
        entry1 = wrap180(fwd.loc[gid].to_numpy() + RNG.normal(0, width1, n_part))
        rel = RNG.normal(0, 25, n_part)  # focused release near rear (0)
        ev = pd.DataFrame({
            "particle_id": np.arange(n_part), "condition": cond, "seed": 0,
            "grain_id": gid, "event_index": 0,
            "theta_rel_deg": wrap180(rel),
            "dtheta_net_deg": wrap180(rel) - entry1})
        nxt = RNG.integers(0, n_grain, n_part)
        same = RNG.random(n_part) < 0.5  # half same-grain wake re-encounters
        nxt = np.where(same, gid, nxt)
        entry2 = np.where(
            same,
            RNG.normal(0, 20, n_part),  # wake re-entry near rear
            wrap180(fwd.loc[nxt].to_numpy() + RNG.normal(0, width2, n_part)))
        rst = pd.DataFrame({
            "condition": cond, "release_grain_id": gid,
            "theta_rel_deg": wrap180(rel), "next_grain_id": nxt.astype(float),
            "theta_next_deg": wrap180(entry2),
            "time_to_next_s": RNG.exponential(1.0, n_part)})
        return ev, rst

    e1, r1 = make("tightening", width1=45.0, width2=15.0)
    e2, r2 = make("memoryless", width1=45.0, width2=45.0)
    res = run(pd.concat([e1, e2]), pd.concat([r1, r2]), stag,
              make_plots=False)
    t = {r["condition"]: r["entry_tightening"] for r in res}
    assert t["tightening"] > 0.15, f"expected tightening, got {t}"
    assert abs(t["memoryless"]) < 0.08, f"expected flat, got {t}"
    print("SELFTEST PASSED: tightening detected where present, "
          "absent where absent.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", help="production event table (gen 1)")
    ap.add_argument("--restarts", help="release-to-next-interception table (gen 2)")
    ap.add_argument("--stagnation", help="per-grain stagnation point table")
    ap.add_argument("--out", default="entry_ffsz", help="output file stem")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not (args.events and args.restarts and args.stagnation):
        ap.error("--events, --restarts and --stagnation are required "
                 "(or use --selftest)")
    run(load_table(args.events), load_table(args.restarts),
        load_table(args.stagnation), outstem=args.out)


if __name__ == "__main__":
    main()
