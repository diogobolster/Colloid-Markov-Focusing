# Entry-FFSZ focusing: PRODUCTION-DATA result

**Date:** 2026-08-18 · Computed from the actual production outputs (OpenFOAM
flow, production restart diagnostic, production stagnation points) staged from
the simulation working directory. Supersedes the LBM preview (v1/v2 reports)
as the citable result; the preview's numbers cross-validate throughout.

## Inputs (all stock production artifacts, read-only)

`outputs/next_interception_kernel/release_to_next_interception_events.csv`
(2500 restarted releases/profile), `random_grain_focusing_production_selected/
grain_stagnation_points.csv` (34 grains, forward + rear), the production
OpenFOAM VTU + geometry (periodic-window shift: disabled), and the production
payload. Generation 1 = first-passage propagation of 3000 particles from the
inlet through the production flow (same integrator style as the restart
diagnostic; chemistry-independent since DLVO forces vanish beyond 200 nm).

## Result

Generation-1 first interception: **E30(forward) = 0.292 [0.275, 0.309]**,
median 44.9° from the nearest forward point (n = 2810). [LBM preview: 0.278,
50.4° — the preview flow was faithful.]

Generation-2 next interception (distinct grain, forward-referenced):

| condition | gen-2 E30 [95% CI] | n | ΔE30 vs no-DLVO gen-2 | exit-side ΔF30 vs no-DLVO |
|---|---|---|---|---|
| no-DLVO            | 0.272 [0.231, 0.315] | 416 | —      | —      |
| unfav 50 mM        | 0.287 [0.232, 0.343] | 254 | +0.015 | +0.310 |
| unfav 75 mM        | 0.342 [0.282, 0.402] | 234 | +0.070 (z ≈ 1.9) | +0.488 |
| unfav 100 mM       | 0.358 [0.297, 0.424] | 229 | +0.086 (z ≈ 2.2) | +0.537 |
| unfav 50 mM 100×D₀ | 0.203 [0.157, 0.253] | 281 | −0.069 | +0.103 |

Same-grain fraction of next events: 0.785 (no-DLVO) → 0.879 / 0.894 / 0.898
(50/75/100 mM); production release F30: 0.080 → 0.390 / 0.568 / 0.617.

## Reading — the three-part answer to Bill's question

1. **The entry-angle baseline is generation-stationary and set by flow
   topology.** The no-DLVO next-grain entry distribution (0.272) matches the
   first-passage distribution from the inlet (0.292). Entry angles are
   substantially reset between grains; they are not the primary carrier of
   encounter memory.

2. **But a weak entry-side focusing signal DOES emerge at production
   statistics** — invisible in the preview's thin samples. Gen-2 E30 rises
   monotonically with ionic strength (0.272 → 0.287 → 0.342 → 0.358), the
   75/100 mM increments over no-DLVO are ΔE30 ≈ +0.07–0.09 at z ≈ 1.9–2.2,
   and the 100×D₀ control falls *below* baseline (−0.069) — extra diffusion
   erases even the geometric released-state bias. Direction, monotonicity,
   and control all behave as Bill's picture would predict.

3. **It is a second-order effect.** The entry-side enhancement is 6–8×
   smaller than the exit-side enhancement at every chemistry (+0.086 vs
   +0.537 at 100 mM). The memory hierarchy in this system is: same-grain
   wake re-encounter (largest) > exit-side rear-zone alignment (F30, M30) >
   entry-side forward-zone alignment (small but real, monotone in IS).

So the honest one-liner for Bill: *entry-FFSZ focusing from one grain to the
next exists in the homogeneous model, but as a weak second-order echo of the
exit-side mechanism — the entry angle is mostly reset by inter-grain
transport, and the focusing memory travels primarily through where releases
happen and how quickly the same grain recaptures them.* The colleague's model
not showing it is consistent: at typical model resolutions/samples a
ΔE30 ≈ 0.07 effect is below the noise floor (our own preview missed it at
n ≈ 40–130).

## Supporting anatomy (production-confirmed)

- The favorable case's per-particle "entry" field (attach on first contact →
  last entry = first entry) is forward-biased (median 59° from forward),
  while unfavorable cases' last-crossing entries are rear-shifted (117–133°):
  production confirmation of the multi-crossing episode cascade and of the
  first-vs-last-crossing distinction flagged in v2.
- Statistical significance caveat: the 75/100 mM increments are ~2σ single
  comparisons; pooling the second production seeds (seed20260508/9 release
  dirs exist) and/or more restarts would firm this up. The monotone ordering
  plus the diffusion control's sign reversal is the more persuasive pattern.

Artifacts: `entry_ffsz_production.png` (comparison + magnitude figure),
`entry_ffsz_results.json` (production gen-2), `first_passage_gen1.json`
(production gen-1). All analysis code on the `entry-ffsz-analysis` branch.
