#!/usr/bin/env python3
"""Stage 2 particle pilots for screened random pore geometries."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage2_particle_pilots"
GEOM_ROOT = ROOT / "outputs" / "stage1_geometry_screen" / "geometries"
FLOW_ROOT = ROOT / "outputs" / "stage1_openfoam_flow"
SUMMARY = OUT / "stage2_particle_pilot_summary.csv"
REPORT = OUT / "stage2_particle_pilot_report.md"
SANITY = OUT / "stage2_flow_transport_sanity.csv"
FLOW_SCREEN = ROOT / "outputs" / "stage1_geometry_screen" / "stage1_flow_screen_summary.csv"

CANDIDATES = [
    "through_random_00",
    "through_few_large_grains",
    "through_tight_throats",
    "through_high_polydispersity",
    "through_pockety_high_cv",
]

PROFILES = [
    "neutral_resolved",
    "favorable_50mM_z70",
    "unfavorable_50mM_z70",
    "unfavorable_100mM_z70",
    "unfavorable_50mM_z70_100xD",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def merge_candidate_rows(existing: list[dict[str, object]], updates: list[dict[str, object]]) -> list[dict[str, object]]:
    update_candidates = {str(row.get("candidate_id", "")) for row in updates if row.get("candidate_id")}
    retained = [row for row in existing if str(row.get("candidate_id", "")) not in update_candidates]
    return retained + updates


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def flow_screen_status() -> dict[str, dict[str, str]]:
    if not FLOW_SCREEN.exists():
        return {}
    rows = read_csv(FLOW_SCREEN)
    return {str(row["candidate_id"]): row for row in rows}


def fallback_flow_case(candidate: str) -> Path:
    return FLOW_ROOT / f"{candidate}_gmsh_screen"


def run_candidate(
    candidate: str,
    particles: int,
    resume: bool,
    flow_case: Path,
    out_root: Path,
    *,
    max_time: float,
    dt: float,
    chunk_size: int,
    grid_nx: int,
    grid_ny: int,
    matrix_bins: int,
    profiles: list[str],
) -> list[dict[str, object]]:
    candidate_out = out_root / candidate
    summary_path = candidate_out / "random_condition_screen_summary.csv"
    if summary_path.exists() and resume:
        rows = read_csv(summary_path)
    else:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_random_condition_screen.py"),
            "--geometry-path",
            str(GEOM_ROOT / candidate / "geometry.json"),
            "--flow-case",
            str(flow_case),
            "--out-dir",
            str(candidate_out),
            "--backend",
            "compiled",
            "--particles",
            str(particles),
            "--max-time",
            str(max_time),
            "--dt",
            str(dt),
            "--grid-nx",
            str(grid_nx),
            "--grid-ny",
            str(grid_ny),
            "--matrix-bins",
            str(matrix_bins),
            "--injection-mode",
            "flux_weighted",
            "--profiles",
            *profiles,
        ]
        if chunk_size > 0:
            cmd.extend(["--chunk-size", str(chunk_size)])
        if resume:
            cmd.append("--resume")
        subprocess.run(cmd, cwd=ROOT, check=True)
        rows = read_csv(summary_path)
    for row in rows:
        row["candidate_id"] = candidate
        row["candidate_out_dir"] = str(candidate_out.relative_to(ROOT))
        row["flow_case"] = str(flow_case.relative_to(ROOT)) if flow_case.is_relative_to(ROOT) else str(flow_case)
    return rows


def sanity_status() -> dict[str, str]:
    if not SANITY.exists():
        return {}
    return {str(row["candidate_id"]): str(row["status"]) for row in read_csv(SANITY)}


def float_or_nan(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def profile_row(rows: list[dict[str, object]], profile: str) -> dict[str, object] | None:
    for row in rows:
        if row.get("profile") == profile:
            return row
    return None


def write_report(rows: list[dict[str, object]], out_root: Path, report_path: Path, profiles: list[str]) -> None:
    by_candidate: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_candidate.setdefault(str(row["candidate_id"]), []).append(row)

    lines = [
        "# Stage 2 Particle Pilot Screen",
        "",
        f"Each geometry was run with {profiles} using the compiled random-packing tracker.",
        "Only Stage 1 flows that passed the passive-throughflow gate were promoted to DLVO particle pilots.",
        "The OpenFOAM velocity field was rescaled to the common 4 m/day mean velocity and the tracker used the same periodic-origin shift as the flow mesh.",
        "",
        "| candidate | status | no-DLVO intercepted | favorable attached | 50 mM intercepted | 50 mM median near time (s) | 100 mM median near time (s) | 100D median near time (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate, candidate_rows in by_candidate.items():
        neutral = profile_row(candidate_rows, "neutral_resolved") or {}
        fav = profile_row(candidate_rows, "favorable_50mM_z70") or {}
        unf50 = profile_row(candidate_rows, "unfavorable_50mM_z70") or {}
        unf100 = profile_row(candidate_rows, "unfavorable_100mM_z70") or {}
        high_d = profile_row(candidate_rows, "unfavorable_50mM_z70_100xD") or {}
        if candidate_rows and candidate_rows[0].get("status") == "skipped_flow_sanity":
            lines.append(
                f"| {candidate} | skipped: {candidate_rows[0].get('skip_reason', candidate_rows[0].get('flow_sanity_status', 'flow gate'))} |  |  |  |  |  |  |"
            )
            continue
        n_particles = max(float_or_nan(neutral.get("particles")), 1.0)
        lines.append(
            "| {candidate} | completed | {neutral_int:.3f} | {fav_att:.3f} | {unf50_int:.3f} | {unf50_near:.3g} | {unf100_near:.3g} | {highd_near:.3g} |".format(
                candidate=candidate,
                neutral_int=float_or_nan(neutral.get("intercepted")) / n_particles,
                fav_att=float_or_nan(fav.get("attached")) / max(float_or_nan(fav.get("particles")), 1.0),
                unf50_int=float_or_nan(unf50.get("intercepted")) / max(float_or_nan(unf50.get("particles")), 1.0),
                unf50_near=float_or_nan(unf50.get("median_near_time_intercepted_s")),
                unf100_near=float_or_nan(unf100.get("median_near_time_intercepted_s")),
                highd_near=float_or_nan(high_d.get("median_near_time_intercepted_s")),
            )
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
        ]
    )
    for candidate in by_candidate:
        candidate_rows = by_candidate[candidate]
        if candidate_rows and candidate_rows[0].get("status") == "skipped_flow_sanity":
            lines.append(
                f"- `{candidate}`: skipped by flow gate ({candidate_rows[0].get('skip_reason', candidate_rows[0].get('flow_sanity_status', ''))})"
            )
            continue
        candidate_out = out_root / candidate
        lines.append(f"- `{candidate}`: `{candidate_out.relative_to(ROOT)}`")
    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "Good Stage 3 candidates should separate no-DLVO and unfavorable residence metrics while avoiding overwhelming finite-horizon censoring in the unfavorable cases.",
            "The 100D control should reduce residence even when it samples near-surface regions frequently.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", nargs="*", default=CANDIDATES)
    parser.add_argument("--out-root", type=Path, default=OUT)
    parser.add_argument("--particles", type=int, default=1000)
    parser.add_argument("--max-time", type=float, default=90.0)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--grid-nx", type=int, default=384)
    parser.add_argument("--grid-ny", type=int, default=256)
    parser.add_argument("--matrix-bins", type=int, default=40)
    parser.add_argument("--profiles", nargs="+", default=PROFILES)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-failed-flow", action="store_true")
    args = parser.parse_args()

    resume = not args.force
    allow_failed_flow = args.allow_failed_flow
    candidates = args.candidates or CANDIDATES
    out_root = args.out_root if args.out_root.is_absolute() else ROOT / args.out_root
    summary = out_root / SUMMARY.name
    report = out_root / REPORT.name
    flow_screen = flow_screen_status()
    legacy_sanity = sanity_status()
    out_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = read_csv(summary) if summary.exists() else []
    for candidate in candidates:
        print(f"stage2 pilot: {candidate}", flush=True)
        try:
            screen_row = flow_screen.get(candidate, {})
            passive_status = str(screen_row.get("passive_gate_status", ""))
            flow_ready = truthy(screen_row.get("flow_ready", False))
            if screen_row and "flow_ready" not in screen_row:
                flow_ready = passive_status == "pass"
            if not screen_row and candidate in legacy_sanity:
                flow_ready = legacy_sanity[candidate] == "pass"
                passive_status = legacy_sanity[candidate]
            if not allow_failed_flow and screen_row and not flow_ready:
                coupling_status = str(screen_row.get("periodic_coupling_status", ""))
                if coupling_status == "fail":
                    skip_reason = "periodic coupling fail"
                elif passive_status:
                    skip_reason = f"passive gate {passive_status}"
                else:
                    skip_reason = "flow_ready false"
                rows = [
                    {
                        "candidate_id": candidate,
                        "status": "skipped_flow_sanity",
                        "flow_sanity_status": passive_status,
                        "periodic_coupling_status": coupling_status,
                        "skip_reason": skip_reason,
                        "flow_case": screen_row.get("flow_case", ""),
                    }
                ]
            elif not allow_failed_flow and not screen_row and not fallback_flow_case(candidate).exists():
                rows = [
                    {
                        "candidate_id": candidate,
                        "status": "skipped_missing_flow",
                        "flow_sanity_status": "missing_stage1_flow_screen",
                        "flow_case": str(fallback_flow_case(candidate).relative_to(ROOT)),
                    }
                ]
            else:
                flow_case = ROOT / str(screen_row.get("flow_case", "")) if screen_row.get("flow_case") else fallback_flow_case(candidate)
                rows = run_candidate(
                    candidate,
                    particles=args.particles,
                    resume=resume,
                    flow_case=flow_case,
                    out_root=out_root,
                    max_time=args.max_time,
                    dt=args.dt,
                    chunk_size=args.chunk_size,
                    grid_nx=args.grid_nx,
                    grid_ny=args.grid_ny,
                    matrix_bins=args.matrix_bins,
                    profiles=args.profiles,
                )
            all_rows = merge_candidate_rows(all_rows, rows)
            write_csv(summary, all_rows)
        except Exception as exc:
            all_rows = merge_candidate_rows(all_rows, [{"candidate_id": candidate, "status": "failed", "error": str(exc)}])
            write_csv(summary, all_rows)
            print(f"failed {candidate}: {exc}", flush=True)
    write_report(all_rows, out_root, report, args.profiles)
    print(summary)
    print(report)


if __name__ == "__main__":
    main()
