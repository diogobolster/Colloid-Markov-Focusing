#!/usr/bin/env python3
"""Stage 1 OpenFOAM flow screen for a small geometry subset.

This intentionally runs only a diverse subset of the geometry candidates.  The
goal is not final production flow; it is to identify candidate pore structures
worth expensive particle tracking.
"""

from __future__ import annotations

import csv
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GEOM_ROOT = ROOT / "outputs" / "stage1_geometry_screen" / "geometries"
OUT_ROOT = ROOT / "outputs" / "stage1_openfoam_flow"
SUMMARY = ROOT / "outputs" / "stage1_geometry_screen" / "stage1_flow_screen_summary.csv"
REPORT = ROOT / "outputs" / "stage1_geometry_screen" / "stage1_flow_screen_report.md"
CONTACT_SHEET = ROOT / "outputs" / "stage1_geometry_screen" / "stage1_flow_screen_contact_sheet.png"
SANITY_SCRIPT = ROOT / "scripts" / "run_stage2_flow_transport_sanity.py"
spec = importlib.util.spec_from_file_location("stage2_flow_transport_sanity", SANITY_SCRIPT)
sanity = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = sanity
spec.loader.exec_module(sanity)

DEFAULT_CANDIDATES = [
    "through_random_00",
    "through_few_large_grains",
    "through_tight_throats",
    "through_high_polydispersity",
    "through_pockety_high_cv",
]

FLOW_SUFFIX = "_gmsh_screen"
PASSIVE_GATE_EXIT_FRACTION = 0.5
GMSH_SIZE_WALL = "2.0e-6"
GMSH_SIZE_FAR = "5.0e-6"
GMSH_REFINE_DIST_MIN = "8.0e-6"
GMSH_REFINE_DIST_MAX = "3.5e-5"
END_TIME = "180"


def parse_report(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "cells": r"Parsed cell count: ([0-9,]+)",
        "mean_ux_m_s": r"Area-weighted mean ux: ([0-9.eE+-]+) m/s",
        "mean_uy_m_s": r"Area-weighted mean uy: ([0-9.eE+-]+) m/s",
        "mean_speed_m_s": r"Area-weighted mean speed: ([0-9.eE+-]+) m/s",
        "max_speed_m_s": r"Maximum cell speed: ([0-9.eE+-]+) m/s",
        "mean_ux_over_target": r"Mean ux / target ux: ([0-9.eE+-]+)",
        "max_speed_over_target": r"Maximum speed / target ux: ([0-9.eE+-]+)",
    }
    row: dict[str, object] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match is None:
            continue
        value = match.group(1).replace(",", "")
        row[key] = int(value) if key == "cells" else float(value)
    return row


def ami_summary(log_path: Path) -> dict[str, object]:
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    weights = [float(value) for value in re.findall(r"average:([0-9.eE+-]+)", text)]
    mins = [float(value) for value in re.findall(r"sum\(weights\) min:([0-9.eE+-]+)", text)]
    return {
        "ami_average_min": min(weights) if weights else "",
        "ami_zero_weight_faces_present": any(value == 0.0 for value in mins),
    }


def passive_gate(candidate: str, geometry_path: Path, flow_dir: Path) -> dict[str, object]:
    try:
        row = sanity.run_case(candidate, geometry_path, flow_dir)
    except Exception as exc:
        return {
            "passive_gate_status": "failed",
            "passive_gate_error": str(exc),
            "flow_ready": False,
        }
    status = str(row.get("status", "failed"))
    exit_fraction = float(row.get("exit_fraction_300s", 0.0))
    return {
        "passive_gate_status": status,
        "passive_exit_fraction_300s": exit_fraction,
        "passive_tracking_origin_shift_um": row.get("tracking_origin_shift_um", ""),
        "passive_median_exit_time_s": row.get("median_exit_time_s", ""),
        "passive_median_x_300s_um": row.get("median_x_300s_um", ""),
        "passive_max_x_300s_um": row.get("max_x_300s_um", ""),
        "flow_ready": status == "pass" and exit_fraction >= PASSIVE_GATE_EXIT_FRACTION,
    }


def finalize_promotion_status(row: dict[str, object]) -> None:
    zero_weight_ami = row.get("ami_zero_weight_faces_present") is True
    passive_ready = row.get("passive_gate_status") == "pass" and float(row.get("passive_exit_fraction_300s", 0.0)) >= PASSIVE_GATE_EXIT_FRACTION
    row["periodic_coupling_status"] = "fail" if zero_weight_ami else "pass"
    row["flow_ready"] = bool(passive_ready and not zero_weight_ami)


def run_candidate(candidate: str, force: bool = False) -> dict[str, object]:
    out_name = f"stage1_openfoam_flow/{candidate}{FLOW_SUFFIX}"
    out_dir = OUT_ROOT / f"{candidate}{FLOW_SUFFIX}"
    report_path = out_dir / "random_openfoam_flow_report.md"
    geometry_path = GEOM_ROOT / candidate / "geometry.json"
    row: dict[str, object] = {
        "candidate_id": candidate,
        "flow_case": str(out_dir.relative_to(ROOT)),
        "status": "pending",
    }
    if report_path.exists() and not force:
        row["status"] = "cached"
        row.update(parse_report(report_path))
        row.update(ami_summary(out_dir / "log.checkMesh"))
        if geometry_path.exists():
            row.update(passive_gate(candidate, geometry_path, out_dir))
            finalize_promotion_status(row)
        return row

    if not geometry_path.exists():
        row["status"] = "missing_geometry"
        row["error"] = str(geometry_path)
        return row

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_random_openfoam_flow.py"),
        "--geometry-path",
        str(geometry_path),
        "--out-name",
        out_name,
        "--mesh-backend",
        "gmsh",
        "--forcing-mode",
        "mean_velocity",
        "--gmsh-size-wall",
        GMSH_SIZE_WALL,
        "--gmsh-size-far",
        GMSH_SIZE_FAR,
        "--gmsh-refine-dist-min",
        GMSH_REFINE_DIST_MIN,
        "--gmsh-refine-dist-max",
        GMSH_REFINE_DIST_MAX,
        "--end-time",
        END_TIME,
    ]
    try:
        subprocess.run(cmd, cwd=ROOT, check=True, timeout=1800)
    except subprocess.TimeoutExpired as exc:
        row["status"] = "timeout"
        row["error"] = str(exc)
        return row
    except subprocess.CalledProcessError as exc:
        row["status"] = "failed"
        row["error"] = str(exc)
        return row

    row["status"] = "completed"
    if report_path.exists():
        row.update(parse_report(report_path))
    row.update(ami_summary(out_dir / "log.checkMesh"))
    row.update(passive_gate(candidate, geometry_path, out_dir))
    finalize_promotion_status(row)
    return row


def write_csv(rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def merge_rows(existing: list[dict[str, object]], updates: list[dict[str, object]]) -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    order: list[str] = []
    for row in existing:
        candidate = str(row.get("candidate_id", ""))
        if not candidate:
            continue
        if candidate not in merged:
            order.append(candidate)
        merged[candidate] = row
    for row in updates:
        candidate = str(row.get("candidate_id", ""))
        if not candidate:
            continue
        if candidate not in merged:
            order.append(candidate)
        merged[candidate] = row
    return [merged[candidate] for candidate in order]


def format_float(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def write_report(rows: list[dict[str, object]]) -> None:
    lines = [
        "# Stage 1 OpenFOAM Flow Screen",
        "",
        "This screen runs body-fitted Gmsh/OpenFOAM solves on a deliberately diverse subset of production-like candidate periodic packings.",
        "Each completed flow is immediately tested with deterministic passive, flux-weighted tracers before any DLVO statistics are allowed downstream.",
        "",
        "| candidate | solve status | passive gate | periodic coupling | promoted | exit fraction, 300 s | cells | mean ux / target | max speed / target | min AMI avg | zero-weight AMI faces |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {candidate_id} | {status} | {passive_gate_status} | {periodic_coupling_status} | {flow_ready} | {passive_exit_fraction_300s} | {cells} | {mean_ux_over_target} | {max_speed_over_target} | {ami_average_min} | {ami_zero_weight_faces_present} |".format(
                candidate_id=row.get("candidate_id", ""),
                status=row.get("status", ""),
                passive_gate_status=row.get("passive_gate_status", ""),
                periodic_coupling_status=row.get("periodic_coupling_status", ""),
                flow_ready=row.get("flow_ready", ""),
                passive_exit_fraction_300s=format_float(row.get("passive_exit_fraction_300s")),
                cells=row.get("cells", ""),
                mean_ux_over_target=format_float(row.get("mean_ux_over_target")),
                max_speed_over_target=format_float(row.get("max_speed_over_target")),
                ami_average_min=format_float(row.get("ami_average_min")),
                ami_zero_weight_faces_present=row.get("ami_zero_weight_faces_present", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Cases are flow-ready for Stage 2 particle pilots only when the passive gate and periodic-coupling check both pass.",
            "Solved cases that fail passive through-passage or have zero-weight AMI faces are retained as geometry/flow diagnostics, but are not promoted to DLVO particle statistics.",
            "",
            f"Contact-sheet figure: `{CONTACT_SHEET.relative_to(ROOT)}`.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def write_contact_sheet(rows: list[dict[str, object]]) -> None:
    completed = [
        row
        for row in rows
        if row.get("status") in {"completed", "cached", "cached_alias"}
        and (ROOT / str(row.get("flow_case", "")) / "random_openfoam_flow.png").exists()
    ]
    if not completed:
        return
    ncols = 2
    nrows = (len(completed) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5.4 * nrows), dpi=160)
    axes_array = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax, row in zip(axes_array, completed, strict=False):
        image_path = ROOT / str(row["flow_case"]) / "random_openfoam_flow.png"
        image = plt.imread(image_path)
        if image.ndim == 3:
            rgb = image[:, :, :3]
            mask = np.any(rgb < 0.985, axis=2)
            ys, xs = np.where(mask)
            if ys.size and xs.size:
                pad = 16
                y0 = max(0, int(ys.min()) - pad)
                y1 = min(image.shape[0], int(ys.max()) + pad)
                x0 = max(0, int(xs.min()) - pad)
                x1 = min(image.shape[1], int(xs.max()) + pad)
                image = image[y0:y1, x0:x1]
        ax.imshow(image)
        ax.set_title(str(row["candidate_id"]).replace("_", " "), fontsize=11, weight="bold")
        ax.axis("off")
    for ax in axes_array[len(completed) :]:
        ax.axis("off")
    fig.tight_layout()
    CONTACT_SHEET.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CONTACT_SHEET)
    plt.close(fig)


def main() -> None:
    force = "--force" in sys.argv
    candidates = [arg for arg in sys.argv[1:] if not arg.startswith("--")] or DEFAULT_CANDIDATES
    updates = []
    for candidate in candidates:
        print(f"flow screen: {candidate}", flush=True)
        updates.append(run_candidate(candidate, force=force))
    rows = merge_rows(read_csv(SUMMARY), updates)
    write_csv(rows)
    write_report(rows)
    write_contact_sheet(rows)
    print(SUMMARY)
    print(REPORT)
    print(CONTACT_SHEET)


if __name__ == "__main__":
    main()
