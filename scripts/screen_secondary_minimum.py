#!/usr/bin/env python3
"""Screen implemented unfavorable DLVO parameters for secondary-minimum depth."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colloid_tsm.physical import PhysicalParams


KB = 1.380649e-23
H = np.logspace(-9, -6, 50_000)


def metrics(params: PhysicalParams) -> tuple[float, float, float, float, float, int]:
    eps0 = 8.8541878128e-12
    eps = params.relative_permittivity * eps0
    kappa = 1.0 / params.debye_length
    c_edl = (
        2.0
        * np.pi
        * eps
        * params.particle_radius
        * kappa
        * params.zeta_particle
        * params.zeta_collector_unfavorable
    )
    potential = c_edl / kappa * np.exp(-kappa * H) - params.hamaker * params.particle_radius / (6.0 * H)
    force = c_edl * np.exp(-kappa * H) - params.hamaker * params.particle_radius / (6.0 * H * H)
    sign_changes = int(np.sum(np.diff(np.sign(force)) != 0))
    min_id = int(np.argmin(potential))
    max_id = int(np.argmax(potential))
    thermal = KB * params.temperature
    return (
        params.debye_length * 1.0e9,
        float(potential[min_id] / thermal),
        float(H[min_id] * 1.0e9),
        float(potential[max_id] / thermal),
        float(H[max_id] * 1.0e9),
        sign_changes,
    )


def main() -> None:
    print("I_mM zc_mV A_J Debye_nm Umin_kBT hmin_nm Umax_kBT hmax_nm roots")
    for ionic_strength_m_molar in (1, 3, 6, 10, 20, 50, 100, 200):
        for zeta_collector_m_v in (-70, -50, -30, -20):
            params = replace(
                PhysicalParams(),
                ionic_strength_molar=ionic_strength_m_molar / 1000.0,
                zeta_collector_unfavorable=zeta_collector_m_v / 1000.0,
            )
            debye_nm, umin, hmin_nm, umax, hmax_nm, roots = metrics(params)
            print(
                f"{ionic_strength_m_molar:4.0f} "
                f"{zeta_collector_m_v:6.0f} "
                f"{params.hamaker:.1e} "
                f"{debye_nm:8.2f} "
                f"{umin:9.2f} "
                f"{hmin_nm:8.2f} "
                f"{umax:10.1f} "
                f"{hmax_nm:8.2f} "
                f"{roots}"
            )


if __name__ == "__main__":
    main()
