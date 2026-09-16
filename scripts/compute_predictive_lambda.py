#!/usr/bin/env python3
"""Predictive angular residence number and the streamline-nesting argument (main text Sections 3.3 and 3.4).

Run from the repository root after the center/corner production suite exists:

    python scripts/compute_predictive_lambda.py

Reads  outputs/openfoam_flow/openfoam_flow_N512.npz                      (resolved center/corner flow raster)
       outputs/openfoam_full_suite/trajectory_library_refinement_<profile>.npz  (central-core production libraries)
Writes outputs/predictive_lambda/predictive_lambda.csv   (tau_K, measured residence per entry, N_e, Lambda_theta, net travel, F30)
       outputs/predictive_lambda/lambda_table.tex        (LaTeX body of the Lambda_theta table)
       outputs/predictive_lambda/streamline_offsets.csv  (outlet offset |y_out - L/2| versus release angle and release gap)

Part 1 (Lambda_theta). tau_K is the mean first-passage time from the well minimum h_min to the basin edge
h_exit (U = -kT) with the wall-corrected normal diffusivity D_perp(h) of Appendix A and a reflecting boundary
at the contact gap, Eq. (mfpt) of the main text. <|u_t|> is the circumferential mean of the tangential fluid
speed at the particle-center radius in the well, read from the raster. N_e is the measured mean number of
well entries per completed center-well release. Lambda_theta = <|u_t|> N_e tau_K / (R + a_p).

Part 2 (streamline nesting). Streamlines are integrated (RK4 on the bilinearly interpolated raster) from
release points at gap h on the center collector, for release angles 0..120 degrees from the rear stagnation
point, to the cell outlet x = L. The outlet offset from the centerline is reported for h = 200 nm and 1 um.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FLOW = ROOT / "outputs" / "openfoam_flow" / "openfoam_flow_N512.npz"
SUITE = ROOT / "outputs" / "openfoam_full_suite"
OUT = ROOT / "outputs" / "predictive_lambda"

kB = 1.380649e-23
T = 298.15
kT = kB * T
EPS = 78.5 * 8.8541878128e-12
E_CHARGE = 1.602176634e-19
NA = 6.02214076e23
AP = 0.55e-6
R = 1.0e-4
L = 4.0e-4
MU = 1e-3
ZETA_P = -0.0501
D0 = kT / (6 * np.pi * MU * AP)

# label -> (library profile, ionic strength mM, collector zeta V, Hamaker J, diffusivity factor)
CASES = {
    "6 mM, -70 mV": ("unfavorable_6mM_z70", 6, -0.070, 3.83e-21, 1),
    "20 mM, -70 mV": ("unfavorable_20mM_z70", 20, -0.070, 3.83e-21, 1),
    "50 mM, -70 mV": ("unfavorable_50mM_z70", 50, -0.070, 3.83e-21, 1),
    "50 mM, -50 mV": ("unfavorable_50mM_z50", 50, -0.050, 3.83e-21, 1),
    "50 mM, -30 mV": ("unfavorable_50mM_z30", 50, -0.030, 3.83e-21, 1),
    "100 mM, -70 mV": ("unfavorable_100mM_z70", 100, -0.070, 3.83e-21, 1),
    "100 mM, -30 mV": ("unfavorable_100mM_z30", 100, -0.030, 3.83e-21, 1),
    "50 mM, -70 mV, 100x D0": ("unfavorable_50mM_z70_100xD", 50, -0.070, 3.83e-21, 100),
}


# ----------------------------------------------------------------------------- DLVO and first-passage time
def kappa(ionic_strength_mM: float) -> float:
    ionic = ionic_strength_mM * 1e-3 * 1e3  # mol/m^3
    return np.sqrt(2 * NA * E_CHARGE**2 * ionic / (EPS * kT))


def potential(h, ionic_mM, zeta_c, hamaker):
    """Sphere-plane linearized constant-potential DLVO potential, Eq. (dlvo-potential); h in m, result in J."""
    hs = np.maximum(h, 1e-9)
    return 2 * np.pi * EPS * AP * ZETA_P * zeta_c * np.exp(-kappa(ionic_mM) * hs) - hamaker * AP / (6 * hs)


def d_perp(h):
    """Wall-corrected normal diffusivity, lubrication scaling with the 1e-4 floor (Appendix A)."""
    hs = np.maximum(h, 1e-9)
    return D0 * np.maximum(hs / (hs + AP), 1e-4)


def well_geometry(ionic_mM, zeta_c, hamaker, hmax=200e-9):
    h = np.logspace(-9, np.log10(hmax), 40000)
    u = potential(h, ionic_mM, zeta_c, hamaker) / kT
    i = int(np.argmin(u))
    inside = u <= -1.0
    j = i
    while j < h.size - 1 and inside[j + 1]:
        j += 1
    return h[i], u[i], h[j]


def mean_first_passage_time(ionic_mM, zeta_c, hamaker, diffusivity_factor=1, contact_gap=1e-9, n=60000):
    """Eq. (mfpt): tau_K = int_{hmin}^{hexit} e^{U/kT}/D_perp int_{hc}^{y} e^{-U/kT} dz dy.

    For the 100x D0 control the Brownian amplitude and thermal drift are scaled but the DLVO drift mobility is
    not, so the control samples the effective potential U/100 with diffusivity 100 D_perp.
    """
    hmin, umin, hexit = well_geometry(ionic_mM, zeta_c, hamaker)
    z = np.linspace(contact_gap, hexit, n)
    u = potential(z, ionic_mM, zeta_c, hamaker) / kT / diffusivity_factor
    u = u - u.min()
    inner = np.concatenate(([0.0], np.cumsum(0.5 * (np.exp(-u[1:]) + np.exp(-u[:-1])) * np.diff(z))))
    i0 = int(np.searchsorted(z, hmin))
    integrand = np.exp(u[i0:]) / (diffusivity_factor * d_perp(z[i0:])) * inner[i0:]
    return float(np.trapezoid(integrand, z[i0:])), hmin, umin, hexit


# ----------------------------------------------------------------------------- resolved flow
class Raster:
    def __init__(self, path: Path):
        f = np.load(path)
        self.ux, self.uy = f["ux"], f["uy"]
        self.n = self.ux.shape[0]

    def velocity(self, x, y):
        n = self.n
        gx = (x % L) / L * n - 0.5
        gy = (y % L) / L * n - 0.5
        i0 = int(np.floor(gx)) % n
        j0 = int(np.floor(gy)) % n
        tx = gx - np.floor(gx)
        ty = gy - np.floor(gy)
        i1 = (i0 + 1) % n
        j1 = (j0 + 1) % n
        w00 = (1 - tx) * (1 - ty)
        w10 = tx * (1 - ty)
        w01 = (1 - tx) * ty
        w11 = tx * ty
        vx = w00 * self.ux[i0, j0] + w10 * self.ux[i1, j0] + w01 * self.ux[i0, j1] + w11 * self.ux[i1, j1]
        vy = w00 * self.uy[i0, j0] + w10 * self.uy[i1, j0] + w01 * self.uy[i0, j1] + w11 * self.uy[i1, j1]
        return float(vx), float(vy)

    def tangential_profile(self, gap, nang=720):
        """Tangential and normal fluid speed at particle-center radius R + a_p + gap around the center collector.

        Angle 0 is the rear (downstream, +x) stagnation point; positive counterclockwise.
        """
        r = R + AP + gap
        th = np.linspace(-np.pi, np.pi, nang, endpoint=False)
        ut = np.empty(nang)
        un = np.empty(nang)
        for k, t in enumerate(th):
            vx, vy = self.velocity(L / 2 + r * np.cos(t), L / 2 + r * np.sin(t))
            ut[k] = -vx * np.sin(t) + vy * np.cos(t)
            un[k] = vx * np.cos(t) + vy * np.sin(t)
        return th, ut, un

    def streamline_to_outlet(self, x0, y0, dt=2e-3, max_steps=4_000_000):
        """RK4 streamline from (x0, y0) in the unwrapped plane until x >= L (the cell outlet). Returns y at x = L."""
        x, y = x0, y0
        for _ in range(max_steps):
            k1 = self.velocity(x, y)
            k2 = self.velocity(x + 0.5 * dt * k1[0], y + 0.5 * dt * k1[1])
            k3 = self.velocity(x + 0.5 * dt * k2[0], y + 0.5 * dt * k2[1])
            k4 = self.velocity(x + dt * k3[0], y + dt * k3[1])
            dx = dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
            dy = dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
            if x + dx >= L:
                frac = (L - x) / dx if dx > 0 else 1.0
                return y + frac * dy
            x += dx
            y += dy
        raise RuntimeError("streamline did not reach the outlet")


# ----------------------------------------------------------------------------- measured residence
def measured_residence(profile: str) -> dict:
    d = np.load(SUITE / f"trajectory_library_refinement_{profile}.npz")
    m = (d["exited"] & ~d["censored"] & (d["center_well_interceptions"] > 0)
         & (d["collector_well_exit"] == 1) & np.isfinite(d["theta_well_exit"]))
    well_time = d["well_time"][m]
    entries = np.maximum(d["well_interceptions"][m], 1)
    net = np.abs(d["well_net_angular_travel"][m])
    th = d["theta_well_exit"][m]
    abs_theta = np.degrees(np.abs(np.arctan2(np.sin(th), np.cos(th))))
    return {
        "n_released": int(m.sum()),
        "mean_residence_per_entry_s": float(np.mean(well_time / entries)),
        "mean_entries": float(np.mean(entries)),
        "median_abs_net_travel_rad": float(np.median(net)),
        "F30": float(np.mean(abs_theta <= 30.0)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raster = Raster(FLOW)

    # Part 1: Lambda_theta table
    rows = []
    for label, (profile, ionic, zeta_c, hamaker, dfac) in CASES.items():
        tau, hmin, umin, hexit = mean_first_passage_time(ionic, zeta_c, hamaker, dfac)
        _, ut, _ = raster.tangential_profile(hmin)
        ut_mean = float(np.mean(np.abs(ut)))
        ut_max = float(np.max(np.abs(ut)))
        meas = measured_residence(profile)
        lam = ut_mean * meas["mean_entries"] * tau / (R + AP)
        rows.append({"case": label, "profile": profile, "Umin_kT": umin, "hmin_nm": hmin * 1e9, "hexit_nm": hexit * 1e9,
                     "tau_K_s": tau, "ut_mean_um_s": ut_mean * 1e6, "ut_max_um_s": ut_max * 1e6,
                     "advective_time_s": (R + AP) / ut_mean, "Lambda_theta": lam, **meas})
        print(f"{label:24s} Umin={umin:6.2f} kT  tau_K={tau:9.3g} s  measured/entry={meas['mean_residence_per_entry_s']:9.3g} s  "
              f"N_e={meas['mean_entries']:5.1f}  Lambda={lam:8.3g}  |dtheta_net|={meas['median_abs_net_travel_rad']:.2f} rad  F30={meas['F30']:.3f}")
    with (OUT / "predictive_lambda.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with (OUT / "lambda_table.tex").open("w") as fh:
        for r in rows:
            fh.write(f"{r['case']} & {r['Umin_kT']:.2f} & {r['tau_K_s']:.3g} & {r['mean_residence_per_entry_s']:.3g} & "
                     f"{r['mean_entries']:.1f} & {r['Lambda_theta']:.3g} & {r['median_abs_net_travel_rad']:.2f} & {r['F30']:.3f} \\\\\n")

    # Part 2: streamline nesting (Section 3.4)
    srows = []
    print("\nstreamline outlet offsets |y_out - L/2| (um):")
    for gap in (200e-9, 1e-6):
        r0 = R + AP + gap
        offsets = []
        for angle in range(0, 121, 15):
            t = np.radians(angle)
            y_out = raster.streamline_to_outlet(L / 2 + r0 * np.cos(t), L / 2 + r0 * np.sin(t))
            off = abs(y_out - L / 2)
            offsets.append(off)
            srows.append({"release_gap_nm": gap * 1e9, "release_angle_deg": angle, "outlet_offset_um": off * 1e6})
        print(f"  gap {gap*1e9:6.0f} nm: max over 0-120 deg = {max(offsets)*1e6:.3f} um")
    with (OUT / "streamline_offsets.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(srows[0].keys()))
        w.writeheader()
        w.writerows(srows)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
