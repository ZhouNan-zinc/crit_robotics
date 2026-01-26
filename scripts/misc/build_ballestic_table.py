import math
import numpy as np
import logging
from tqdm import tqdm

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("type", type=str, choices=["small", "big"])
    parser.add_argument("velocity", type=float, nargs="?", default=None)

    parser.add_argument("--k", type=float, default=None)
    parser.add_argument("--g", type=float, default=9.79)

    parser.add_argument("--Rmax", type=float, default=20.0)
    parser.add_argument("--zmin", type=float, default=-3.0)
    parser.add_argument("--zmax", type=float, default=3.0)
    parser.add_argument("--dl", type=float, default=0.0025)

    parser.add_argument("--theta_min_deg", type=float, default=0.1)
    parser.add_argument("--theta_max_deg", type=float, default=30.0)
    parser.add_argument("--theta_samples", type=int, default=2000)

    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--save_valid_mask", action="store_true")

    args = parser.parse_args()

    if args.velocity is None:
        logging.warning("Ballestic velocity not set, assuming from ballestic type. This could cause potential danger.")
        args.velocity = 23.3 if args.type == "small" else 15.6

    if args.type == "small":
        args.d = 16.8e-3
        args.m = 3.2e-3
        k_default = 0.01485
    elif args.type == "big":
        args.d = 42.5e-3
        args.m = 44.5e-3
        k_default = 0.00683
    else:
        raise ValueError("Unsupported ballestic type.")

    if args.k is None:
        logging.warning("k not set. Using default k based on type. This may be inaccurate.")
        args.k = k_default

    if args.out is None:
        args.out = f"LUT_theta_t_RZ_{args.type}_v{args.velocity:.2f}_k{args.k:.6f}.npz"

    return args


def deriv_R(s, k, g):
    z, vx, vz, t = s
    if vx <= 1e-8:
        vx = 1e-8
    v = math.hypot(vx, vz)
    dz_dR = vz / vx
    dvx_dR = -k * v
    dvz_dR = (-k * v * vz - g) / vx
    dt_dR = 1.0 / vx
    return np.array([dz_dR, dvx_dR, dvz_dR, dt_dR], dtype=np.float64)


def rk4_step_R(s, dR, k, g):
    k1 = deriv_R(s, k, g)
    k2 = deriv_R(s + 0.5 * dR * k1, k, g)
    k3 = deriv_R(s + 0.5 * dR * k2, k, g)
    k4 = deriv_R(s + dR * k3, k, g)
    return s + (dR / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_family(v0, thetas_rad, Rgrid, k, g):
    NR = len(Rgrid)
    NT = len(thetas_rad)
    Z = np.empty((NT, NR), dtype=np.float32)
    T = np.empty((NT, NR), dtype=np.float32)
    dR = float(Rgrid[1] - Rgrid[0])

    for i, th in enumerate(tqdm(thetas_rad, desc="Simulating theta family")):
        vx0 = v0 * math.cos(float(th))
        vz0 = v0 * math.sin(float(th))
        s = np.array([0.0, vx0, vz0, 0.0], dtype=np.float64)

        Z[i, 0] = s[0]
        T[i, 0] = s[3]

        for j in range(1, NR):
            if s[1] <= 1e-5:
                Z[i, j:] = np.nan
                T[i, j:] = np.nan
                break
            s = rk4_step_R(s, dR, k, g)
            Z[i, j] = s[0]
            T[i, j] = s[3]

    return Z, T


def build_LUT_from_family_low_arc_only(Z, T, thetas_rad, zgrid, save_valid_mask=False):
    NT, NR = Z.shape
    Nz = len(zgrid)

    theta_LUT = np.full((NR, Nz), np.nan, dtype=np.float32)
    t_LUT = np.full((NR, Nz), np.nan, dtype=np.float32)

    valid_mask = None
    if save_valid_mask:
        valid_mask = np.zeros((NR, Nz), dtype=np.uint8)

    thetas_f = thetas_rad.astype(np.float32)

    for r_idx in tqdm(range(NR), desc="Building (R,z)->(theta,t) LUT (low arc)"):
        z_list = Z[:, r_idx]
        t_list = T[:, r_idx]

        mask = np.isfinite(z_list) & np.isfinite(t_list)
        if mask.sum() < 8:
            continue

        zv = z_list[mask].astype(np.float32)
        tv = t_list[mask].astype(np.float32)
        thv = thetas_f[mask].astype(np.float32)

        peak_idx = int(np.argmax(zv))

        zv_low = zv[:peak_idx + 1]
        tv_low = tv[:peak_idx + 1]
        th_low = thv[:peak_idx + 1]

        if len(zv_low) < 8:
            continue

        keep = np.ones_like(zv_low, dtype=bool)
        last = float(zv_low[0])
        for i in range(1, len(zv_low)):
            if float(zv_low[i]) <= last:
                keep[i] = False
            else:
                last = float(zv_low[i])

        zv_low = zv_low[keep]
        tv_low = tv_low[keep]
        th_low = th_low[keep]

        if len(zv_low) < 8:
            continue

        zmin = float(zv_low[0])
        zmax = float(zv_low[-1])

        inside = (zgrid >= zmin) & (zgrid <= zmax)
        if not np.any(inside):
            continue

        zq = zgrid[inside].astype(np.float32)
        theta_LUT[r_idx, inside] = np.interp(zq, zv_low, th_low).astype(np.float32)
        t_LUT[r_idx, inside] = np.interp(zq, zv_low, tv_low).astype(np.float32)

        if valid_mask is not None:
            valid_mask[r_idx, inside] = 1

    return theta_LUT, t_LUT, valid_mask


def main(args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    dl = float(args.dl)
    Rgrid = np.arange(0.0, float(args.Rmax) + dl, dl, dtype=np.float32)
    zgrid = np.arange(float(args.zmin), float(args.zmax) + dl, dl, dtype=np.float32)

    theta_min = math.radians(float(args.theta_min_deg))
    theta_max = math.radians(float(args.theta_max_deg))
    thetas = np.linspace(theta_min, theta_max, int(args.theta_samples), dtype=np.float32)

    v0 = float(args.velocity)
    k = float(args.k)
    g = float(args.g)

    logging.info(f"type={args.type}, v0={v0}, k={k}, g={g}, d={args.d}, m={args.m}")
    logging.info(f"R=[{Rgrid[0]:.4f},{Rgrid[-1]:.4f}] step={dl:.4f} N={len(Rgrid)}")
    logging.info(f"z=[{zgrid[0]:.4f},{zgrid[-1]:.4f}] step={dl:.4f} N={len(zgrid)}")
    logging.info(f"theta=[{args.theta_min_deg},{args.theta_max_deg}] deg, samples={args.theta_samples}")

    Z, T = simulate_family(v0=v0, thetas_rad=thetas, Rgrid=Rgrid, k=k, g=g)
    theta_LUT, t_LUT, valid_mask = build_LUT_from_family_low_arc_only(
        Z=Z, T=T, thetas_rad=thetas, zgrid=zgrid, save_valid_mask=args.save_valid_mask
    )

    out = args.out

    if valid_mask is None:
        np.savez_compressed(
            out,
            Rgrid=Rgrid,
            zgrid=zgrid,
            theta_LUT=theta_LUT,
            t_LUT=t_LUT,
            v0=np.float32(v0),
            k=np.float32(k),
            g=np.float32(g),
            dl=np.float32(dl),
            d=np.float32(args.d),
            m=np.float32(args.m),
            theta_min_deg=np.float32(args.theta_min_deg),
            theta_max_deg=np.float32(args.theta_max_deg),
            theta_samples=np.int32(args.theta_samples),
        )
    else:
        np.savez_compressed(
            out,
            Rgrid=Rgrid,
            zgrid=zgrid,
            theta_LUT=theta_LUT,
            t_LUT=t_LUT,
            valid_mask=valid_mask,
            v0=np.float32(v0),
            k=np.float32(k),
            g=np.float32(g),
            dl=np.float32(dl),
            d=np.float32(args.d),
            m=np.float32(args.m),
            theta_min_deg=np.float32(args.theta_min_deg),
            theta_max_deg=np.float32(args.theta_max_deg),
            theta_samples=np.int32(args.theta_samples),
        )

    finite_ratio = np.isfinite(theta_LUT).mean() * 100.0
    logging.info(f"Saved: {out}")
    logging.info(f"theta_LUT shape={theta_LUT.shape}, t_LUT shape={t_LUT.shape}, reachable={finite_ratio:.2f}%")


if __name__ == "__main__":
    args = parse_args()
    main(args)
