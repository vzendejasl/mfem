#!/usr/bin/env python3
# tgv_nufft_with_analytic.py
#
# Non-uniform NUFFT spectrum   +   analytic uniform-FFT reference
# ----------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import finufft, argparse, textwrap

# ---------- analytic velocity field -----------------------------------
def tgv_velocity(x, y, z, A=1.0):
    u =  A * np.sin(x) * np.cos(y) * np.cos(z)
    v = -A * np.cos(x) * np.sin(y) * np.cos(z)
    w =  np.zeros_like(u)
    return u, v, w

# ---------- tools ------------------------------------------------------
def nufft_cube(x, y, z, strengths, shape, eps=1e-6):
    F = finufft.nufft3d1(x, y, z, strengths.astype(np.complex128),
                         shape, eps=eps, isign=+1, modeord=1)
    return np.fft.fftshift(F.reshape(shape))          # no extra scaling

def iso_spectrum(E3d):
    N = E3d.shape[0]
    k = np.arange(-N//2, N//2)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    kmag  = np.sqrt(KX**2 + KY**2 + KZ**2).ravel()
    edges = np.arange(0, N) - 0.5
    kcent = 0.5*(edges[:-1] + edges[1:])
    Ek, _ = np.histogram(kmag, bins=edges, weights=E3d.ravel())
    return kcent, Ek

# ---------- CLI --------------------------------------------------------
p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent("""\
    NUFFT spectrum + analytic uniform-FFT reference for the 3-D TGV.
      --N      base lattice size (default 64)
      --jitter lattice perturbation amplitude 0–1
      --random M  use a cloud of M random points instead of a lattice
    """))
p.add_argument("--N",      type=int,   default=64)
p.add_argument("--jitter", type=float, default=0.0)
p.add_argument("--random", type=int,   default=None, metavar="M")
p.add_argument("--A",      type=float, default=1.0)
p.add_argument("--eps",    type=float, default=1e-12)
args = p.parse_args()

# ---------- build point cloud -----------------------------------------
rng = np.random.default_rng(0)
if args.random is None:
    x0 = np.linspace(-np.pi, np.pi, args.N, endpoint=False)
    dx = x0[1] - x0[0]
    X0, Y0, Z0 = np.meshgrid(x0, x0, x0, indexing="ij")
    xs, ys, zs = X0.ravel(), Y0.ravel(), Z0.ravel()
    if args.jitter > 0:
        xs += (rng.random(xs.size)-.5)*args.jitter*dx
        ys += (rng.random(xs.size)-.5)*args.jitter*dx
        zs += (rng.random(xs.size)-.5)*args.jitter*dx
        for arr in (xs, ys, zs):                  # periodic wrap
            arr[:] = np.where(arr < -np.pi, arr+2*np.pi, arr)
            arr[:] = np.where(arr >=  np.pi, arr-2*np.pi, arr)
else:
    xs = rng.uniform(-np.pi, np.pi, args.random)
    ys = rng.uniform(-np.pi, np.pi, args.random)
    zs = rng.uniform(-np.pi, np.pi, args.random)

M, V = xs.size, (2*np.pi)**3
print(f"Point cloud built:  M = {M:,d}")

# ---------- velocity & unit weights (1/M) -----------------------------
u, v, w = tgv_velocity(xs, ys, zs, args.A)
w_j      = np.full_like(xs, 1.0/M, dtype=float)      # ← unit weights
tke_avg  = 0.5*np.sum((u**2 + v**2 + w**2)*w_j)      # density
print(f"⟨KE⟩ density (MC) = {tke_avg:.6e}   analytic A²/8 = {args.A**2/8:.6e}")

# ---------- NUFFT spectrum --------------------------------------------
shape = (args.N,)*3 if args.random is None else (64,)*3
Fx = nufft_cube(xs, ys, zs, u*w_j, shape, args.eps)
Fy = nufft_cube(xs, ys, zs, v*w_j, shape, args.eps)
Fz = nufft_cube(xs, ys, zs, w*w_j, shape, args.eps)
E3d_nu = 0.5*(np.abs(Fx)**2 + np.abs(Fy)**2 + np.abs(Fz)**2)
kcent, Ek_nu = iso_spectrum(E3d_nu)
print(f"ΣE(k) NUFFT      = {E3d_nu.sum():.6e}")

# ---------- NEW SECTION: analytic uniform-FFT reference ---------------
Nref = shape[0]                       # same resolution as NUFFT cube
x_ref = np.linspace(-np.pi, np.pi, Nref, endpoint=False)
Xr, Yr, Zr = np.meshgrid(x_ref, x_ref, x_ref, indexing="ij")
ur, vr, wr = tgv_velocity(Xr, Yr, Zr, args.A)
Mref = Nref**3
# coefficients with the same 1/M weighting
Fx_ref = np.fft.fftshift(np.fft.fftn(ur)) / Mref
Fy_ref = np.fft.fftshift(np.fft.fftn(vr)) / Mref
Fz_ref = np.fft.fftshift(np.fft.fftn(wr)) / Mref
E3d_ref = 0.5*(np.abs(Fx_ref)**2 + np.abs(Fy_ref)**2 + np.abs(Fz_ref)**2)
_, Ek_ref = iso_spectrum(E3d_ref)
print(f"ΣE(k) analytic   = {E3d_ref.sum():.6e}")

# ---------- plot velocity magnitude -----------------------------------
velmag = np.sqrt(u**2 + v**2 + w**2)
fig1 = plt.figure(figsize=(8,6))
ax1 = fig1.add_subplot(111, projection="3d")
show = slice(None) if M<=60_000 else rng.choice(M, 60_000, replace=False)
sc = ax1.scatter(xs[show], ys[show], zs[show], c=velmag[show],
                 cmap="viridis", s=4, alpha=0.7)
fig1.colorbar(sc, ax=ax1, label="|u|")
ax1.set(xlabel="x", ylabel="y", zlabel="z",
        title=f"|u|  (jitter={args.jitter}, M={M:,d})")
fig1.tight_layout()

# ---------- plot spectrum ---------------------------------------------
fig2 = plt.figure(figsize=(8,6))
kfit = kcent[1:]                      # skip k=0 for guide
plt.loglog(kcent, kcent*Ek_ref, "k-.", label="analytic FFT")
plt.loglog(kcent, kcent*Ek_nu, "b-", label="NUFFT (non-uniform)")
plt.loglog(kfit, 0.1*(kfit/1)**(-5/3), "r--", label=r"$k^{-5/3}$")
plt.xlabel("wavenumber k");  plt.ylabel(r"$k\,E(k)$")
plt.title("Isotropic spectrum: NUFFT vs. analytic")
plt.grid(True, which="both", ls=":")
plt.legend();  fig2.tight_layout()

plt.show()

