#!/usr/bin/env python3
import numpy as np
import finufft

# ---------- Helper: unstructured points & synthetic field ----------
def make_unstructured_points(M, L=1.0, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.random((M, 3)) * L
    w_x = np.full(M, L**3 / M)     # simple MC weights
    return X, w_x

def synthetic_velocity(X, L=1.0):
    two_pi = 2.0 * np.pi
    x, y, z = (X[:,0]/L, X[:,1]/L, X[:,2]/L)
    u = np.empty((X.shape[0], 3), dtype=float)
    u[:,0] = np.sin(two_pi*(2*x + 1*y + 0*z)) + 0.5*np.cos(two_pi*(3*x))
    u[:,1] = np.cos(two_pi*(0*x + 2*y + 1*z)) + 0.5*np.sin(two_pi*(3*y))
    u[:,2] = np.sin(two_pi*(1*x + 0*y + 2*z)) + 0.5*np.cos(two_pi*(3*z))
    return u

# ---------- FINUFFT mode indices (no fftshift needed) ----------
def finufft_mode_indices(N):
    m = N // 2
    return np.arange(-m, m + (N % 2), dtype=int)

# ---------- Build theta coords and weights (θ-space) ----------
def to_theta_and_wtheta(X, w_x, L):
    two_pi = 2.0 * np.pi
    theta = two_pi * (X / L) - np.pi     # map to [-pi, pi)
    x_ang = np.asarray(theta[:,0], dtype=np.float64, order='C')
    y_ang = np.asarray(theta[:,1], dtype=np.float64, order='C')
    z_ang = np.asarray(theta[:,2], dtype=np.float64, order='C')
    w_theta = w_x * (two_pi / L)**3      # Jacobian
    return x_ang, y_ang, z_ang, w_theta

# ---------- Linear operators: A (modes→points), A* (points→modes) ----------
def A_apply(x_ang, y_ang, z_ang, modes, Nxyz, isign=+1, eps=1e-12):
    # Type-2: uniform modes -> nonuniform values
    return finufft.nufft3d2(x_ang, y_ang, z_ang, modes, isign=isign, eps=eps)

def AH_apply(x_ang, y_ang, z_ang, vals, Nxyz, isign=-1, eps=1e-12):
    # Type-1: nonuniform values -> uniform modes (adjoint)
    return finufft.nufft3d1(x_ang, y_ang, z_ang, vals, Nxyz, isign=isign, eps=eps)

# ---------- CG solve for (A* W A) m = A* W u ----------
def cg_normal_eq(x_ang, y_ang, z_ang, w_theta, rhs, Nxyz, tol=1e-10, maxit=30):
    """
    Solve for 'modes' minimizing ||W^(1/2)(A modes - u)||_2.
    rhs = A* (W u).
    Matvec: v -> A* (W (A v)).
    """
    modes = np.zeros(Nxyz, dtype=np.complex128)
    r = rhs - AH_apply(x_ang, y_ang, z_ang, w_theta * A_apply(x_ang, y_ang, z_ang, modes, Nxyz), Nxyz)
    p = r.copy()
    rr_old = np.vdot(r, r)

    for _ in range(maxit):
        Ap = AH_apply(x_ang, y_ang, z_ang, w_theta * A_apply(x_ang, y_ang, z_ang, p, Nxyz), Nxyz)
        denom = np.vdot(p, Ap)
        alpha = rr_old / (denom if np.abs(denom) > 1e-300 else 1e-300)
        modes = modes + alpha * p
        r = r - alpha * Ap
        rr_new = np.vdot(r, r)
        if np.sqrt(rr_new.real) < tol:
            break
        beta = rr_new / (rr_old if rr_old.real > 1e-300 else 1e-300)
        p = r + beta * p
        rr_old = rr_new
    return modes

# ---------- Isotropic PSD ----------
def isotropic_psd(Uhatx, Uhaty, Uhatz, L=1.0, k_units="cycles/length"):
    N1, N2, N3 = Uhatx.shape
    k1 = finufft_mode_indices(N1)
    k2 = finufft_mode_indices(N2)
    k3 = finufft_mode_indices(N3)
    K1, K2, K3 = np.meshgrid(k1, k2, k3, indexing='ij')
    K = np.sqrt(K1**2 + K2**2 + K3**2, dtype=float)
    E_mode = 0.5 * (np.abs(Uhatx)**2 + np.abs(Uhaty)**2 + np.abs(Uhatz)**2)
    Kmax = int(np.ceil(K.max()))
    shell = np.clip(np.floor(K).astype(int), 0, Kmax)
    counts = np.bincount(shell.ravel(), minlength=Kmax+1)
    sums   = np.bincount(shell.ravel(), weights=E_mode.ravel(), minlength=Kmax+1)
    E_shell = np.zeros(Kmax+1, dtype=float)
    np.divide(sums, counts, out=E_shell, where=counts>0)
    k_shell = (np.arange(Kmax+1) + 0.5).astype(float)
    if k_units == "cycles/length":
        k_shell = k_shell / L
    elif k_units == "radians/length":
        k_shell = (2*np.pi) * k_shell / L
    else:
        raise ValueError("k_units must be 'cycles/length' or 'radians/length'")
    return k_shell, E_shell, counts

# ---------- Main ----------
if __name__ == "__main__":
    # Sizes
    N1 = N2 = N3 = 64
    Nxyz = (N1, N2, N3)
    M = 200_000
    L = 1.0

    print(f"Generating {M} unstructured points in a {L}×{L}×{L} periodic box...")
    X, w_x = make_unstructured_points(M, L=L, seed=1)
    u = synthetic_velocity(X, L=L)

    # Physical KE
    KE_phys = 0.5 * np.sum(w_x * np.sum(u**2, axis=1))
    print(f"Physical-space KE ≈ {KE_phys:.6e}")

    # θ-coords & weights
    x_ang, y_ang, z_ang, w_theta = to_theta_and_wtheta(X, w_x, L)

    # Right-hand sides: rhs_c = A* (W u_c)
    print(f"Solving LSQ projection with CG (tol=1e-10, maxit=20)...")
    rhs_x = AH_apply(x_ang, y_ang, z_ang, (w_theta * u[:,0]).astype(np.complex128), Nxyz)
    rhs_y = AH_apply(x_ang, y_ang, z_ang, (w_theta * u[:,1]).astype(np.complex128), Nxyz)
    rhs_z = AH_apply(x_ang, y_ang, z_ang, (w_theta * u[:,2]).astype(np.complex128), Nxyz)

    mx = cg_normal_eq(x_ang, y_ang, z_ang, w_theta, rhs_x, Nxyz, tol=1e-10, maxit=20)
    my = cg_normal_eq(x_ang, y_ang, z_ang, w_theta, rhs_y, Nxyz, tol=1e-10, maxit=20)
    mz = cg_normal_eq(x_ang, y_ang, z_ang, w_theta, rhs_z, Nxyz, tol=1e-10, maxit=20)

    # IMPORTANT: LSQ modes are already the Fourier series coefficients g_k (no extra scaling!)
    Uhatx, Uhaty, Uhatz = mx, my, mz

    # Parseval KE: ∫|u|^2 dx = L^3 * ∑|g_k|^2
    KE_spec = 0.5 * (L**3) * (
        np.sum(np.abs(Uhatx)**2) +
        np.sum(np.abs(Uhaty)**2) +
        np.sum(np.abs(Uhatz)**2)
    )
    rel = (KE_spec - KE_phys) / max(KE_phys, 1e-300)
    print(f"Spectral-space KE ≈ {KE_spec:.6e}")
    print(f"Relative KE error = {rel:+.3e}")

    # PSD
    k_shell, E_shell, counts = isotropic_psd(Uhatx, Uhaty, Uhatz, L=L, k_units="cycles/length")
    print("\nIsotropic PSD E(k) (first 12 shells):")
    for i in range(min(12, len(k_shell))):
        print(f"k ~ {k_shell[i]:6.3f}  |  E(k) = {E_shell[i]:.6e}  (modes: {counts[i]})")

    # # Optional quick plot:
    # import matplotlib.pyplot as plt
    # mask = counts > 0
    # plt.loglog(k_shell[mask][1:], E_shell[mask][1:], marker='o')
    # plt.xlabel("k (cycles/length)"); plt.ylabel("E(k)")
    # plt.title("Isotropic PSD from NUFFT (3D, LSQ projection)")
    # plt.grid(True, which='both'); plt.show()
