#!/usr/bin/env python3
# iso_turb_serial.py  –  faithful to TurboGenPY/isoturbo.py
#
# * one patch: no multiprocessing, no files
# * pure NumPy, runs in seconds for 64³
# * prints E(k) table + fitted slope

import numpy as np
import matplotlib.pyplot as plt
import argparse

# ------------- user-supplied Kolmogorov spectrum ----------------------
def spectrum_k53(k):                 # E(k) = C k^-5/3
    return k**(-5/3)

# ------------- build random mode list (exact isoturbo) ----------------
def random_modes(nmodes, k1, kN, dx, dy, dz, rng):
    # random directions and phases
    phi   = rng.uniform(0, 2*np.pi, nmodes)
    nu    = rng.random(nmodes)
    theta = np.arccos(2*nu - 1)
    psi   = rng.uniform(-np.pi/2, np.pi/2, nmodes)
    alfa  = rng.uniform(0, 2*np.pi, nmodes)

    # continuous k uniformly spaced in [k1, kN]
    dk = (kN - k1) / nmodes
    kn = k1 + np.arange(nmodes)*dk
    dkn= np.full(nmodes, dk)

    kx = np.sin(theta)*np.cos(phi)*kn
    ky = np.sin(theta)*np.sin(phi)*kn
    kz = np.cos(theta)          *kn

    # divergence-free polarisation σ = ζ × k̃
    phi1   = rng.uniform(0, 2*np.pi, nmodes)
    nu1    = rng.random(nmodes)
    theta1 = np.arccos(2*nu1 - 1)
    zx = np.sin(theta1)*np.cos(phi1)
    zy = np.sin(theta1)*np.sin(phi1)
    zz = np.cos(theta1)

    ktx = np.sin(kx*dx/2)/dx
    kty = np.sin(ky*dy/2)/dy
    ktz = np.sin(kz*dz/2)/dz

    sx =  zy*ktz - zz*kty
    sy = -(zx*ktz - zz*ktx)
    sz =  zx*kty - zy*ktx
    s_norm = np.sqrt(sx*sx + sy*sy + sz*sz)
    sx, sy, sz = sx/s_norm, sy/s_norm, sz/s_norm

    # amplitude √(E(k) Δk)
    um = np.sqrt(np.clip(spectrum_k53(kn), 0, None) * dkn)

    return kx, ky, kz, sx, sy, sz, psi, um

# ------------- velocity field on uniform grid -------------------------
def velocity_field(Nx, Ny, Nz, lx, ly, lz, modes):
    dx, dy, dz = lx/Nx, ly/Ny, lz/Nz
    xc = dx/2 + np.arange(Nx)*dx
    yc = dy/2 + np.arange(Ny)*dy
    zc = dz/2 + np.arange(Nz)*dz

    kx,ky,kz,sx,sy,sz,psi,um = modes
    u = np.zeros((Nx,Ny,Nz))
    v = np.zeros_like(u)
    w = np.zeros_like(u)

    for m in range(len(kx)):
        arg = kx[m]*xc[:,None,None] + ky[m]*yc[None,:,None] + kz[m]*zc[None,None,:] - psi[m]
        cosA = np.cos(arg)
        u += 2*um[m]*sx[m]*cosA
        v += 2*um[m]*sy[m]*cosA
        w += 2*um[m]*sz[m]*cosA
    return u,v,w

# ------------- isotropic spectrum ------------------------------------
def iso_spectrum(u,v,w):
    Nx = u.shape[0]
    Ux = np.fft.fftshift(np.fft.fftn(u))/Nx**3
    Uy = np.fft.fftshift(np.fft.fftn(v))/Nx**3
    Uz = np.fft.fftshift(np.fft.fftn(w))/Nx**3
    E3d=0.5*(np.abs(Ux)**2+np.abs(Uy)**2+np.abs(Uz)**2)

    k   = np.arange(-Nx//2, Nx//2)
    KX, KY, KZ = np.meshgrid(k,k,k,indexing='ij')
    km  = np.sqrt(KX**2+KY**2+KZ**2).ravel()
    edges = np.arange(0, Nx) - .5
    Ek,_ = np.histogram(km,bins=edges,weights=E3d.ravel())
    kcent = .5*(edges[:-1]+edges[1:])
    return kcent, Ek, E3d.sum()

# ------------- main ---------------------------------------------------
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--N", type=int, default=64, help="grid points/side")
    pa.add_argument("--nmodes", type=int, default=2000, help="# Fourier modes")
    args = pa.parse_args()

    Nx = Ny = Nz = args.N
    lx = ly = lz = 2*np.pi
    dx = lx/Nx; dy = ly/Ny; dz = lz/Nz

    rng = np.random.default_rng(0)
    modes = random_modes(args.nmodes, k1=1.0, kN=args.N/2-1,
                         dx=dx, dy=dy, dz=dz, rng=rng)

    print("…building velocity field")
    u,v,w = velocity_field(Nx,Ny,Nz,lx,ly,lz,modes)

    k,Ek,Esum = iso_spectrum(u,v,w)
    ke_avg = 0.5*np.mean(u**2+v**2+w**2)

    print(f"\nPhysical ⟨KE⟩  = {ke_avg:.6e}")
    print(f"Spectral ΣE(k) = {Esum:.6e}")

    print("\nFirst 10 shells:")
    for kk,Ekx in zip(k[1:11],Ek[1:11]):
        print(f"  k={int(kk):2d}   E(k)={Ekx:.3e}")

    # slope fit on k=4…N/3
    mask = (k>=4)&(k<=args.N//3)&(Ek>0)
    slope,_ = np.polyfit(np.log10(k[mask]), np.log10(Ek[mask]), 1)
    print(f"\nLeast-squares slope ≈ {slope:.3f}  (target −1.667)")

    plt.figure(figsize=(6,5))
    plt.loglog(k,Ek,'b-',label='E(k)')
    plt.loglog(k[1:],0.5*k[1:]**(-5/3),'k--',label=r'$k^{-5/3}$')
    plt.xlabel('k'); plt.ylabel('E(k)')
    plt.grid(True,which='both',ls=':')
    plt.legend(); plt.tight_layout(); plt.show()

# === add at the bottom of iso_turb_serial.py ==========================
if __name__ == "__main__":
    from finufft import nufft3d1
    Nx = Ny = Nz = 64
    lx = ly = lz = 2*np.pi
    rng = np.random.default_rng(1)

    # 1) build lattice-compatible field (integer k’s)
    Ux,Uy,Uz = build_field(Nx, A=1.0, kmin=1, seed=0)   # <-- integer shells
    u,v,w    = ifft_vec(Ux,Uy,Uz)

    # ---------- helper to bin spectrum --------------------------------
    def spectrum_from_cube(Fx,Fy,Fz):
        E3d = 0.5*(np.abs(Fx)**2+np.abs(Fy)**2+np.abs(Fz)**2)
        k   = np.arange(-Nx//2,Nx//2)
        KX,KY,KZ = np.meshgrid(k,k,k,indexing='ij')
        Km  = np.sqrt(KX**2+KY**2+KZ**2).ravel()
        edges = np.arange(0,Nx)-.5
        Ek,_ = np.histogram(Km,bins=edges,weights=E3d.ravel())
        kcent = .5*(edges[:-1]+edges[1:])
        msk = (kcent>=4)&(kcent<=Nx/3)&(Ek>0)
        slope,_ = np.polyfit(np.log10(kcent[msk]),np.log10(Ek[msk]),1)
        return Ek.sum(), slope

    # ---------- (A) plain FFT on uniform grid -------------------------
    Fx = np.fft.fftshift(np.fft.fftn(u))/Nx**3
    Fy = np.fft.fftshift(np.fft.fftn(v))/Nx**3
    Fz = np.fft.fftshift(np.fft.fftn(w))/Nx**3
    Einv, slope_fft = spectrum_from_cube(Fx,Fy,Fz)
    print(f"\n[Uniform FFT]     ΣE={Einv:.6f},  slope={slope_fft:.3f}")

    # ---------- (B) jittered grid + NUFFT -----------------------------
    dx = lx/Nx
    X,Y,Z = np.meshgrid(np.linspace(0,lx,Nx,endpoint=False),
                        np.linspace(0,ly,Ny,endpoint=False),
                        np.linspace(0,lz,Nz,endpoint=False),
                        indexing='ij')
    X += (rng.random(X.shape)-.5)*0.5*dx
    Y += (rng.random(Y.shape)-.5)*0.5*dx
    Z += (rng.random(Z.shape)-.5)*0.5*dx
    xs,ys,zs = X.ravel(),Y.ravel(),Z.ravel()
    wj = np.full_like(xs,1.0/xs.size)

    def coeff(comp):
        return np.fft.fftshift(
            nufft3d1(xs,ys,zs,(comp.ravel()*wj).astype(np.complex128),
                     (Nx,Ny,Nz), eps=1e-12, isign=+1, modeord=1)
        ).reshape((Nx,Ny,Nz))
    Fx,Fy,Fz = coeff(u), coeff(v), coeff(w)
    Einv, slope_jit = spectrum_from_cube(Fx,Fy,Fz)
    print(f"[50%-jitter NUFFT] ΣE={Einv:.6f},  slope={slope_jit:.3f}")

    # ---------- (C) random cloud + NUFFT ------------------------------
    M = 500_000
    xs = rng.uniform(0,lx,M); ys = rng.uniform(0,ly,M); zs = rng.uniform(0,lz,M)
    ux = nufft3d2(xs,ys,zs,Ux, isign=-1, eps=1e-12).real
    vy = nufft3d2(xs,ys,zs,Uy, isign=-1, eps=1e-12).real
    wz = nufft3d2(xs,ys,zs,Uz, isign=-1, eps=1e-12).real
    wj = np.full(M, 1.0/M)
    Fx = nufft3d1(xs,ys,zs,(ux*wj).astype(np.complex128),(Nx,Ny,Nz),
                  eps=1e-12,isign=+1,modeord=1).reshape((Nx,Ny,Nz))
    Fy = nufft3d1(xs,ys,zs,(vy*wj).astype(np.complex128),(Nx,Ny,Nz),
                  eps=1e-12,isign=+1,modeord=1).reshape((Nx,Ny,Nz))
    Fz = nufft3d1(xs,ys,zs,(wz*wj).astype(np.complex128),(Nx,Ny,Nz),
                  eps=1e-12,isign=+1,modeord=1).reshape((Nx,Ny,Nz))
    Fx = np.fft.fftshift(Fx); Fy=np.fft.fftshift(Fy); Fz=np.fft.fftshift(Fz)
    Einv, slope_rand = spectrum_from_cube(Fx,Fy,Fz)
    print(f"[Random-cloud NUFFT] ΣE={Einv:.6f}, slope={slope_rand:.3f}")
    

