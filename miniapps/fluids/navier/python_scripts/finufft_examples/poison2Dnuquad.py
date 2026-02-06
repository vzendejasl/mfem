import numpy as np
import matplotlib.pyplot as plt
import finufft
from scipy.fft import fft2, ifft2, fftshift, ifftshift

# Source function: two bumps, zero mean
w0 = 0.1
def src(x, y):
    return np.exp(-0.5*((x-1)**2 + (y-2)**2)/w0**2) - np.exp(-0.5*((x-3)**2 + (y-5)**2)/w0**2)

# Section A: Solve -Δu = f on regular grid via FFT
print('FFT on regular grid...')
ns = np.arange(40, 121, 20)
ns = 2*np.ceil(ns/2).astype(int)  # ensure even
fft_fhat = None
x_last = None
u_last = None
for n in ns:
    x = 2*np.pi * np.arange(n) / n
    xx, yy = np.meshgrid(x, x, indexing='ij')
    f = src(xx, yy)
    fhat = ifft2(f)
    k = np.fft.fftshift(np.arange(-n//2, n//2))
    kx, ky = np.meshgrid(k, k, indexing='ij')
    kfilter = np.zeros((n, n))
    mask = (kx != 0) | (ky != 0)
    kfilter[mask] = 1.0 / (kx[mask]**2 + ky[mask]**2)
    # Zero out the mean and Nyquist frequencies
    kfilter[n//2, :] = 0
    kfilter[:, n//2] = 0
    kfilter[0, 0] = 0
    u = fft2(kfilter * fhat)
    u = np.real(u)
    print(f'n={n}:\tu(0,0) = {u[0,0]:.15e}')
    if n == ns[-1]:
        fft_fhat = fhat
        x_last = x
        u_last = u
        f_last = f

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(np.log10(np.abs(fft_fhat)), aspect='auto', origin='lower')
plt.title('FFT: log10 abs fhat')
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(f_last.T, extent=(x_last[0], x_last[-1], x_last[0], x_last[-1]), origin='lower')
plt.title('source term f')
plt.colorbar()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(f_last.T, extent=(x_last[0], x_last[-1], x_last[0], x_last[-1]), origin='lower')
plt.title('source term f')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(u_last.T, extent=(x_last[0], x_last[-1], x_last[0], x_last[-1]), origin='lower')
plt.title('FFT solution u')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.tight_layout()
plt.show()

# Section B: Solve on tensor-product nonuniform grid (with known quadrature)
print('\nNUFFT on tensor-prod NU (known-quadrature) grid...')
def mapx(t): return t + 0.5*np.sin(t)
def mapxp(t): return 1 + 0.5*np.cos(t)
def mapy(t): return t + 0.4*np.sin(2*t)
def mapyp(t): return 1 + 0.8*np.cos(2*t)
tol = 1e-12
ns = np.arange(80, 241, 40)
ns = 2*np.ceil(ns/2).astype(int)  # ensure even

for n in ns:
    t = 2*np.pi * np.arange(n) / n
    xm = mapx(t)
    ym = mapy(t)
    xw = mapxp(t)
    yw = mapyp(t)
    ww = np.outer(xw, yw) / n**2
    xx, yy = np.meshgrid(xm, ym, indexing='ij')
    f = src(xx, yy)
    if n == ns[0]:
        plt.figure()
        plt.pcolormesh(xm, ym, f.T)
        plt.title('f on mesh')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.colorbar()
        plt.show()
    Nk = int(0.5 * n)
    Nk = 2 * int(np.ceil(Nk / 2))  # ensure even
    # Type 1 NUFFT
    fhat = finufft.nufft2d1(
        xx.ravel(), yy.ravel(), (f * ww).ravel().astype(np.complex128),
        isign=1, eps=tol, n_modes=(Nk, Nk)
    )
    k = np.fft.fftshift(np.arange(-Nk//2, Nk//2))
    kx, ky = np.meshgrid(k, k, indexing='ij')
    kfilter = np.zeros((Nk, Nk))
    mask = (kx != 0) | (ky != 0)
    kfilter[mask] = 1.0 / (kx[mask]**2 + ky[mask]**2)
    kfilter[Nk//2, :] = 0
    kfilter[:, Nk//2] = 0
    kfilter[0, 0] = 0
    uhat = fhat * kfilter
    # Type 2 NUFFT
    u = finufft.nufft2d2(
        xx.ravel(), yy.ravel(), uhat.astype(np.complex128),
        isign=-1, eps=tol
    )
    u = np.real(u).reshape(n, n)
    print(f'n={n}:\tNk={Nk}\tu(0,0) = {u[0,0]:.15e}')
    if n == ns[-1]:
        xm_last, ym_last, f_last, u_last = xm, ym, f, u
        fhat_last = fhat

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(np.log10(np.abs(fhat_last)).reshape(Nk,Nk), aspect='auto', origin='lower')
plt.title('NUFFT: log10 abs fhat')
plt.colorbar()
plt.subplot(1,2,2)
plt.pcolormesh(xm_last, ym_last, f_last.T)
plt.title('source term f')
plt.axis('equal')
plt.colorbar()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.pcolormesh(xm_last, ym_last, f_last.T)
plt.title('source term f')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.colorbar()
plt.subplot(1,2,2)
plt.pcolormesh(xm_last, ym_last, u_last.T)
plt.title('NUFFT solution u')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.colorbar()
plt.tight_layout()
plt.show()

# Section C: Solve on general nonuniform grid (with Jacobian quadrature)
print('\nNUFFT on general NU (known-quadrature) grid...')
def map(t, s):
    return np.array([
        t + 0.5*np.sin(t) + 0.2*np.sin(2*s),
        s + 0.3*np.sin(2*s) + 0.3*np.sin(s-t)
    ])
def mapJ(t, s):
    # Returns 2x2 Jacobian matrix at each (t,s)
    J11 = 1 + 0.5*np.cos(t)
    J12 = 0.4*np.cos(2*s)
    J21 = -0.3*np.cos(s-t)
    J22 = 1 + 0.6*np.cos(2*s) + 0.3*np.cos(s-t)
    return np.array([[J11, J12], [J21, J22]])

ns = np.arange(80, 241, 40)
ns = 2*np.ceil(ns/2).astype(int)  # ensure even

for n in ns:
    t = 2*np.pi * np.arange(n) / n
    tt, ss = np.meshgrid(t, t, indexing='ij')
    tt_flat = tt.ravel()
    ss_flat = ss.ravel()
    mapped = map(tt_flat, ss_flat)
    xx = mapped[0, :].reshape(n, n)
    yy = mapped[1, :].reshape(n, n)
    # Compute determinant of Jacobian at each point
    J = mapJ(tt_flat, ss_flat)
    detJ = J[0,0]*J[1,1] - J[0,1]*J[1,0]
    ww = detJ / n**2
    f = src(xx, yy)
    if n == ns[0]:
        plt.figure()
        plt.pcolormesh(xx, yy, f)
        plt.title('f on mesh')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.colorbar()
        plt.show()
    Nk = int(0.5 * n)
    Nk = 2 * int(np.ceil(Nk / 2))  # ensure even
    # Type 1 NUFFT
    fhat = finufft.nufft2d1(
        xx.ravel(), yy.ravel(), (f.ravel() * ww).astype(np.complex128),
        isign=1, eps=tol, n_modes=(Nk, Nk)
    )
    k = np.fft.fftshift(np.arange(-Nk//2, Nk//2))
    kx, ky = np.meshgrid(k, k, indexing='ij')
    kfilter = np.zeros((Nk, Nk))
    mask = (kx != 0) | (ky != 0)
    kfilter[mask] = 1.0 / (kx[mask]**2 + ky[mask]**2)
    kfilter[Nk//2, :] = 0
    kfilter[:, Nk//2] = 0
    kfilter[0, 0] = 0
    uhat = fhat * kfilter
    # Type 2 NUFFT
    u = finufft.nufft2d2(
        xx.ravel(), yy.ravel(), uhat.astype(np.complex128),
        isign=-1, eps=tol
    )
    u = np.real(u).reshape(n, n)
    print(f'n={n}:\tNk={Nk}\tu(0,0) = {u[0,0]:.15e}')
    if n == ns[-1]:
        xx_last, yy_last, f_last, u_last = xx, yy, f, u
        fhat_last = fhat

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(np.log10(np.abs(fhat_last)).reshape(Nk,Nk), aspect='auto', origin='lower')
plt.title('NUFFT: log10 abs fhat')
plt.colorbar()
plt.subplot(1,2,2)
plt.pcolormesh(xx_last, yy_last, f_last)
plt.title('source term f')
plt.axis('equal')
plt.colorbar()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.pcolormesh(xx_last, yy_last, f_last)
plt.title('source term f')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.colorbar()
plt.subplot(1,2,2)
plt.pcolormesh(xx_last, yy_last, u_last)
plt.title('NUFFT solution u')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.colorbar()
plt.tight_layout()
plt.show()
