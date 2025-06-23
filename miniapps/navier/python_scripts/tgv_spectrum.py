#!/usr/bin/env python3
# compare_spectrum.py
# ---------------------------------------------------------
#  usage:  python compare_spectrum.py SampledData42.txt
# ---------------------------------------------------------
import numpy as np, matplotlib.pyplot as plt
import argparse, re, os, textwrap

def tgv_velocity(x,y,z,A=1.0):
    u =  A*np.sin(x)*np.cos(y)*np.cos(z)
    v = -A*np.cos(x)*np.sin(y)*np.cos(z)
    w =  np.zeros_like(u)
    return u,v,w

def isotropic_spectrum(u,v,w,dx,dy,dz):
    nx,ny,nz = u.shape
    N = nx*ny*nz
    Ux = np.fft.fftshift(np.fft.fftn(u))/N
    Uy = np.fft.fftshift(np.fft.fftn(v))/N
    Uz = np.fft.fftshift(np.fft.fftn(w))/N
    E_xyz = 0.5*(np.abs(Ux)**2+np.abs(Uy)**2+np.abs(Uz)**2)

    kx = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(nx,d=dx))
    ky = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(ny,d=dy))
    kz = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(nz,d=dz))
    KX,KY,KZ = np.meshgrid(kx,ky,kz,indexing='ij')
    kmag = np.sqrt(KX**2+KY**2+KZ**2).ravel()

    kmax = int(kx.max())+1
    edges = np.arange(0,kmax+1)-0.5
    Ek, _ = np.histogram(kmag,bins=edges,weights=E_xyz.ravel())
    kcent = 0.5*(edges[:-1]+edges[1:])
    return kcent, Ek                 #  <<< pure E(k)

# ---------------------------------------------------------
p = argparse.ArgumentParser(description='Compare MFEM vs analytic TGV spectrum')
p.add_argument('file',type=str)
args = p.parse_args()
fname = args.file

# -------- header -----------------
with open(fname) as f:
    hdr = [next(f) for _ in range(6)]
step = next((re.search(r'Step\s*=\s*(\d+)',l).group(1) for l in hdr if 'Step' in l),'0')
time = float(next((re.search(r'Time\s*=\s*([0-9.eE+-]+)',l).group(1) for l in hdr if 'Time' in l),'0'))

# -------- data -------------------
d = np.genfromtxt(fname,skip_header=6)
x,y,z = d[:,0],d[:,1],d[:,2]
u,v,w = d[:,3],d[:,4],d[:,5]

xr = np.unique(np.round(x,10))
yr = np.unique(np.round(y,10))
zr = np.unique(np.round(z,10))
nx,ny,nz = len(xr),len(yr),len(zr)
dx,dy,dz = xr[1]-xr[0], yr[1]-yr[0], zr[1]-zr[0]

ux = np.full((nx,ny,nz),np.nan)
vy = np.full_like(ux,np.nan); wz = np.full_like(ux,np.nan)
xi = {v:i for i,v in enumerate(xr)}
yi = {v:i for i,v in enumerate(yr)}
zi = {v:i for i,v in enumerate(zr)}
for xx,yy,zz,uu,vv,ww in zip(x,y,z,u,v,w):
    ux[xi[round(xx,10)], yi[round(yy,10)], zi[round(zz,10)]] = uu
    vy[xi[round(xx,10)], yi[round(yy,10)], zi[round(zz,10)]] = vv
    wz[xi[round(xx,10)], yi[round(yy,10)], zi[round(zz,10)]] = ww
ux = np.nan_to_num(ux); vy = np.nan_to_num(vy); wz = np.nan_to_num(wz)

tke_phys = 0.5*np.mean(ux**2+vy**2+wz**2)
print(f'TKE physical  = {tke_phys:.6e}')

k_num, Ek_num = isotropic_spectrum(ux,vy,wz,dx,dy,dz)
print(f'TKE Fourier   = {Ek_num.sum():.6e}')        # matches physical

# -------- analytic field ----------
xa = np.linspace(0,2*np.pi,nx,endpoint=False)
ya = np.linspace(0,2*np.pi,ny,endpoint=False)
za = np.linspace(0,2*np.pi,nz,endpoint=False)
Xa,Ya,Za = np.meshgrid(xa,ya,za,indexing='ij')
u0,v0,w0 = tgv_velocity(Xa,Ya,Za)
k_ic, Ek_ic = isotropic_spectrum(u0,v0,w0,xa[1]-xa[0],ya[1]-ya[0],za[1]-za[0])

# --- analytic field -------------------------------------------------
xa = np.linspace(0,2*np.pi,nx,endpoint=False)
ya = np.linspace(0,2*np.pi,ny,endpoint=False)
za = np.linspace(0,2*np.pi,nz,endpoint=False)
Xa,Ya,Za = np.meshgrid(xa,ya,za,indexing='ij')
u0,v0,w0  = tgv_velocity(Xa,Ya,Za)

k_ic, Ek_ic = isotropic_spectrum(u0,v0,w0, xa[1]-xa[0], ya[1]-ya[0], za[1]-za[0])
tke_ic      = Ek_ic.sum()                # <── analytic Fourier energy
print(f'TKE Fourier (analytic IC) = {tke_ic:.6e}')


# -------- save & plot -------------
outtxt = os.path.join(os.path.dirname(fname),f'energy_spectrum_step_{step}.txt')
np.savetxt(outtxt,np.column_stack((k_num,Ek_num)),
           header=textwrap.dedent(f'''
           k   E(k)
           Step {step}  Time {time:.5e}''').strip(),fmt='%.6e')
print('wrote',outtxt)

plt.figure(figsize=(8,6))
plt.loglog(k_ic, k_ic*Ek_ic,'k-.',label='TGV analytic')
plt.loglog(k_num,k_num*Ek_num,'b-', label=f'MFEM step {step}')
plt.loglog(k_ic, 1e-1*(k_ic/1)**(-5/3),'r--',label='k^-5/3')
plt.xlabel('wavenumber k'); plt.ylabel(r'$k\,E(k)$')
plt.title('Isotropic energy spectrum'); plt.grid(True,which='both',ls='--')
plt.legend(); plt.tight_layout(); plt.show()

