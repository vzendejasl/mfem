// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. All Rights reserved.
// See files LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// Single-domain Navier shear-layer demo (2D/3D via -dim, default 2):
// - Programmatic rectangular mesh (channel)
// - Inflow: hyperbolic tangent shear profile
// - Top/bottom: free-slip walls (v=0, tangential velocity unconstrained)
// - Outflow: natural/Neumann (do not mark as essential)
// - VisIt output (VisItDataCollection)
// - Scalar AMR is supported. With -amr -pa, flow operators use partial
//   assembly while scalar DG transport uses full assembly: the current DG
//   face partial-assembly path is not safe to rebuild on a nonconforming mesh.
//
// Example runs:
// 2D (default):
//   mpirun -n 4 ./navier_shear_layer -no-vis -visit -dc 10 -dt 0.001 -Re 50 -o 1 -eps 0.005 -pa
// 2D with more frequent snapshots:
//   mpirun -n 4 ./navier_shear_layer -no-vis -visit -dc 0 -dt 0.001 -Re 50 -o 1
// 2D with stronger perturbation + faster forcing:
//   mpirun -n 4 ./navier_shear_layer -no-vis -visit -eps 0.02 -nper 8
// 2D high Re with tighter shear layer:
//   mpirun -n 4 ./navier_shear_layer -no-vis -visit -dc 100 -dt 0.001 -Re 2000 -o 2 -eps 0.08 -pa -delta 0.01 -tf 8
// 3D (spanwise modes, periodic in z):
//   mpirun -n 8 ./navier_shear_layer -dim 3 -nx 64 -ny 32 -nz 32 -lx 4 -ly 1 -lz 1 -nmodes 3 -nper 4 -no-vis -visit
// mpirun -n 8 ./navier_shear_layer -dim 3 -nx 32 -ny 16 -nz 8 -lx 4 -ly 1 -lz 0.5 -nmodes 3 -nper 4 -no-vis -visit -dt 0.001 -tf 1.0 -Re 2000 -pa -delta 0.01 -eps 0.08
// mpirun -n 8 ./navier_shear_layer -dim 3 -nx 128 -ny 32 -nz 16 -lx 4 -ly 1 -lz 0.5 -nmodes 3 -nper 4 -no-vis -visit -dt 0.0005 -tf 8.0 -Re 2000 -pa -delta 0.01 -eps 0.08
//
// 2D AMR smoke test (vorticity/enstrophy marker with passive-dye transfer):
//   mpirun -n 2 ./navier_shear_layer -dim 2 -nx 16 -ny 8 -o 2 -Re 100 \
//     -dt 0.001 -tf 0.02 -amr -amr-t 0.005 -amr-th 0.5 \
//     -amr-check -amr-rebalance -no-vis -visit -dc 5 \
//     -run shear_layer_amr_smoke
// 2D AMR using the dye mixing band 0.3 <= c <= 0.7:
//   mpirun -n 2 ./navier_shear_layer -dim 2 -nx 16 -ny 8 -o 2 -Re 100 \
//     -dt 0.001 -tf 0.02 -amr -amr-t 0.005 -amr-dye-mixing \
//     -amr-dye-min 0.3 -amr-dye-max 0.7 -amr-check -amr-rebalance \
//     -no-vis -visit -dc 5 -run shear_layer_dye_amr
// Simple moving dye AMR: no DOF/event/marking caps; dye range is 0.3 <= c <= 0.8.
//   mpirun -n 2 ./navier_shear_layer -dim 2 -nx 16 -ny 8 -o 2 -Re 500 \
//     -dt 0.001 -tf 3.0 -pa -amr-dye-dynamic -amr-t 0.005 \
//     -amr-max-level 1 -amr-check -amr-rebalance -no-vis -visit -dc 5 \
//     -run shear_layer_dye_dynamic -nper 4 -delta 0.01 -eps 0.08
// Add -amr-dye-enstrophy to also refine high-enstrophy vortex regions.
// Moving 2D scalar-gradient AMR (unlimited events, bounded mesh growth):
//   mpirun -n 2 ./navier_shear_layer -dim 2 -nx 16 -ny 8 -o 2 -Re 500 \
//     -dt 0.0005 -tf 3.0 -pa -amr -amr-t 0.005 -amr-i 25 \
//     -amr-max-events 0 -amr-max-marked 32 -amr-max-dofs 100000 \
//     -amr-dye-gradient -amr-dye-grad-th 0.5 \
//     -amr-dye-grad-coarsen-th 0.1 -amr-dye-grad-inlet-buffer 0.75 -amr-coarsen \
//     -amr-check -amr-rebalance -no-vis -visit -dc 5 \
//     -run shear_layer_moving_amr -nmodes 3 -nper 4 -delta 0.01 -eps 0.08

// 3D AMR using a broad dye mixing band for this coarse smoke mesh:
//   mpirun -n 2 ./navier_shear_layer -dim 3 -nx 8 -ny 4 -nz 4 \
//     -lx 4 -ly 1 -lz 0.5 -o 2 -Re 100 -dt 0.0005 -tf 0.005 \
//     -amr -amr-t 0.002 -amr-dye-mixing -amr-dye-min 0.05 -amr-dye-max 0.95 \
//     -amr-check -amr-rebalance -no-per-z -no-vis -visit -dc 5 \
//     -run shear_layer_3d_dye_amr

#include "mfem.hpp"
#include "navier_solver.hpp"
#include "navier_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cerrno>
#include <iostream>
#include <memory>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <vector>
#include <cstdint>

using namespace mfem;
using namespace navier;

// -----------------------------------------------------------------------------
// Minimal Navier context + accessors so that navier_utils.cpp
// (checkpointing / sampling helpers) can link against this miniapp.
// -----------------------------------------------------------------------------
struct s_NavierContext
{
   int    ref_levels                 = 0;
   int    element_subdivisions       = 0; // (serial) not used here except as nx
   int    element_subdivisions_parallel = 0; // not used here
   int    order                      = 2;
   real_t reynum                     = 100.0;
   real_t kinvis                     = 1.0 / reynum;
   real_t t_final                    = 1.0;
   real_t dt                         = 0.01;
   bool   pa                         = false;
   bool   ni                         = false;
   bool   visualization              = true; // GLVis
   bool   checkres                   = false;
   int    num_pts                    = 0;
   bool   visit                      = true;  // <-- default ON for this demo
   bool   paraview                   = false;
   bool   binary                     = false; // VisItDataCollection uses this
   bool   conduit                    = false;
   bool   restart                    = false;
   int    element_center_cycle       = 0;
   int    data_dump_cycle            = 10;     // write VisIt every N steps
   bool   filter                     = false;
   bool   oversample                 = false;
   real_t alpha                      = 0.0;
   bool   amr                        = false;
   real_t amr_time                   = 0.01;
   real_t amr_threshold              = 0.5;
   bool   amr_dye_mixing             = false;
   real_t amr_dye_min                = 0.3;
   real_t amr_dye_max                = 0.8;
   bool   amr_dye_dynamic            = false;
   bool   amr_dye_enstrophy          = false;
   int    amr_max_level              = 0;
   bool   amr_dye_gradient           = false;
   real_t amr_dye_gradient_threshold = 0.5;
   real_t amr_dye_gradient_coarsen_threshold = 0.1;
   real_t amr_dye_gradient_inlet_buffer = 0.0;
   bool   amr_coarsen                = false;
   int    amr_interval               = 0;
   int    amr_max_events             = 1;
   int    amr_max_dofs               = 0;
   int    amr_max_marked_global      = 0;
   bool   amr_preserve_history       = true;
   bool   amr_rebalance              = false;
   bool   amr_check                  = false;
   std::string run_name              = "navier_shear_layer";


} ctx;

static int    g_dim   = 2;    // 2D or 3D
static real_t g_eps   = 0.02; // perturbation amplitude (try 0.01–0.05)
static int    g_k     = 2;    // number of waves in x (2D only)
static real_t g_sigma = 0.10; // y-localization width (absolute in y-units)
static int    g_nper  = 4;    // oscillations per convective time (Lx/Uc)
static int    g_nmodes = 1;   // spanwise Fourier modes (3D only)
static real_t g_phi0  = 0.0;  // base phase (radians)
static bool   g_per_z = true; // periodic in z (3D only)

static bool   g_use_scalar = true; // DG passive scalar on by default; use -no-scalar to disable
static real_t g_scalar_kappa = -1.0; // < 0 => use nu
static real_t g_scalar_top = 1.0;
static real_t g_scalar_bottom = 0.0;
static real_t g_scalar_delta = -1.0; // < 0 => use velocity shear thickness
static int    g_scalar_order = -1; // < 0 => use velocity order
static bool   g_scalar_limit = true; // bound-preserving DG limiter
static bool   g_scalar_cg = false;    // use explicit H1/CG scalar instead of DG
static bool   g_scalar_supg = true;   // CG-only streamline stabilization
static real_t g_scalar_supg_c = 1.0; // CG-only SUPG strength

// Optional turbulence-like inflow perturbation: deterministic pseudo-random,
// periodic in time, localized around the shear layer, and vanishing at the
// top/bottom walls. Disabled by default so the old inflow path remains intact.
static bool   g_turb_inflow = false;
static int    g_noise_modes = 8;
static int    g_noise_seed  = 12345;
static real_t g_noise_period = -1.0; // < 0 => use convective time Lx/Uc
static real_t g_eps_u = -1.0;        // < 0 => 0.25*g_eps
static real_t g_eps_w = -1.0;        // < 0 => g_eps (3D only)

// Accessors expected by navier_utils.[hpp,cpp].
bool   GetVisit(const s_NavierContext *c) { return c && c->visit; }
bool   GetConduit(const s_NavierContext *c) { return c && c->conduit; }
real_t GetReynum(const s_NavierContext *c) { return c ? c->reynum : 0.0; }
int    GetNumPts(const s_NavierContext *c) { return c ? c->num_pts : 0; }
int    GetElementSubdivisions(const s_NavierContext *c)
{ return c ? c->element_subdivisions : 0; }
int    GetElementSubdivisionsParallel(const s_NavierContext *c)
{ return c ? c->element_subdivisions_parallel : 0; }
int    GetOrder(const s_NavierContext *c) { return c ? c->order : 1; }
real_t GetKinvis(const s_NavierContext *c) { return c ? c->kinvis : 0.0; }
bool   GetPA(const s_NavierContext *c) { return c && c->pa; }
bool   GetNI(const s_NavierContext *c) { return c && c->ni; }
real_t GetDt(const s_NavierContext *c) { return c ? c->dt : 0.0; }
bool   GetOverSample(const s_NavierContext *c) { return c && c->oversample; }
bool   GetFilter(const s_NavierContext *c) { return c && c->filter; }
real_t GetAlpha(const s_NavierContext *c) { return c ? c->alpha : 0.0; }

// -----------------------------------------------------------------------------
// Shear-layer boundary data (set from main).
// -----------------------------------------------------------------------------
static real_t g_xmin   = 0.0;
static real_t g_xmax   = 4.0;
static real_t g_ymin   = 0.0;
static real_t g_ymax   = 1.0;
static real_t g_zmin   = 0.0;
static real_t g_zmax   = 1.0;

static real_t g_Ubot   = 1.0;
static real_t g_Utop   = 3.0;
static real_t g_delta  = 0.05; // shear thickness (absolute in y-units)

// In MFEM cartesian meshes:
// 2D: 1=bottom, 2=right, 3=top, 4=left
// 3D: 1=zmin, 2=ymin, 3=xmax, 4=ymax, 5=xmin, 6=zmax
enum BdrAttr2D
{
   BDR_BOTTOM = 1,
   BDR_RIGHT  = 2,
   BDR_TOP    = 3,
   BDR_LEFT   = 4
};
enum BdrAttr3D
{
   BDR_ZMIN = 1,
   BDR_YMIN = 2,
   BDR_XMAX = 3,
   BDR_YMAX = 4,
   BDR_XMIN = 5,
   BDR_ZMAX = 6
};

static inline real_t shear_profile_u(real_t y)
{
   // u(y) = Umean + 0.5*(Utop-Ubot)*tanh((y-y0)/delta)
   const real_t y0    = 0.5*(g_ymin + g_ymax);
   const real_t Umean = 0.5*(g_Utop + g_Ubot);
   const real_t dU    = (g_Utop - g_Ubot);
   const real_t arg   = (y - y0) / g_delta;
   return Umean + 0.5*dU*std::tanh(arg);
}

static inline real_t inflow_convective_velocity()
{
   return 0.5 * (g_Utop + g_Ubot);
}

static inline real_t omega_from_convective()
{
   const real_t Lx = g_xmax - g_xmin;
   const real_t Uc = inflow_convective_velocity();
   return (g_nper > 0 && Lx > 0.0 && Uc != 0.0)
          ? (2.0*M_PI * real_t(g_nper) * Uc / Lx)
          : 0.0;
}

static inline real_t wall_envelope(real_t y)
{
   const real_t Ly = g_ymax - g_ymin;
   if (Ly <= 0.0) { return 0.0; }
   const real_t eta = (y - g_ymin) / Ly;
   return 4.0 * eta * (1.0 - eta);
}


static inline real_t shear_envelope(real_t y)
{
   const real_t y0 = 0.5*(g_ymin + g_ymax);
   const real_t gauss = std::exp(-std::pow((y - y0)/g_sigma, 2));
   return gauss * wall_envelope(y);
}

static inline real_t inflow_repeat_period()
{
   if (g_noise_period > 0.0) { return g_noise_period; }
   const real_t Lx = g_xmax - g_xmin;
   const real_t Uc = inflow_convective_velocity();
   return (Lx > 0.0 && Uc != 0.0) ? (Lx / std::abs(Uc)) : 1.0;
}

static inline std::uint64_t mix_bits(std::uint64_t x)
{
   x ^= x >> 30;
   x *= 0xbf58476d1ce4e5b9ULL;
   x ^= x >> 27;
   x *= 0x94d049bb133111ebULL;
   x ^= x >> 31;
   return x;
}

static inline real_t hash_unit(int a, int b, int c, int d)
{
   std::uint64_t x = 0x9e3779b97f4a7c15ULL;
   x ^= mix_bits(static_cast<std::uint64_t>(g_noise_seed + 0x9e37 * (a + 17)));
   x ^= mix_bits(static_cast<std::uint64_t>(b + 131 * (c + 17)));
   x ^= mix_bits(static_cast<std::uint64_t>(d + 911));
   const std::uint64_t mantissa = mix_bits(x) & ((1ULL << 53) - 1ULL);
   return real_t(mantissa) / real_t(1ULL << 53);
}

static inline real_t hash_signed(int a, int b, int c, int d)
{
   return 2.0 * hash_unit(a, b, c, d) - 1.0;
}

static inline real_t periodic_random_series(real_t xi_hat, real_t eta,
                                           real_t zeta, int component)
{
   const bool use_z = (g_dim == 3) && (g_zmax > g_zmin);
   if (g_noise_modes <= 0) { return 0.0; }

   real_t sum = 0.0;
   real_t norm = 0.0;
   const real_t two_pi = 2.0 * M_PI;

   for (int m = 1; m <= g_noise_modes; ++m)
   {
      const real_t amp = hash_signed(component, m, 0, 1);
      const int kx = 1 + ((3 * m + component) % std::max(1, g_noise_modes));
      const int ky = 1 + ((5 * m + component + 1) % std::max(1, g_noise_modes));
      const real_t phi = two_pi * hash_unit(component, m, 0, 2);
      real_t phase = two_pi * (real_t(kx) * xi_hat + real_t(ky) * eta) + phi;
      if (use_z)
      {
         const int kz = 1 + ((7 * m + component + 2) % std::max(1, g_nmodes));
         phase += two_pi * real_t(kz) * zeta;
      }
      sum += amp * std::sin(phase);
      norm += std::abs(amp);
   }

   return (norm > 0.0) ? (sum / norm) : 0.0;
}

static inline real_t turbulence_like_noise(int component, real_t x, real_t y,
                                           real_t z, real_t t)
{
   const real_t Ly = g_ymax - g_ymin;
   if (Ly <= 0.0) { return 0.0; }

   const real_t T = inflow_repeat_period();
   const real_t Uc = inflow_convective_velocity();
   const real_t Lc = (T > 0.0) ? std::abs(Uc) * T : 0.0;
   const real_t xi = (x - g_xmin) - Uc * t;
   const real_t xi_hat = (Lc > 0.0) ? (xi / Lc) : 0.0;
   const real_t eta = (y - g_ymin) / Ly;

   real_t zeta = 0.0;
   if (g_dim == 3)
   {
      const real_t Lz = g_zmax - g_zmin;
      zeta = (Lz > 0.0) ? ((z - g_zmin) / Lz) : 0.0;
   }

   return shear_envelope(y) * periodic_random_series(xi_hat, eta, zeta, component);
}

static inline real_t perturb_uy_2d(real_t x, real_t y, real_t t)
{
   const real_t y0 = 0.5*(g_ymin + g_ymax);
   const real_t phase = 2.0*M_PI*real_t(g_k) * (x - g_xmin) / (g_xmax - g_xmin);
   const real_t gauss = std::exp(-std::pow((y - y0)/g_sigma, 2));
   const real_t omega = omega_from_convective();
   return g_eps * std::sin(phase - omega*t) * gauss * wall_envelope(y);
}

static inline real_t spanwise_mode_sum(real_t z, real_t t)
{
   if (g_nmodes <= 0) { return 0.0; }
   const real_t Lz = g_zmax - g_zmin;
   const real_t omega = omega_from_convective();
   real_t sum = 0.0;
   const int nm = g_nmodes;
   for (int n = 1; n <= nm; ++n)
   {
      const real_t alpha = (Lz > 0.0) ? (2.0*M_PI * real_t(n) / Lz) : 0.0;
      const real_t phase = alpha * (z - g_zmin) + real_t(n) * g_phi0;
      sum += std::cos(phase + omega*t);
   }
   return sum / real_t(nm);
}

static inline real_t perturb_uy_3d(real_t y, real_t z, real_t t)
{
   const real_t y0 = 0.5*(g_ymin + g_ymax);
   const real_t gauss = std::exp(-std::pow((y - y0)/g_sigma, 2));
   return g_eps * spanwise_mode_sum(z, t) * gauss * wall_envelope(y);
}

void vel_inflow(const Vector &x, real_t t, Vector &u)
{
   u.SetSize(g_dim);
   u = 0.0;

   const real_t eps_u = (g_eps_u >= 0.0) ? g_eps_u : (0.25 * g_eps);
   const real_t eps_v = g_eps;
   const real_t eps_w = (g_eps_w >= 0.0) ? g_eps_w : g_eps;

   if (!g_turb_inflow)
   {
      u(0) = shear_profile_u(x(1));
      if (g_dim == 2)
      {
         u(1) = perturb_uy_2d(x(0), x(1), t);
      }
      else
      {
         u(1) = perturb_uy_3d(x(1), x(2), t);
      }
      return;
   }

   const real_t z = (g_dim == 3) ? x(2) : 0.0;
   const real_t nu = turbulence_like_noise(/*component=*/0, x(0), x(1), z, t);
   const real_t nv = turbulence_like_noise(/*component=*/1, x(0), x(1), z, t);
   const real_t nw = turbulence_like_noise(/*component=*/2, x(0), x(1), z, t);

   u(0) = shear_profile_u(x(1)) + eps_u * nu;
   u(1) = eps_v * nv;
   if (g_dim == 3)
   {
      u(2) = eps_w * nw;
   }
}


real_t zero_scalar(const Vector &, real_t)
{
   return 0.0;
}

static inline real_t scalar_profile(real_t y)
{
   const real_t y0 = 0.5 * (g_ymin + g_ymax);
   const real_t delta = (g_scalar_delta > 0.0) ? g_scalar_delta : g_delta;
   const real_t cmean = 0.5 * (g_scalar_top + g_scalar_bottom);
   const real_t dc = (g_scalar_top - g_scalar_bottom);
   const real_t arg = (delta > 0.0) ? ((y - y0) / delta) : 0.0;
   return cmean + 0.5 * dc * std::tanh(arg);
}

real_t scalar_inflow(const Vector &x)
{
   return scalar_profile(x(1));
}

real_t scalar_ic(const Vector &x)
{
   return scalar_profile(x(1));
}

// Initial condition: start from the inflow shear profile everywhere (simple)
void vel_ic(const Vector &x, Vector &u)
{
   u.SetSize(g_dim);
   u = 0.0;
   u(0) = shear_profile_u(x(1));
   if (g_dim > 1)
   {
      u(1) = 0.0;
   }
   if (g_dim > 2)
   {
      u(2) = 0.0;
   }
}



static void ComputeQCriterion(ParGridFunction &u, ParGridFunction &q)
{
   FiniteElementSpace *v_fes = u.FESpace();
   FiniteElementSpace *fes = q.FESpace();

   Array<int> zones_per_vdof(fes->GetVSize());
   zones_per_vdof = 0;
   q = 0.0;

   int elndofs;
   Array<int> v_dofs, dofs;
   Vector vals, loc_data;
   const int vdim = v_fes->GetVDim();
   DenseMatrix grad_hat, dshape, grad;

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      fes->GetElementVDofs(e, dofs);
      v_fes->GetElementVDofs(e, v_dofs);
      u.GetSubVector(v_dofs, loc_data);
      vals.SetSize(dofs.Size());
      ElementTransformation *tr = fes->GetElementTransformation(e);
      const FiniteElement *el = fes->GetFE(e);
      elndofs = el->GetDof();
      const int dim = el->GetDim();
      dshape.SetSize(elndofs, dim);

      for (int dof = 0; dof < elndofs; ++dof)
      {
         const IntegrationPoint &ip = el->GetNodes().IntPoint(dof);
         tr->SetIntPoint(&ip);

         el->CalcDShape(tr->GetIntPoint(), dshape);
         grad_hat.SetSize(vdim, dim);
         DenseMatrix loc_data_mat(loc_data.GetData(), elndofs, vdim);
         MultAtB(loc_data_mat, dshape, grad_hat);

         const DenseMatrix &Jinv = tr->InverseJacobian();
         grad.SetSize(grad_hat.Height(), Jinv.Width());
         Mult(grad_hat, Jinv, grad);

         real_t q_val = 0.0;
         if (dim == 2)
         {
            q_val = -0.5 * (((grad(0, 0))*(grad(0, 0))) + ((grad(1, 1))*(grad(1, 1))))
                    - grad(0, 1) * grad(1, 0);
         }
         else
         {
            q_val = -0.5 * (((grad(0, 0))*(grad(0, 0))) + ((grad(1, 1))*(grad(1, 1))) + ((grad(2, 2))*(grad(2, 2))))
                    - grad(0, 1) * grad(1, 0) - grad(0, 2) * grad(2, 0)
                    - grad(1, 2) * grad(2, 1);
         }
         vals(dof) = q_val;
      }

      for (int j = 0; j < dofs.Size(); j++)
      {
         const int ldof = dofs[j];
         q(ldof) += vals[j];
         zones_per_vdof[ldof]++;
      }
   }

   GroupCommunicator &gcomm = q.ParFESpace()->GroupComm();
   gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
   gcomm.Bcast(zones_per_vdof);
   gcomm.Reduce<real_t>(q.GetData(), GroupCommunicator::Sum);
   gcomm.Bcast<real_t>(q.GetData());

   for (int i = 0; i < q.Size(); i++)
   {
      const int nz = zones_per_vdof[i];
      if (nz) { q(i) /= nz; }
   }
}

struct MarkCandidate
{
   real_t priority;
   int element;
};

// On this Cartesian shear-layer mesh, isotropic refinement reduces element
// volume by 2^dim. This provides a simple global AMR-depth cap without using
// a DOF budget or a per-event marking cap.
static bool IsBelowAMRLevel(ParMesh &pmesh, int element,
                            real_t root_element_volume,
                            int max_level)
{
   if (max_level <= 0) { return true; }
   const int dim = pmesh.Dimension();
   const real_t finest_allowed_volume = root_element_volume
                                        / std::pow(real_t(2), dim * max_level);
   return pmesh.GetElementVolume(element)
          > finest_allowed_volume * (real_t(1.0) + real_t(1e-12));
}

// Select the same globally highest-priority candidates on every rank, then
// return only the local element indices owned by this rank. This makes the
// AMR cap exact rather than a per-rank approximation.
static Array<int> SelectGlobalCandidates(const std::vector<MarkCandidate> &local,
                                         int global_limit,
                                         MPI_Comm comm)
{
   int rank = 0;
   int nranks = 0;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &nranks);

   const int local_count = static_cast<int>(local.size());
   std::vector<int> counts(nranks), displacements(nranks);
   MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);

   int global_count = 0;
   for (int r = 0; r < nranks; ++r)
   {
      displacements[r] = global_count;
      global_count += counts[r];
   }

   std::vector<real_t> local_priorities(local_count);
   std::vector<int> local_elements(local_count);
   for (int i = 0; i < local_count; ++i)
   {
      local_priorities[i] = local[i].priority;
      local_elements[i] = local[i].element;
   }

   std::vector<real_t> all_priorities(global_count);
   std::vector<int> all_elements(global_count);
   MPI_Allgatherv(local_priorities.data(), local_count,
                  MPITypeMap<real_t>::mpi_type, all_priorities.data(),
                  counts.data(), displacements.data(),
                  MPITypeMap<real_t>::mpi_type, comm);
   MPI_Allgatherv(local_elements.data(), local_count, MPI_INT,
                  all_elements.data(), counts.data(), displacements.data(),
                  MPI_INT, comm);

   struct GlobalCandidate
   {
      real_t priority;
      int rank;
      int element;
   };
   std::vector<GlobalCandidate> all_candidates;
   all_candidates.reserve(global_count);
   for (int r = 0; r < nranks; ++r)
   {
      for (int i = 0; i < counts[r]; ++i)
      {
         const int j = displacements[r] + i;
         all_candidates.push_back({all_priorities[j], r, all_elements[j]});
      }
   }
   std::sort(all_candidates.begin(), all_candidates.end(),
             [](const GlobalCandidate &a, const GlobalCandidate &b)
   {
      if (a.priority != b.priority) { return a.priority > b.priority; }
      if (a.rank != b.rank) { return a.rank < b.rank; }
      return a.element < b.element;
   });

   const int selected_count = global_limit > 0
                              ? std::min(global_limit, global_count)
                              : global_count;
   Array<int> selected;
   for (int i = 0; i < selected_count; ++i)
   {
      if (all_candidates[i].rank == rank)
      {
         selected.Append(all_candidates[i].element);
      }
   }
   return selected;
}

// Elementwise, volume-normalized enstrophy:
//   int_K |curl(u)|^2 / |K|.
// This is evaluated directly from the velocity gradient so the AMR marker
// remains valid before and after a nonconforming mesh update.
static void ComputeElementEnstrophy(ParGridFunction &u, Vector &indicator)
{
   FiniteElementSpace *fes = u.FESpace();
   const int dim = fes->GetMesh()->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3,
               "The shear-layer AMR marker requires a 2D or 3D mesh.");
   MFEM_VERIFY(fes->GetVDim() == dim,
               "The shear-layer AMR marker requires a physical velocity field.");

   indicator.SetSize(fes->GetNE());
   Array<int> vdofs;
   Vector local_u;
   DenseMatrix dshape, grad_hat, grad;

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      const FiniteElement *fe = fes->GetFE(e);
      const int ndof = fe->GetDof();
      const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                2 * fe->GetOrder() + 2);
      fes->GetElementVDofs(e, vdofs);
      u.GetSubVector(vdofs, local_u);
      DenseMatrix u_dofs(local_u.GetData(), ndof, fes->GetVDim());
      ElementTransformation *T = fes->GetElementTransformation(e);
      dshape.SetSize(ndof, dim);

      real_t enstrophy = 0.0;
      real_t volume = 0.0;
      for (int q = 0; q < ir.GetNPoints(); ++q)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T->SetIntPoint(&ip);
         fe->CalcDShape(ip, dshape);
         grad_hat.SetSize(fes->GetVDim(), dim);
         MultAtB(u_dofs, dshape, grad_hat);
         grad.SetSize(fes->GetVDim(), dim);
         Mult(grad_hat, T->InverseJacobian(), grad);

         real_t omega2 = 0.0;
         if (dim == 2)
         {
            const real_t omega = grad(1, 0) - grad(0, 1);
            omega2 = omega * omega;
         }
         else
         {
            const real_t omega_x = grad(2, 1) - grad(1, 2);
            const real_t omega_y = grad(0, 2) - grad(2, 0);
            const real_t omega_z = grad(1, 0) - grad(0, 1);
            omega2 = omega_x * omega_x + omega_y * omega_y
                     + omega_z * omega_z;
         }
         const real_t weight = ip.weight * T->Weight();
         enstrophy += weight * omega2;
         volume += weight;
      }
      indicator(e) = volume > 0.0 ? enstrophy / volume : 0.0;
   }
}

static void ComputeVorticity(NavierSolver &flowsolver,
                             ParGridFunction &u,
                             ParGridFunction &w)
{
   if (u.ParFESpace()->GetParMesh()->Dimension() == 2)
   {
      flowsolver.ComputeCurl2D(u, w);
   }
   else
   {
      flowsolver.ComputeCurl3D(u, w);
   }
}

// Mark an element when its sampled dye range overlaps the requested mixing
// band. For the default 0 <= dye <= 1, [0.3, 0.7] tracks the interface rather
// than either pure stream.
static void ComputeDyeMixingCandidates(ParGridFunction &dye,
                                        real_t lower,
                                        real_t upper,
                                        std::vector<MarkCandidate> &candidates)
{
   FiniteElementSpace *fes = dye.FESpace();
   candidates.clear();

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      const FiniteElement *fe = fes->GetFE(e);
      const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                2 * fe->GetOrder() + 2);
      ElementTransformation *T = fes->GetElementTransformation(e);
      real_t dye_min = std::numeric_limits<real_t>::infinity();
      real_t dye_max = -std::numeric_limits<real_t>::infinity();

      for (int q = 0; q < ir.GetNPoints(); ++q)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T->SetIntPoint(&ip);
         const real_t dye_value = dye.GetValue(*T, ip);
         dye_min = std::min(dye_min, dye_value);
         dye_max = std::max(dye_max, dye_value);
      }

      if (dye_max >= lower && dye_min <= upper)
      {
         // Prefer elements whose sampled dye range has the greatest overlap
         // with the requested mixing band. This preserves the sharpest/more
         // strongly mixed portions when a global marking cap is active.
         const real_t overlap = std::min(dye_max, upper)
                                - std::max(dye_min, lower);
         candidates.push_back({std::max(overlap, real_t(0.0)), e});
      }
   }
}

// Elementwise, volume-normalized scalar-gradient magnitude:
//   int_K |grad(c)|^2 / |K|.
// Unlike a broad dye-value band, this follows the sharp scalar sheets that
// wrap around the shear-layer rollers.
static void ComputeElementDyeGradient(ParGridFunction &dye, Vector &indicator)
{
   FiniteElementSpace *fes = dye.FESpace();
   const int dim = fes->GetMesh()->Dimension();
   indicator.SetSize(fes->GetNE());

   Array<int> vdofs;
   Vector local_dye;
   DenseMatrix dshape, dye_dofs, grad_hat, grad;
   for (int e = 0; e < fes->GetNE(); ++e)
   {
      const FiniteElement *fe = fes->GetFE(e);
      const int ndof = fe->GetDof();
      const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                2 * fe->GetOrder() + 2);
      fes->GetElementVDofs(e, vdofs);
      dye.GetSubVector(vdofs, local_dye);
      dye_dofs.UseExternalData(local_dye.GetData(), ndof, 1);
      ElementTransformation *T = fes->GetElementTransformation(e);
      dshape.SetSize(ndof, dim);

      real_t gradient_energy = 0.0;
      real_t volume = 0.0;
      for (int q = 0; q < ir.GetNPoints(); ++q)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T->SetIntPoint(&ip);
         fe->CalcDShape(ip, dshape);
         grad_hat.SetSize(1, dim);
         MultAtB(dye_dofs, dshape, grad_hat);
         grad.SetSize(1, dim);
         Mult(grad_hat, T->InverseJacobian(), grad);

         real_t gradient_squared = 0.0;
         for (int d = 0; d < dim; ++d)
         {
            gradient_squared += grad(0, d) * grad(0, d);
         }
         const real_t weight = ip.weight * T->Weight();
         gradient_energy += weight * gradient_squared;
         volume += weight;
      }
      indicator(e) = volume > 0.0 ? gradient_energy / volume : 0.0;
   }
}

// The prescribed inflow profile remains sharp for the whole calculation and
// otherwise dominates a global gradient threshold. Zero its contribution in
// a user-selected buffer so the adaptive budget can follow downstream rollers.
static void MaskDyeGradientInletBuffer(ParGridFunction &dye,
                                       real_t inlet_buffer,
                                       Vector &indicator)
{
   if (inlet_buffer <= 0.0) { return; }

   FiniteElementSpace *fes = dye.FESpace();
   Vector center(fes->GetMesh()->SpaceDimension());
   for (int e = 0; e < fes->GetNE(); ++e)
   {
      const FiniteElement *fe = fes->GetFE(e);
      ElementTransformation *T = fes->GetElementTransformation(e);
      T->Transform(Geometries.GetCenter(fe->GetGeomType()), center);
      if (center(0) < g_xmin + inlet_buffer)
      {
         indicator(e) = 0.0;
      }
   }
}

// Use a binary keep/error field for derefinement. With the max aggregation
// below, a refined parent is coarsened only when every child is outside the
// dye mixing band, providing hysteresis against immediate refine/coarsen
// oscillation at an active interface.
static void ComputeDyeMixingKeepError(ParGridFunction &dye,
                                       real_t lower,
                                       real_t upper,
                                       Array<real_t> &element_error)
{
   FiniteElementSpace *fes = dye.FESpace();
   element_error.SetSize(fes->GetNE());

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      const FiniteElement *fe = fes->GetFE(e);
      const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                2 * fe->GetOrder() + 2);
      ElementTransformation *T = fes->GetElementTransformation(e);
      real_t dye_min = std::numeric_limits<real_t>::infinity();
      real_t dye_max = -std::numeric_limits<real_t>::infinity();

      for (int q = 0; q < ir.GetNPoints(); ++q)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T->SetIntPoint(&ip);
         const real_t dye_value = dye.GetValue(*T, ip);
         dye_min = std::min(dye_min, dye_value);
         dye_max = std::max(dye_max, dye_value);
      }
      element_error[e] = (dye_max >= lower && dye_min <= upper) ? 1.0 : 0.0;
   }
}



static void ComputeWallDiagnostics(ParGridFunction &u,
                                   int attr,
                                   real_t &max_abs_un,
                                   real_t &l2_un,
                                   real_t &max_ut)
{
   ParFiniteElementSpace *pfes = u.ParFESpace();
   ParMesh *pmesh = pfes->GetParMesh();
   const int dim = pmesh->Dimension();

   real_t local_max_abs_un = 0.0;
   real_t local_int_un2 = 0.0;
   real_t local_int_ds = 0.0;
   real_t local_max_ut = 0.0;

   Vector uval(dim), nor(dim);
   for (int be = 0; be < pfes->GetNBE(); ++be)
   {
      if (pmesh->GetBdrAttribute(be) != attr) { continue; }

      const FiniteElement *fe = pfes->GetBE(be);
      ElementTransformation *T = pfes->GetBdrElementTransformation(be);
      const int intorder = std::max(2, 2 * fe->GetOrder() + 2);
      const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(), intorder);

      for (int j = 0; j < ir.GetNPoints(); ++j)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         T->SetIntPoint(&ip);
         u.GetVectorValue(*T, ip, uval);

         CalcOrtho(T->Jacobian(), nor);
         const real_t nor_norm = nor.Norml2();
         if (nor_norm <= 0.0) { continue; }
         nor /= nor_norm;

         const real_t un = uval * nor;
         const real_t ut2 = std::max(real_t(0.0), uval * uval - un * un);
         const real_t ut = std::sqrt(ut2);
         const real_t ds = ip.weight * T->Weight();

         local_max_abs_un = std::max(local_max_abs_un, std::abs(un));
         local_int_un2 += ds * un * un;
         local_int_ds += ds;
         local_max_ut = std::max(local_max_ut, ut);
      }
   }

   MPI_Allreduce(&local_max_abs_un, &max_abs_un, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MAX, pmesh->GetComm());
   MPI_Allreduce(&local_int_un2, &l2_un, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_SUM, pmesh->GetComm());
   real_t global_int_ds = 0.0;
   MPI_Allreduce(&local_int_ds, &global_int_ds, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_SUM, pmesh->GetComm());
   MPI_Allreduce(&local_max_ut, &max_ut, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MAX, pmesh->GetComm());

   l2_un = (global_int_ds > 0.0) ? std::sqrt(l2_un / global_int_ds) : 0.0;
}

static void PrintSlipWallDiagnostics(ParGridFunction &u,
                                     int top_attr,
                                     int bottom_attr,
                                     int step,
                                     real_t t)
{
   real_t top_max_abs_un = 0.0, top_l2_un = 0.0, top_max_ut = 0.0;
   real_t bot_max_abs_un = 0.0, bot_l2_un = 0.0, bot_max_ut = 0.0;

   ComputeWallDiagnostics(u, top_attr, top_max_abs_un, top_l2_un, top_max_ut);
   ComputeWallDiagnostics(u, bottom_attr, bot_max_abs_un, bot_l2_un, bot_max_ut);

   const int myid = u.ParFESpace()->GetParMesh()->GetMyRank();
   if (myid == 0)
   {
      std::cout << "[BC debug] step " << step
                << "  t = " << t
                << "  dim = " << g_dim
                << "\n"
                << "  top wall    : max|u.n| = " << top_max_abs_un
                << "  L2(u.n) = " << top_l2_un
                << "  max|u_t| = " << top_max_ut << "\n"
                << "  bottom wall : max|u.n| = " << bot_max_abs_un
                << "  L2(u.n) = " << bot_l2_un
                << "  max|u_t| = " << bot_max_ut << "\n";
   }
}



class StreamlineDiffusionMatrixCoefficient : public MatrixCoefficient
{
private:
   VectorGridFunctionCoefficient &vel_coeff;
   real_t c_supg;
   real_t small;

public:
   StreamlineDiffusionMatrixCoefficient(int dim,
                                        VectorGridFunctionCoefficient &vel_coeff_,
                                        real_t c_supg_)
      : MatrixCoefficient(dim),
        vel_coeff(vel_coeff_),
        c_supg(c_supg_),
        small(1.0e-14)
   { }

   void Eval(DenseMatrix &M, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      Vector u(GetVDim());
      vel_coeff.Eval(u, T, ip);

      const real_t umag = u.Norml2();
      const real_t h = 2.0 * std::pow(std::abs(T.Weight()), 1.0 / GetVDim());

      M.SetSize(GetVDim());
      M = 0.0;

      if (umag <= small || h <= small || c_supg <= 0.0)
      {
         return;
      }

      const real_t tau = c_supg * h / (2.0 * umag + small);
      for (int i = 0; i < GetVDim(); ++i)
      {
         for (int j = 0; j < GetVDim(); ++j)
         {
            M(i, j) = tau * u(i) * u(j);
         }
      }
   }
};

static void ComputeScalarTransportCFL(const ParFiniteElementSpace &fes,
                                      const ParGridFunction &velocity,
                                      int order,
                                      real_t diff_kappa,
                                      real_t dt,
                                      real_t &adv_cfl,
                                      real_t &diff_cfl)
{
   ParMesh *pmesh = fes.GetParMesh();
   real_t local_adv = 0.0;
   real_t local_diff = 0.0;
   Array<int> verts;
   Vector vel(g_dim);
   const real_t pfac = real_t(2 * order + 1);

   for (int e = 0; e < pmesh->GetNE(); ++e)
   {
      pmesh->GetElementVertices(e, verts);
      Vector xmin(g_dim), xmax(g_dim);
      xmin = std::numeric_limits<real_t>::infinity();
      xmax = -std::numeric_limits<real_t>::infinity();
      for (int j = 0; j < verts.Size(); ++j)
      {
         const real_t *vx = pmesh->GetVertex(verts[j]);
         for (int d = 0; d < g_dim; ++d)
         {
            xmin(d) = std::min(xmin(d), vx[d]);
            xmax(d) = std::max(xmax(d), vx[d]);
         }
      }
      real_t hmin = std::numeric_limits<real_t>::infinity();
      for (int d = 0; d < g_dim; ++d)
      {
         const real_t hd = xmax(d) - xmin(d);
         if (hd > 0.0) { hmin = std::min(hmin, hd); }
      }
      if (!std::isfinite(hmin) || hmin <= 0.0) { continue; }

      ElementTransformation *T = fes.GetElementTransformation(e);
      const IntegrationPoint &ip = Geometries.GetCenter(fes.GetFE(e)->GetGeomType());
      velocity.GetVectorValue(*T, ip, vel);
      const real_t speed = vel.Norml2();

      local_adv = std::max(local_adv, dt * pfac * speed / hmin);
      if (diff_kappa > 0.0)
      {
         local_diff = std::max(local_diff,
                               dt * diff_kappa * pfac * pfac / (hmin * hmin));
      }
   }

   MPI_Allreduce(&local_adv, &adv_cfl, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MAX, pmesh->GetComm());
   MPI_Allreduce(&local_diff, &diff_cfl, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MAX, pmesh->GetComm());
}

class PassiveScalarSolverBase
{
public:
   virtual ~PassiveScalarSolverBase() = default;
   virtual void Initialize() = 0;
   virtual void Step(real_t dt) = 0;
   virtual void PrepareForMeshChange() = 0;
   virtual void UpdateAfterMeshChange() = 0;
   virtual ParGridFunction *GetField() = 0;
   virtual void ComputeMinMaxMass(real_t &global_min,
                                  real_t &global_max,
                                  real_t &global_mass) const = 0;
   virtual void PrintDiagnostics(int step, real_t t, real_t dt) const = 0;
   virtual const char *DiscName() const = 0;
};

class DGScalarTransportOperator : public TimeDependentOperator
{
private:
   ParFiniteElementSpace &fes;
   ParGridFunction &velocity;
   FunctionCoefficient inflow_coeff;
   VectorGridFunctionCoefficient vel_coeff;
   Array<int> inflow_marker;
   real_t diff_kappa;
   real_t sigma;
   real_t ip_penalty;
   bool use_pa;

   ParBilinearForm mass_form;
   std::unique_ptr<ParBilinearForm> adv_form;
   std::unique_ptr<ParBilinearForm> diff_form;

   OperatorHandle M_op;
   OperatorHandle Adv_op;
   OperatorHandle Diff_op;

   Solver *M_prec;
   mutable CGSolver M_solver;
   mutable Vector z;
   mutable Vector z2;
   mutable Vector rhs;
   Vector adv_rhs;
   Vector diff_rhs;

   void BuildMassOperator()
   {
      if (use_pa)
      {
         mass_form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      }
      mass_form.AddDomainIntegrator(new MassIntegrator());
      mass_form.Assemble();

      M_op.SetType(mass_form.GetAssemblyLevel() == AssemblyLevel::LEGACY ?
                   Operator::Hypre_ParCSR : Operator::ANY_TYPE);
      Array<int> empty;
      mass_form.FormSystemMatrix(empty, M_op);

      if (mass_form.GetAssemblyLevel() == AssemblyLevel::LEGACY)
      {
         HypreParMatrix &M_mat = *M_op.As<HypreParMatrix>();
         M_prec = new HypreSmoother(M_mat, HypreSmoother::Jacobi);
         M_solver.SetPreconditioner(*M_prec);
      }
      else
      {
         Array<int> empty;
         M_prec = new OperatorJacobiSmoother(mass_form, empty);
         M_solver.SetPreconditioner(*M_prec);
      }

      M_solver.SetOperator(*M_op);
      M_solver.iterative_mode = false;
      M_solver.SetRelTol(1e-10);
      M_solver.SetAbsTol(0.0);
      M_solver.SetMaxIter(200);
      M_solver.SetPrintLevel(0);
   }

   void BuildDiffusionOperator()
   {
      diff_rhs.SetSize(height);
      diff_rhs = 0.0;
      if (diff_kappa <= 0.0) { return; }

      diff_form = std::make_unique<ParBilinearForm>(&fes);
      if (use_pa)
      {
         diff_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      }

      ConstantCoefficient diff_coeff(diff_kappa);
      diff_form->AddDomainIntegrator(new DiffusionIntegrator(diff_coeff));
      diff_form->AddInteriorFaceIntegrator(
         new DGDiffusionIntegrator(diff_coeff, sigma, ip_penalty));
      diff_form->AddBdrFaceIntegrator(
         new DGDiffusionIntegrator(diff_coeff, sigma, ip_penalty),
         inflow_marker);
      diff_form->Assemble();
      diff_form->Finalize();

      Diff_op.SetType(diff_form->GetAssemblyLevel() == AssemblyLevel::LEGACY ?
                      Operator::Hypre_ParCSR : Operator::ANY_TYPE);
      Array<int> empty;
      diff_form->FormSystemMatrix(empty, Diff_op);

      ParLinearForm diff_lf(&fes);
      diff_lf.AddBdrFaceIntegrator(
         new DGDirichletLFIntegrator(inflow_coeff, diff_coeff,
                                     sigma, ip_penalty),
         inflow_marker);
      diff_lf.Assemble();
      diff_rhs = diff_lf;
   }

public:
   DGScalarTransportOperator(ParFiniteElementSpace &fes_,
                             ParGridFunction &velocity_,
                             const Array<int> &inflow_attr,
                             real_t kappa,
                             int order,
                             bool pa)
      : TimeDependentOperator(fes_.GetTrueVSize(), 0.0),
        fes(fes_),
        velocity(velocity_),
        inflow_coeff(scalar_inflow),
        vel_coeff(&velocity_),
        inflow_marker(inflow_attr),
        diff_kappa(kappa),
        sigma(-1.0),
        ip_penalty((order + 1) * (order + 1)),
        use_pa(pa),
        mass_form(&fes_),
        M_prec(nullptr),
        M_solver(fes_.GetComm()),
        z(height), z2(height), rhs(height)
   {
      BuildMassOperator();
      BuildDiffusionOperator();
      UpdateVelocityDependentTerms();
   }

   ~DGScalarTransportOperator() override
   {
      delete M_prec;
   }

   void UpdateVelocityDependentTerms()
   {
      adv_form = std::make_unique<ParBilinearForm>(&fes);
      if (use_pa)
      {
         adv_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      }
      constexpr real_t alpha = -1.0;
      adv_form->AddDomainIntegrator(new ConvectionIntegrator(vel_coeff, alpha));
      adv_form->AddInteriorFaceIntegrator(
         new NonconservativeDGTraceIntegrator(vel_coeff, alpha));
      adv_form->AddBdrFaceIntegrator(
         new NonconservativeDGTraceIntegrator(vel_coeff, alpha));
      adv_form->Assemble();
      adv_form->Finalize();

      Adv_op.SetType(adv_form->GetAssemblyLevel() == AssemblyLevel::LEGACY ?
                     Operator::Hypre_ParCSR : Operator::ANY_TYPE);
      Array<int> empty;
      adv_form->FormSystemMatrix(empty, Adv_op);

      ParLinearForm adv_lf(&fes);
      adv_lf.AddBdrFaceIntegrator(
         new BoundaryFlowIntegrator(inflow_coeff, vel_coeff, alpha),
         inflow_marker);
      adv_lf.Assemble();
      adv_rhs.SetSize(height);
      adv_rhs = adv_lf;
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      rhs = 0.0;
      if (Adv_op.Ptr())
      {
         Adv_op.Ptr()->Mult(x, z);
         rhs += z;
      }
      if (Diff_op.Ptr())
      {
         Diff_op.Ptr()->Mult(x, z2);
         rhs -= z2;
      }
      rhs += adv_rhs;
      if (diff_kappa > 0.0)
      {
         rhs += diff_rhs;
      }
      M_solver.Mult(rhs, y);
   }
};


class CGScalarTransportOperator : public TimeDependentOperator
{
private:
   ParFiniteElementSpace &fes;
   ParGridFunction &velocity;
   VectorGridFunctionCoefficient vel_coeff;
   real_t diff_kappa;
   bool use_supg;
   real_t supg_c;
   bool use_pa;
   Array<int> ess_tdof_list;

   ParBilinearForm mass_form;
   std::unique_ptr<ParBilinearForm> adv_form;
   std::unique_ptr<ParBilinearForm> diff_form;
   std::unique_ptr<ParBilinearForm> supg_form;

   OperatorHandle M_op;
   OperatorHandle Adv_op;
   OperatorHandle Diff_op;
   OperatorHandle Supg_op;

   mutable CGSolver M_solver;
   mutable Vector z, z2, z3, rhs;

   void BuildMassOperator()
   {
      if (use_pa)
      {
         mass_form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      }
      mass_form.AddDomainIntegrator(new MassIntegrator());
      mass_form.Assemble();
      mass_form.Finalize();

      M_op.SetType(mass_form.GetAssemblyLevel() == AssemblyLevel::LEGACY ?
                   Operator::Hypre_ParCSR : Operator::ANY_TYPE);
      mass_form.FormSystemMatrix(ess_tdof_list, M_op);

      M_solver.SetOperator(*M_op.Ptr());
      M_solver.iterative_mode = false;
      M_solver.SetRelTol(1e-12);
      M_solver.SetAbsTol(0.0);
      M_solver.SetMaxIter(200);
      M_solver.SetPrintLevel(0);
   }

   void BuildDiffusionOperator()
   {
      if (diff_kappa <= 0.0) { return; }
      diff_form = std::make_unique<ParBilinearForm>(&fes);
      if (use_pa)
      {
         diff_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      }
      ConstantCoefficient diff_coeff(diff_kappa);
      diff_form->AddDomainIntegrator(new DiffusionIntegrator(diff_coeff));
      diff_form->Assemble();
      diff_form->Finalize();
      Diff_op.SetType(diff_form->GetAssemblyLevel() == AssemblyLevel::LEGACY ?
                     Operator::Hypre_ParCSR : Operator::ANY_TYPE);
      Array<int> empty;
      diff_form->FormSystemMatrix(empty, Diff_op);
   }

public:
   CGScalarTransportOperator(ParFiniteElementSpace &fes_,
                             ParGridFunction &velocity_,
                             const Array<int> &ess_tdof_list_,
                             real_t kappa,
                             bool use_supg_in,
                             real_t supg_c_in,
                             bool pa)
      : TimeDependentOperator(fes_.GetTrueVSize(), 0.0),
        fes(fes_),
        velocity(velocity_),
        vel_coeff(&velocity_),
        diff_kappa(kappa),
        use_supg(use_supg_in),
        supg_c(supg_c_in),
        use_pa(pa),
        ess_tdof_list(ess_tdof_list_),
        mass_form(&fes_),
        M_solver(fes_.GetComm()),
        z(height), z2(height), z3(height), rhs(height)
   {
      BuildMassOperator();
      BuildDiffusionOperator();
      UpdateVelocityDependentTerms();
   }

   void UpdateVelocityDependentTerms()
   {
      adv_form = std::make_unique<ParBilinearForm>(&fes);
      if (use_pa)
      {
         adv_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      }
      adv_form->AddDomainIntegrator(new ConvectionIntegrator(vel_coeff, 1.0));
      adv_form->Assemble();
      adv_form->Finalize();
      Adv_op.SetType(adv_form->GetAssemblyLevel() == AssemblyLevel::LEGACY ?
                    Operator::Hypre_ParCSR : Operator::ANY_TYPE);
      Array<int> empty;
      adv_form->FormSystemMatrix(empty, Adv_op);

      supg_form.reset();
      Supg_op.Clear();
      if (use_supg && supg_c > 0.0)
      {
         supg_form = std::make_unique<ParBilinearForm>(&fes);
         if (use_pa)
         {
            supg_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
         }
         auto *supg_coeff = new StreamlineDiffusionMatrixCoefficient(fes.GetParMesh()->Dimension(), vel_coeff, supg_c);
         supg_form->AddDomainIntegrator(new DiffusionIntegrator(*supg_coeff));
         supg_form->Assemble();
         supg_form->Finalize();
         Supg_op.SetType(supg_form->GetAssemblyLevel() == AssemblyLevel::LEGACY ?
                         Operator::Hypre_ParCSR : Operator::ANY_TYPE);
         supg_form->FormSystemMatrix(empty, Supg_op);
      }
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      rhs = 0.0;
      if (Adv_op.Ptr())
      {
         Adv_op.Ptr()->Mult(x, z);
         rhs -= z;
      }
      if (Diff_op.Ptr())
      {
         Diff_op.Ptr()->Mult(x, z2);
         rhs -= z2;
      }
      if (Supg_op.Ptr())
      {
         Supg_op.Ptr()->Mult(x, z3);
         rhs -= z3;
      }
      // Hold essential inflow true dofs fixed while still allowing
      // their nonzero values in x to influence the interior through the
      // full-space operator action above.
      for (int i = 0; i < ess_tdof_list.Size(); ++i)
      {
         rhs(ess_tdof_list[i]) = 0.0;
      }
      M_solver.Mult(rhs, y);
      for (int i = 0; i < ess_tdof_list.Size(); ++i)
      {
         y(ess_tdof_list[i]) = 0.0;
      }
   }
};

class DGPassiveScalarSolver : public PassiveScalarSolverBase
{
private:
   DG_FECollection scalar_fec;
   ParFiniteElementSpace scalar_fes;
   ParGridFunction scalar_gf;
   ParGridFunction &velocity;
   FunctionCoefficient ic_coeff;
   ConstantCoefficient one;
   ParLinearForm unit_mass_lf;
   Array<int> inflow_marker;
   real_t diff_kappa;
   real_t lower_bound;
   real_t upper_bound;
   bool use_limiter;
   int scalar_order;
   bool use_pa;

   std::unique_ptr<DGScalarTransportOperator> oper;
   std::unique_ptr<ODESolver> ode_solver;
   Vector scalar_tdof;
   real_t scalar_time = 0.0;

   void BuildTransportOperator()
   {
      oper = std::make_unique<DGScalarTransportOperator>(scalar_fes, velocity,
                                                         inflow_marker,
                                                         diff_kappa,
                                                         scalar_order,
                                                         use_pa);
      ode_solver = std::make_unique<RK3SSPSolver>();
      ode_solver->Init(*oper);
   }

public:
   DGPassiveScalarSolver(ParMesh *pmesh,
                         int order,
                         const Array<int> &inflow_attr,
                         ParGridFunction &velocity_,
                         real_t kappa,
                         bool use_limiter_in,
                         bool pa)
      : scalar_fec(order, pmesh->Dimension(), BasisType::GaussLobatto),
        scalar_fes(pmesh, &scalar_fec),
        scalar_gf(&scalar_fes),
        velocity(velocity_),
        ic_coeff(scalar_ic),
        one(1.0),
        unit_mass_lf(&scalar_fes),
        inflow_marker(inflow_attr),
        diff_kappa(kappa),
        lower_bound(std::min(g_scalar_top, g_scalar_bottom)),
        upper_bound(std::max(g_scalar_top, g_scalar_bottom)),
        use_limiter(use_limiter_in),
        scalar_order(order),
        use_pa(pa)
   {
      scalar_gf = 0.0;
      unit_mass_lf.AddDomainIntegrator(new DomainLFIntegrator(one));
      unit_mass_lf.Assemble();
      BuildTransportOperator();
      scalar_tdof.SetSize(scalar_fes.GetTrueVSize());
   }

   const char *DiscName() const override { return "DG"; }

   void Initialize() override
   {
      scalar_gf.ProjectCoefficient(ic_coeff);
      scalar_gf.GetTrueDofs(scalar_tdof);
      scalar_time = 0.0;
   }

   void Step(real_t dt) override
   {
      oper->UpdateVelocityDependentTerms();
      scalar_gf.GetTrueDofs(scalar_tdof);
      ode_solver->Step(scalar_tdof, scalar_time, dt);
      scalar_gf.SetFromTrueDofs(scalar_tdof);
      ApplyBoundPreservingLimiter();
      scalar_gf.GetTrueDofs(scalar_tdof);
   }

   void PrepareForMeshChange() override
   {
      // PA scalar operators own element data associated with the old mesh.
      // Destroy them before GeneralRefinement/Rebalance changes that mesh.
      ode_solver.reset();
      oper.reset();
   }

   void UpdateAfterMeshChange() override
   {
      // The field is updated in place, allowing MFEM to prolong the existing
      // transported dye rather than projecting the initial condition again.
      // The scalar time is intentionally preserved across this operation.
      scalar_fes.Update();
      scalar_gf.Update();
      unit_mass_lf.Update();
      unit_mass_lf.Assemble();
      scalar_tdof.SetSize(scalar_fes.GetTrueVSize());
      scalar_gf.GetTrueDofs(scalar_tdof);
      BuildTransportOperator();

      if (use_limiter)
      {
         ApplyBoundPreservingLimiter();
         scalar_gf.GetTrueDofs(scalar_tdof);
      }
   }

   void ApplyBoundPreservingLimiter()
   {
      if (!use_limiter) { return; }

      Array<int> vdofs;
      Vector el_dofs;
      const real_t eps = 1e-14;

      for (int e = 0; e < scalar_fes.GetNE(); ++e)
      {
         const FiniteElement *fe = scalar_fes.GetFE(e);
         ElementTransformation *T = scalar_fes.GetElementTransformation(e);
         const int intorder = std::max(2, 2 * fe->GetOrder() + 4);
         const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(), intorder);

         real_t cell_int = 0.0;
         real_t cell_vol = 0.0;
         real_t sample_min = std::numeric_limits<real_t>::infinity();
         real_t sample_max = -std::numeric_limits<real_t>::infinity();

         for (int j = 0; j < ir.GetNPoints(); ++j)
         {
            const IntegrationPoint &ip = ir.IntPoint(j);
            T->SetIntPoint(&ip);
            const real_t val = scalar_gf.GetValue(*T, ip);
            const real_t w = ip.weight * T->Weight();

            cell_int += w * val;
            cell_vol += w;
            sample_min = std::min(sample_min, val);
            sample_max = std::max(sample_max, val);
         }

         if (cell_vol <= 0.0) { continue; }

         const real_t avg_old = cell_int / cell_vol;
         const real_t avg_limited =
            std::min(std::max(avg_old, lower_bound), upper_bound);

         real_t theta = 1.0;
         if (sample_max - avg_old > eps)
         {
            theta = std::min(theta,
                             (upper_bound - avg_limited) / (sample_max - avg_old));
         }
         if (avg_old - sample_min > eps)
         {
            theta = std::min(theta,
                             (avg_limited - lower_bound) / (avg_old - sample_min));
         }
         theta = std::max(real_t(0.0), std::min(real_t(1.0), theta));

         if (theta >= 1.0 - 1e-14 &&
             std::abs(avg_limited - avg_old) <= 1e-14)
         {
            continue;
         }

         scalar_fes.GetElementDofs(e, vdofs);
         scalar_gf.GetSubVector(vdofs, el_dofs);
         for (int i = 0; i < el_dofs.Size(); ++i)
         {
            el_dofs(i) = theta * (el_dofs(i) - avg_old) + avg_limited;
         }
         scalar_gf.SetSubVector(vdofs, el_dofs);
      }

      scalar_gf.SetTrueVector();
   }

   ParGridFunction *GetField() override { return &scalar_gf; }

   void ComputeMinMaxMass(real_t &global_min,
                          real_t &global_max,
                          real_t &global_mass) const override
   {
      const Vector &loc = scalar_gf;
      real_t local_min = std::numeric_limits<real_t>::infinity();
      real_t local_max = -std::numeric_limits<real_t>::infinity();
      for (int i = 0; i < loc.Size(); ++i)
      {
         local_min = std::min(local_min, loc(i));
         local_max = std::max(local_max, loc(i));
      }
      if (loc.Size() == 0)
      {
         local_min = std::numeric_limits<real_t>::infinity();
         local_max = -std::numeric_limits<real_t>::infinity();
      }

      const real_t local_mass = scalar_gf * unit_mass_lf;
      MPI_Allreduce(&local_min, &global_min, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_MIN, scalar_fes.GetParMesh()->GetComm());
      MPI_Allreduce(&local_max, &global_max, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_MAX, scalar_fes.GetParMesh()->GetComm());
      MPI_Allreduce(&local_mass, &global_mass, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM,
                    scalar_fes.GetParMesh()->GetComm());
   }

   void PrintDiagnostics(int step, real_t t, real_t dt) const override
   {
      real_t global_min = 0.0, global_max = 0.0, global_mass = 0.0;
      real_t adv_cfl = 0.0, diff_cfl = 0.0;
      ComputeMinMaxMass(global_min, global_max, global_mass);
      ComputeScalarTransportCFL(scalar_fes, velocity, scalar_order, diff_kappa, dt,
                                adv_cfl, diff_cfl);
      if (scalar_fes.GetParMesh()->GetMyRank() == 0)
      {
         std::cout << "[Scalar DG] step " << step
                   << "  t = " << t
                   << "  min(dye) = " << global_min
                   << "  max(dye) = " << global_max
                   << "  mass(dye) = " << global_mass
                   << "  CFL_adv = " << adv_cfl
                   << "  CFL_diff = " << diff_cfl
                   << std::endl;
      }
   }
};

class CGPassiveScalarSolver : public PassiveScalarSolverBase
{
private:
   H1_FECollection scalar_fec;
   ParFiniteElementSpace scalar_fes;
   ParGridFunction scalar_gf;
   ParGridFunction &velocity;
   FunctionCoefficient ic_coeff;
   FunctionCoefficient inflow_coeff;
   ConstantCoefficient one;
   ParLinearForm unit_mass_lf;
   Array<int> inflow_marker;
   Array<int> ess_tdof_list;
   real_t diff_kappa;
   int scalar_order;
   bool use_pa;
   bool use_supg;
   real_t supg_c;

   std::unique_ptr<CGScalarTransportOperator> oper;
   std::unique_ptr<ODESolver> ode_solver;
   Vector scalar_tdof;
   real_t scalar_time = 0.0;

   void BuildTransportOperator()
   {
      oper = std::make_unique<CGScalarTransportOperator>(scalar_fes, velocity,
                                                         ess_tdof_list,
                                                         diff_kappa,
                                                         use_supg,
                                                         supg_c,
                                                         use_pa);
      ode_solver = std::make_unique<RK3SSPSolver>();
      ode_solver->Init(*oper);
   }

public:
   CGPassiveScalarSolver(ParMesh *pmesh,
                         int order,
                         const Array<int> &inflow_attr,
                         ParGridFunction &velocity_,
                         real_t kappa,
                         bool pa,
                         bool use_supg_in,
                         real_t supg_c_in)
      : scalar_fec(order, pmesh->Dimension()),
        scalar_fes(pmesh, &scalar_fec),
        scalar_gf(&scalar_fes),
        velocity(velocity_),
        ic_coeff(scalar_ic),
        inflow_coeff(scalar_inflow),
        one(1.0),
        unit_mass_lf(&scalar_fes),
        inflow_marker(inflow_attr),
        diff_kappa(kappa),
        scalar_order(order),
        use_pa(pa),
        use_supg(use_supg_in),
        supg_c(supg_c_in)
   {
      scalar_gf = 0.0;
      scalar_fes.GetEssentialTrueDofs(inflow_marker, ess_tdof_list);
      unit_mass_lf.AddDomainIntegrator(new DomainLFIntegrator(one));
      unit_mass_lf.Assemble();
      BuildTransportOperator();
      scalar_tdof.SetSize(scalar_fes.GetTrueVSize());
   }

   const char *DiscName() const override { return "CG"; }

   void Initialize() override
   {
      scalar_gf.ProjectCoefficient(ic_coeff);
      scalar_gf.ProjectBdrCoefficient(inflow_coeff, inflow_marker);
      scalar_gf.GetTrueDofs(scalar_tdof);
      scalar_time = 0.0;
   }

   void Step(real_t dt) override
   {
      oper->UpdateVelocityDependentTerms();
      scalar_gf.ProjectBdrCoefficient(inflow_coeff, inflow_marker);
      scalar_gf.GetTrueDofs(scalar_tdof);
      ode_solver->Step(scalar_tdof, scalar_time, dt);
      scalar_gf.SetFromTrueDofs(scalar_tdof);
      scalar_gf.ProjectBdrCoefficient(inflow_coeff, inflow_marker);
      scalar_gf.SetTrueVector();
      scalar_gf.GetTrueDofs(scalar_tdof);
   }

   void PrepareForMeshChange() override
   {
      ode_solver.reset();
      oper.reset();
   }

   void UpdateAfterMeshChange() override
   {
      scalar_fes.Update();
      scalar_gf.Update();
      scalar_fes.GetEssentialTrueDofs(inflow_marker, ess_tdof_list);

      // Preserve the transported interior field while restoring the newly
      // numbered nonzero inflow degrees of freedom.
      scalar_gf.ProjectBdrCoefficient(inflow_coeff, inflow_marker);
      scalar_gf.SetTrueVector();
      unit_mass_lf.Update();
      unit_mass_lf.Assemble();
      scalar_tdof.SetSize(scalar_fes.GetTrueVSize());
      scalar_gf.GetTrueDofs(scalar_tdof);
      BuildTransportOperator();
   }

   ParGridFunction *GetField() override { return &scalar_gf; }

   void ComputeMinMaxMass(real_t &global_min,
                          real_t &global_max,
                          real_t &global_mass) const override
   {
      const Vector &loc = scalar_gf;
      real_t local_min = std::numeric_limits<real_t>::infinity();
      real_t local_max = -std::numeric_limits<real_t>::infinity();
      for (int i = 0; i < loc.Size(); ++i)
      {
         local_min = std::min(local_min, loc(i));
         local_max = std::max(local_max, loc(i));
      }
      if (loc.Size() == 0)
      {
         local_min = std::numeric_limits<real_t>::infinity();
         local_max = -std::numeric_limits<real_t>::infinity();
      }
      const real_t local_mass = scalar_gf * unit_mass_lf;
      MPI_Allreduce(&local_min, &global_min, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_MIN, scalar_fes.GetParMesh()->GetComm());
      MPI_Allreduce(&local_max, &global_max, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_MAX, scalar_fes.GetParMesh()->GetComm());
      MPI_Allreduce(&local_mass, &global_mass, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM,
                    scalar_fes.GetParMesh()->GetComm());
   }

   void PrintDiagnostics(int step, real_t t, real_t dt) const override
   {
      real_t global_min = 0.0, global_max = 0.0, global_mass = 0.0;
      real_t adv_cfl = 0.0, diff_cfl = 0.0;
      ComputeMinMaxMass(global_min, global_max, global_mass);
      ComputeScalarTransportCFL(scalar_fes, velocity, scalar_order, diff_kappa, dt,
                                adv_cfl, diff_cfl);
      if (scalar_fes.GetParMesh()->GetMyRank() == 0)
      {
         std::cout << "[Scalar CG] step " << step
                   << "  t = " << t
                   << "  min(dye) = " << global_min
                   << "  max(dye) = " << global_max
                   << "  mass(dye) = " << global_mass
                   << "  CFL_adv = " << adv_cfl
                   << "  CFL_diff = " << diff_cfl
                   << std::endl;
      }
   }
};

// Optional GLVis helper (same pattern as MFEM examples)
static void VisualizeField(socketstream &sock,
                           const char *vishost, int visport,
                           ParGridFunction &gf, const char *title,
                           int x, int y, int w, int h, bool vec)
{
   gf.HostRead();
   ParMesh &pmesh = *gf.ParFESpace()->GetParMesh();
   MPI_Comm comm = pmesh.GetComm();
   int num_procs, myid;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);

   bool newly_opened = false;
   int connection_failed;

   do
   {
      if (myid == 0)
      {
         if (!sock.is_open() || !sock)
         {
            sock.open(vishost, visport);
            sock.precision(8);
            newly_opened = true;
         }
         sock << "solution\n";
      }

      pmesh.PrintAsOne(sock);
      gf.SaveAsOne(sock);

      if (myid == 0 && newly_opened)
      {
         const char* keys =
            (gf.FESpace()->GetMesh()->Dimension() == 2) ? "mAcRjlmm" : "mmaaAcl";

         sock << "window_title '" << title << "'\n"
              << "window_geometry " << x << " " << y << " " << w << " " << h
              << "\nkeys " << keys;
         if (vec) { sock << "vvv"; }
         sock << std::endl;
      }

      if (myid == 0) { connection_failed = !sock && !newly_opened; }
      MPI_Bcast(&connection_failed, 1, MPI_INT, 0, comm);
   }
   while (connection_failed);
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const int myid = Mpi::WorldRank();

   // -------------------------
   // Runtime / mesh parameters
   // -------------------------
   int dim = 2;
   int nx = 128;
   int ny = 64;
   int nz = 32;
   real_t lx = 4.0;
   real_t ly = 1.0;
   real_t lz = 1.0;

   int visport = 19916;
   bool glvis  = true;

   OptionsParser args(argc, argv);
   args.AddOption(&dim, "-dim", "--dim", "Problem dimension: 2 or 3.");
   args.AddOption(&nx, "-nx", "--nx", "Number of elements in x.");
   args.AddOption(&ny, "-ny", "--ny", "Number of elements in y.");
   args.AddOption(&nz, "-nz", "--nz", "Number of elements in z (3D only).");
   args.AddOption(&lx, "-lx", "--lx", "Domain length in x.");
   args.AddOption(&ly, "-ly", "--ly", "Domain length in y.");
   args.AddOption(&lz, "-lz", "--lz", "Domain length in z (3D only).");

   args.AddOption(&ctx.order, "-o", "--order", "Velocity polynomial order.");
   args.AddOption(&ctx.reynum, "-Re", "--reynolds", "Reynolds number.");
   args.AddOption(&ctx.dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&ctx.t_final, "-tf", "--t-final", "Final time.");
   args.AddOption(&ctx.ref_levels, "-r", "--refine-serial",
                  "Number of uniform serial refinements.");
   args.AddOption(&ctx.pa, "-pa", "--pa", "-no-pa", "--no-pa",
                  "Enable/disable partial assembly.");
   args.AddOption(&glvis, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable/disable GLVis.");
   args.AddOption(&visport, "-p", "--send-port", "Socket port for GLVis.");

   args.AddOption(&ctx.visit, "-visit", "--visit", "-no-visit", "--no-visit",
                  "Enable/disable VisIt output.");
   args.AddOption(&ctx.binary, "-bin", "--binary", "-no-bin", "--no-binary",
                  "Write VisIt output in binary (if supported).");
   args.AddOption(&ctx.data_dump_cycle, "-dc", "--dump-cycle",
                  "Write VisIt output every N steps (0 => every step).");
   args.AddOption(&ctx.amr, "-amr", "--enable-amr", "-no-amr", "--disable-amr",
                  "Enable vorticity/enstrophy-driven AMR (2D/3D)." );
   args.AddOption(&ctx.amr_time, "-amr-t", "--amr-time",
                  "Time at or after which to begin AMR.");
   args.AddOption(&ctx.amr_threshold, "-amr-th", "--amr-threshold",
                  "Refine elements at or above this fraction of maximum enstrophy.");
   args.AddOption(&ctx.amr_dye_mixing,
                  "-amr-dye-mixing", "--amr-dye-mixing",
                  "-amr-enstrophy", "--amr-enstrophy",
                  "Use dye mixing-band AMR instead of the default enstrophy sensor.");
   args.AddOption(&ctx.amr_dye_min, "-amr-dye-min", "--amr-dye-min",
                  "Lower dye value defining the AMR mixing band.");
   args.AddOption(&ctx.amr_dye_max, "-amr-dye-max", "--amr-dye-max",
                  "Upper dye value defining the AMR mixing band.");
   args.AddOption(&ctx.amr_dye_dynamic,
                  "-amr-dye-dynamic", "--amr-dye-dynamic",
                  "-no-amr-dye-dynamic", "--no-amr-dye-dynamic",
                  "Use simple moving AMR: refine/coarsen from the dye band with no event, DOF, or marking cap.");
   args.AddOption(&ctx.amr_dye_enstrophy,
                  "-amr-dye-enstrophy", "--amr-dye-enstrophy",
                  "-no-amr-dye-enstrophy", "--no-amr-dye-enstrophy",
                  "In simple moving-dye AMR, also refine high-enstrophy cells and retain them while enstrophy is high.");
   args.AddOption(&ctx.amr_max_level, "-amr-max-level", "--amr-max-level",
                  "Maximum refinement level relative to the starting Cartesian mesh; zero disables the level cap.");
   args.AddOption(&ctx.amr_dye_gradient,
                  "-amr-dye-gradient", "--amr-dye-gradient",
                  "-no-amr-dye-gradient", "--no-amr-dye-gradient",
                  "Use the scalar-gradient AMR sensor instead of dye-band or enstrophy marking.");
   args.AddOption(&ctx.amr_dye_gradient_threshold,
                  "-amr-dye-grad-th", "--amr-dye-gradient-threshold",
                  "Refine above this fraction of the global maximum elementwise dye-gradient magnitude.");
   args.AddOption(&ctx.amr_dye_gradient_coarsen_threshold,
                  "-amr-dye-grad-coarsen-th", "--amr-dye-gradient-coarsen-threshold",
                  "Coarsen below this lower fraction of the global maximum dye-gradient magnitude.");
   args.AddOption(&ctx.amr_dye_gradient_inlet_buffer,
                  "-amr-dye-grad-inlet-buffer", "--amr-dye-gradient-inlet-buffer",
                  "Ignore the prescribed inflow scalar gradient over this x-distance when using -amr-dye-gradient.");
   args.AddOption(&ctx.amr_coarsen,
                  "-amr-coarsen", "--amr-coarsen",
                  "-no-amr-coarsen", "--no-amr-coarsen",
                  "Coarsen refined dye regions after they leave the mixing band.");
   args.AddOption(&ctx.amr_interval, "-amr-i", "--amr-interval",
                  "Repeat AMR every this many steps after the first event; zero performs one event.");
   args.AddOption(&ctx.amr_max_events, "-amr-max-events", "--amr-max-events",
                  "Maximum number of AMR events; zero permits unlimited events.");
   args.AddOption(&ctx.amr_max_dofs, "-amr-max-dofs", "--amr-max-dofs",
                  "Do not start AMR when velocity true DOFs reach this ceiling; zero disables it.");
   args.AddOption(&ctx.amr_max_marked_global,
                  "-amr-max-marked", "--amr-max-marked-global",
                  "Globally refine at most this many marked elements per AMR event; zero disables the cap.");
   args.AddOption(&ctx.amr_preserve_history,
                  "-amr-history", "--amr-preserve-history",
                  "-amr-restart-bdf", "--amr-restart-bdf",
                  "Transfer BDF history instead of restarting BDF1 after AMR.");
   args.AddOption(&ctx.amr_rebalance,
                  "-amr-rebalance", "--amr-rebalance",
                  "-no-amr-rebalance", "--no-amr-rebalance",
                  "Rebalance after refinement and transfer state a second time.");
   args.AddOption(&ctx.amr_check,
                  "-amr-check", "--amr-check",
                  "-no-amr-check", "--no-amr-check",
                  "Fail if no AMR event occurs, the post-AMR flow solve fails, or dye mass transfer fails.");
   args.AddOption(&ctx.run_name, "-run", "--run-name",
                  "Prefix for the VisIt output directory and collection.");

   args.AddOption(&g_Utop, "-Utop", "--Utop", "Upper-stream asymptotic speed in the shear profile.");
   args.AddOption(&g_Ubot, "-Ubot", "--Ubot", "Lower-stream asymptotic speed in the shear profile.");
   args.AddOption(&g_delta, "-delta", "--delta", "Shear thickness in y-units.");
   args.AddOption(&g_eps, "-eps", "--eps", "Perturbation amplitude for uy at inflow.");
   args.AddOption(&g_k, "-k", "--k", "Perturbation wavenumber in x (2D only).");
   args.AddOption(&g_sigma, "-sig", "--sigma", "Perturbation Gaussian width in y-units.");
   args.AddOption(&g_nmodes, "-nmodes", "--nmodes",
                  "Number of spanwise Fourier modes (3D only).");
   args.AddOption(&g_phi0, "-phi0", "--phi0", "Base phase (radians).");
   args.AddOption(&g_nper, "-nper", "--nper",
                  "Oscillations per convective time (Lx/Uc).");
   args.AddOption(&g_per_z, "-per-z", "--per-z", "-no-per-z", "--no-per-z",
                  "Enable/disable periodicity in z (3D only).");

   args.AddOption(&g_use_scalar, "-scalar", "--scalar", "-no-scalar", "--no-scalar",
                  "Enable/disable passive scalar transport (DG by default, optional explicit H1/CG with -sc-cg).");
   args.AddOption(&g_scalar_kappa, "-sc-kappa", "--scalar-kappa",
                  "Passive scalar diffusivity (< 0 uses nu).");
   args.AddOption(&g_scalar_top, "-sc-top", "--scalar-top",
                  "Top-stream dye value.");
   args.AddOption(&g_scalar_bottom, "-sc-bottom", "--scalar-bottom",
                  "Bottom-stream dye value.");
   args.AddOption(&g_scalar_delta, "-sc-delta", "--scalar-delta",
                  "Scalar interface thickness (< 0 uses velocity shear thickness).");
   args.AddOption(&g_scalar_order, "-sc-order", "--scalar-order",
                  "DG scalar polynomial order (< 0 uses velocity order).");
   args.AddOption(&g_scalar_limit, "-sc-limit", "--scalar-limit",
                  "-no-sc-limit", "--no-scalar-limit",
                  "Enable/disable bound-preserving DG dye limiter.");
   args.AddOption(&g_scalar_cg, "-sc-cg", "--scalar-cg",
                  "-no-sc-cg", "--no-scalar-cg",
                  "Use explicit H1/CG passive scalar instead of the default DG scalar.");
   args.AddOption(&g_scalar_supg, "-sc-supg", "--scalar-supg",
                  "-no-sc-supg", "--no-scalar-supg",
                  "Enable/disable SUPG-like streamline stabilization for the CG scalar path.");
   args.AddOption(&g_scalar_supg_c, "-sc-supg-c", "--scalar-supg-c",
                  "SUPG-like streamline stabilization strength for the CG scalar path.");
   args.AddOption(&g_turb_inflow, "-turb-inflow", "--turb-inflow",
                  "-no-turb-inflow", "--no-turb-inflow",
                  "Enable/disable a turbulence-like pseudo-random periodic inflow perturbation.");
   args.AddOption(&g_noise_modes, "-noise-modes", "--noise-modes",
                  "Number of temporal/spatial pseudo-random inflow modes.");
   args.AddOption(&g_noise_seed, "-noise-seed", "--noise-seed",
                  "Integer seed for the deterministic pseudo-random inflow generator.");
   args.AddOption(&g_noise_period, "-noise-period", "--noise-period",
                  "Repeat period for the convected turbulence-like inflow template (<=0 uses convective time Lx/Uc).");
   args.AddOption(&g_eps_u, "-epsu", "--epsu",
                  "Streamwise pseudo-random inflow perturbation amplitude (<0 uses 0.25*eps).");
   args.AddOption(&g_eps_w, "-epsw", "--epsw",
                  "Spanwise pseudo-random inflow perturbation amplitude in 3D (<0 uses eps).");

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(std::cout); }
      return 1;
   }
   if (ctx.amr_dye_dynamic)
   {
      // Simple moving-mesh mode: the current dye band is the only sensor.
      // It intentionally removes the event, DOF, and marking caps.
      ctx.amr = true;
      ctx.amr_dye_mixing = true;
      ctx.amr_dye_gradient = false;
      ctx.amr_coarsen = true;
      ctx.amr_max_events = 0;
      ctx.amr_max_dofs = 0;
      ctx.amr_max_marked_global = 0;
      if (ctx.amr_interval <= 0) { ctx.amr_interval = 25; }
      if (ctx.amr_max_level <= 0) { ctx.amr_max_level = 1; }
   }
   if (myid == 0) { args.PrintOptions(std::cout); }
   if (myid == 0 && ctx.amr_dye_dynamic)
   {
      std::cout << "[AMR] Simple moving-dye mode: band ["
                << ctx.amr_dye_min << ", " << ctx.amr_dye_max
                << "], max level " << ctx.amr_max_level
                << ", cadence " << ctx.amr_interval
                << " steps, unlimited events"
                << (ctx.amr_dye_enstrophy
                    ? ", plus enstrophy at " + std::to_string(ctx.amr_threshold)
                    : "")
                << ".\n";
   }

   if (dim != 2 && dim != 3)
   {
      if (myid == 0) { std::cout << "Unsupported -dim " << dim << "\n"; }
      return 2;
   }
   g_dim = dim;

   MFEM_VERIFY(!ctx.amr || (ctx.amr_threshold > 0.0 && ctx.amr_threshold <= 1.0),
               "The AMR enstrophy threshold must be in (0, 1].");
   MFEM_VERIFY(!ctx.amr || g_dim != 3 || !g_per_z,
               "3D AMR currently requires -no-per-z: periodic-z AMR is not "
               "yet valid for the shear-layer flow/scalar operator path.");
   MFEM_VERIFY(!ctx.amr_dye_mixing || g_use_scalar,
               "Dye-mixing AMR requires the passive scalar; remove -no-scalar.");
   MFEM_VERIFY(!ctx.amr_dye_mixing || ctx.amr_dye_min < ctx.amr_dye_max,
               "The dye-mixing lower bound must be smaller than the upper bound.");
   MFEM_VERIFY(!ctx.amr_dye_mixing || !ctx.amr_dye_gradient,
               "Choose either -amr-dye-mixing or -amr-dye-gradient, not both.");
   MFEM_VERIFY(!ctx.amr_dye_gradient || g_use_scalar,
               "Dye-gradient AMR requires the passive scalar; remove -no-scalar.");
   MFEM_VERIFY(!ctx.amr_dye_gradient
               || (ctx.amr_dye_gradient_threshold > 0.0
                   && ctx.amr_dye_gradient_threshold <= 1.0),
               "The dye-gradient refinement threshold must be in (0, 1].");
   MFEM_VERIFY(!ctx.amr_dye_gradient
               || (ctx.amr_dye_gradient_coarsen_threshold >= 0.0
                   && ctx.amr_dye_gradient_coarsen_threshold
                      < ctx.amr_dye_gradient_threshold),
               "The dye-gradient coarsening threshold must be nonnegative and below the refinement threshold.");
   MFEM_VERIFY(!ctx.amr_dye_gradient || ctx.amr_dye_gradient_inlet_buffer >= 0.0,
               "The dye-gradient inlet buffer cannot be negative.");
   MFEM_VERIFY(!ctx.amr_coarsen
               || (ctx.amr_dye_mixing || ctx.amr_dye_gradient),
               "Dye-based AMR coarsening requires a dye AMR sensor.");
   MFEM_VERIFY(!ctx.amr || ctx.amr_max_events >= 0,
               "The AMR maximum event count cannot be negative.");
   MFEM_VERIFY(!ctx.amr || ctx.amr_max_marked_global >= 0,
               "The global AMR marking cap cannot be negative.");
   MFEM_VERIFY(!ctx.amr || ctx.amr_max_level >= 0,
               "The AMR maximum level cannot be negative.");
   MFEM_VERIFY(!ctx.amr_dye_enstrophy || ctx.amr_dye_dynamic,
               "-amr-dye-enstrophy requires -amr-dye-dynamic.");

   ctx.kinvis = 1.0 / ctx.reynum;

   // Domain bounds for BC functions
   g_xmin = 0.0; g_xmax = lx;
   g_ymin = 0.0; g_ymax = ly;
   g_zmin = 0.0; g_zmax = lz;

   // -------------------------
   // Build a cartesian mesh
   // -------------------------
   Mesh mesh;
   if (g_dim == 2)
   {
      mesh = Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL,
                                   /*gen_edges=*/true, lx, ly);
   }
   else
   {
      mesh = Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON,
                                   /*sx=*/lx, /*sy=*/ly, /*sz=*/lz,
                                   /*sfc_ordering=*/true);
   }

   mesh.SetCurvature(ctx.order, /*discont=*/false);

   for (int lev = 0; lev < ctx.ref_levels; lev++) { mesh.UniformRefinement(); }

   if (ctx.amr)
   {
      // The currently validated 3D AMR path uses the non-periodic z mesh.
      mesh.EnsureNCMesh();
   }

   std::unique_ptr<Mesh> periodic_mesh;
   Mesh *mesh_ptr = &mesh;
   if (g_dim == 3 && g_per_z)
   {
      Vector z_translation(3);
      z_translation = 0.0;
      z_translation(2) = lz;
      std::vector<Vector> translations = { z_translation };
      periodic_mesh = std::make_unique<Mesh>(
         Mesh::MakePeriodic(mesh, mesh.CreatePeriodicVertexMapping(translations)));
      mesh_ptr = periodic_mesh.get();
   }

   ParMesh pmesh(MPI_COMM_WORLD, *mesh_ptr);

   real_t local_root_element_volume = 0.0;
   for (int e = 0; e < pmesh.GetNE(); ++e)
   {
      local_root_element_volume = std::max(local_root_element_volume,
                                           pmesh.GetElementVolume(e));
   }
   real_t root_element_volume = 0.0;
   MPI_Allreduce(&local_root_element_volume, &root_element_volume, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_MAX, pmesh.GetComm());

   if (myid == 0)
   {
      std::cout << "NE = " << pmesh.GetNE()
                << ", order = " << ctx.order
                << ", Re = " << ctx.reynum
                << ", nu = " << ctx.kinvis << "\n";
   }

   // -------------------------
   // Navier solver
   // -------------------------
// -------------------------
// Navier solver
// -------------------------
NavierSolver flowsolver(&pmesh, ctx.order, ctx.kinvis);
flowsolver.EnablePA(ctx.pa);

// Dirichlet markers and BC registration (do this BEFORE Setup)
Array<int> bdr(pmesh.bdr_attributes.Max());

int inflow_attr = (g_dim == 2) ? BDR_LEFT : BDR_XMIN;
int top_attr    = (g_dim == 2) ? BDR_TOP : BDR_YMAX;
int bottom_attr = (g_dim == 2) ? BDR_BOTTOM : BDR_YMIN;
const int wall_normal_component = 1; // y-component in both 2D and 3D

// Inflow (x-min): full velocity Dirichlet.
bdr = 0; bdr[inflow_attr - 1] = 1;
flowsolver.AddVelDirichletBC(vel_inflow, bdr);

// Top (y-max): free-slip wall, enforce only zero normal velocity.
bdr = 0; bdr[top_attr - 1] = 1;
flowsolver.AddVelDirichletBC(zero_scalar, bdr, wall_normal_component);
flowsolver.AddPrescribedNormalVelocityBC(bdr);

// Bottom (y-min): free-slip wall, enforce only zero normal velocity.
bdr = 0; bdr[bottom_attr - 1] = 1;
flowsolver.AddVelDirichletBC(zero_scalar, bdr, wall_normal_component);
flowsolver.AddPrescribedNormalVelocityBC(bdr);

// ---- Initialize velocity BEFORE Setup ----
ParGridFunction *u = flowsolver.GetCurrentVelocity();
ParGridFunction *w = flowsolver.GetCurrentVorticity();
*w = 0.0;

VectorFunctionCoefficient u0(g_dim, vel_ic);
u->ProjectCoefficient(u0);
u->SetTrueVector(); // keep tdofs consistent

// Now finalize solver setup (internalizes operators/state)
flowsolver.Setup(ctx.dt);

// (optional) re-grab pointers in case Setup swaps internal storage
u = flowsolver.GetCurrentVelocity();
ParGridFunction *p = flowsolver.GetCurrentPressure();
w = flowsolver.GetCurrentVorticity();
ParGridFunction q_gf(p->ParFESpace());
q_gf = 0.0;
ComputeVorticity(flowsolver, *u, *w);
ComputeQCriterion(*u, q_gf);

if (myid == 0)
{
   std::cout << "[BC debug] Free-slip walls enabled on boundary attributes "
             << bottom_attr << " and " << top_attr
             << "; enforcing zero wall-normal component index "
             << wall_normal_component << " (y-direction)." << std::endl;
   if (g_turb_inflow)
   {
      const real_t period = inflow_repeat_period();
      std::cout << "[Inflow] turbulence-like pseudo-random convected inflow enabled"
                << ", modes = " << g_noise_modes
                << ", seed = " << g_noise_seed
                << ", period = " << period
                << ", Uc = " << inflow_convective_velocity()
                << ", eps_u = " << ((g_eps_u >= 0.0) ? g_eps_u : (0.25 * g_eps))
                << ", eps_v = " << g_eps
                << ", eps_w = " << ((g_eps_w >= 0.0) ? g_eps_w : g_eps)
                << std::endl;
   }
}
PrintSlipWallDiagnostics(*u, top_attr, bottom_attr, /*step=*/0, /*t=*/0.0);

   std::unique_ptr<PassiveScalarSolverBase> scalar_solver;
   ParGridFunction *dye = nullptr;
   if (g_use_scalar)
   {
      // DG face PA data is not safely rebuilt on the current NC AMR path.
      // Keep -pa enabled for the flow solver, but use robust full assembly
      // for scalar transport whenever the mesh can change.
      const bool scalar_pa = ctx.pa && !ctx.amr;
      const real_t scalar_kappa = (g_scalar_kappa >= 0.0) ? g_scalar_kappa : ctx.kinvis;
      const int scalar_order = (g_scalar_order >= 0) ? g_scalar_order : ctx.order;
      Array<int> inflow_marker(pmesh.bdr_attributes.Max());
      inflow_marker = 0;
      inflow_marker[inflow_attr - 1] = 1;
      if (g_scalar_cg)
      {
         scalar_solver = std::make_unique<CGPassiveScalarSolver>(&pmesh, scalar_order,
                                                                 inflow_marker, *u,
                                                                 scalar_kappa,
                                                                 scalar_pa,
                                                                 g_scalar_supg,
                                                                 g_scalar_supg_c);
      }
      else
      {
         scalar_solver = std::make_unique<DGPassiveScalarSolver>(&pmesh, scalar_order,
                                                                 inflow_marker, *u,
                                                                 scalar_kappa,
                                                                 g_scalar_limit,
                                                                 scalar_pa);
      }
      scalar_solver->Initialize();
      dye = scalar_solver->GetField();
      if (myid == 0)
      {
         std::cout << "[Scalar " << scalar_solver->DiscName() << "] Passive dye enabled with kappa = " << scalar_kappa
                   << ", top = " << g_scalar_top
                   << ", bottom = " << g_scalar_bottom
                   << ", delta = "
                   << ((g_scalar_delta > 0.0) ? g_scalar_delta : g_delta)
                   << ", order = " << scalar_order
                   << ", time integrator = explicit RK3SSP"
                   << ", assembly = " << (scalar_pa ? "PA" : "full");
         if (g_scalar_cg)
         {
            std::cout << ", discretization = H1/CG"
                      << ", SUPG = " << (g_scalar_supg ? "on" : "off")
                      << ", SUPG_c = " << g_scalar_supg_c;
         }
         else
         {
            std::cout << ", discretization = DG"
                      << ", upwinding = on, diffusion = SIPG"
                      << ", limiter = "
                      << (g_scalar_limit ? "bound-preserving on" : "off");
         }
         if (ctx.pa && ctx.amr)
         {
            std::cout << ", scalar PA disabled for NC AMR";
         }
         std::cout << std::endl;
      }
      scalar_solver->PrintDiagnostics(/*step=*/0, /*t=*/0.0, ctx.dt);
   }

   // -------------------------
   // VisIt output
   // -------------------------
   std::unique_ptr<VisItDataCollection> visit_dc;
   if (ctx.visit)
   {
      const std::string visit_dir = ctx.run_name + "_visit";
      if (myid == 0)
      {
         const int rc = mkdir(visit_dir.c_str(), 0755);
         if (rc != 0 && errno != EEXIST)
         {
            MFEM_ABORT("Failed to create VisIt output directory.");
         }
      }
      MPI_Barrier(MPI_COMM_WORLD);

      const std::string visit_prefix =
         visit_dir + "/" + ctx.run_name;
      visit_dc = std::make_unique<VisItDataCollection>(visit_prefix.c_str(), &pmesh);
      visit_dc->SetPrecision(8);
      // Some MFEM builds support binary toggle; if not, this is harmless.
      // visit_dc->SetBinary(ctx.binary);

      visit_dc->RegisterField("velocity", u);
      visit_dc->RegisterField("vorticity", w);
      visit_dc->RegisterField("qcriterion", &q_gf);
      if (dye) { visit_dc->RegisterField("dye", dye); }

      visit_dc->SetCycle(0);
      visit_dc->SetTime(0.0);
      visit_dc->Save();
   }

   // -------------------------
   // GLVis (optional)
   // -------------------------
   socketstream vis_sock;
   if (glvis)
   {
      char vishost[] = "localhost";
      VisualizeField(vis_sock, vishost, visport, *u, "Velocity",
                     /*x=*/10, /*y=*/10, /*w=*/500, /*h=*/350, /*vec=*/true);
   }

   // -------------------------
   // Time integration loop
   // -------------------------
   real_t t = 0.0;
   real_t dt = ctx.dt;
   const real_t t_final = ctx.t_final;

   int step = 0;
   int amr_events = 0;
   int last_amr_step = -1;
   bool saw_post_amr_step = false;
   bool post_amr_solves_converged = true;
   real_t max_scalar_mass_transfer_change = 0.0;
   while (t < t_final - 0.5*dt)
   {
      flowsolver.Step(t, dt, step);
      const int cycle = step + 1;
      bool amr_this_step = false;

      if (last_amr_step >= 0 && cycle > last_amr_step)
      {
         saw_post_amr_step = true;
         post_amr_solves_converged = post_amr_solves_converged
                                     && flowsolver.LastStepConverged();
      }

      ComputeVorticity(flowsolver, *u, *w);
      ComputeQCriterion(*u, q_gf);

      const bool first_amr_event = amr_events == 0;
      const bool recurring_amr_event = ctx.amr_interval > 0
                                       && cycle % ctx.amr_interval == 0;
      const bool amr_is_scheduled = ctx.amr
                                    && (ctx.amr_max_events == 0
                                        || amr_events < ctx.amr_max_events)
                                    && t >= ctx.amr_time
                                    && (first_amr_event || recurring_amr_event);
      if (amr_is_scheduled)
      {
         // MFEM selects complete sibling groups internally, so derefinement
         // cannot use the flow solver's pre-change history-transfer callback.
         // Restart BDF after a successful coarsen; the current velocity,
         // pressure, and dye fields still transfer through GridFunction::Update().
         bool coarsened = false;
         bool refined = false;
         long long coarsen_elements_before = 0;
         int coarsen_velocity_dofs_before = 0;
         real_t coarsen_scalar_min_before = 0.0;
         real_t coarsen_scalar_max_before = 0.0;
         real_t coarsen_scalar_mass_before = 0.0;
         if (ctx.amr_coarsen && amr_events > 0)
         {
            MFEM_VERIFY(dye != nullptr,
                        "Dye-based AMR coarsening requires an initialized scalar field.");
            coarsen_elements_before = pmesh.GetGlobalNE();
            coarsen_velocity_dofs_before = u->ParFESpace()->GlobalTrueVSize();
            if (scalar_solver)
            {
               scalar_solver->ComputeMinMaxMass(coarsen_scalar_min_before,
                                                coarsen_scalar_max_before,
                                                coarsen_scalar_mass_before);
            }
            Array<real_t> keep_error;
            if (ctx.amr_dye_gradient)
            {
               Vector gradient_indicator;
               ComputeElementDyeGradient(*dye, gradient_indicator);
               MaskDyeGradientInletBuffer(*dye,
                                          ctx.amr_dye_gradient_inlet_buffer,
                                          gradient_indicator);
               const real_t local_max = gradient_indicator.Size()
                                        ? gradient_indicator.Max() : 0.0;
               real_t global_max = 0.0;
               MPI_Allreduce(&local_max, &global_max, 1,
                             MPITypeMap<real_t>::mpi_type, MPI_MAX,
                             pmesh.GetComm());
               keep_error.SetSize(gradient_indicator.Size());
               const real_t inverse_global_max = global_max > 0.0
                                                 ? 1.0 / global_max : 0.0;
               for (int e = 0; e < keep_error.Size(); ++e)
               {
                  keep_error[e] = gradient_indicator(e) * inverse_global_max;
               }
               coarsened = pmesh.DerefineByError(
                  keep_error, ctx.amr_dye_gradient_coarsen_threshold,
                  /*nc_limit=*/0, /*op=*/2);
            }
            else
            {
               ComputeDyeMixingKeepError(*dye, ctx.amr_dye_min,
                                          ctx.amr_dye_max, keep_error);
               if (ctx.amr_dye_enstrophy)
               {
                  Vector enstrophy;
                  ComputeElementEnstrophy(*u, enstrophy);
                  const real_t local_max = enstrophy.Size() ? enstrophy.Max() : 0.0;
                  real_t global_max = 0.0;
                  MPI_Allreduce(&local_max, &global_max, 1,
                                MPITypeMap<real_t>::mpi_type, MPI_MAX,
                                pmesh.GetComm());
                  const real_t keep_threshold = 0.5 * ctx.amr_threshold * global_max;
                  for (int e = 0; e < keep_error.Size(); ++e)
                  {
                     if (enstrophy(e) >= keep_threshold && enstrophy(e) > 0.0)
                     {
                        keep_error[e] = 1.0;
                     }
                  }
               }
               coarsened = pmesh.DerefineByError(keep_error, 0.5,
                                                  /*nc_limit=*/0, /*op=*/2);
            }
            if (coarsened)
            {
               if (scalar_solver)
               {
                  scalar_solver->PrepareForMeshChange();
               }

               StopWatch amr_timer;
               amr_timer.Start();
               flowsolver.UpdateAfterMeshChange(dt, /*restart_time_integrator=*/true);
               if (scalar_solver) { scalar_solver->UpdateAfterMeshChange(); }

               if (ctx.amr_rebalance)
               {
                  if (scalar_solver) { scalar_solver->PrepareForMeshChange(); }
                  flowsolver.PrepareForMeshChange(/*preserve_time_history=*/false);
                  pmesh.Rebalance();
                  flowsolver.UpdateAfterMeshChange(dt, /*restart_time_integrator=*/true);
                  if (scalar_solver) { scalar_solver->UpdateAfterMeshChange(); }
               }
               amr_timer.Stop();

               u = flowsolver.GetCurrentVelocity();
               p = flowsolver.GetCurrentPressure();
               w = flowsolver.GetCurrentVorticity();
               q_gf.SetSpace(p->ParFESpace());
               q_gf = 0.0;
               ComputeVorticity(flowsolver, *u, *w);
               ComputeQCriterion(*u, q_gf);
               if (scalar_solver) { dye = scalar_solver->GetField(); }
               if (ctx.visit) { visit_dc->SetMesh(MPI_COMM_WORLD, &pmesh); }

               const long long elements_after = pmesh.GetGlobalNE();
               real_t scalar_min_after = 0.0, scalar_max_after = 0.0;
               real_t scalar_mass_after = 0.0, scalar_mass_relative_change = 0.0;
               if (scalar_solver)
               {
                  scalar_solver->ComputeMinMaxMass(scalar_min_after,
                                                   scalar_max_after,
                                                   scalar_mass_after);
                  scalar_mass_relative_change =
                     std::abs(scalar_mass_after - coarsen_scalar_mass_before)
                     / std::max(std::abs(coarsen_scalar_mass_before), real_t(1e-30));
                  max_scalar_mass_transfer_change = std::max(
                     max_scalar_mass_transfer_change, scalar_mass_relative_change);
               }
               if (myid == 0)
               {
                  std::cout << "\n[AMR] event " << (amr_events + 1)
                            << " at t = " << t << ": coarsened elements "
                            << coarsen_elements_before << " -> " << elements_after
                            << ", velocity true DOFs " << coarsen_velocity_dofs_before
                            << " -> " << u->ParFESpace()->GlobalTrueVSize()
                            << ", BDF restarted\n";
                  if (scalar_solver)
                  {
                     std::cout << "[AMR] dye min/max " << coarsen_scalar_min_before
                               << " / " << coarsen_scalar_max_before << " -> "
                               << scalar_min_after << " / " << scalar_max_after
                               << ", mass " << coarsen_scalar_mass_before << " -> "
                               << scalar_mass_after << " (relative change "
                               << scalar_mass_relative_change << ")\n";
                  }
               }
            }
         }

         // Re-evaluate and refine on the transferred mesh even after a
         // successful coarsen. This lets the AMR region move with the roller
         // instead of alternating between coarsening and refinement events.
         const int velocity_dofs_before = u->ParFESpace()->GlobalTrueVSize();
         const bool can_refine = ctx.amr_max_dofs == 0
                                 || velocity_dofs_before < ctx.amr_max_dofs;
         std::vector<MarkCandidate> local_candidates;
         if (ctx.amr_dye_gradient)
         {
            MFEM_VERIFY(dye != nullptr,
                        "Dye-gradient AMR requires an initialized scalar field.");
            Vector indicator;
            ComputeElementDyeGradient(*dye, indicator);
            MaskDyeGradientInletBuffer(*dye,
                                       ctx.amr_dye_gradient_inlet_buffer,
                                       indicator);
            const real_t local_max = indicator.Size() ? indicator.Max() : 0.0;
            real_t global_max = 0.0;
            MPI_Allreduce(&local_max, &global_max, 1,
                          MPITypeMap<real_t>::mpi_type, MPI_MAX, pmesh.GetComm());
            const real_t threshold = ctx.amr_dye_gradient_threshold * global_max;
            for (int e = 0; e < pmesh.GetNE(); ++e)
            {
               if (indicator(e) >= threshold && indicator(e) > 0.0)
               {
                  local_candidates.push_back({indicator(e), e});
               }
            }
         }
         else if (ctx.amr_dye_enstrophy)
         {
            std::vector<MarkCandidate> dye_candidates;
            ComputeDyeMixingCandidates(*dye, ctx.amr_dye_min,
                                       ctx.amr_dye_max, dye_candidates);
            std::vector<real_t> priority(pmesh.GetNE(), 0.0);
            for (const MarkCandidate &candidate : dye_candidates)
            {
               priority[candidate.element] = 1.0;
            }

            Vector enstrophy;
            ComputeElementEnstrophy(*u, enstrophy);
            const real_t local_max = enstrophy.Size() ? enstrophy.Max() : 0.0;
            real_t global_max = 0.0;
            MPI_Allreduce(&local_max, &global_max, 1,
                          MPITypeMap<real_t>::mpi_type, MPI_MAX, pmesh.GetComm());
            const real_t threshold = ctx.amr_threshold * global_max;
            for (int e = 0; e < pmesh.GetNE(); ++e)
            {
               if (enstrophy(e) >= threshold && enstrophy(e) > 0.0)
               {
                  priority[e] = std::max(priority[e], enstrophy(e) / global_max);
               }
            }
            for (int e = 0; e < pmesh.GetNE(); ++e)
            {
               if (priority[e] > 0.0) { local_candidates.push_back({priority[e], e}); }
            }
         }
         else if (ctx.amr_dye_mixing)
         {
            ComputeDyeMixingCandidates(*dye, ctx.amr_dye_min,
                                       ctx.amr_dye_max, local_candidates);
            for (MarkCandidate &candidate : local_candidates)
            {
               candidate.priority *= pmesh.GetElementSize(candidate.element, 1);
            }
         }
         else
         {
            Vector indicator;
            ComputeElementEnstrophy(*u, indicator);
            const real_t local_max = indicator.Size() ? indicator.Max() : 0.0;
            real_t global_max = 0.0;
            MPI_Allreduce(&local_max, &global_max, 1,
                          MPITypeMap<real_t>::mpi_type, MPI_MAX, pmesh.GetComm());
            const real_t threshold = ctx.amr_threshold * global_max;
            for (int e = 0; e < pmesh.GetNE(); ++e)
            {
               if (indicator(e) >= threshold && indicator(e) > 0.0)
               {
                  local_candidates.push_back(
                     {indicator(e) * pmesh.GetElementSize(e, 1), e});
               }
            }
         }

         if (ctx.amr_max_level > 0)
         {
            local_candidates.erase(
               std::remove_if(local_candidates.begin(), local_candidates.end(),
                              [&](const MarkCandidate &candidate)
            {
               return !IsBelowAMRLevel(pmesh, candidate.element,
                                       root_element_volume, ctx.amr_max_level);
            }), local_candidates.end());
         }

         const int local_eligible = static_cast<int>(local_candidates.size());
         int global_eligible = 0;
         MPI_Allreduce(&local_eligible, &global_eligible, 1, MPI_INT, MPI_SUM,
                       pmesh.GetComm());
         if (global_eligible > 0 && can_refine)
         {
            Array<int> refine_elements = SelectGlobalCandidates(
               local_candidates, ctx.amr_max_marked_global, pmesh.GetComm());
            refine_elements.Sort();
            int local_marked = refine_elements.Size();
            int global_marked = 0;
            MPI_Allreduce(&local_marked, &global_marked, 1, MPI_INT, MPI_SUM,
                          pmesh.GetComm());
            if (global_marked > 0)
            {
               const long long elements_before = pmesh.GetGlobalNE();
               real_t scalar_min_before = 0.0, scalar_max_before = 0.0;
               real_t scalar_mass_before = 0.0;
               if (scalar_solver)
               {
                  scalar_solver->ComputeMinMaxMass(scalar_min_before,
                                                   scalar_max_before,
                                                   scalar_mass_before);
                  scalar_solver->PrepareForMeshChange();
               }
               StopWatch amr_timer;
               amr_timer.Start();
               flowsolver.PrepareForMeshChange(ctx.amr_preserve_history);
               pmesh.GeneralRefinement(refine_elements);
               flowsolver.UpdateAfterMeshChange(dt, !ctx.amr_preserve_history);
               if (scalar_solver) { scalar_solver->UpdateAfterMeshChange(); }
               refined = true;

               if (ctx.amr_rebalance)
               {
                  if (scalar_solver) { scalar_solver->PrepareForMeshChange(); }
                  flowsolver.PrepareForMeshChange(ctx.amr_preserve_history);
                  pmesh.Rebalance();
                  flowsolver.UpdateAfterMeshChange(dt, !ctx.amr_preserve_history);
                  if (scalar_solver) { scalar_solver->UpdateAfterMeshChange(); }
               }
               amr_timer.Stop();

               u = flowsolver.GetCurrentVelocity();
               p = flowsolver.GetCurrentPressure();
               w = flowsolver.GetCurrentVorticity();
               q_gf.SetSpace(p->ParFESpace());
               q_gf = 0.0;
               ComputeVorticity(flowsolver, *u, *w);
               ComputeQCriterion(*u, q_gf);
               if (scalar_solver) { dye = scalar_solver->GetField(); }
               if (ctx.visit) { visit_dc->SetMesh(MPI_COMM_WORLD, &pmesh); }

               const long long elements_after = pmesh.GetGlobalNE();
               real_t scalar_min_after = 0.0, scalar_max_after = 0.0;
               real_t scalar_mass_after = 0.0, scalar_mass_relative_change = 0.0;
               if (scalar_solver)
               {
                  scalar_solver->ComputeMinMaxMass(scalar_min_after,
                                                   scalar_max_after,
                                                   scalar_mass_after);
                  scalar_mass_relative_change =
                     std::abs(scalar_mass_after - scalar_mass_before)
                     / std::max(std::abs(scalar_mass_before), real_t(1e-30));
                  max_scalar_mass_transfer_change = std::max(
                     max_scalar_mass_transfer_change, scalar_mass_relative_change);
               }
               if (myid == 0)
               {
                  std::cout << "\n[AMR] event " << (amr_events + 1)
                            << " at t = " << t << ": elements "
                            << elements_before << " -> " << elements_after
                            << ", marked " << global_marked << " of "
                            << global_eligible << " eligible"
                            << (ctx.amr_max_marked_global > 0
                                ? " (global cap "
                                + std::to_string(ctx.amr_max_marked_global) + ")"
                                : " (all eligible)")
                            << ", sensor "
                            << (ctx.amr_dye_gradient ? "dye gradient"
                                : (ctx.amr_dye_enstrophy ? "dye band OR enstrophy"
                                   : (ctx.amr_dye_mixing ? "dye mixing band" : "enstrophy")))
                            << ", velocity true DOFs " << velocity_dofs_before
                            << " -> " << u->ParFESpace()->GlobalTrueVSize()
                            << "\n";
               }
            }
         }
         if (coarsened || refined)
         {
            ++amr_events;
            last_amr_step = cycle;
            amr_this_step = true;
         }
      }

      if (scalar_solver)
      {
         scalar_solver->Step(dt);
      }
      const real_t cfl = flowsolver.ComputeCFL(*u, dt);

      // VisIt output
      if (ctx.visit)
      {
         const int dump_every = (ctx.data_dump_cycle <= 0) ? 1 : ctx.data_dump_cycle;
         if (amr_this_step || cycle % dump_every == 0)
         {
            visit_dc->SetCycle(cycle);
            visit_dc->SetTime(t);
            visit_dc->Save();
         }
      }

      // GLVis refresh
      if (glvis)
      {
         char vishost[] = "localhost";
         VisualizeField(vis_sock, vishost, visport, *u, "Velocity",
                        /*x=*/10, /*y=*/10, /*w=*/500, /*h=*/350, /*vec=*/true);
      }

      if (cycle % 10 == 0)
      {
         PrintSlipWallDiagnostics(*u, top_attr, bottom_attr, cycle, t);
         if (scalar_solver)
         {
            scalar_solver->PrintDiagnostics(cycle, t, dt);
         }
      }

      if (myid == 0 && (cycle % 10 == 0))
      {
         std::cout << "step " << cycle
                   << "  t = " << t
                   << "  dt = " << dt
                   << "  CFL(u) = " << cfl << "\n";
      }

      step++;
   }

   flowsolver.PrintTimingData();
   if (ctx.amr_check)
   {
      const bool passed = amr_events > 0 && saw_post_amr_step
                          && post_amr_solves_converged
                          && max_scalar_mass_transfer_change <= 1e-10;
      if (!passed)
      {
         if (myid == 0)
         {
            std::cerr << "[AMR] check failed: events=" << amr_events
                      << ", post-AMR step observed=" << saw_post_amr_step
                      << ", post-AMR solves converged="
                      << post_amr_solves_converged
                      << ", max scalar mass transfer change="
                      << max_scalar_mass_transfer_change << std::endl;
         }
         return 1;
      }
   }
   return 0;
}
