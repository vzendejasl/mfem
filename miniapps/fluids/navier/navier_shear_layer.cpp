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


#include "mfem.hpp"
#include "navier_solver.hpp"
#include "navier_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cerrno>
#include <iostream>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <vector>

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


} ctx;

static int    g_dim   = 2;    // 2D or 3D
static real_t g_eps   = 0.02; // perturbation amplitude (try 0.01–0.05)
static int    g_k     = 2;    // number of waves in x (2D only)
static real_t g_sigma = 0.10; // y-localization width (absolute in y-units)
static int    g_nper  = 4;    // oscillations per convective time (Lx/Uc)
static int    g_nmodes = 1;   // spanwise Fourier modes (3D only)
static real_t g_phi0  = 0.0;  // base phase (radians)
static bool   g_per_z = true; // periodic in z (3D only)

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

static inline real_t omega_from_convective()
{
   const real_t Lx = g_xmax - g_xmin;
   const real_t Uc = 0.5*(g_Utop + g_Ubot);
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
   u(0) = shear_profile_u(x(1));
   if (g_dim == 2)
   {
      u(1) = perturb_uy_2d(x(0), x(1), t);
   }
   else
   {
      u(1) = perturb_uy_3d(x(1), x(2), t);
   }
}


real_t zero_scalar(const Vector &, real_t)
{
   return 0.0;
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


   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(std::cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(std::cout); }

   if (dim != 2 && dim != 3)
   {
      if (myid == 0) { std::cout << "Unsupported -dim " << dim << "\n"; }
      return 2;
   }
   g_dim = dim;


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

// (optional) re-grab pointer in case Setup swaps internal storage
u = flowsolver.GetCurrentVelocity();

if (myid == 0)
{
   std::cout << "[BC debug] Free-slip walls enabled on boundary attributes "
             << bottom_attr << " and " << top_attr
             << "; enforcing zero wall-normal component index "
             << wall_normal_component << " (y-direction)." << std::endl;
}
PrintSlipWallDiagnostics(*u, top_attr, bottom_attr, /*step=*/0, /*t=*/0.0);

   // -------------------------
   // VisIt output
   // -------------------------
   std::unique_ptr<VisItDataCollection> visit_dc;
   if (ctx.visit)
   {
      const char *visit_dir = "navier_shear_layer_visit";
      if (myid == 0)
      {
         const int rc = mkdir(visit_dir, 0755);
         if (rc != 0 && errno != EEXIST)
         {
            MFEM_ABORT("Failed to create VisIt output directory.");
         }
      }
      MPI_Barrier(MPI_COMM_WORLD);

      const std::string visit_prefix =
         std::string(visit_dir) + "/navier_shear_layer";
      visit_dc = std::make_unique<VisItDataCollection>(visit_prefix.c_str(), &pmesh);
      visit_dc->SetPrecision(8);
      // Some MFEM builds support binary toggle; if not, this is harmless.
      // visit_dc->SetBinary(ctx.binary);

      visit_dc->RegisterField("velocity", u);
      visit_dc->RegisterField("vorticity", w);

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
   while (t < t_final - 0.5*dt)
   {
      flowsolver.Step(t, dt, step);
      const real_t cfl = flowsolver.ComputeCFL(*u, dt);

      // VisIt output
      if (ctx.visit)
      {
         const int dump_every = (ctx.data_dump_cycle <= 0) ? 1 : ctx.data_dump_cycle;
         if (step % dump_every == 0)
         {
            visit_dc->SetCycle(step);
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

      if (step % 10 == 0)
      {
         PrintSlipWallDiagnostics(*u, top_attr, bottom_attr, step, t);
      }

      if (myid == 0 && (step % 10 == 0))
      {
         std::cout << "step " << step
                   << "  t = " << t
                   << "  dt = " << dt
                   << "  CFL = " << cfl << "\n";
      }

      step++;
   }

   flowsolver.PrintTimingData();
   return 0;
}
