// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. All Rights reserved.
// See files LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// Single-domain Navier shear-layer demo:
// - Programmatic rectangular mesh (channel)
// - Inflow: hyperbolic tangent shear profile
// - Top/bottom: Dirichlet moving walls (u=3 at top, u=1 at bottom)
// - Outflow: natural/Neumann (do not mark as essential)
// - VisIt output (VisItDataCollection)
// mpirun -n 4 ./navier_shear_layer -no-vis -visit -dc 10 -dt 0.001 -Re 50 -o 1 -eps 0.005 -pa


#include "mfem.hpp"
#include "navier_solver.hpp"
#include "navier_utils.hpp"

#include <cmath>
#include <cerrno>
#include <iostream>
#include <memory>
#include <sys/stat.h>

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

static real_t g_eps   = 0.02; // perturbation amplitude (try 0.01–0.05)
static int    g_k     = 2;    // number of waves in x
static real_t g_sigma = 0.10; // y-localization width (absolute in y-units)

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

static real_t g_Ubot   = 1.0;
static real_t g_Utop   = 3.0;
static real_t g_delta  = 0.05; // shear thickness (absolute in y-units)

// In MFEM cartesian meshes, boundary attributes are typically:
//   1 = bottom, 2 = right, 3 = top, 4 = left
// We'll use:
//   inflow  = left  (attr 4)
//   outflow = right (attr 2)
//   bottom  = attr 1
//   top     = attr 3
enum BdrAttr
{
   BDR_BOTTOM = 1,
   BDR_RIGHT  = 2,
   BDR_TOP    = 3,
   BDR_LEFT   = 4
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

static inline real_t perturb_uy(real_t x, real_t y)
{
   const real_t y0 = 0.5*(g_ymin + g_ymax);
   const real_t phase = 2.0*M_PI*real_t(g_k) * (x - g_xmin) / (g_xmax - g_xmin);
   const real_t gauss = std::exp(-std::pow((y - y0)/g_sigma, 2));
   return g_eps * std::sin(phase) * gauss;
}

void vel_inflow(const Vector &x,real_t, Vector &u)
{
   u.SetSize(2);
   u = 0.0;
   u(0) = shear_profile_u(x(1));
   u(1) = perturb_uy(x(0), x(1));
}


// Dirichlet BC function for top wall: constant u=Utop
void vel_top(const Vector &, real_t, Vector &u)
{
   u.SetSize(2);
   u = 0.0;
   u(0) = g_Utop;
}

// Dirichlet BC function for bottom wall: constant u=Ubot
void vel_bottom(const Vector &, real_t, Vector &u)
{
   u.SetSize(2);
   u = 0.0;
   u(0) = g_Ubot;
}

// Initial condition: start from the inflow shear profile everywhere (simple)
void vel_ic(const Vector &x, Vector &u)
{
   u.SetSize(2);
   u = 0.0;
   u(0) = shear_profile_u(x(1));
   u(1) = 0.0;
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
   int nx = 128;
   int ny = 64;
   real_t lx = 4.0;
   real_t ly = 1.0;

   int visport = 19916;
   bool glvis  = true;

   OptionsParser args(argc, argv);
   args.AddOption(&nx, "-nx", "--nx", "Number of elements in x.");
   args.AddOption(&ny, "-ny", "--ny", "Number of elements in y.");
   args.AddOption(&lx, "-lx", "--lx", "Domain length in x.");
   args.AddOption(&ly, "-ly", "--ly", "Domain length in y.");

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

   args.AddOption(&g_Utop, "-Utop", "--Utop", "Top wall streamwise speed.");
   args.AddOption(&g_Ubot, "-Ubot", "--Ubot", "Bottom wall streamwise speed.");
   args.AddOption(&g_delta, "-delta", "--delta", "Shear thickness in y-units.");
   args.AddOption(&g_eps, "-eps", "--eps", "Perturbation amplitude for uy at inflow/IC.");
   args.AddOption(&g_k, "-k", "--k", "Perturbation wavenumber in x (integer).");
   args.AddOption(&g_sigma, "-sig", "--sigma", "Perturbation Gaussian width in y-units.");


   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(std::cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(std::cout); }

   ctx.kinvis = 1.0 / ctx.reynum;

   // Domain bounds for BC functions
   g_xmin = 0.0; g_xmax = lx;
   g_ymin = 0.0; g_ymax = ly;

   // -------------------------
   // Build a cartesian mesh
   // -------------------------
   Mesh mesh = Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL,
                                    /*gen_edges=*/true, lx, ly);
   mesh.SetCurvature(ctx.order, /*discont=*/false);

   for (int lev = 0; lev < ctx.ref_levels; lev++) { mesh.UniformRefinement(); }

   ParMesh pmesh(MPI_COMM_WORLD, mesh);

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

// Inflow (left boundary attr=4)
bdr = 0; bdr[BDR_LEFT - 1] = 1;
flowsolver.AddVelDirichletBC(vel_inflow, bdr);

// Top (attr=3)
bdr = 0; bdr[BDR_TOP - 1] = 1;
flowsolver.AddVelDirichletBC(vel_top, bdr);

// Bottom (attr=1)
bdr = 0; bdr[BDR_BOTTOM - 1] = 1;
flowsolver.AddVelDirichletBC(vel_bottom, bdr);

// ---- Initialize velocity BEFORE Setup ----
ParGridFunction *u = flowsolver.GetCurrentVelocity();
VectorFunctionCoefficient u0(2, vel_ic);
u->ProjectCoefficient(u0);
u->SetTrueVector(); // keep tdofs consistent

// Now finalize solver setup (internalizes operators/state)
flowsolver.Setup(ctx.dt);

// (optional) re-grab pointer in case Setup swaps internal storage
u = flowsolver.GetCurrentVelocity();

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
