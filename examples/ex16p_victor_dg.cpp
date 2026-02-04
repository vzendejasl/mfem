//                       MFEM Example 16 - Simplified DG Version
//
// Compile with: cd examples && make ex16p_victor_dg
//
// Sample runs:
//   mpirun -np 4 ./ex16p_victor_dg -inline -structured -nx 4 -ny 4 -nz 4 -p -s 4 -dt 1e-4 -tf 0.001 -no-vis
//   mpirun -np 4 ./ex16p_victor_dg -inline -structured -nx 4 -ny 4 -nz 4 -s 2 -dt 1e-4 -tf 0.2 -visit -rs 1 -p --alpha 0.0 --kappa 0.2 -pa
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <memory>

using namespace std;
using namespace mfem;

class DGConductionOperator : public TimeDependentOperator
{
protected:
   ParFiniteElementSpace &fespace;
   Array<int> ess_tdof_list;
   ParBilinearForm *M_bf;
   ParBilinearForm *K_bf;
   OperatorHandle M, K;
   HypreParMatrix *T;
   real_t current_dt;

   CGSolver M_solver;
   Solver *M_prec;
   CGSolver T_solver;
   HypreSmoother T_prec;

   real_t alpha, kappa, sigma, kappa_dg;
   bool periodic, pa;
   mutable Vector z;

   ParGridFunction *u_coeff_gf;
   GridFunctionCoefficient *diff_coeff;

public:
   DGConductionOperator(ParFiniteElementSpace &f, real_t alpha_, real_t kappa_,
                        real_t sigma_ = -1.0, real_t kappa_dg_ = -1.0, 
                        bool periodic_ = false, bool pa_ = false, const Vector &u = Vector());
   void Mult(const Vector &u, Vector &du_dt) const override;
   void ImplicitSolve(const real_t dt, const Vector &u, Vector &k) override;
   void SetParameters(const Vector &u);
   ~DGConductionOperator() override;
};

Vector bb_min, bb_max;
real_t InitialTemperature(const Vector &x);

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   int myid = Mpi::WorldRank();
   Hypre::Init();

   int order = 2;
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   const char *mesh_file = "../data/star.mesh";
   int ode_solver_type = 4;
   real_t t_final = 0.5;
   real_t dt = 1.0e-4;
   real_t alpha = 0.01;
   real_t kappa = 0.1;
   bool visualization = true;
   bool use_inline_mesh = false;
   bool periodic = false;
   bool structured_mesh = false;
   bool pa = false;
   bool visit = false;
   int vis_steps = 10;
   int nx = 8, ny = 8, nz = 1;
   real_t x1 = 0.0, x2 = 1.0, y1 = 0.0, y2 = 1.0, z1 = 0.0, z2 = 1.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the DG finite elements.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of serial uniform refinements.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of parallel uniform refinements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  ODESolver::Types.c_str());
   args.AddOption(&alpha, "-a", "--alpha",
                  "Alpha coefficient.");
   args.AddOption(&kappa, "-k", "--kappa",
                  "Base diffusivity coefficient.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time.");
   args.AddOption(&use_inline_mesh, "-inline", "--inline-mesh", "-no-inline",
                  "--no-inline-mesh",
                  "Enable or disable inline mesh generation.");
   args.AddOption(&periodic, "-p", "--periodic", "-no-p", "--no-periodic",
                  "Enable or disable periodic boundary conditions.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly",
                  "Enable or disable partial assembly.");
   args.AddOption(&structured_mesh, "-structured", "--structured-mesh",
                  "-no-structured", "--no-structured-mesh",
                  "Use structured mesh (hex/quad) for inline generation.");
   args.AddOption(&nx, "-nx", "--nx",
                  "Number of elements in x for inline mesh.");
   args.AddOption(&ny, "-ny", "--ny",
                  "Number of elements in y for inline mesh.");
   args.AddOption(&nz, "-nz", "--nz",
                  "Number of elements in z for inline mesh.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit", "--no-visit",
                  "Save data files for VisIt visualization.");
   args.AddOption(&vis_steps, "-vs", "--vis-steps",
                  "Save/print every n-th time step.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   Mesh *mesh = nullptr;
   if (use_inline_mesh)
   {
      Mesh *init_mesh = nullptr;
      if (nz > 1) {
         if (structured_mesh) init_mesh = new Mesh(Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON, x2-x1, y2-y1, z2-z1));
         else init_mesh = new Mesh(Mesh::MakeCartesian3DWith24TetsPerHex(nx, ny, nz, x2-x1, y2-y1, z2-z1));
         if (periodic) {
            std::vector<Vector> translations = {Vector({x2-x1, 0.0, 0.0}), Vector({0.0, y2-y1, 0.0}), Vector({0.0, 0.0, z2-z1})};
            mesh = new Mesh(Mesh::MakePeriodic(*init_mesh, init_mesh->CreatePeriodicVertexMapping(translations)));
            delete init_mesh;
         } else mesh = init_mesh;
      } else {
         if (structured_mesh) init_mesh = new Mesh(Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL, false, x2-x1, y2-y1));
         else init_mesh = new Mesh(Mesh::MakeCartesian2DWith5QuadsPerQuad(nx, ny, x2-x1, y2-y1));
         if (periodic) {
            std::vector<Vector> translations = {Vector({x2-x1, 0.0}), Vector({0.0, y2-y1})};
            mesh = new Mesh(Mesh::MakePeriodic(*init_mesh, init_mesh->CreatePeriodicVertexMapping(translations)));
            delete init_mesh;
         } else mesh = init_mesh;
      }
   } else mesh = new Mesh(mesh_file, 1, 1);

   for (int l = 0; l < ser_ref_levels; l++) mesh->UniformRefinement();

   mesh->GetBoundingBox(bb_min, bb_max);
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   for (int l = 0; l < par_ref_levels; l++) pmesh->UniformRefinement();

   L2_FECollection fe_coll(order, pmesh->Dimension(), BasisType::GaussLobatto);
   ParFiniteElementSpace fespace(pmesh, &fe_coll);

   // Determine h_min for CFL calculation
   real_t h_min = 1e10;
   for (int i = 0; i < pmesh->GetNE(); i++)
   {
      h_min = std::min(h_min, pmesh->GetElementSize(i));
   }
   real_t global_h_min;
   MPI_Allreduce(&h_min, &global_h_min, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MIN, pmesh->GetComm());

   ParGridFunction u_gf(&fespace);
   FunctionCoefficient u_0(InitialTemperature);
   u_gf.ProjectCoefficient(u_0);
   Vector u;
   u_gf.GetTrueDofs(u);

   real_t kappa_dg = (order + 1) * (order + 1);
   DGConductionOperator oper(fespace, alpha, kappa, -1.0, kappa_dg, periodic, pa, u);

   unique_ptr<ODESolver> ode_solver = ODESolver::Select(ode_solver_type);
   MFEM_VERIFY(ode_solver != nullptr, "Unknown ODE solver type.");

   ode_solver->Init(oper);
   real_t t = 0.0;

   VisItDataCollection visit_dc("DataVisit/heat_conduction_example16_dg", pmesh);
   visit_dc.RegisterField("temperature", &u_gf);

   // Linear form for computing the integral of the solution (total heat)
   ParLinearForm LF(&fespace);
   ConstantCoefficient one(1.0);
   LF.AddDomainIntegrator(new DomainLFIntegrator(one));
   LF.Assemble();

   // Compute initial energy and integral
   u_gf.SetFromTrueDofs(u);
   double loc_energy = u_gf * u_gf;
   double energy_init;
   MPI_Allreduce(&loc_energy, &energy_init, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
   
   double loc_integral = LF(u_gf);
   double integral_init;
   MPI_Allreduce(&loc_integral, &integral_init, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());

   if (myid == 0)
   {
      cout << "Initial L2 energy: " << energy_init << endl;
      cout << "Initial total integral: " << integral_init << endl;
   }

   const int output_steps = std::max(vis_steps, 1);
   const real_t p_factor = pow(order + 1.0, 4.0);

   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final - dt/2)
      {
         last_step = true;
      }

      ode_solver->Step(u, t, dt);
      if (last_step || (ti % output_steps) == 0)
      {
         u_gf.SetFromTrueDofs(u);
         loc_energy = u_gf * u_gf;
         double energy;
         MPI_Allreduce(&loc_energy, &energy, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());

         loc_integral = LF(u_gf);
         double integral;
         MPI_Allreduce(&loc_integral, &integral, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());

         // Compute Suggested dt for explicit DG diffusion
         // Heuristic: dt < h^2 / (kappa * (p+1)^4)
         real_t u_max = u.Max();
         real_t global_u_max;
         MPI_Allreduce(&u_max, &global_u_max, 1, MPITypeMap<real_t>::mpi_type,
                       MPI_MAX, pmesh->GetComm());
         real_t kappa_max = kappa + alpha * global_u_max;

         // p_factor accounts for the clustering of DOFs in high-order DG
         real_t suggested_dt = (global_h_min * global_h_min) / (kappa_max * p_factor);

         if (myid == 0)
         {
            cout << "step " << ti << ", t = " << t 
                 << ", energy = " << energy 
                 << ", integral = " << integral 
                 << ", suggested dt = " << suggested_dt << endl;
         }

         if (visit)
         {
            visit_dc.SetCycle(ti);
            visit_dc.SetTime(t);
            visit_dc.Save();
         }
      }
      oper.SetParameters(u);
   }

   u_gf.SetFromTrueDofs(u);
   loc_energy = u_gf * u_gf;
   double energy_final;
   MPI_Allreduce(&loc_energy, &energy_final, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());

   loc_integral = LF(u_gf);
   double integral_final;
   MPI_Allreduce(&loc_integral, &integral_final, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());

   if (myid == 0)
   {
      cout << "\n=== Final Results ===" << endl;
      cout << "Final L2 energy:   " << energy_final << " (should decrease)" << endl;
      cout << "Energy change:     " << energy_final - energy_init << endl;
      cout << "Final integral:    " << integral_final << " (should be conserved for periodic BC)" << endl;
      cout << "Integral change:   " << integral_final - integral_init << endl;
   }

   delete pmesh;
   return 0;
}

DGConductionOperator::DGConductionOperator(ParFiniteElementSpace &f, real_t alpha_,
                                           real_t kappa_, real_t sigma_,
                                           real_t kappa_dg_, bool periodic_,
                                           bool pa_, const Vector &u)
   : TimeDependentOperator(f.GetTrueVSize(), 0.0), fespace(f), M_bf(nullptr),
     K_bf(nullptr), T(nullptr), current_dt(0.0), M_solver(f.GetComm()),
     M_prec(nullptr), T_solver(f.GetComm()), alpha(alpha_), kappa(kappa_),
     sigma(sigma_), kappa_dg(kappa_dg_), periodic(periodic_), pa(pa_),
     z(height), u_coeff_gf(nullptr), diff_coeff(nullptr)
{
   const real_t rel_tol = 1e-8;

   M_bf = new ParBilinearForm(&fespace);
   if (pa) M_bf->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   M_bf->AddDomainIntegrator(new MassIntegrator());
   M_bf->Assemble();
   if (pa)
   {
      M.Reset(M_bf, false);
   }
   else
   {
      M_bf->Finalize();
      M.Reset(M_bf->ParallelAssemble(), true);
   }

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(rel_tol);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
   if (pa)
   {
      M_prec = new OperatorJacobiSmoother(*M_bf, ess_tdof_list);
   }
   else
   {
      HypreSmoother *h_prec = new HypreSmoother();
      h_prec->SetType(HypreSmoother::Jacobi);
      h_prec->SetOperator(*M.As<HypreParMatrix>());
      M_prec = h_prec;
   }
   M_solver.SetPreconditioner(*M_prec);
   M_solver.SetOperator(*M);

   T_solver.iterative_mode = false;
   T_solver.SetRelTol(rel_tol);
   T_solver.SetAbsTol(0.0);
   T_solver.SetMaxIter(100);
   T_solver.SetPrintLevel(0);
   T_prec.SetType(HypreSmoother::Jacobi);
   if (!pa)
   {
      T_solver.SetPreconditioner(T_prec);
   }

   u_coeff_gf = new ParGridFunction(&fespace);
   diff_coeff = new GridFunctionCoefficient(u_coeff_gf);

   SetParameters(u);
}

void DGConductionOperator::Mult(const Vector &u, Vector &du_dt) const
{
   K->Mult(u, z); z.Neg();
   M_solver.Mult(z, du_dt);
}

void DGConductionOperator::ImplicitSolve(const real_t dt, const Vector &u, Vector &k)
{
   if (pa) mfem_error("ImplicitSolve not supported with PA yet");
   if (!T)
   {
      T = Add(1.0, *M.As<HypreParMatrix>(), dt, *K.As<HypreParMatrix>());
      current_dt = dt;
      T_prec.SetOperator(*T);
      T_solver.SetOperator(*T);
   }
   MFEM_VERIFY(dt == current_dt, "dt changed");
   K->Mult(u, z); z.Neg();
   T_solver.Mult(z, k);
}

void DGConductionOperator::SetParameters(const Vector &u)
{
   u_coeff_gf->SetFromTrueDofs(u);
   for (int i = 0; i < u_coeff_gf->Size(); i++)
   {
      (*u_coeff_gf)(i) = kappa + alpha * (*u_coeff_gf)(i);
   }
   u_coeff_gf->ExchangeFaceNbrData();

   K.Clear();
   delete K_bf;
   K_bf = new ParBilinearForm(&fespace);
   if (pa) K_bf->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   K_bf->AddDomainIntegrator(new DiffusionIntegrator(*diff_coeff));
   K_bf->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(*diff_coeff, sigma, kappa_dg));
   if (!periodic) K_bf->AddBdrFaceIntegrator(new DGDiffusionIntegrator(*diff_coeff, sigma, kappa_dg));
   K_bf->Assemble();
   if (pa)
   {
      K.Reset(K_bf, false);
   }
   else
   {
      K_bf->Finalize();
      K.Reset(K_bf->ParallelAssemble(), true);
   }
   delete T; T = nullptr;
}

DGConductionOperator::~DGConductionOperator() 
{ 
   delete M_bf; delete K_bf; delete T; 
   delete M_prec;
   delete diff_coeff; delete u_coeff_gf;
}

real_t InitialTemperature(const Vector &x)
{
   Vector center(x.Size());
   real_t min_box_span = std::numeric_limits<real_t>::max();

   for (int i = 0; i < x.Size(); i++)
   {
      center(i) = 0.5 * (bb_min(i) + bb_max(i));
      min_box_span = std::min(min_box_span, bb_max(i) - bb_min(i));
   }

   // For a unit box this gives a centered sphere (circle in 2D) of radius 0.2.
   const real_t radius = 0.2 * min_box_span;
   const real_t radius2 = radius * radius;

   real_t r2 = 0.0;
   for (int i = 0; i < x.Size(); i++)
   {
      const real_t dx = x(i) - center(i);
      r2 += dx * dx;
   }

   return (r2 <= radius2) ? 2.0 : 1.0;
}
