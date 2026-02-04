//                       MFEM Example 16 - Simplified DG Version
//
// Compile with: cd examples && make ex16p_victor_dg
//
// Sample runs:
//   mpirun -np 4 ./ex16p_victor_dg -inline -structured -nx 4 -ny 4 -nz 4 -p -s 4 -dt 1e-4 -tf 0.001 -no-vis
//   mpirun -np 4 ./ex16p_victor_dg -inline -structured -nx 4 -ny 4 -nz 4 -s 2 -dt 1e-4 -tf 0.2 -visit -rs 1 -p --alpha 0.0 --kappa 0.2 
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;
using namespace mfem;

class DGConductionOperator : public TimeDependentOperator
{
protected:
   ParFiniteElementSpace &fespace;
   ParBilinearForm *M;
   ParBilinearForm *K;
   HypreParMatrix Mmat, Kmat;
   HypreParMatrix *T;
   real_t current_dt;
   CGSolver M_solver;
   HypreSmoother M_prec;
   CGSolver T_solver;
   HypreSmoother T_prec;
   real_t alpha, kappa, sigma, kappa_dg;
   bool periodic;
   mutable Vector z;

   ParGridFunction *u_coeff_gf;
   GridFunctionCoefficient *diff_coeff;

public:
   DGConductionOperator(ParFiniteElementSpace &f, real_t alpha_, real_t kappa_,
                        real_t sigma_ = -1.0, real_t kappa_dg_ = -1.0, 
                        bool periodic_ = false, const Vector &u = Vector());
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
   bool visit = false;
   int vis_steps = 10;
   int nx = 8, ny = 8, nz = 1;
   real_t x1 = 0.0, x2 = 1.0, y1 = 0.0, y2 = 1.0, z1 = 0.0, z2 = 1.0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Order.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial", "Serial refinement.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel", "Parallel refinement.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver", "Solver.");
   args.AddOption(&alpha, "-a", "--alpha", "Alpha.");
   args.AddOption(&kappa, "-k", "--kappa", "Kappa.");
   args.AddOption(&dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&t_final, "-tf", "--t-final", "Final time.");
   args.AddOption(&use_inline_mesh, "-inline", "--inline-mesh", "-no-inline", "--no-inline-mesh", "Inline mesh.");
   args.AddOption(&periodic, "-p", "--periodic", "-no-p", "--no-periodic", "Periodic.");
   args.AddOption(&structured_mesh, "-structured", "--structured-mesh", "-no-structured", "--no-structured-mesh", "Structured.");
   args.AddOption(&nx, "-nx", "--nx", "nx.");
   args.AddOption(&ny, "-ny", "--ny", "ny.");
   args.AddOption(&nz, "-nz", "--nz", "nz.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis", "--no-visualization", "Vis.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit", "--no-visit", "Visit.");
   args.AddOption(&vis_steps, "-vs", "--vis-steps", "Vis steps.");
   args.Parse();
   if (!args.Good()) { if (myid == 0) args.PrintUsage(cout); return 1; }

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
   MPI_Allreduce(&h_min, &global_h_min, 1, MPI_DOUBLE, MPI_MIN, pmesh->GetComm());

   ParGridFunction u_gf(&fespace);
   FunctionCoefficient u_0(InitialTemperature);
   u_gf.ProjectCoefficient(u_0);
   Vector u; u_gf.GetTrueDofs(u);

   real_t kappa_dg = (order + 1) * (order + 1);
   DGConductionOperator oper(fespace, alpha, kappa, -1.0, kappa_dg, periodic, u);

   ODESolver *ode_solver = nullptr;
   switch (ode_solver_type) {
      case 2: ode_solver = new RK2Solver(0.5); break;
      case 4: ode_solver = new RK4Solver; break;
      default: ode_solver = new ForwardEulerSolver;
   }

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

   for (int ti = 1; t < t_final - dt/2; ti++)
   {
      ode_solver->Step(u, t, dt);
      if (ti % vis_steps == 0)
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
         MPI_Allreduce(&u_max, &global_u_max, 1, MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
         real_t kappa_max = kappa + alpha * global_u_max;
         
         // p_factor accounts for the clustering of DOFs in high-order DG
         real_t p_factor = pow(order + 1.0, 4.0);
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

   delete ode_solver;
   delete pmesh;
   return 0;
}

DGConductionOperator::DGConductionOperator(ParFiniteElementSpace &f, real_t alpha_, real_t kappa_, real_t sigma_, real_t kappa_dg_, bool periodic_, const Vector &u)
   : TimeDependentOperator(f.GetTrueVSize(), 0.0), fespace(f), M(nullptr), K(nullptr), T(nullptr), current_dt(0.0), M_solver(f.GetComm()), T_solver(f.GetComm()), alpha(alpha_), kappa(kappa_), sigma(sigma_), kappa_dg(kappa_dg_), periodic(periodic_), z(height), u_coeff_gf(nullptr), diff_coeff(nullptr)
{
   M = new ParBilinearForm(&fespace);
   M->AddDomainIntegrator(new MassIntegrator());
   M->Assemble(); M->Finalize();
   M->FormSystemMatrix(Array<int>(), Mmat);

   M_solver.SetOperator(Mmat);
   M_solver.SetRelTol(1e-8); M_solver.SetMaxIter(100); M_solver.SetPrintLevel(0);
   M_prec.SetType(HypreSmoother::Jacobi);
   M_prec.SetOperator(Mmat);
   M_solver.SetPreconditioner(M_prec);

   T_solver.SetRelTol(1e-8); T_solver.SetMaxIter(100); T_solver.SetPrintLevel(0);
   T_solver.SetPreconditioner(T_prec);

   u_coeff_gf = new ParGridFunction(&fespace);
   diff_coeff = new GridFunctionCoefficient(u_coeff_gf);

   SetParameters(u);
}

void DGConductionOperator::Mult(const Vector &u, Vector &du_dt) const
{
   Kmat.Mult(u, z); z.Neg();
   M_solver.Mult(z, du_dt);
}

void DGConductionOperator::ImplicitSolve(const real_t dt, const Vector &u, Vector &k)
{
   if (!T)
   {
      T = Add(1.0, Mmat, dt, Kmat);
      current_dt = dt;
      T_solver.SetOperator(*T);
   }
   MFEM_VERIFY(dt == current_dt, "dt changed");
   Kmat.Mult(u, z); z.Neg();
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

   delete K;
   K = new ParBilinearForm(&fespace);
   K->AddDomainIntegrator(new DiffusionIntegrator(*diff_coeff));
   K->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(*diff_coeff, sigma, kappa_dg));
   if (!periodic) K->AddBdrFaceIntegrator(new DGDiffusionIntegrator(*diff_coeff, sigma, kappa_dg));
   K->Assemble(); K->Finalize();
   K->FormSystemMatrix(Array<int>(), Kmat);
   delete T; T = nullptr;
}

DGConductionOperator::~DGConductionOperator() 
{ 
   delete M; delete K; delete T; 
   delete diff_coeff; delete u_coeff_gf;
}

real_t InitialTemperature(const Vector &x)
{
   real_t r2 = 0.0;
   real_t L = bb_max(0) - bb_min(0);
   real_t sigma_gauss = 0.15 * L;
   for (int i = 0; i < x.Size(); i++) {
      real_t mid = (bb_min(i) + bb_max(i)) * 0.5 + 0.2;
      r2 += (x(i) - mid) * (x(i) - mid);
   }
   return 1.0 + exp(-r2 / (2.0 * sigma_gauss * sigma_gauss));
}