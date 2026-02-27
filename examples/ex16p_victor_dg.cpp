//                       MFEM Example 16 - Simplified DG Version
//
// Compile with: cd examples && make ex16p_victor_dg
//
// Sample runs:
//   mpirun -np 4 ./ex16p_victor_dg -inline -structured -nx 4 -ny 4 -nz 4 -p -s 4 -dt 1e-4 -tf 0.001 -no-vis
//   mpirun -np 4 ./ex16p_victor_dg -inline -structured -nx 4 -ny 4 -nz 4 -s 2 -dt 1e-4 -tf 0.2 -visit -rs 1 -p --alpha 0.0 --kappa 0.2 -pa
//   mpirun -np 8 ./ex16p_victor_dg -inline -structured -nx 4 -ny 4 -nz 4 -rs 1 -rp 1 -o 1 -s 2 -dt 1e-4 -tf 0.1 --alpha 0.0 --kappa 0.0056035 -pa -visit
//   mpirun -np 8 ./ex16p_victor_dg -inline -structured -nx 4 -ny 4 -nz 4 -rs 1 -rp 1 -o 1 -s 2 -dt 1e-4 -tf 0.1 --alpha 0.0 --kappa 0.0056035 -pa -visit
//   mpirun -np 4 ./ex16p_victor_dg -inline -structured -nx 4 -ny 4 -nz 4 -rs 2   -s 2 -dt 1e-4 -tf 0.2 --kappa 7.845666208755404e-4 -o 1 -pa -a 0.0 -visit 
// With mesh perturbation (time-dependent mesh oscillation):
//   mpirun -np 4 ./ex16p_victor_dg -inline -structured -nx 4 -ny 4 -nz 4 -s 2 -dt 1e-4 -tf 0.1 --kappa 0.1 -mamp 0.02 -momega 20.0 -visit
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
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

// Mesh perturbation: apply a time-dependent sinusoidal displacement to mesh nodes
// Nodes oscillate radially from the center: x_new = x_orig + A*sin(ω*t)*r_hat
// where r_hat is the unit radial direction from the center
void ApplyMeshPerturbation(ParMesh *pmesh, const Vector &orig_nodes,
                           real_t amp, real_t omega, real_t t)
{
   const int dim = pmesh->Dimension();
   Vector *mesh_nodes = pmesh->GetNodes();
   if (!mesh_nodes) return;  // Linear mesh without nodes GridFunction

   const int num_nodes = mesh_nodes->Size() / dim;

   Vector center(dim);
   for (int d = 0; d < dim; d++)
   {
      center(d) = 0.5 * (bb_min(d) + bb_max(d));
   }

   real_t temporal = amp * sin(omega * t);

   for (int i = 0; i < num_nodes; i++)
   {
      // Get original position
      Vector x_orig(dim), x_new(dim);
      for (int d = 0; d < dim; d++)
      {
         x_orig(d) = orig_nodes(i * dim + d);
      }

      // Compute radial direction from center
      Vector r_vec(dim);
      real_t r_mag = 0.0;
      for (int d = 0; d < dim; d++)
      {
         r_vec(d) = x_orig(d) - center(d);
         r_mag += r_vec(d) * r_vec(d);
      }
      r_mag = sqrt(r_mag);

      // Apply displacement (avoid division by zero at center)
      if (r_mag > 1e-12)
      {
         for (int d = 0; d < dim; d++)
         {
            (*mesh_nodes)(i * dim + d) = x_orig(d) + temporal * (r_vec(d) / r_mag);
         }
      }
      else
      {
         for (int d = 0; d < dim; d++)
         {
            (*mesh_nodes)(i * dim + d) = x_orig(d);
         }
      }
   }

   // Exchange face neighbor node data for parallel consistency
   pmesh->ExchangeFaceNbrNodes();
}

// Compute the L2 integral of a scalar field squared: integral(f^2 dV)
// Uses proper quadrature integration over the domain
double ComputeL2FieldSquared(ParFiniteElementSpace &fes, const Vector &field_tdofs)
{
   ParMesh *pmesh = fes.GetParMesh();
   const int dim = pmesh->Dimension();
   const int order = fes.GetOrder(0);

   // Use quadrature rule with sufficient accuracy for L2 norm
   const int ir_order = 2 * order + 2;

   ParGridFunction field_gf(&fes);
   field_gf.SetFromTrueDofs(field_tdofs);

   double local_integral = 0.0;

   for (int e = 0; e < fes.GetNE(); e++)
   {
      ElementTransformation *Tr = fes.GetElementTransformation(e);
      const IntegrationRule &ir = IntRules.Get(fes.GetFE(e)->GetGeomType(), ir_order);

      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr->SetIntPoint(&ip);

         double f_val = field_gf.GetValue(*Tr, ip);
         double w = ip.weight * Tr->Weight();

         local_integral += f_val * f_val * w;
      }
   }

   double global_integral;
   MPI_Allreduce(&local_integral, &global_integral, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());

   return global_integral;
}

// Compute the domain volume
double ComputeDomainVolume(ParFiniteElementSpace &fes)
{
   ParMesh *pmesh = fes.GetParMesh();
   const int order = fes.GetOrder(0);
   const int ir_order = 2 * order;

   double local_vol = 0.0;

   for (int e = 0; e < fes.GetNE(); e++)
   {
      ElementTransformation *Tr = fes.GetElementTransformation(e);
      const IntegrationRule &ir = IntRules.Get(fes.GetFE(e)->GetGeomType(), ir_order);

      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr->SetIntPoint(&ip);
         local_vol += ip.weight * Tr->Weight();
      }
   }

   double global_vol;
   MPI_Allreduce(&local_vol, &global_vol, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());

   return global_vol;
}

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
   real_t mesh_perturb_amp = 0.0;    // Mesh perturbation amplitude (0 = disabled)
   real_t mesh_perturb_omega = 10.0; // Angular frequency for mesh perturbation

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
   args.AddOption(&mesh_perturb_amp, "-mamp", "--mesh-perturb-amp",
                  "Amplitude of time-dependent mesh perturbation (0 = disabled).");
   args.AddOption(&mesh_perturb_omega, "-momega", "--mesh-perturb-omega",
                  "Angular frequency for mesh perturbation.");
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

   // Set mesh curvature to ensure nodes exist (needed for mesh perturbation)
   // Using order 2 geometry for smooth deformations
   if (mesh_perturb_amp != 0.0)
   {
      pmesh->SetCurvature(std::max(2, order));
   }

   // Store original mesh nodes for perturbation
   Vector orig_mesh_nodes;
   if (mesh_perturb_amp != 0.0 && pmesh->GetNodes())
   {
      orig_mesh_nodes = *pmesh->GetNodes();
      if (myid == 0)
      {
         cout << "Mesh perturbation enabled: amp = " << mesh_perturb_amp
              << ", omega = " << mesh_perturb_omega << endl;
      }
   }

   // L2_FECollection fe_coll(order, pmesh->Dimension(), BasisType::Positive);
   // ParFiniteElementSpace fespace(pmesh, &fe_coll);

   L2_FECollection fe_coll(order, pmesh->Dimension(), BasisType::Positive);
   ParFiniteElementSpace fespace(pmesh, &fe_coll);

   // Laghos-style: non-positive L2 basis (default is GaussLegendre).
   L2_FECollection fe_coll_np(order, pmesh->Dimension(), BasisType::GaussLegendre);
   ParFiniteElementSpace fespace_np(pmesh, &fe_coll_np);

   ParGridFunction u_gf(&fespace);
   ParGridFunction u_np(&fespace_np);

   FunctionCoefficient u_0(InitialTemperature);
   u_np.ProjectCoefficient(u_0);
   u_gf.ProjectGridFunction(u_np);
   if (myid == 0)
   {
      cout << "u min/max: " << u_gf.Min() << " " << u_gf.Max() << endl;
   }


   // Determine h_min for CFL calculation
   real_t h_min = 1e10;
   for (int i = 0; i < pmesh->GetNE(); i++)
   {
      h_min = std::min(h_min, pmesh->GetElementSize(i));
   }
   real_t global_h_min;
   MPI_Allreduce(&h_min, &global_h_min, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MIN, pmesh->GetComm());

   // ParGridFunction u_gf(&fespace);
   // FunctionCoefficient u_0(InitialTemperature);
   // u_gf.ProjectCoefficient(u_0);
   Vector u;
   u_gf.GetTrueDofs(u);

   real_t kappa_dg = (order + 1) * (order + 1);
   DGConductionOperator oper(fespace, alpha, kappa, -1.0, kappa_dg, periodic, pa, u);

   unique_ptr<ODESolver> ode_solver = ODESolver::Select(ode_solver_type);
   MFEM_VERIFY(ode_solver != nullptr, "Unknown ODE solver type.");

   ode_solver->Init(oper);
   real_t t = 0.0;

   VisItDataCollection visit_dc("DataVisit/heat_conduction_DG", pmesh);
   visit_dc.RegisterField("temperature", &u_gf);

   // Open CSV file for diagnostics output
   std::ofstream csv_file;
   const int csv_width = 26;  // Column width for alignment
   if (myid == 0)
   {
      csv_file.open("output_dg.csv");
      csv_file << std::scientific << std::setprecision(16) << std::right;
      csv_file << std::setw(csv_width) << "Cycle" << ","
               << std::setw(csv_width) << "Time" << ","
               << std::setw(csv_width) << "dt" << ","
               << std::setw(csv_width) << "L2Energy" << ","
               << std::setw(csv_width) << "RHS_RMS" << endl;
   }

   // Vector for storing RHS
   Vector rhs(u.Size());

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

   // Compute domain volume for RMS calculations
   double domain_volume = ComputeDomainVolume(fespace);
   if (myid == 0)
   {
      cout << "Domain volume: " << domain_volume << endl;
   }

   // Compute initial RHS RMS using proper L2 quadrature integration
   // RMS = sqrt( integral(f^2 dV) / Volume )
   oper.Mult(u, rhs);
   double rhs_l2_sq_init = ComputeL2FieldSquared(fespace, rhs);
   double rhs_rms_init = sqrt(rhs_l2_sq_init / domain_volume);

   if (myid == 0)
   {
      cout << "step 0, t = 0"
           << ", energy = " << energy_init
           << ", integral = " << integral_init
           << ", rhs_rms = " << rhs_rms_init << endl;

      // Write initial state to CSV (cycle 0)
      csv_file << std::setw(csv_width) << 0 << ","
               << std::setw(csv_width) << 0.0 << ","
               << std::setw(csv_width) << dt << ","
               << std::setw(csv_width) << energy_init << ","
               << std::setw(csv_width) << rhs_rms_init << endl;
   }

   // Save initial state to VisIt (cycle 0, t=0)
   if (visit)
   {
      visit_dc.SetCycle(0);
      visit_dc.SetTime(0.0);
      visit_dc.Save();
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

      // Apply time-dependent mesh perturbation before each step
      if (mesh_perturb_amp != 0.0 && orig_mesh_nodes.Size() > 0)
      {
         ApplyMeshPerturbation(pmesh, orig_mesh_nodes, mesh_perturb_amp,
                               mesh_perturb_omega, t + dt);
         // Rebuild operators after mesh change
         oper.SetParameters(u);
      }

      ode_solver->Step(u, t, dt);

      // Compute diagnostics every time step for CSV output
      u_gf.SetFromTrueDofs(u);
      loc_energy = u_gf * u_gf;
      double energy;
      MPI_Allreduce(&loc_energy, &energy, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());

      loc_integral = LF(u_gf);
      double integral;
      MPI_Allreduce(&loc_integral, &integral, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());

      // Compute RHS and its RMS using proper L2 quadrature integration
      oper.Mult(u, rhs);
      double rhs_l2_sq = ComputeL2FieldSquared(fespace, rhs);
      double rhs_rms = sqrt(rhs_l2_sq / domain_volume);

      // Write to CSV every time step
      if (myid == 0)
      {
         csv_file << std::setw(csv_width) << ti << ","
                  << std::setw(csv_width) << t << ","
                  << std::setw(csv_width) << dt << ","
                  << std::setw(csv_width) << energy << ","
                  << std::setw(csv_width) << rhs_rms << endl;
      }

      // Console and VisIt output at vis_steps intervals
      if (last_step || (ti % output_steps) == 0)
      {
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
                 << ", rhs_rms = " << rhs_rms
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

      csv_file.close();
      cout << "Diagnostics written to output_dg.csv" << endl;
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

   return (r2 <= radius2) ? 2.0 : 0.0;
}
