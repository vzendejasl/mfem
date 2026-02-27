//                       MFEM Example 16 - Parallel Version
//
// Compile with: make ex16p
//
// Sample runs:  mpirun -np 4 ex16p
//               mpirun -np 4 ex16p -m ../data/inline-tri.mesh
//               mpirun -np 4 ex16p -m ../data/disc-nurbs.mesh -tf 2
//               mpirun -np 4 ex16p -s 21 -a 0.0 -k 1.0
//               mpirun -np 4 ex16p -s 22 -a 1.0 -k 0.0
//               mpirun -np 8 ex16p -s 23 -a 0.5 -k 0.5 -o 4
//               mpirun -np 4 ex16p -s 4 -dt 1.0e-4 -tf 4.0e-2 -vs 40
//               mpirun -np 16 ex16p -m ../data/fichera-q2.mesh
//               mpirun -np 16 ex16p -m ../data/fichera-mixed.mesh
//               mpirun -np 16 ex16p -m ../data/escher-p2.mesh
//               mpirun -np 8 ex16p -m ../data/beam-tet.mesh -tf 10 -dt 0.1
//               mpirun -np 4 ex16p -m ../data/amr-quad.mesh -o 4 -rs 0 -rp 0
//               mpirun -np 4 ex16p -m ../data/amr-hex.mesh -o 2 -rs 0 -rp 0
//
// Description:  This example solves a time dependent nonlinear heat equation
//               problem of the form du/dt = C(u), with a non-linear diffusion
//               operator C(u) = \nabla \cdot (\kappa + \alpha u) \nabla u.
//
//               The example demonstrates the use of nonlinear operators (the
//               class ConductionOperator defining C(u)), as well as their
//               implicit time integration. Note that implementing the method
//               ConductionOperator::ImplicitSolve is the only requirement for
//               high-order implicit (SDIRK) time integration. In this example,
//               the diffusion operator is linearized by evaluating with the
//               lagged solution from the previous timestep, so there is only
//               a linear solve. Optional saving with ADIOS2
//               (adios2.readthedocs.io) is also illustrated.
//
//               We recommend viewing examples 2, 9 and 10 before viewing this
//               example.

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

/** After spatial discretization, the conduction model can be written as:
 *
 *     du/dt = M^{-1}(-Ku)
 *
 *  where u is the vector representing the temperature, M is the mass matrix,
 *  and K is the diffusion operator with diffusivity depending on u:
 *  (\kappa + \alpha u).
 *
 *  Class ConductionOperator represents the right-hand side of the above ODE.
 */
class ConductionOperator : public TimeDependentOperator
{
protected:
   ParFiniteElementSpace &fespace;
   Array<int> ess_tdof_list; // this list remains empty for pure Neumann b.c.

   ParBilinearForm *M;
   ParBilinearForm *K;

   HypreParMatrix Mmat;
   HypreParMatrix Kmat;
   HypreParMatrix *T; // T = M + dt K
   real_t current_dt;

   CGSolver M_solver;    // Krylov solver for inverting the mass matrix M
   HypreSmoother M_prec; // Preconditioner for the mass matrix M

   CGSolver T_solver;    // Implicit solver for T = M + dt K
   HypreSmoother T_prec; // Preconditioner for the implicit solver

   real_t alpha, kappa;

   mutable Vector z; // auxiliary vector

public:
   ConductionOperator(ParFiniteElementSpace &f, real_t alpha, real_t kappa,
                      const Vector &u);

   void Mult(const Vector &u, Vector &du_dt) const override;
   /** Solve the Backward-Euler equation: k = f(u + dt*k, t), for the unknown k.
       This is the only requirement for high-order SDIRK implicit integration.*/
   void ImplicitSolve(const real_t dt, const Vector &u, Vector &k) override;

   /// Update the diffusion BilinearForm K using the given true-dof vector `u`.
   void SetParameters(const Vector &u);

   ~ConductionOperator() override;
};

// Mesh bounding box
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
      Vector x_orig(dim);
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
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   int order = 2;

   int ode_solver_type = 23;  // SDIRK33Solver
   real_t t_final = 0.5;
   real_t dt = 1.0e-2;
   real_t alpha = 1.0e-2;
   real_t kappa = 0.5;

   bool visualization = true;
   bool visit = false;
   int vis_steps = 5;
   bool adios2 = false;

   // Mesh generation options
   bool use_inline_mesh = false;
   bool periodic = false;
   bool structured_mesh = false;
   int nx = 8;
   int ny = 8;
   int nz = 1;
   real_t x1 = 0.0, x2 = 1.0;
   real_t y1 = 0.0, y2 = 1.0;
   real_t z1 = 0.0, z2 = 1.0;
   real_t mesh_perturb_amp = 0.0;    // Mesh perturbation amplitude (0 = disabled)
   real_t mesh_perturb_omega = 10.0; // Angular frequency for mesh perturbation

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  ODESolver::Types.c_str());
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&alpha, "-a", "--alpha",
                  "Alpha coefficient.");
   args.AddOption(&kappa, "-k", "--kappa",
                  "Kappa coefficient offset.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&adios2, "-adios2", "--adios2-streams", "-no-adios2",
                  "--no-adios2-streams",
                  "Save data using adios2 streams.");
   args.AddOption(&use_inline_mesh, "-inline", "--inline-mesh", "-no-inline",
                  "--no-inline-mesh", "Enable inline mesh generation.");
   args.AddOption(&periodic, "-p", "--periodic", "-no-p",
                  "--no-periodic", "Enable periodic boundary conditions.");
   args.AddOption(&structured_mesh, "-structured", "--structured-mesh",
                  "-no-structured", "--no-structured-mesh",
                  "Use structured mesh (hex/quad) for inline generation.");
   args.AddOption(&nx, "-nx", "--grid-points-x", "Number of grid points in x.");
   args.AddOption(&ny, "-ny", "--grid-points-y", "Number of grid points in y.");
   args.AddOption(&nz, "-nz", "--grid-points-z", "Number of grid points in z.");
   args.AddOption(&x1, "-x1", "--x-min", "Min x coordinate.");
   args.AddOption(&x2, "-x2", "--x-max", "Max x coordinate.");
   args.AddOption(&y1, "-y1", "--y-min", "Min y coordinate.");
   args.AddOption(&y2, "-y2", "--y-max", "Max y coordinate.");
   args.AddOption(&z1, "-z1", "--z-min", "Min z coordinate.");
   args.AddOption(&z2, "-z2", "--z-max", "Max z coordinate.");
   args.AddOption(&mesh_perturb_amp, "-mamp", "--mesh-perturb-amp",
                  "Amplitude of time-dependent mesh perturbation (0 = disabled).");
   args.AddOption(&mesh_perturb_omega, "-momega", "--mesh-perturb-omega",
                  "Angular frequency for mesh perturbation.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }

   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Read the serial mesh from the given mesh file on all processors or
   //    generate it inline.
   Mesh *mesh = nullptr;
   if (use_inline_mesh)
   {
      Mesh *init_mesh = nullptr;
      // 3D Mesh
      if (nz > 1)
      {
         if (structured_mesh)
         {
            init_mesh = new Mesh(Mesh::MakeCartesian3D(nx, ny, nz,
                                                       Element::HEXAHEDRON,
                                                       x2 - x1, y2 - y1, z2 - z1));
         }
         else
         {
            init_mesh = new Mesh(Mesh::MakeCartesian3DWith24TetsPerHex(nx, ny, nz,
                                                                       x2 - x1, y2 - y1, z2 - z1));
         }

         if (periodic)
         {
            Vector x_translation({x2 - x1, 0.0, 0.0});
            Vector y_translation({0.0, y2 - y1, 0.0});
            Vector z_translation({0.0, 0.0, z2 - z1});
            std::vector<Vector> translations = {x_translation, y_translation, z_translation};
            mesh = new Mesh(Mesh::MakePeriodic(*init_mesh,
                                               init_mesh->CreatePeriodicVertexMapping(translations)));
            delete init_mesh;
         }
         else
         {
            mesh = init_mesh;
         }
      }
      // 2D Mesh
      else
      {
         if (structured_mesh)
         {
            init_mesh = new Mesh(Mesh::MakeCartesian2D(nx, ny,
                                                       Element::QUADRILATERAL,
                                                       false,
                                                       x2 - x1, y2 - y1));
         }
         else
         {
            init_mesh = new Mesh(Mesh::MakeCartesian2DWith5QuadsPerQuad(nx, ny,
                                                                        x2 - x1, y2 - y1));
         }

         if (periodic)
         {
            Vector x_translation({x2 - x1, 0.0, 0.0});
            Vector y_translation({0.0, y2 - y1, 0.0});
            std::vector<Vector> translations = {x_translation, y_translation};
            mesh = new Mesh(Mesh::MakePeriodic(*init_mesh,
                                               init_mesh->CreatePeriodicVertexMapping(translations)));
            delete init_mesh;
         }
         else
         {
            mesh = init_mesh;
         }
      }

      // Shift the mesh to (x1, y1, z1)
      VectorFunctionCoefficient translate_mesh(mesh->Dimension(),
                                               [&](const Vector &x_in, Vector &x_out)
      {
         x_out[0] = x_in[0] + x1;
         x_out[1] = x_in[1] + y1;
         if (mesh->Dimension() == 3)
         {
            x_out[2] = x_in[2] + z1;
         }
      });
      mesh->Transform(translate_mesh);
   }
   else
   {
      mesh = new Mesh(mesh_file, 1, 1);
   }

   mesh->GetBoundingBox(bb_min, bb_max);

   int dim = mesh->Dimension();

   // 4. Define the ODE solver used for time integration. Several implicit
   //    singly diagonal implicit Runge-Kutta (SDIRK) methods, as well as
   //    explicit Runge-Kutta methods are available.
   unique_ptr<ODESolver> ode_solver = ODESolver::Select(ode_solver_type);

   // 5. Refine the mesh in serial to increase the resolution. In this example
   //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   //    a command-line parameter.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   // Set mesh curvature to ensure nodes exist (needed for mesh perturbation)
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

   // 7. Define the vector finite element space representing the current and the
   //    initial temperature, u_ref.
   H1_FECollection fe_coll(order, dim);
   ParFiniteElementSpace fespace(pmesh, &fe_coll);

   HYPRE_BigInt fe_size = fespace.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of temperature unknowns: " << fe_size << endl;
   }

   ParGridFunction u_gf(&fespace);

   // 8. Set the initial conditions for u. All boundaries are considered
   //    natural.
   FunctionCoefficient u_0(InitialTemperature);
   u_gf.ProjectCoefficient(u_0);
   Vector u;
   u_gf.GetTrueDofs(u);

   // 9. Initialize the conduction operator and the VisIt visualization.
   ConductionOperator oper(fespace, alpha, kappa, u);

   u_gf.SetFromTrueDofs(u);
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "ex16_cg-mesh." << setfill('0') << setw(6) << myid;
      sol_name << "ex16_cg-init." << setfill('0') << setw(6) << myid;
      ofstream omesh(mesh_name.str().c_str());
      omesh.precision(precision);
      pmesh->Print(omesh);
      ofstream osol(sol_name.str().c_str());
      osol.precision(precision);
      u_gf.Save(osol);
   }

   VisItDataCollection visit_dc("DataVisit/heat_conduction_CG", pmesh);
   visit_dc.RegisterField("temperature", &u_gf);
   if (visit)
   {
      visit_dc.SetCycle(0);
      visit_dc.SetTime(0.0);
      visit_dc.Save();
   }

   // Optionally output a BP (binary pack) file using ADIOS2. This can be
   // visualized with the ParaView VTX reader.
#ifdef MFEM_USE_ADIOS2
   ADIOS2DataCollection* adios2_dc = NULL;
   if (adios2)
   {
      std::string postfix(mesh_file);
      postfix.erase(0, std::string("../data/").size() );
      postfix += "_o" + std::to_string(order);
      postfix += "_solver" + std::to_string(ode_solver_type);
      const std::string collection_name = "ex16-p-" + postfix + ".bp";

      adios2_dc = new ADIOS2DataCollection(MPI_COMM_WORLD, collection_name, pmesh);
      adios2_dc->SetParameter("SubStreams", std::to_string(num_procs/2) );
      adios2_dc->RegisterField("temperature", &u_gf);
      adios2_dc->SetCycle(0);
      adios2_dc->SetTime(0.0);
      adios2_dc->Save();
   }
#endif

   socketstream sout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      sout.open(vishost, visport);
      sout << "parallel " << num_procs << " " << myid << endl;
      int good = sout.good(), all_good;
      MPI_Allreduce(&good, &all_good, 1, MPI_INT, MPI_MIN, pmesh->GetComm());
      if (!all_good)
      {
         sout.close();
         visualization = false;
         if (myid == 0)
         {
            cout << "Unable to connect to GLVis server at "
                 << vishost << ':' << visport << endl;
            cout << "GLVis visualization disabled.\n";
         }
      }
      else
      {
         sout.precision(precision);
         sout << "solution\n" << *pmesh << u_gf;
         sout << "pause\n";
         sout << flush;
         if (myid == 0)
         {
            cout << "GLVis visualization paused."
                 << " Press space (in the GLVis window) to resume it.\n";
         }
      }
   }

   // Open CSV file for diagnostics output
   std::ofstream csv_file;
   const int csv_width = 26;  // Column width for alignment
   if (myid == 0)
   {
      csv_file.open("output_cg.csv");
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

   // 10. Perform time-integration (looping over the time iterations, ti, with a
   //     time-step dt).
   ode_solver->Init(oper);
   real_t t = 0.0;

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
      if (last_step || (ti % vis_steps) == 0)
      {
         if (myid == 0)
         {
            cout << "step " << ti << ", t = " << t
                 << ", energy = " << energy
                 << ", integral = " << integral
                 << ", rhs_rms = " << rhs_rms << endl;
         }

         if (visualization)
         {
            sout << "parallel " << num_procs << " " << myid << "\n";
            sout << "solution\n" << *pmesh << u_gf << flush;
         }

         if (visit)
         {
            visit_dc.SetCycle(ti);
            visit_dc.SetTime(t);
            visit_dc.Save();
         }

#ifdef MFEM_USE_ADIOS2
         if (adios2)
         {
            adios2_dc->SetCycle(ti);
            adios2_dc->SetTime(t);
            adios2_dc->Save();
         }
#endif
      }
      oper.SetParameters(u);
   }

   // Compute final diagnostics
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
      cout << "Diagnostics written to output_cg.csv" << endl;
   }

#ifdef MFEM_USE_ADIOS2
   if (adios2)
   {
      delete adios2_dc;
   }
#endif

   // 11. Save the final solution in parallel. This output can be viewed later
   //     using GLVis: "glvis -np <np> -m ex16_cg-mesh -g ex16_cg-final".
   {
      ostringstream sol_name;
      sol_name << "ex16_cg-final." << setfill('0') << setw(6) << myid;
      ofstream osol(sol_name.str().c_str());
      osol.precision(precision);
      u_gf.Save(osol);
   }

   // 12. Free the used memory.
   delete pmesh;

   return 0;
}

ConductionOperator::ConductionOperator(ParFiniteElementSpace &f, real_t al,
                                       real_t kap, const Vector &u)
   : TimeDependentOperator(f.GetTrueVSize(), (real_t) 0.0), fespace(f),
     M(NULL), K(NULL), T(NULL), current_dt(0.0),
     M_solver(f.GetComm()), T_solver(f.GetComm()), z(height)
{
   const real_t rel_tol = 1e-8;

   M = new ParBilinearForm(&fespace);
   M->AddDomainIntegrator(new MassIntegrator());
   M->Assemble(0); // keep sparsity pattern of M and K the same
   M->FormSystemMatrix(ess_tdof_list, Mmat);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(rel_tol);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
   M_prec.SetType(HypreSmoother::Jacobi);
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(Mmat);

   alpha = al;
   kappa = kap;

   T_solver.iterative_mode = false;
   T_solver.SetRelTol(rel_tol);
   T_solver.SetAbsTol(0.0);
   T_solver.SetMaxIter(100);
   T_solver.SetPrintLevel(0);
   T_solver.SetPreconditioner(T_prec);

   SetParameters(u);
}

void ConductionOperator::Mult(const Vector &u, Vector &du_dt) const
{
   // Compute:
   //    du_dt = M^{-1}*-Ku
   // for du_dt, where K is linearized by using u from the previous timestep
   Kmat.Mult(u, z);
   z.Neg(); // z = -z
   M_solver.Mult(z, du_dt);
}

void ConductionOperator::ImplicitSolve(const real_t dt,
                                       const Vector &u, Vector &du_dt)
{
   // Solve the equation:
   //    du_dt = M^{-1}*[-K(u + dt*du_dt)]
   // for du_dt, where K is linearized by using u from the previous timestep
   if (!T)
   {
      T = Add(1.0, Mmat, dt, Kmat);
      current_dt = dt;
      T_solver.SetOperator(*T);
   }
   MFEM_VERIFY(dt == current_dt, ""); // SDIRK methods use the same dt
   Kmat.Mult(u, z);
   z.Neg();
   T_solver.Mult(z, du_dt);
}

void ConductionOperator::SetParameters(const Vector &u)
{
   ParGridFunction u_alpha_gf(&fespace);
   u_alpha_gf.SetFromTrueDofs(u);
   for (int i = 0; i < u_alpha_gf.Size(); i++)
   {
      u_alpha_gf(i) = kappa + alpha*u_alpha_gf(i);
   }

   delete K;
   K = new ParBilinearForm(&fespace);

   GridFunctionCoefficient u_coeff(&u_alpha_gf);

   K->AddDomainIntegrator(new DiffusionIntegrator(u_coeff));
   K->Assemble(0); // keep sparsity pattern of M and K the same
   K->FormSystemMatrix(ess_tdof_list, Kmat);
   delete T;
   T = NULL; // re-compute T on the next ImplicitSolve
}

ConductionOperator::~ConductionOperator()
{
   delete T;
   delete M;
   delete K;
}

real_t InitialTemperature(const Vector &x)
{
   Vector center(x.Size());
   for (int i = 0; i < x.Size(); i++)
   {
      center(i) = (bb_min(i) + bb_max(i)) * 0.5 + 0.2;
   }

   Vector x_centered(x);
   x_centered -= center;

   if (x_centered.Norml2() < 0.2)
   {
      return 2.0;
   }
   else
   {
      return 1.0;
   }
}
