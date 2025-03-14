//                       MFEM Example 9 - Parallel Version
//
// Compile with: make victor_periodic_p
//
// Sample runs:
//    srun -n36 -ppdebug victor_periodic_p -dt 0.0001 -tf 2 -visit
//    srun -n36 victor_periodic_p -rs 0 -rp 2 -dt 0.0005 -tf 2.0 -visit -nx 8 -ny 8 -nz 8 -structured_mesh
//
// Description:  This example code solves the time-dependent advection equation
//               du/dt + v.grad(u) = 0, where v is a given fluid velocity, and
//               u0(x)=u(0,x) is a given initial condition.
//
//               The example demonstrates the use of Discontinuous Galerkin (DG)
//               bilinear forms in MFEM (face integrators), the use of implicit
//               and explicit ODE time integrators, the definition of periodic
//               boundary conditions through periodic meshes, as well as the use
//               of GLVis for persistent visualization of a time-evolving
//               solution. Saving of time-dependent data files for visualization
//               with VisIt (visit.llnl.gov) and ParaView (paraview.org), as
//               well as the optional saving with ADIOS2 (adios2.readthedocs.io)
//               are also illustrated.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

// For outputing dat to 1D
#include <map>
#include <vector>
#include <cmath>

using namespace std;
using namespace mfem;

// Choice for the problem setup. The fluid velocity, initial condition and
// inflow boundary condition are chosen based on this parameter.
int problem;

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v);

// Initial condition
real_t u0_function(const Vector &x);

// Inflow boundary condition
real_t inflow_function(const Vector &x);

// Compute the planar average of solution
void ComputeAverageSolution(ParGridFunction* sol,ParFiniteElementSpace* fes, ParMesh* mesh);

// Compute the total average value of the solution
void ComputeTotalIntegral(ParGridFunction* sol, ParFiniteElementSpace* fes, ParMesh* pmesh);


void ComputeElementCenterValues(ParGridFunction* sol, ParFiniteElementSpace* fes, ParMesh* pmesh);
// void ComputeElementFaceAverageValues(ParGridFunction* sol, ParFiniteElementSpace* fes, ParMesh* pmesh);
// void ComputeElementSurfaceAverageValues(ParGridFunction* sol, ParFiniteElementSpace* fes, ParMesh* pmesh);
void ComputeElementCenterAndLeftFaceValues(ParGridFunction* sol, ParFiniteElementSpace* fes, ParMesh* pmesh);

// Mesh bounding box
Vector bb_min, bb_max;

// Define mixedwidth of using tanh profile
real_t integral_width;

// Define grid points
int nx, ny, nz;

// Define order
int order;

// Type of preconditioner for implicit time integrator
enum class PrecType : int
{
   ILU = 0,
   AIR = 1
};

#if MFEM_HYPRE_VERSION >= 21800
// Algebraic multigrid preconditioner for advective problems based on
// approximate ideal restriction (AIR). Most effective when matrix is
// first scaled by DG block inverse, and AIR applied to scaled matrix.
// See https://doi.org/10.1137/17M1144350.
class AIR_prec : public Solver
{
private:
   const HypreParMatrix *A;
   // Copy of A scaled by block-diagonal inverse
   HypreParMatrix A_s;

   HypreBoomerAMG *AIR_solver;
   int blocksize;

public:
   AIR_prec(int blocksize_) : AIR_solver(NULL), blocksize(blocksize_) { }

   void SetOperator(const Operator &op) override
   {
      width = op.Width();
      height = op.Height();

      A = dynamic_cast<const HypreParMatrix *>(&op);
      MFEM_VERIFY(A != NULL, "AIR_prec requires a HypreParMatrix.")

      // Scale A by block-diagonal inverse
      BlockInverseScale(A, &A_s, NULL, NULL, blocksize,
                        BlockInverseScaleJob::MATRIX_ONLY);
      delete AIR_solver;
      AIR_solver = new HypreBoomerAMG(A_s);
      AIR_solver->SetAdvectiveOptions(1, "", "FA");
      AIR_solver->SetPrintLevel(0);
      AIR_solver->SetMaxLevels(50);
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      // Scale the rhs by block inverse and solve system
      HypreParVector z_s;
      BlockInverseScale(A, NULL, &x, &z_s, blocksize,
                        BlockInverseScaleJob::RHS_ONLY);
      AIR_solver->Mult(z_s, y);
   }

   ~AIR_prec() override
   {
      delete AIR_solver;
   }
};
#endif


class DG_Solver : public Solver
{
private:
   HypreParMatrix &M, &K;
   SparseMatrix M_diag;
   HypreParMatrix *A;
   GMRESSolver linear_solver;
   Solver *prec;
   real_t dt;
public:
   DG_Solver(HypreParMatrix &M_, HypreParMatrix &K_, const FiniteElementSpace &fes,
             PrecType prec_type)
      : M(M_),
        K(K_),
        A(NULL),
        linear_solver(M.GetComm()),
        dt(-1.0)
   {
      int block_size = fes.GetFE(0)->GetDof();
      if (prec_type == PrecType::ILU)
      {
         prec = new BlockILU(block_size,
                             BlockILU::Reordering::MINIMUM_DISCARDED_FILL);
      }
      else if (prec_type == PrecType::AIR)
      {
#if MFEM_HYPRE_VERSION >= 21800
         prec = new AIR_prec(block_size);
#else
         MFEM_ABORT("Must have MFEM_HYPRE_VERSION >= 21800 to use AIR.\n");
#endif
      }
      linear_solver.iterative_mode = false;
      linear_solver.SetRelTol(1e-9);
      linear_solver.SetAbsTol(0.0);
      linear_solver.SetMaxIter(100);
      linear_solver.SetPrintLevel(0);
      linear_solver.SetPreconditioner(*prec);

      M.GetDiag(M_diag);
   }

   void SetTimeStep(real_t dt_)
   {
      if (dt_ != dt)
      {
         dt = dt_;
         // Form operator A = M - dt*K
         delete A;
         A = Add(-dt, K, 0.0, K);
         SparseMatrix A_diag;
         A->GetDiag(A_diag);
         A_diag.Add(1.0, M_diag);
         // this will also call SetOperator on the preconditioner
         linear_solver.SetOperator(*A);
      }
   }

   void SetOperator(const Operator &op) override
   {
      linear_solver.SetOperator(op);
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      linear_solver.Mult(x, y);
   }

   ~DG_Solver() override
   {
      delete prec;
      delete A;
   }
};


/** A time-dependent operator for the right-hand side of the ODE. The DG weak
    form of du/dt = -v.grad(u) is M du/dt = K u + b, where M and K are the mass
    and advection matrices, and b describes the flow on the boundary. This can
    be written as a general ODE, du/dt = M^{-1} (K u + b), and this class is
    used to evaluate the right-hand side. */
class FE_Evolution : public TimeDependentOperator
{
private:
   OperatorHandle M, K;
   const Vector &b;
   Solver *M_prec;
   CGSolver M_solver;
   DG_Solver *dg_solver;

   mutable Vector z;

public:
   FE_Evolution(ParBilinearForm &M_, ParBilinearForm &K_, const Vector &b_,
                PrecType prec_type);

   void Mult(const Vector &x, Vector &y) const override;
   void ImplicitSolve(const real_t dt, const Vector &x, Vector &k) override;

   ~FE_Evolution() override;
};

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   problem = 0;
   const char *mesh_file = "../data/periodic-hexagon.mesh";
   int ser_ref_levels = 2;
   int par_ref_levels = 0;
   order = 2;
   bool pa = false;
   bool ea = false;
   bool fa = false;
   const char *device_config = "cpu";
   int ode_solver_type = 4;
   real_t t_final = 10.0;
   real_t dt = 0.01;
   bool visualization = false;
   bool visit = false;
   bool paraview = false;
   bool adios2 = false;
   bool binary = true;
   int vis_steps = 100;
   nx = 5;
   ny = 5;
   nz = 5;
   bool structured_mesh = false;
   integral_width = 0.2;

   double x2 = M_PI;
   double x1 = 0.0;
   double y2 = M_PI;
   double y1 = 0.0;
   double z2 = M_PI;
   double z1 = 0.0;


#if MFEM_HYPRE_VERSION >= 21800
   PrecType prec_type = PrecType::AIR;
#else
   PrecType prec_type = PrecType::ILU;
#endif
   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Meshg file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use. See options in velocity_function().");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&ea, "-ea", "--element-assembly", "-no-ea",
                  "--no-element-assembly", "Enable Element Assembly.");
   args.AddOption(&fa, "-fa", "--full-assembly", "-no-fa",
                  "--no-full-assembly", "Enable Full Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6,\n\t"
                  "            11 - Backward Euler,\n\t"
                  "            12 - SDIRK23 (L-stable), 13 - SDIRK33,\n\t"
                  "            22 - Implicit Midpoint Method,\n\t"
                  "            23 - SDIRK23 (A-stable), 24 - SDIRK34");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption((int *)&prec_type, "-pt", "--prec-type", "Preconditioner for "
                  "implicit solves. 0 for ILU, 1 for pAIR-AMG.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&paraview, "-paraview", "--paraview-datafiles", "-no-paraview",
                  "--no-paraview-datafiles",
                  "Save data files for ParaView (paraview.org) visualization.");
   args.AddOption(&adios2, "-adios2", "--adios2-streams", "-no-adios2",
                  "--no-adios2-streams",
                  "Save data using adios2 streams.");
   args.AddOption(&binary, "-binary", "--binary-datafiles", "-ascii",
                  "--ascii-datafiles",
                  "Use binary (Sidre) or ascii format for VisIt data files.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&nx, "-nx", "--grid-points-x",
                  "Number of grid points in x.");
   args.AddOption(&ny, "-ny", "--grid-points-y",
                  "Number of grid points in y.");
   args.AddOption(&nz, "-nz", "--grid-points-z",
                  "Number of grid points in z.");
    args.AddOption(&structured_mesh, "-structured_mesh", "--structured_mesh","no-structured_mesh",
                   "--no-structured_mesh",
                   "Run a structured mesh.");
   args.AddOption(&integral_width, "-int_width", "--width of profile",
                  "Integral width.");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root())
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (Mpi::Root())
   {
      args.PrintOptions(cout);
   }

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   Mesh *mesh;
   Mesh *init_mesh;

   // 3D Mesh
   if (nz > 1){
     if (structured_mesh){
       init_mesh = new Mesh(Mesh::MakeCartesian3D(nx,
                                                  ny,
                                                  nz,
                                                  Element::HEXAHEDRON,
                                                  x2 - x1,
                                                  y2 - y1, 
                                                  z2 - z1));
     }else
     {
       init_mesh = new Mesh(Mesh::MakeCartesian3DWith24TetsPerHex(nx,
                                                                  ny,
                                                                  nz,
                                                                  x2 - x1,
                                                                  y2 - y1,
                                                                  z2 - z1));
     }

     Vector x_translation({x2 - x1, 0.0, 0.0});
     Vector y_translation({0.0, y2 - y1, 0.0});
     Vector z_translation({0.0, 0.0, z2 - z1});

     std::vector<Vector> translations = {x_translation, y_translation,z_translation};

     mesh = new Mesh(Mesh::MakePeriodic(*init_mesh, init_mesh->CreatePeriodicVertexMapping(translations)));

   }
   // 2D Mesh
   else{
     if (structured_mesh){
       init_mesh = new Mesh(Mesh::MakeCartesian2D(nx, 
                                                  ny, 
                                                  Element::QUADRILATERAL, 
                                                  false, 
                                                  x2 - x1,
                                                  y2 - y1));
     }else{
       init_mesh = new Mesh(Mesh::MakeCartesian2DWith5QuadsPerQuad(nx, 
                                                                   ny, 
                                                                   x2 - x1,
                                                                   y2 - y1));
     }

     // // Shift the vertices (note this only works on low order meshes!)
     // for (int i = 0; i < init_mesh->GetNV(); i++)
     // {
     //    double *v = init_mesh->GetVertex(i);
     //    v[0] += -0.1;
     //    v[1] += -0.1;
     // }

     Vector x_translation({x2 - x1, 0.0, 0.0});
     Vector y_translation({0.0, y2 - y1, 0.0});

     std::vector<Vector> translations = {x_translation, y_translation};
     mesh = new Mesh(Mesh::MakePeriodic(*init_mesh, init_mesh->CreatePeriodicVertexMapping(translations)));

   }

   // GridFunction *nodes = mesh->GetNodes();
   // // Ensure the GridFunction is not null
   // if (nodes)
   // {
   //    // Get the number of nodes
   //    int num_nodes = nodes->Size() / mesh->Dimension();

   //    // Modify the node positions
   //    for (int i = 0; i < num_nodes; i++)
   //    {
   //      if(i == 0){
   //        // Access the node coordinates
   //        double *node = nodes->GetData() + i * mesh->Dimension();
   //        node[0] += -0.51;
   //      }

   //    }
   // }
   //
   // Define a translation function for the mesh nodes
   VectorFunctionCoefficient translate_set_mesh(mesh->Dimension(), [&](const Vector &x_in, Vector &x_out){

      x_out[0] = x_in[0] + x1; // Translate x-coordinate 
      x_out[1] = x_in[1] + y1; // Translate y-coordinate 
      if (mesh->Dimension() == 3){
        x_out[2] = x_in[2] + z1; // Translate y-coordinate 
      }
   });

   VectorFunctionCoefficient rotate(mesh->Dimension(), [&](const Vector &x_in, Vector &x_out){
      // double theta = M_PI/4;
      double theta = 0.0;

      // Compute the rotation matrix elements
      double cos_theta = cos(theta);
      double sin_theta = sin(theta);

      double x_new = cos_theta * x_in[0] - sin_theta * x_in[1];
      double y_new = sin_theta * x_in[0] + cos_theta * x_in[1];

      x_out[0] = x_new; // Rotate x-coordinate 
      x_out[1] = y_new; // Rotate y-coordinate 
      if (mesh->Dimension() == 3){
        x_out[2] = x_in[2]; // No Rotation on z-coordinate 
      }
   });

   mesh->mfem::Mesh::Transform(translate_set_mesh);
   mesh->Transform(rotate);

   int dim = mesh->Dimension();

   // 4. Define the ODE solver used for time integration. Several explicit
   //    Runge-Kutta methods are available.
   ODESolver *ode_solver = NULL;
   switch (ode_solver_type)
   {
      // Explicit methods
      case 1: ode_solver = new ForwardEulerSolver; break;
      case 2: ode_solver = new RK2Solver(1.0); break;
      case 3: ode_solver = new RK3SSPSolver; break;
      case 4: ode_solver = new RK4Solver; break;
      case 6: ode_solver = new RK6Solver; break;
      // Implicit (L-stable) methods
      case 11: ode_solver = new BackwardEulerSolver; break;
      case 12: ode_solver = new SDIRK23Solver(2); break;
      case 13: ode_solver = new SDIRK33Solver; break;
      // Implicit A-stable methods (not L-stable)
      case 22: ode_solver = new ImplicitMidpointSolver; break;
      case 23: ode_solver = new SDIRK23Solver; break;
      case 24: ode_solver = new SDIRK34Solver; break;
      default:
         if (Mpi::Root())
         {
            cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         }
         delete mesh;
         return 3;
   }

   // 5. Refine the mesh in serial to increase the resolution. In this example
   //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   //    a command-line parameter. If the mesh is of NURBS type, we convert it
   //    to a (piecewise-polynomial) high-order mesh.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }
   if (mesh->NURBSext)
   {
      mesh->SetCurvature(max(order, 1));
   }
   mesh->GetBoundingBox(bb_min, bb_max, max(order, 1));

   // 6. Define the parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   // Shift the vertices
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   // 7. Define the parallel discontinuous DG finite element space on the
   //    parallel refined mesh of the given polynomial order.
   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   ParFiniteElementSpace *fes = new ParFiniteElementSpace(pmesh, &fec);

   HYPRE_BigInt global_vSize = fes->GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "Number of unknowns: " << global_vSize << endl;
   }

   // 8. Set up and assemble the parallel bilinear and linear forms (and the
   //    parallel hypre matrices) corresponding to the DG discretization. The
   //    DGTraceIntegrator involves integrals over mesh interior faces.
   VectorFunctionCoefficient velocity(dim, velocity_function);
   FunctionCoefficient inflow(inflow_function);
   FunctionCoefficient u0(u0_function);
  

   ParBilinearForm *m = new ParBilinearForm(fes);
   ParBilinearForm *k = new ParBilinearForm(fes);
   if (pa)
   {
      m->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      k->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   else if (ea)
   {
      m->SetAssemblyLevel(AssemblyLevel::ELEMENT);
      k->SetAssemblyLevel(AssemblyLevel::ELEMENT);
   }
   else if (fa)
   {
      m->SetAssemblyLevel(AssemblyLevel::FULL);
      k->SetAssemblyLevel(AssemblyLevel::FULL);
   }

   m->AddDomainIntegrator(new MassIntegrator);
   constexpr real_t alpha = -1.0;
   k->AddDomainIntegrator(new ConvectionIntegrator(velocity, alpha));
   k->AddInteriorFaceIntegrator(
      new NonconservativeDGTraceIntegrator(velocity, alpha));
   k->AddBdrFaceIntegrator(
      new NonconservativeDGTraceIntegrator(velocity, alpha));

   ParLinearForm *b = new ParLinearForm(fes);
   b->AddBdrFaceIntegrator(
      new BoundaryFlowIntegrator(inflow, velocity, alpha));

   int skip_zeros = 0;
   m->Assemble();
   k->Assemble(skip_zeros);
   b->Assemble();
   m->Finalize();
   k->Finalize(skip_zeros);

   HypreParVector *B = b->ParallelAssemble();

   // 9. Define the initial conditions, save the corresponding grid function to
   //    a file and (optionally) save data in the VisIt format and initialize
   //    GLVis visualization.
   ParGridFunction *u = new ParGridFunction(fes);
   u->ProjectCoefficient(u0);
   HypreParVector *U = u->GetTrueDofs();

   // ComputeAverageSolution(u, fes, pmesh);
   // ComputeTotalIntegral(u, fes, pmesh);
   
   ComputeElementCenterValues(u, fes, pmesh);
   // ComputeElementCenterAndLeftFaceValues(u,fes,pmesh);

   // ComputeElementFaceAverageValues(u,fes,pmesh);
   // ComputeElementSurfaceAverageValues(u,fes,pmesh);

   {
      ostringstream mesh_name, sol_name;
      mesh_name << "ex9-mesh." << setfill('0') << setw(6) << myid;
      sol_name << "ex9-init." << setfill('0') << setw(6) << myid;
      ofstream omesh(mesh_name.str().c_str());
      omesh.precision(precision);
      pmesh->Print(omesh);
      ofstream osol(sol_name.str().c_str());
      osol.precision(precision);
      u->Save(osol);
   }

   // Create data collection for solution output: either VisItDataCollection for
   // ascii data files, or SidreDataCollection for binary data files.
   DataCollection *dc = NULL;
   if (visit)
   {
      if (binary)
      {
#ifdef MFEM_USE_SIDRE
         dc = new SidreDataCollection("Example9-Parallel", pmesh);
#else
         MFEM_ABORT("Must build with MFEM_USE_SIDRE=YES for binary output.");
#endif
      }
      else
      {
         dc = new VisItDataCollection("DataVisit/Example9-Parallel", pmesh);
         dc->SetPrecision(precision);
         // To save the mesh using MFEM's parallel mesh format:
         // dc->SetFormat(DataCollection::PARALLEL_FORMAT);
      }
      dc->RegisterField("solution", u);
      dc->SetCycle(0);
      dc->SetTime(0.0);
      dc->Save();
   }

   ParaViewDataCollection *pd = NULL;
   if (paraview)
   {
      pd = new ParaViewDataCollection("DataParaview/Example9P", pmesh);
      pd->SetPrefixPath("ParaView");
      pd->RegisterField("solution", u);
      pd->SetLevelsOfDetail(order);
      pd->SetDataFormat(VTKFormat::BINARY);
      pd->SetHighOrderOutput(true);
      pd->SetCycle(0);
      pd->SetTime(0.0);
      pd->Save();
   }

   // Optionally output a BP (binary pack) file using ADIOS2. This can be
   // visualized with the ParaView VTX reader.
#ifdef MFEM_USE_ADIOS2
   ADIOS2DataCollection *adios2_dc = NULL;
   if (adios2)
   {
      std::string postfix(mesh_file);
      postfix.erase(0, std::string("../data/").size() );
      postfix += "_o" + std::to_string(order);
      const std::string collection_name = "ex9-p-" + postfix + ".bp";

      adios2_dc = new ADIOS2DataCollection(MPI_COMM_WORLD, collection_name, pmesh);
      // output data substreams are half the number of mpi processes
      adios2_dc->SetParameter("SubStreams", std::to_string(num_procs/2) );
      // adios2_dc->SetLevelsOfDetail(2);
      adios2_dc->RegisterField("solution", u);
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
      if (!sout)
      {
         if (Mpi::Root())
         {
            cout << "Unable to connect to GLVis server at "
                 << vishost << ':' << visport << endl;
         }
         visualization = false;
         if (Mpi::Root())
         {
            cout << "GLVis visualization disabled.\n";
         }
      }
      else
      {
         sout << "parallel " << num_procs << " " << myid << "\n";
         sout.precision(precision);
         sout << "solution\n" << *pmesh << *u;
         sout << "pause\n";
         sout << flush;
         if (Mpi::Root())
         {
            cout << "GLVis visualization paused."
                 << " Press space (in the GLVis window) to resume it.\n";
         }
      }
   }

   // 10. Define the time-dependent evolution operator describing the ODE
   //     right-hand side, and perform time-integration (looping over the time
   //     iterations, ti, with a time-step dt).
   FE_Evolution adv(*m, *k, *B, prec_type);

   real_t t = 0.0;
   adv.SetTime(t);
   ode_solver->Init(adv);

   bool done = false;
   for (int ti = 0; !done; )
   {
      real_t dt_real = min(dt, t_final - t);
      ode_solver->Step(*U, t, dt_real);
      ti++;
     

      done = (t >= t_final - 1e-8*dt);

      if (done || ti % vis_steps == 0)
      {
         if (Mpi::Root())
         {
            cout << "time step: " << ti << ", time: " << t << endl;
         }

         // 11. Extract the parallel grid function corresponding to the finite
         //     element approximation U (the local solution on each processor).
         *u = *U;
   

         if (visualization)
         {
            sout << "parallel " << num_procs << " " << myid << "\n";
            sout << "solution\n" << *pmesh << *u << flush;
         }

         if (visit)
         {
            dc->SetCycle(ti);
            dc->SetTime(t);
            dc->Save();
            ComputeTotalIntegral(u, fes, pmesh);
         }

         if (paraview)
         {
            pd->SetCycle(ti);
            pd->SetTime(t);
            pd->Save();
         }

#ifdef MFEM_USE_ADIOS2
         // transient solutions can be visualized with ParaView
         if (adios2)
         {
            adios2_dc->SetCycle(ti);
            adios2_dc->SetTime(t);
            adios2_dc->Save();
         }
#endif
      }
   }

   // 12. Save the final solution in parallel. This output can be viewed later
   //     using GLVis: "glvis -np <np> -m ex9-mesh -g ex9-final".
   {
      *u = *U;
      ostringstream sol_name;
      sol_name << "ex9-final." << setfill('0') << setw(6) << myid;
      ofstream osol(sol_name.str().c_str());
      osol.precision(precision);
      u->Save(osol);
   }

   // 13. Free the used memory.
   delete U;
   delete u;
   delete B;
   delete b;
   delete k;
   delete m;
   delete fes;
   delete pmesh;
   delete ode_solver;
   delete pd;
   delete init_mesh;
#ifdef MFEM_USE_ADIOS2
   if (adios2)
   {
      delete adios2_dc;
   }
#endif
   delete dc;

   return 0;
}


// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(ParBilinearForm &M_, ParBilinearForm &K_,
                           const Vector &b_, PrecType prec_type)
   : TimeDependentOperator(M_.ParFESpace()->GetTrueVSize()), b(b_),
     M_solver(M_.ParFESpace()->GetComm()),
     z(height)
{
   if (M_.GetAssemblyLevel()==AssemblyLevel::LEGACY)
   {
      M.Reset(M_.ParallelAssemble(), true);
      K.Reset(K_.ParallelAssemble(), true);
   }
   else
   {
      M.Reset(&M_, false);
      K.Reset(&K_, false);
   }

   M_solver.SetOperator(*M);

   Array<int> ess_tdof_list;
   if (M_.GetAssemblyLevel()==AssemblyLevel::LEGACY)
   {
      HypreParMatrix &M_mat = *M.As<HypreParMatrix>();
      HypreParMatrix &K_mat = *K.As<HypreParMatrix>();
      HypreSmoother *hypre_prec = new HypreSmoother(M_mat, HypreSmoother::Jacobi);
      M_prec = hypre_prec;

      dg_solver = new DG_Solver(M_mat, K_mat, *M_.FESpace(), prec_type);
   }
   else
   {
      M_prec = new OperatorJacobiSmoother(M_, ess_tdof_list);
      dg_solver = NULL;
   }

   M_solver.SetPreconditioner(*M_prec);
   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
}

// Solve the equation:
//    u_t = M^{-1}(Ku + b),
// by solving associated linear system
//    (M - dt*K) d = K*u + b
void FE_Evolution::ImplicitSolve(const real_t dt, const Vector &x, Vector &k)
{
   K->Mult(x, z);
   z += b;
   dg_solver->SetTimeStep(dt);
   dg_solver->Mult(z, k);
}

void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
   // y = M^{-1} (K x + b)
   K->Mult(x, z);
   z += b;
   M_solver.Mult(z, y);
}

FE_Evolution::~FE_Evolution()
{
   delete M_prec;
   delete dg_solver;
}


// Velocity coefficient
void velocity_function(const Vector &x, Vector &v)
{
  int dim = x.Size();
  
  if (dim == 3){
    v(0) = 1.0;
    v(1) = 1.0;
    v(2) = 0.0;
  }
  else if(dim==2){
    v(0) = 1.0;
    v(1) = 1.0;
  }

}

// Initial condition
double u0_function(const Vector &x)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

   // double rx = 0.45, ry = 0.25, cx = 0., cy = -0.2, w = 10.;
   // return ( erfc(w*(X(0)-cx-rx))*erfc(-w*(X(0)-cx+rx)) *
   //          erfc(w*(X(1)-cy-ry))*erfc(-w*(X(1)-cy+ry)) )/16;
   
   // double center = (bb_min[0] + bb_max[0]) * 0.5;
   // return 0.5*(1 + tanh((x(0)- center)/integral_width));
   return sin(x(0))*cos(x(1))*cos(x(2));
   
   // return sin(2*M_PI*x(0));

}

// Inflow boundary condition (zero for the problems considered in this example)
real_t inflow_function(const Vector &x)
{
   switch (problem)
   {
      case 0:
      case 1:
      case 2:
      case 3: return 0.0;
   }
   return 0.0;
}


// This routine only works in serial
void ComputeAverageSolution(ParGridFunction* sol,ParFiniteElementSpace* fes, ParMesh* pmesh)
{

  double x =  0.5;
  double x_min = bb_min[0];
  double x_max = bb_max[0];
  double y_min = bb_min[1];
  double y_max = bb_max[1];
  double z_min = bb_min[2];
  double z_max = bb_max[2];

  // You get x_min and x_max
  // No Averaging -- we are selecting one of the 
  // elements at the boundary (can add average later)
  int ix = floor(((x-x_min)/(x_max-x_min))*nx);

  // To handle edge cases
  ix = min(ix,nx-1);
  ix = max(ix,0);
  double hx = (x_max-x_min)/nx;
  double hy = (y_max-y_min)/ny;
  double hz = (z_max-z_min)/nz;

  // if ix = 0 you get xmin
  // if ix = nx you get xmax
  // This is physical space
  double x_hat = (x-x_min)/hx - ix;

  // Create 2D integration rule
  const IntegrationRule &ir2D = IntRules.Get(Geometry::SQUARE, 2*order+2);

  // Inialiaze 3D integration rule with the name number of points as ir2D
  IntegrationRule ir3D(ir2D.GetNPoints());

  for (int i = 0; i < ir2D.GetNPoints(); i++){
    ir3D.IntPoint(i).x = x_hat;
    ir3D.IntPoint(i).y = ir2D.IntPoint(i).x;
    ir3D.IntPoint(i).z = ir2D.IntPoint(i).y;
    ir3D.IntPoint(i).weight = ir2D.IntPoint(i).weight;
  }

  double integral = 0.0;
  double cross_sectional_area = 0.0;
  double area_per_element = hy*hz;

  for (int iz = 0; iz < nz; iz++){
    for (int iy = 0; iy < ny; iy++){
      // MeshMakeCartesian3D ordering
      int i = ix + nx*(iy+ny*iz); // Map cartesian index to scalar index

      ElementTransformation *Trans = pmesh->GetElementTransformation(i);
      const IntegrationRule &ir = ir3D;

      for (int j = 0; j < ir.GetNPoints(); j++){
        const IntegrationPoint &ip = ir.IntPoint(j);
        Trans->SetIntPoint(&ip);

        double weight = ip.weight*area_per_element;
      
        double value = sol->GetValue(i,ip);
        integral += value * weight;
        cross_sectional_area += weight;
      }
    }
  }

  double average = integral / cross_sectional_area;
  std::cout<< "Average solution at x = " << x << " is " << average << std::endl;
  std::cout<< "cross_sectional_area at x = " << x << " is " << cross_sectional_area << std::endl;

}

void ComputeTotalIntegral(ParGridFunction* sol, ParFiniteElementSpace* fes, ParMesh* pmesh)
{
   // Inialize total integral and volume
   double total_volume   = 0.0;
   double total_integral = 0.0;

   // Get the order of the finite element space
   int order = fes->GetOrder(0);

   // Loop over the elements
   for (int e = 0; e < pmesh->GetNE(); e++)
   {

     // Determine the element type: triangle, quadrilateral, tetrahedron, etc.
     ElementTransformation *Trans = pmesh->GetElementTransformation(e);
     Geometry::Type geom_type = pmesh->GetElementBaseGeometry(e);

     // Create an integration rule appropriate for the elements
     const IntegrationRule * ir = &IntRules.Get(Trans->GetGeometryType(), 2*order+2);

     // Prepare to compute the integral and volume over this element
     double volume_per_cell = 0.0;
     double elem_integral = 0.0;

     // Loop over each quadrature point in the reference element
     for (int i = 0; i < ir->GetNPoints(); i++)
     {
        // Extract the current quadrature point from the integration rule
        const IntegrationPoint &ip = ir->IntPoint(i);
     
        // Prepare to evaluate the coordinate transformation at the current
        // quadrature point
        Trans->SetIntPoint(&ip);

        // Evaluate the solution at this integration point
        double sol_value = sol->GetValue(e, ip);
     
        // Compute the Jacobian determinant at the current integration point
        double detJ = Trans->Weight();
        double weight = ip.weight;

        volume_per_cell += weight*detJ;
        elem_integral   += sol_value * weight * detJ;

     }
     // Add the element's integral and volume to the total
     total_volume +=volume_per_cell;
     total_integral += elem_integral;
   }

  // If running in parallel, reduce the integral across all processors
  #ifdef MFEM_USE_MPI
  double global_integral = 0.0;
  double global_volume   = 0.0;

  MPI_Allreduce(&total_integral, &global_integral, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
  MPI_Allreduce(&total_volume,   &global_volume,   1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());

  double volume_averaged_sol; 
  if (pmesh->GetMyRank() == 0)
  {
    // This is the analytic solution for a tanh profile integrated 
    // from xmin to xmax
    double center = (bb_min[0] + bb_max[0]) * 0.5;
    double analytic_sol = 0.5*(bb_max[0] - bb_min[0]) 
                        + 0.5*integral_width*log(
                           (cosh(bb_max[0] - center)/integral_width)
                          /(cosh(bb_min[0] - center)/integral_width));

    volume_averaged_sol = global_integral/global_volume;

    std::cout << "Anaylicial solution: "                            << std::setprecision(4) << analytic_sol        << std::endl; 
    std::cout << "The volume-averaged solution is:  "               << std::setprecision(4) << volume_averaged_sol << std::endl;
    std::cout << "Total integral of the solution over the domain: " << std::setprecision(4) << global_integral     << std::endl;
    std::cout << "Total volume of the solution over the domain: "   << std::setprecision(4) << global_volume       << std::endl;
    std::cout << "Error : "                                         << std::setprecision(4) << std::abs(volume_averaged_sol-analytic_sol) << std::endl; 
  }
  #else
    // This is the analytic solution for a tanh profile integrated 
    // from xmin to xmax
    double center = (bb_min[0] + bb_max[0]) * 0.5;
    double analytic_sol = 0.5*(bb_max[0] - bb_min[0]) 
                        + 0.5*integral_width*log(
                           (cosh(bb_max[0] - center)/integral_width)
                          /(cosh(bb_min[0] - center)/integral_width));

    volume_averaged_sol = total_integral/total_volume;

    std::cout << "Anaylicial solution: "                            << std::setprecision(4) << analytic_sol        << std::endl; 
    std::cout << "The volume-averaged solution is:  "               << std::setprecision(4) << volume_averaged_sol << std::endl;
    std::cout << "Total integral of the solution over the domain: " << std::setprecision(4) << total_integral      << std::endl;
    std::cout << "Total volume of the solution over the domain: "   << std::setprecision(4) << total_volume        << std::endl;
    std::cout << "Error : "                                         << std::setprecision(4) << std::abs(volume_averaged_sol-analytic_sol) << std::endl; 
  #endif
}


void ComputeElementCenterValues(ParGridFunction* sol, ParFiniteElementSpace* fes, ParMesh* pmesh)
{
    // Local arrays to store the data
    std::vector<double> local_x, local_y, local_z, local_value;

    // Set the integration point to the center of the reference element
    IntegrationPoint ip;
    ip.Set3(0.5, 0.5, 0.5);  // Center of the reference element

    // Loop over local elements
    for (int i = 0; i < pmesh->GetNE(); i++)
    {
        // Get the element transformation
        ElementTransformation *Trans = pmesh->GetElementTransformation(i);

        // Evaluate the solution at the element center
        Trans->SetIntPoint(&ip);
        double value = sol->GetValue(*Trans, ip);

        // Transform the reference point to physical coordinates
        Vector phys_coords(3);
        Trans->Transform(ip, phys_coords);

        double x_center = phys_coords[0];
        double y_center = phys_coords[1];
        double z_center = phys_coords[2];

        // Store the data
        local_x.push_back(x_center);
        local_y.push_back(y_center);
        local_z.push_back(z_center);
        local_value.push_back(value);
    }

    // MPI setup
    MPI_Comm comm = pmesh->GetComm();
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Gather all data on rank 0
    std::vector<double> all_x, all_y, all_z, all_value;
    int local_num_elements = local_x.size();
    std::vector<int> all_num_elements(size);
    std::vector<int> displs(size);

    MPI_Gather(&local_num_elements, 1, MPI_INT, 
        all_num_elements.data(), 1, MPI_INT, 0, comm);

    if (rank == 0)
    {
        int total_elements = 0;
        displs[0] = 0;
        for (int i = 0; i < size; ++i)
        {
            total_elements += all_num_elements[i];
            if (i > 0)
            {
                displs[i] = displs[i - 1] + all_num_elements[i - 1];
            }
        }

        all_x.resize(total_elements);
        all_y.resize(total_elements);
        all_z.resize(total_elements);
        all_value.resize(total_elements);
    }

    MPI_Gatherv(local_x.data(), local_num_elements, MPI_DOUBLE, 
        all_x.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(local_y.data(), local_num_elements, MPI_DOUBLE, 
        all_y.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(local_z.data(), local_num_elements, MPI_DOUBLE, 
        all_z.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(local_value.data(), local_num_elements, MPI_DOUBLE, 
        all_value.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);

    // Write the data to a file in a human-readable format on rank 0
    if (rank == 0)
    {
        std::ofstream ofs("element_centers.txt");
        for (size_t i = 0; i < all_x.size(); ++i)
        {
            ofs << all_x[i] << " " << all_y[i] << " " << all_z[i] << " " << all_value[i] << std::endl;
        }
        ofs.close();
    }
    
}



void ComputeElementCenterAndLeftFaceValues(ParGridFunction* sol, ParFiniteElementSpace* fes, ParMesh* pmesh)
{
    // Local arrays to store the data
    std::vector<double> local_x_center, local_y_center, local_z_center, local_value_center;
    std::vector<double> local_x_left, local_y_left, local_z_left, local_value_left;

    // Set the integration points
    IntegrationPoint ip_center, ip_left;
    ip_center.Set3(0.5, 0.5, 0.5);  // Center of the reference element
    ip_left.Set3(0.0, 0.5, 0.5);    // Left face of the reference element

    // Loop over local elements
    for (int i = 0; i < pmesh->GetNE(); i++)
    {
        // Get the element transformation
        ElementTransformation *Trans = pmesh->GetElementTransformation(i);

        // Evaluate the solution at the element center
        Trans->SetIntPoint(&ip_center);
        double value_center = sol->GetValue(*Trans, ip_center);

        // Transform the reference point to physical coordinates
        Vector phys_coords_center(3);
        Trans->Transform(ip_center, phys_coords_center);

        // Store the center data
        local_x_center.push_back(phys_coords_center[0]);
        local_y_center.push_back(phys_coords_center[1]);
        local_z_center.push_back(phys_coords_center[2]);
        local_value_center.push_back(value_center);

        // Evaluate the solution at the left face
        Trans->SetIntPoint(&ip_left);
        double value_left = sol->GetValue(*Trans, ip_left);

        // Transform the reference point to physical coordinates
        Vector phys_coords_left(3);
        Trans->Transform(ip_left, phys_coords_left);

        // Store the left face data
        local_x_left.push_back(phys_coords_left[0]);
        local_y_left.push_back(phys_coords_left[1]);
        local_z_left.push_back(phys_coords_left[2]);
        local_value_left.push_back(value_left);
    }

    // MPI setup
    MPI_Comm comm = pmesh->GetComm();
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Gather all data on rank 0
    std::vector<double> all_x_center, all_y_center, all_z_center, all_value_center;
    std::vector<double> all_x_left, all_y_left, all_z_left, all_value_left;
    int local_num_elements = local_x_center.size();
    std::vector<int> all_num_elements(size);
    std::vector<int> displs(size);

    MPI_Gather(&local_num_elements, 1, MPI_INT, 
        all_num_elements.data(), 1, MPI_INT, 0, comm);

    if (rank == 0)
    {
        int total_elements = 0;
        displs[0] = 0;
        for (int i = 0; i < size; ++i)
        {
            total_elements += all_num_elements[i];
            if (i > 0)
            {
                displs[i] = displs[i - 1] + all_num_elements[i - 1];
            }
        }

        all_x_center.resize(total_elements);
        all_y_center.resize(total_elements);
        all_z_center.resize(total_elements);
        all_value_center.resize(total_elements);

        all_x_left.resize(total_elements);
        all_y_left.resize(total_elements);
        all_z_left.resize(total_elements);
        all_value_left.resize(total_elements);
    }

    MPI_Gatherv(local_x_center.data(), local_num_elements, MPI_DOUBLE, 
        all_x_center.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(local_y_center.data(), local_num_elements, MPI_DOUBLE, 
        all_y_center.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(local_z_center.data(), local_num_elements, MPI_DOUBLE, 
        all_z_center.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(local_value_center.data(), local_num_elements, MPI_DOUBLE, 
        all_value_center.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);

    MPI_Gatherv(local_x_left.data(), local_num_elements, MPI_DOUBLE, 
        all_x_left.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(local_y_left.data(), local_num_elements, MPI_DOUBLE, 
        all_y_left.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(local_z_left.data(), local_num_elements, MPI_DOUBLE, 
        all_z_left.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(local_value_left.data(), local_num_elements, MPI_DOUBLE, 
        all_value_left.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);

    // Write the data to a file in a human-readable format on rank 0
    if (rank == 0)
    {
        std::ofstream ofs("element_centers_and_left_faces.txt");
        for (size_t i = 0; i < all_x_center.size(); ++i)
        {
            // ofs << "Left Face: " << all_x_left[i] << " " << all_y_left[i] << " " << all_z_left[i] << " " << all_value_left[i] << std::endl;
            // ofs << "Center: " << all_x_center[i] << " " << all_y_center[i] << " " << all_z_center[i] << " " << all_value_center[i] << std::endl;
            ofs << all_x_left[i] << " " << all_y_left[i] << " " << all_z_left[i] << " " << all_value_left[i] << std::endl;
            ofs << all_x_center[i] << " " << all_y_center[i] << " " << all_z_center[i] << " " << all_value_center[i] << std::endl;
        }
        ofs.close();
    }
}

/* Need to implement
void ComputeElementCenterLeftAndRightFaceValues(ParGridFunction* sol, ParFiniteElementSpace* fes, ParMesh* pmesh)
{
    // Local arrays to store the data
    std::vector<double> local_x_center, local_y_center, local_z_center, local_value_center;
    std::vector<double> local_x_left, local_y_left, local_z_left, local_value_left;
    std::vector<double> local_x_right, local_y_right, local_z_right, local_value_right,

    // Set the integration points
    IntegrationPoint ip_center, ip_left;
    ip_center.Set3(0.5, 0.5, 0.5);  // Center of the reference element
    ip_left.Set3(0.0, 0.5, 0.5);    // Left face of the reference element
    ip_right.Set3(1.0, 0.5, 0.5);   // Right face of the reference element

    // Loop over local elements
    for (int i = 0; i < pmesh->GetNE(); i++)
    {
        // Get the element transformation
        ElementTransformation *Trans = pmesh->GetElementTransformation(i);

        // Evaluate the solution at the element center
        Trans->SetIntPoint(&ip_center);
        double value_center = sol->GetValue(*Trans, ip_center);

        // Transform the reference point to physical coordinates
        Vector phys_coords_center(3);
        Trans->Transform(ip_center, phys_coords_center);

        // Store the center data
        local_x_center.push_back(phys_coords_center[0]);
        local_y_center.push_back(phys_coords_center[1]);
        local_z_center.push_back(phys_coords_center[2]);
        local_value_center.push_back(value_center);

        // Evaluate the solution at the left face
        Trans->SetIntPoint(&ip_left);
        double value_left = sol->GetValue(*Trans, ip_left);

        // Transform the reference point to physical coordinates
        Vector phys_coords_left(3);
        Trans->Transform(ip_left, phys_coords_left);

        // Store the left face data
        local_x_left.push_back(phys_coords_left[0]);
        local_y_left.push_back(phys_coords_left[1]);
        local_z_left.push_back(phys_coords_left[2]);
        local_value_left.push_back(value_left);
    }

    // MPI setup
    MPI_Comm comm = pmesh->GetComm();
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Gather all data on rank 0
    std::vector<double> all_x_center, all_y_center, all_z_center, all_value_center;
    std::vector<double> all_x_left, all_y_left, all_z_left, all_value_left;
    int local_num_elements = local_x_center.size();
    std::vector<int> all_num_elements(size);
    std::vector<int> displs(size);

    MPI_Gather(&local_num_elements, 1, MPI_INT, 
        all_num_elements.data(), 1, MPI_INT, 0, comm);

    if (rank == 0)
    {
        int total_elements = 0;
        displs[0] = 0;
        for (int i = 0; i < size; ++i)
        {
            total_elements += all_num_elements[i];
            if (i > 0)
            {
                displs[i] = displs[i - 1] + all_num_elements[i - 1];
            }
        }

        all_x_center.resize(total_elements);
        all_y_center.resize(total_elements);
        all_z_center.resize(total_elements);
        all_value_center.resize(total_elements);

        all_x_left.resize(total_elements);
        all_y_left.resize(total_elements);
        all_z_left.resize(total_elements);
        all_value_left.resize(total_elements);
    }

    MPI_Gatherv(local_x_center.data(), local_num_elements, MPI_DOUBLE, 
        all_x_center.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(local_y_center.data(), local_num_elements, MPI_DOUBLE, 
        all_y_center.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(local_z_center.data(), local_num_elements, MPI_DOUBLE, 
        all_z_center.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(local_value_center.data(), local_num_elements, MPI_DOUBLE, 
        all_value_center.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);

    MPI_Gatherv(local_x_left.data(), local_num_elements, MPI_DOUBLE, 
        all_x_left.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(local_y_left.data(), local_num_elements, MPI_DOUBLE, 
        all_y_left.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(local_z_left.data(), local_num_elements, MPI_DOUBLE, 
        all_z_left.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(local_value_left.data(), local_num_elements, MPI_DOUBLE, 
        all_value_left.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);

    // Write the data to a file in a human-readable format on rank 0
    if (rank == 0)
    {
        std::ofstream ofs("element_centers_and_left_faces.txt");
        for (size_t i = 0; i < all_x_center.size(); ++i)
        {
            // ofs << "Left Face: " << all_x_left[i] << " " << all_y_left[i] << " " << all_z_left[i] << " " << all_value_left[i] << std::endl;
            // ofs << "Center: " << all_x_center[i] << " " << all_y_center[i] << " " << all_z_center[i] << " " << all_value_center[i] << std::endl;
            ofs << all_x_left[i] << " " << all_y_left[i] << " " << all_z_left[i] << " " << all_value_left[i] << std::endl;
            ofs << all_x_center[i] << " " << all_y_center[i] << " " << all_z_center[i] << " " << all_value_center[i] << std::endl;
        }
        ofs.close();
    }
}
*/

/* Needs to be debugged
void ComputeElementSurfaceAverageValues(ParGridFunction* sol, ParFiniteElementSpace* fes, ParMesh* pmesh)
{
    // Synchronize the ParGridFunction data across processors
    sol->ExchangeFaceNbrData();

    // Local arrays to store the data
    std::vector<double> local_x, local_y, local_z, local_value;

    // Determine the integration order
    int intorder = 2 * fes->GetOrder(0) + 3;

    // Loop over local elements
    for (int i = 0; i < pmesh->GetNE(); i++)
    {
        // Get the element transformation
        ElementTransformation *Trans = pmesh->GetElementTransformation(i);

        // Compute the physical coordinates of the element center
        IntegrationPoint ip_center;
        ip_center.Set3(0.5, 0.5, 0.5);  // Center of the reference hexahedron

        Vector phys_coords(3);
        Trans->Transform(ip_center, phys_coords);

        double x_center = phys_coords[0];
        double y_center = phys_coords[1];
        double z_center = phys_coords[2];

        // Initialize variables for surface integration
        double surface_integral = 0.0;
        double total_surface_area = 0.0;

        // Get the global element number
        int global_elem_no = pmesh->GetGlobalElementNum(i);

        // Get the face indices associated with this element
        Array<int> face_indices, orientation;
        pmesh->GetElementFaces(i, face_indices, orientation);

        // Loop over faces of the element
        for (int f = 0; f < face_indices.Size(); f++)
        {
            int face_idx = face_indices[f];

            // Get the face transformation, including neighbor elements
            FaceElementTransformations *face_trans = pmesh->GetFaceElementTransformations(face_idx, true);

            // Ensure face_trans is not null
            if (!face_trans)
            {
                mfem::mfem_error("FaceElementTransformations is null.");
                continue;
            }

            // Determine which side of the face to use
            ElementTransformation *face_elem_trans = nullptr;

            if (face_trans->Elem1No == global_elem_no)
            {
                face_elem_trans = face_trans->Elem1;
                if (!face_elem_trans)
                {
                    // Try to get face neighbor element transformation
                    face_elem_trans = pmesh->GetFaceNbrElementTransformation(face_trans->Elem1No);
                }
            }
            else if (face_trans->Elem2No == global_elem_no)
            {
                face_elem_trans = face_trans->Elem2;
                if (!face_elem_trans)
                {
                    // Try to get face neighbor element transformation
                    face_elem_trans = pmesh->GetFaceNbrElementTransformation(face_trans->Elem2No);
                }
            }
            else
            {
                // This face is not adjacent to the current element
                continue;
            }

            if (!face_elem_trans)
            {
                mfem::mfem_error("ElementTransformation is null.");
                continue;
            }

            // Get the face geometry type
            Geometry::Type face_geom = face_trans->FaceGeom;

            // Get the integration rule for the face
            const IntegrationRule &face_ir = IntRules.Get(face_geom, intorder);

            // Integrate over the face
            double face_integral = 0.0;
            double face_area = 0.0;

            for (int q = 0; q < face_ir.GetNPoints(); q++)
            {
                const IntegrationPoint &ip = face_ir.IntPoint(q);

                // Set the integration point in the face transformation
                face_trans->SetIntPoint(&ip);

                // Evaluate the solution at this point
                double value = sol->GetValue(*face_elem_trans, ip);

                // Compute the face Jacobian determinant (for area weighting)
                double weight = ip.weight * face_trans->Weight();

                // Accumulate the weighted solution value
                face_integral += value * weight;

                // Accumulate the face area
                face_area += weight;
            }

            // Accumulate the face integral and area to the total
            surface_integral += face_integral;
            total_surface_area += face_area;
        }

        // Compute the average over the surfaces
        double average_value = 0.0;
        if (total_surface_area != 0.0)
        {
            average_value = surface_integral / total_surface_area;
        }

        // Store the data
        local_x.push_back(x_center);
        local_y.push_back(y_center);
        local_z.push_back(z_center);
        local_value.push_back(average_value);
    }

    // MPI data gathering (same as your working code)
    MPI_Comm comm = pmesh->GetComm();
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Gather all data on rank 0
    std::vector<double> all_x, all_y, all_z, all_value;
    int local_num_elements = local_x.size();
    std::vector<int> all_num_elements(size);
    std::vector<int> displs(size);

    MPI_Gather(&local_num_elements, 1, MPI_INT,
               all_num_elements.data(), 1, MPI_INT, 0, comm);

    if (rank == 0)
    {
        int total_elements = 0;
        displs[0] = 0;
        for (int i = 0; i < size; ++i)
        {
            total_elements += all_num_elements[i];
            if (i > 0)
            {
                displs[i] = displs[i - 1] + all_num_elements[i - 1];
            }
        }

        all_x.resize(total_elements);
        all_y.resize(total_elements);
        all_z.resize(total_elements);
        all_value.resize(total_elements);
    }

    MPI_Gatherv(local_x.data(), local_num_elements, MPI_DOUBLE,
                all_x.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);

    MPI_Gatherv(local_y.data(), local_num_elements, MPI_DOUBLE,
                all_y.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);

    MPI_Gatherv(local_z.data(), local_num_elements, MPI_DOUBLE,
                all_z.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);

    MPI_Gatherv(local_value.data(), local_num_elements, MPI_DOUBLE,
                all_value.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);

    // Write the data to a file in a human-readable format on rank 0
    if (rank == 0)
    {
        std::ofstream ofs("element_surface_averages.txt");
        for (size_t i = 0; i < all_x.size(); ++i)
        {
            ofs << all_x[i] << " " << all_y[i] << " " << all_z[i] << " " << all_value[i] << std::endl;
        }
        ofs.close();
    }
}*/




// void ComputeElementCenterValues(ParGridFunction* sol, ParFiniteElementSpace* fes, ParMesh* pmesh)
// {
//     /*
//     // Get the mesh bounding box
//     Vector bb_min, bb_max;
//     pmesh->GetBoundingBox(bb_min, bb_max);
// 
//     double x_min = bb_min[0], x_max = bb_max[0];
//     double y_min = bb_min[1], y_max = bb_max[1];
//     double z_min = bb_min[2], z_max = bb_max[2];
// 
//     // Compute element sizes
//     double hx = (x_max - x_min) / nx;
//     double hy = (y_max - y_min) / ny;
//     double hz = (z_max - z_min) / nz;
// 
//     // Local arrays to store the data
//     std::vector<double> local_x, local_y, local_z, local_value;
// 
//     // Loop over local elements
//     for (int i = 0; i < pmesh->GetNE(); i++)
//     {
//         // Get the global element number
//         int global_elem_id = pmesh->GetGlobalElementNum(i);
// 
//         // Compute ix, iy, iz from global_elem_id
//         int nx_ny = nx * ny;
//         int iz = global_elem_id / nx_ny;
//         int rem = global_elem_id % nx_ny;
//         int iy = rem / nx;
//         int ix = rem % nx;
// 
//         // Compute physical coordinates of the element center
//         double x_center = x_min + (ix + 0.5) * hx;
//         double y_center = y_min + (iy + 0.5) * hy;
//         double z_center = z_min + (iz + 0.5) * hz;
// 
//         // Evaluate the solution at the center of the element
//         // Use reference coordinates at the center (0.5, 0.5, 0.5)
//         IntegrationPoint ip;
//         ip.Set3(0.5, 0.5, 0.5);
// 
//         // Get the element transformation
//         ElementTransformation *Trans = pmesh->GetElementTransformation(i);
//         Trans->SetIntPoint(&ip);
// 
//         // Evaluate the solution at the reference point
//         double value = sol->GetValue(*Trans, ip);
// 
//         // Store the data
//         local_x.push_back(x_center);
//         local_y.push_back(y_center);
//         local_z.push_back(z_center);
//         local_value.push_back(value);
//     }
// 
//     // Write the data to a file per processor
//     int rank = pmesh->GetMyRank();
//     std::ostringstream filename;
//     filename << "element_centers_rank_" << rank << ".txt";
//     std::ofstream ofs(filename.str());
//     for (size_t i = 0; i < local_x.size(); ++i)
//     {
//         ofs << local_x[i] << " " << local_y[i] << " " << local_z[i] << " " << local_value[i] << std::endl;
//     }
//     ofs.close();
//     */ 
// }

