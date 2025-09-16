#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Example run
// mpirun -n 8 ./exVectorDecomp -n 16 -o 2 -rs 1e-8 -pa

// Exact solution, E, and r.h.s., f. See below for implementation.
void A_exact(const Vector &x, Vector &A);
void curl_A_exact(const Vector &x, Vector &Acurl);
void w_exact(const Vector &x, Vector &f);
void u_exact(const Vector &x, Vector &A);
void grad_phi_exact(const Vector &x, Vector &u);

real_t freq = 1.0, kappa;
real_t delta_const = 1e-8;
bool static_cond = false;
int dim;
int sdim;
int order;

FiniteElementCollection *nd_fec = nullptr;
FiniteElementCollection *rt_fec = nullptr;
FiniteElementCollection *l2_fec = nullptr;
FiniteElementCollection *h1_fec = nullptr;

ParFiniteElementSpace *l2_fespace_scalar = nullptr; 
ParFiniteElementSpace *l2_fespace_vector = nullptr;

ParFiniteElementSpace *nd_fespace = nullptr;
ParFiniteElementSpace *rt_fespace = nullptr;

ParFiniteElementSpace *h1_fespace_scalar = nullptr;
ParFiniteElementSpace *h1_fespace_vector = nullptr;

class H1ToL2OrHdivProjector
{
private:
    ParFiniteElementSpace *test_fes;
    bool pa;
    
    // Store the mass matrix and solver for reuse
    std::unique_ptr<ParBilinearForm> mass_form;
    std::unique_ptr<CGSolver> cg_solver;
    std::unique_ptr<OperatorJacobiSmoother> jacobi_prec;
    std::unique_ptr<HypreParMatrix> mass_matrix;
    std::unique_ptr<HypreDiagScale> hypre_prec;
    std::unique_ptr<HyprePCG> hypre_solver;
    OperatorPtr mass_op;

    void SetUpOperator();
    
public:
    H1ToL2OrHdivProjector(ParFiniteElementSpace *test_space, 
                       bool partial_assembly); 

    void Apply(ParGridFunction &result, const ParGridFunction &u_h1) const;
};

class H1ToHdivOrHcurlProjector
{
private:
    ParFiniteElementSpace *test_fes;
    bool pa;
    
    // Store the mass matrix and solver for reuse
    std::unique_ptr<ParBilinearForm> mass_form;
    std::unique_ptr<CGSolver> cg_solver;
    std::unique_ptr<OperatorJacobiSmoother> jacobi_prec;
    std::unique_ptr<HypreParMatrix> mass_matrix;
    std::unique_ptr<HypreDiagScale> hypre_prec;
    std::unique_ptr<HyprePCG> hypre_solver;
    OperatorPtr mass_op;

    void SetUpOperator();
    
public:
   H1ToHdivOrHcurlProjector(ParFiniteElementSpace *test_space, 
                      bool partial_assembly); 

    void Apply(ParGridFunction &result, const ParGridFunction &u_h1) const;
};

class HcurlHdivProjector 
{
private:
    ParFiniteElementSpace *trial_fes;
    ParFiniteElementSpace *test_fes;
    bool pa;
    
    // Store the mass matrix and solver for reuse
    std::unique_ptr<ParBilinearForm> mass_form;
    std::unique_ptr<ParMixedBilinearForm> mixed_form;
    std::unique_ptr<CGSolver> cg_solver;
    std::unique_ptr<OperatorJacobiSmoother> jacobi_prec;
    std::unique_ptr<HypreParMatrix> mass_matrix;
    std::unique_ptr<HypreParMatrix> mixed_matrix;
    std::unique_ptr<HypreDiagScale> hypre_prec;
    std::unique_ptr<HyprePCG> hypre_solver;
    OperatorPtr mass_op;
    
    void SetUpOperator();
public:
    HcurlHdivProjector(ParFiniteElementSpace *trial_space, 
                       ParFiniteElementSpace *test_space, 
                       bool partial_assembly);

    void Apply(ParGridFunction &result, const ParGridFunction &gftrial) const;
};

class ComputeDivergenceHdivToL2
{
private:
    ParFiniteElementSpace *trial_fes;
    ParFiniteElementSpace *test_fes;
    bool pa;
    
    // Store the mass matrix and solver for reuse
    std::unique_ptr<ParBilinearForm> mass_form;
    std::unique_ptr<ParMixedBilinearForm> mixed_form;
    std::unique_ptr<CGSolver> cg_solver;
    std::unique_ptr<OperatorJacobiSmoother> jacobi_prec;
    std::unique_ptr<HypreParMatrix> mass_matrix;
    std::unique_ptr<HypreParMatrix> mixed_matrix;
    std::unique_ptr<HypreDiagScale> hypre_prec;
    std::unique_ptr<HyprePCG> hypre_solver;
    OperatorPtr mass_op;
    
    void SetUpOperator();
public:
    ComputeDivergenceHdivToL2(ParFiniteElementSpace *trial_space, 
                       ParFiniteElementSpace *test_space, 
                       bool partial_assembly);
    
    void Apply(ParGridFunction &result, const ParGridFunction &gftrial) const;
};

class ComputeCurlHcurlToHdiv
{
private:
    ParFiniteElementSpace *trial_fes;
    ParFiniteElementSpace *test_fes;
    bool pa;
    
    // Store the mass matrix and solver for reuse
    std::unique_ptr<ParBilinearForm> mass_form;
    std::unique_ptr<ParMixedBilinearForm> mixed_form;
    std::unique_ptr<CGSolver> cg_solver;
    std::unique_ptr<OperatorJacobiSmoother> jacobi_prec;
    std::unique_ptr<HypreParMatrix> mass_matrix;
    std::unique_ptr<HypreParMatrix> mixed_matrix;
    std::unique_ptr<HypreDiagScale> hypre_prec;
    std::unique_ptr<HyprePCG> hypre_solver;
    OperatorPtr mass_op;
    
    void SetUpOperator();

public:
    ComputeCurlHcurlToHdiv(ParFiniteElementSpace *trial_space, 
                       ParFiniteElementSpace *test_space, 
                       bool partial_assembly); 
    
    void Apply(ParGridFunction &result, const ParGridFunction &gftrial) const;
};

class ComputeGradientH1ScalarToHcurl
{
private:
    ParFiniteElementSpace *trial_fes;
    ParFiniteElementSpace *test_fes;
    bool pa;
    
    // Store the mass matrix and solver for reuse
    std::unique_ptr<ParBilinearForm> mass_form;
    std::unique_ptr<ParMixedBilinearForm> mixed_form;
    std::unique_ptr<CGSolver> cg_solver;
    std::unique_ptr<OperatorJacobiSmoother> jacobi_prec;
    std::unique_ptr<HypreParMatrix> mass_matrix;
    std::unique_ptr<HypreParMatrix> mixed_matrix;
    std::unique_ptr<HypreDiagScale> hypre_prec;
    std::unique_ptr<HyprePCG> hypre_solver;
    OperatorPtr mass_op;
        
    void SetUpOperator();
    
public:
    ComputeGradientH1ScalarToHcurl(ParFiniteElementSpace *trial_space, 
                       ParFiniteElementSpace *test_space, 
                       bool partial_assembly);
    
    void Apply(ParGridFunction &result, const ParGridFunction &gftrial) const;
};

struct ProjectorOps {

   H1ToHdivOrHcurlProjector projectorH1ToHdiv;
   H1ToL2OrHdivProjector projectorL2ToH1Scalar;
   ComputeGradientH1ScalarToHcurl projectorComputeGradientH1ScalarToHcurl;
   H1ToL2OrHdivProjector projectorH1ToL2;
   H1ToHdivOrHcurlProjector projectorH1ToHcurl;
   HcurlHdivProjector projectorHcurlToHdiv;
   HcurlHdivProjector projectorHdivToHcurl;
   ComputeDivergenceHdivToL2 projectorDivHdivToL2;
   ComputeCurlHcurlToHdiv projectorCurlHcurlToHdiv;
   H1ToL2OrHdivProjector projectorHdivToL2;
   H1ToL2OrHdivProjector projectorL2ToH1;

   ProjectorOps(ParFiniteElementSpace *h1_fespace_vector,
                ParFiniteElementSpace *h1_fespace_scalar,
                ParFiniteElementSpace *nd_fespace,
                ParFiniteElementSpace *rt_fespace,
                ParFiniteElementSpace *l2_fespace_vector, 
                ParFiniteElementSpace *l2_fespace_scalar, 
                bool pa)

   : projectorH1ToHdiv(rt_fespace, pa),
     projectorL2ToH1Scalar(h1_fespace_scalar, pa),
     projectorComputeGradientH1ScalarToHcurl(h1_fespace_scalar, nd_fespace, pa),
     projectorH1ToL2(l2_fespace_vector, pa),
     projectorH1ToHcurl(nd_fespace, pa),
     projectorHcurlToHdiv(rt_fespace, nd_fespace, pa),
     projectorHdivToHcurl(nd_fespace, rt_fespace, pa),
     projectorDivHdivToL2(rt_fespace,l2_fespace_scalar, pa),
     projectorCurlHcurlToHdiv(nd_fespace, rt_fespace, pa),
     projectorHdivToL2(l2_fespace_vector, pa),
     projectorL2ToH1(h1_fespace_vector, pa)
     {

     }
};

void solve_scalar_potential( const ProjectorOps& ops,
                             const ParGridFunction &u_h1,
                             ParGridFunction &grad_phi_h1,
                             ParMesh *pmesh, bool pa);

void solve_vector_potential( const ProjectorOps& ops,
                             const ParGridFunction &u_h1,
                             ParGridFunction &curl_Ah_h1,
                             ParMesh *pmesh, bool pa);

void verify_vector_potential(const ProjectorOps& ops,
                             const ParGridFunction &curl_Ah_h1,
                             ParMesh *pmesh, bool pa);

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   order = 1;
   static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = true;
   double length = 1.0;
   int num_el = 4;
   delta_const = 1e-4;
#ifdef MFEM_USE_AMGX
   bool useAmgX = false;
#endif

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact"
                  " solution.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&num_el, "-n", "--num_elements",
              "Number of elements per direction.");
   args.AddOption(&delta_const, "-rs", "--regulurization_scale",
              "Scaling for regularization");
#ifdef MFEM_USE_AMGX
   args.AddOption(&useAmgX, "-amgx", "--useAmgX", "-no-amgx",
                  "--no-useAmgX",
                  "Enable or disable AmgX in MatrixFreeAMS.");
#endif

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
   kappa = freq * M_PI;

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // 4. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh init_mesh = Mesh(Mesh::MakeCartesian3D(num_el,
                                              num_el,
                                              num_el,
                                              Element::HEXAHEDRON,
                                              length,
                                              length,
                                              length, false));

   Vector x_translation({length, 0.0, 0.0});
   Vector y_translation({0.0, length, 0.0});
   Vector z_translation({0.0, 0.0, length});

   std::vector<Vector> translations = {x_translation, y_translation, z_translation};

   Mesh *mesh = new Mesh(Mesh::MakePeriodic(init_mesh, init_mesh.CreatePeriodicVertexMapping(translations)));
   dim = mesh->Dimension();
   sdim = mesh->SpaceDimension();

   // VectorFunctionCoefficient translate_set_mesh(mesh->Dimension(), [&](const Vector &x_in, Vector &x_out){

   //    x_out[0] = x_in[0]*0.5; // Translate x-coordinate
   //    x_out[1] = x_in[1]*x_in[1]*2.0; // Translate y-coordinate
   //    if (mesh->Dimension() == 3)
   //    {
   //       x_out[2] = x_in[2]*0.75; // Translate z-coordinate
   //    }
   // });

   // // Apply translation to the mesh
   // mesh->Transform(translate_set_mesh);

   // 5. Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 1,000 elements.

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   {
      int par_ref_levels = 0;
      for (int l = 0; l < par_ref_levels; l++)
      {
         pmesh->UniformRefinement();
      }
   }

   nd_fec = new ND_FECollection(order, dim);
   rt_fec = new RT_FECollection(order-1, dim); // H(div)
   l2_fec = new L2_FECollection(order-1, dim);
   h1_fec = new H1_FECollection(order, dim);

   l2_fespace_scalar = new ParFiniteElementSpace(pmesh, l2_fec);
   l2_fespace_vector = new ParFiniteElementSpace(pmesh, l2_fec, dim);

   nd_fespace = new ParFiniteElementSpace(pmesh, nd_fec);
   rt_fespace = new ParFiniteElementSpace(pmesh, rt_fec);

   h1_fespace_scalar = new ParFiniteElementSpace(pmesh, h1_fec);
   h1_fespace_vector = new ParFiniteElementSpace(pmesh, h1_fec, dim);

   ProjectorOps ops(h1_fespace_vector,
                    h1_fespace_scalar,
                    nd_fespace,
                    rt_fespace,
                    l2_fespace_vector, 
                    l2_fespace_scalar, 
                    pa);



   HYPRE_BigInt size = nd_fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   // \nabla \times Ah in H(div)
   VectorFunctionCoefficient curl_Ah_exact_coeff(sdim, curl_A_exact);
   ParGridFunction curl_Ah_exact_h1(h1_fespace_vector);

   curl_Ah_exact_h1.ProjectCoefficient(curl_Ah_exact_coeff);

   VectorFunctionCoefficient u_coeff(sdim, u_exact);

   // 1. Define velocity spaces
   // Define u in H1
   ParGridFunction u_h1(h1_fespace_vector);
   u_h1.ProjectCoefficient(u_coeff);

   ParGridFunction curl_Ah_h1(h1_fespace_vector);
   solve_vector_potential(ops, u_h1, curl_Ah_h1, pmesh, pa);

   ParGridFunction curl_Ah_l2(l2_fespace_vector);
   ops.projectorH1ToL2.Apply(curl_Ah_l2, curl_Ah_h1);

   // Solve for scalar potential

   // 4. Solve Poisson problem \nabla^2 \phi = div(u)
   ParGridFunction grad_phi_h1(h1_fespace_vector);
   solve_scalar_potential(ops, u_h1, grad_phi_h1, pmesh, pa);

   ParGridFunction grad_phi_l2(l2_fespace_vector);
   ops.projectorH1ToL2.Apply(grad_phi_l2, grad_phi_h1);

   // Use ProjectDiscCoefficient for averaging-based projection from L2 to H1
   // Note that this approach destroys the curl free free property of the 
   // scalar potential. Instead perform an L2 projection.
   // VectorGridFunctionCoefficient grad_phi_l2_coeff(&grad_phi_l2);
   // grad_phi_h1.ProjectDiscCoefficient(grad_phi_l2_coeff);

   // Sanity Checks
   ParGridFunction u_l2(l2_fespace_vector);
   ops.projectorH1ToL2.Apply(u_l2, u_h1);

   ParGridFunction vel_error(l2_fespace_vector);
   vel_error = grad_phi_l2;
   vel_error += curl_Ah_l2;
   vel_error -= u_l2;

   // Subtract grad phi from u -- do we get a better curl Ah field?
   curl_Ah_l2 = u_l2;
   curl_Ah_l2 -= grad_phi_l2;

   ParGridFunction curl_Ah_h1_from_grad_phi(h1_fespace_vector);
   ops.projectorL2ToH1.Apply(curl_Ah_h1_from_grad_phi, curl_Ah_l2);

   verify_vector_potential(ops, curl_Ah_h1_from_grad_phi,
                             pmesh, pa);
   // verify_vector_potential(ops, curl_Ah_h1,
   //                           pmesh, pa);

   // Define ceofficients for comparison for later
   VectorFunctionCoefficient grad_phi_coeff(sdim, grad_phi_exact); // nabla \phi
   ParGridFunction grad_phi_exact_h1(h1_fespace_vector);
   grad_phi_exact_h1.ProjectCoefficient(grad_phi_coeff);

   // 15. Compute and print the L^2 norm of the error.
   {

      ConstantCoefficient zero(0.0);

      Vector zero_v(dim);
      zero_v = 0.0;
      VectorConstantCoefficient zero_vec(zero_v);

      // double curl_grad_phi_computed_error_project = curl_grad_phi_hdiv.ComputeL2Error(zero_vec);
      // double div_curl_A_error_l2 = div_curl_Ah_l2.ComputeL2Error(zero);
      double grad_phi_error_h1 = grad_phi_h1.ComputeL2Error(grad_phi_coeff);
      double curl_Ah_h1_error = curl_Ah_h1.ComputeL2Error(curl_Ah_exact_coeff);
      double curl_Ah_h1_error_from_grad_phi = curl_Ah_h1_from_grad_phi.ComputeL2Error(curl_Ah_exact_coeff);
      double total_vel_error = vel_error.ComputeL2Error(zero);
   

      if (myid == 0)
      {
         // cout << "div(curl A) H1 L2 norm (should be ~0): " << div_curl_A_error_l2 << endl;
         cout << "curl Ah H1 L2 norm: " << curl_Ah_h1_error << endl;
         cout << "curl Ah H1 L2 from grad phi norm: " << curl_Ah_h1_error_from_grad_phi << endl;

         // cout << "curl(grad phi) project L2 error (should be ~0): " << curl_grad_phi_computed_error_project << endl;
         cout << "grad_phi H1 L2 norm: " << grad_phi_error_h1 << endl;
         cout << "vel error from reconstruction: " << total_vel_error << endl;
      }
   }


   /*
   {
   // mesh and solution (already correct)
   ostringstream mesh_name, sol_name;
   mesh_name << "mesh." << setfill('0') << setw(6) << myid;
   sol_name  << "sol."  << setfill('0') << setw(6) << myid;

   ofstream mesh_ofs(mesh_name.str());
   mesh_ofs.precision(8);
   pmesh->Print(mesh_ofs);
   ofstream sol_ofs(sol_name.str());
   sol_ofs.precision(8);
   Ah.Save(sol_ofs);

   // curl(u_h) in H(curl)
   ostringstream curl_name;
   curl_name << "curl_u_hcurl." << setfill('0') << setw(6) << myid;
   ofstream curl_ofs(curl_name.str());
   curl_ofs.precision(8);
   curl_u.Save(curl_ofs);

   // curl(u_h) in H(curl)
   ostringstream curl_lin_sys_name;
   curl_lin_sys_name << "curl_u_lin_sys." << setfill('0') << setw(6) << myid;
   ofstream curl_lin_sys_ofs(curl_lin_sys_name.str());
   curl_lin_sys_ofs.precision(8);
   curl_u_hcurl.Save(curl_lin_sys_ofs);

   // exact curl for comparison
   ostringstream curl_ex_name;
   curl_ex_name << "curl_u_exact_hcurl." << setfill('0') << setw(6) << myid;
   ofstream curl_ex_ofs(curl_ex_name.str());
   curl_ex_ofs.precision(8);
   curl_u_exact.Save(curl_ex_ofs);

   }*/

    VisItDataCollection dc("VelocityDecomposition/output_visit", pmesh);
    dc.SetFormat(DataCollection::PARALLEL_FORMAT);
    dc.SetCycle(0);
    dc.SetTime(0.0);
    dc.SetPrecision(16);
    
    dc.RegisterField("velocity_total_h1", &u_h1);
    dc.RegisterField("curl_Ah_h1", &curl_Ah_h1);
    dc.RegisterField("curl_Ah_h1_from_grad_phi", &curl_Ah_h1_from_grad_phi);
    dc.RegisterField("curl_Ah_exact_h1", &curl_Ah_exact_h1);

    dc.RegisterField("grad_phi_h1",   &grad_phi_h1);
    dc.RegisterField("grad_phi_exact",   &grad_phi_exact_h1);

    dc.Save();


   delete nd_fespace;
   delete rt_fespace;
   delete h1_fespace_vector;
   delete h1_fespace_scalar;
   delete l2_fespace_vector;
   delete l2_fespace_scalar;
   delete nd_fec;
   delete rt_fec;
   delete h1_fec;
   delete l2_fec;
   delete pmesh;

   return 0;
}

void A_exact(const Vector &x, Vector &A)
{
   A(0) = -1/(4*M_PI)*cos(4*M_PI*x(2)) + 1/(6*M_PI)*cos(6*M_PI*x(1));
   A(1) = -1/(4*M_PI)*cos(4*M_PI*x(0)) + 1/(6*M_PI)*cos(6*M_PI*x(2));
   A(2) = -1/(4*M_PI)*cos(4*M_PI*x(1)) + 1/(6*M_PI)*cos(6*M_PI*x(0));
}

void w_exact(const Vector &x, Vector &f)
{
   if (dim == 3)
   {
      f(0) = 6*M_PI*cos(6*M_PI*x(1)) - 4*M_PI*cos(4*M_PI*x(2));
      f(1) = 6*M_PI*cos(6*M_PI*x(2)) - 4*M_PI*cos(4*M_PI*x(0));
      f(2) = 6*M_PI*cos(6*M_PI*x(0)) - 4*M_PI*cos(4*M_PI*x(1));
   }
   else
   {
      f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
      f(1) = (1. + kappa * kappa) * sin(kappa * x(0));
      if (x.Size() == 3) { f(2) = 0.0; }
   }
}

void u_exact(const Vector &x, Vector &A)
{
   real_t xi = 2*M_PI*x(0);
   real_t yi = 2*M_PI*x(1);
   real_t zi = 2*M_PI*x(2);
 
   A(0) = sin(xi) * cos(yi) * cos(zi);
   A(1) = -cos(xi) * sin(yi) * cos(zi);
   A(2) = 0.0;
   if (dim == 3)
   {
      A(0) = sin(2*M_PI*x(0)) + sin(4*M_PI*x(1)) + sin(6*M_PI*x(2));
      A(1) = sin(6*M_PI*x(0)) + sin(2*M_PI*x(1)) + sin(4*M_PI*x(2));
      A(2) = sin(4*M_PI*x(0)) + sin(6*M_PI*x(1)) + sin(2*M_PI*x(2));
   }
    else
    {
        A(0) = sin(kappa * x(1));
        A(1) = sin(kappa * x(0));
        if (x.Size() == 3) { A(2) = 0.0; }
    }    
}


// Compressive part (grad(phi))
void grad_phi_exact(const Vector &x, Vector &u)
{
    u(0) = sin(2*M_PI*x(0));
    u(1) = sin(2*M_PI*x(1));
    u(2) = sin(2*M_PI*x(2));
}

void curl_A_exact(const Vector &x, Vector &Acurl)
{
   if (dim == 3)
   {
      Acurl(0) = sin(4*M_PI*x(1)) + sin(6*M_PI*x(2));
      Acurl(1) = sin(6*M_PI*x(0)) + sin(4*M_PI*x(2));
      Acurl(2) = sin(4*M_PI*x(0)) + sin(6*M_PI*x(1));
   }
   else
   {
      Acurl(0) = (1. + kappa * kappa) * sin(kappa * x(1));
      Acurl(1) = (1. + kappa * kappa) * sin(kappa * x(0));
      if (x.Size() == 3) { Acurl(2) = 0.0; }
   }
}


void solve_vector_potential( const ProjectorOps& ops,
                             const ParGridFunction &u_h1,
                             ParGridFunction &curl_Ah_h1,
                             ParMesh *pmesh, bool pa)
{
   int myid = Mpi::WorldRank();

   // 2. Peform the needed projections
   // Project u in H1 to Hcurl
   ParGridFunction u_hcurl(nd_fespace);

   ops.projectorH1ToHcurl.Apply(u_hcurl, u_h1);

   // Compute the curl of u
   ParGridFunction curl_u(rt_fespace);
   curl_u = 0.0;

   // We can also solve a linear system to move form one space to another
   ops.projectorCurlHcurlToHdiv.Apply(curl_u, u_hcurl);

   // The test space which is being projected to is
   // H(curl) from the trial space H(div)
   // Note that the trial space needs to not be empyt ie.
   // be projected to

   ParGridFunction curl_u_hcurl(nd_fespace);

   ops.projectorHcurlToHdiv.Apply(curl_u_hcurl, curl_u);

   // Project the exact space for comparison later
   VectorFunctionCoefficient curl_u_coeff_exact(dim, w_exact);
   ParGridFunction curl_u_exact(nd_fespace);
   curl_u_exact.ProjectCoefficient(curl_u_coeff_exact);

   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   Array<int> ess_bdr;

   if (pmesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr = 0;
   }

   // 9. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (f,phi_i) where f is given by the function f_exact and phi_i are the
   //    basis functions in the finite element fespace.
   VectorGridFunctionCoefficient f(&curl_u_hcurl);
   ParLinearForm *b = new ParLinearForm(nd_fespace);
   b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
   b->Assemble();

   // 10. Define the solution vector x as a parallel finite element grid function
   //     corresponding to fespace. Initialize x by projecting the exact
   //     solution. Note that only values from the boundary edges will be used
   //     when eliminating the non-homogeneous boundary condition to modify the
   //     r.h.s. vector b.
   ParGridFunction x(nd_fespace);
   x = 0.0;

   // 11. Set up the parallel bilinear form corresponding to the EM diffusion
   //     operator curl muinv curl + sigma I, by adding the curl-curl and the
   //     mass domain integrators.
   Coefficient *muinv = new ConstantCoefficient(1.0);
   Coefficient *sigma = new ConstantCoefficient(delta_const);
   ParBilinearForm *a = new ParBilinearForm(nd_fespace);
   if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a->AddDomainIntegrator(new CurlCurlIntegrator(*muinv));
   a->AddDomainIntegrator(new VectorFEMassIntegrator(*sigma));

   // 12. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   OperatorPtr A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   // 13. Solve the system AX=B using PCG with an AMS preconditioner.
   if (pa)
   {
#ifdef MFEM_USE_AMGX
      MatrixFreeAMS ams(*a, *A, *nd_fespace, muinv, sigma, NULL, ess_bdr, useAmgX);
#else
      MatrixFreeAMS ams(*a, *A, *nd_fespace, muinv, sigma, NULL, ess_bdr);

      // std::unique_ptr<Solver> solver;
      // solver.reset(new LORSolver<HypreBoomerAMG>(*a, ess_tdof_list));
      // solver.reset(new OperatorJacobiSmoother(*a, ess_tdof_list));
#endif
      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(1000);
      cg.SetPrintLevel(1);
      cg.SetOperator(*A);
      cg.SetPreconditioner(ams);
      // cg.SetPreconditioner(*solver);
      cg.Mult(B, X);
   }
   else
   {
      ParFiniteElementSpace *prec_ndfespace =
         (a->StaticCondensationIsEnabled() ? a->SCParFESpace() : nd_fespace);
      HypreAMS ams(*A.As<HypreParMatrix>(), prec_ndfespace);
      // ams.SetSingularProblem();
      HyprePCG pcg(*A.As<HypreParMatrix>());
      pcg.SetTol(1e-12);
      pcg.SetMaxIter(500);
      pcg.SetPrintLevel(2);
      pcg.SetPreconditioner(ams);
      pcg.Mult(B, X);
   }

   // 14. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a->RecoverFEMSolution(X, *b, x);

   // 3. Solve for the vector potential
   ParGridFunction Ah(nd_fespace);
   Ah = x;

   // Compute the curl of the vector potential which is the divergence 
   // free part of the velocity field

   // Compute curl of Ah in H(div)
   ParGridFunction curl_Ah(rt_fespace);
   ops.projectorCurlHcurlToHdiv.Apply(curl_Ah, Ah);

   // 4. Verification part to make sure field is divergence free

   // Verification of divergence free field
   // The test space which is being projected to is
   // H(div) from the trial space H(curl)
   ParGridFunction Ah_hdiv(rt_fespace);
   ops.projectorHdivToHcurl.Apply(Ah_hdiv, Ah);

   // Set \nabla \cdot (\nabla \times Ah) to be in L2
   // Compute \nabla \cdot (\nabla \times Ah) in H(div)
   ParGridFunction div_curl_Ah(l2_fespace_scalar);
   ParGridFunction div_Ah(l2_fespace_scalar);

   ops.projectorDivHdivToL2.Apply(div_curl_Ah, curl_Ah);
   ops.projectorDivHdivToL2.Apply(div_Ah, Ah_hdiv);

   // 5. Move curl of vector potential to H1 for visualization for later
   ParGridFunction curl_Ah_l2(l2_fespace_vector);
   ops.projectorHdivToL2.Apply(curl_Ah_l2, curl_Ah);

   // Project from L2 to H1 by solving linear system
   ops.projectorL2ToH1.Apply(curl_Ah_h1, curl_Ah_l2);

   VectorFunctionCoefficient A_coeff(sdim, A_exact);
   VectorFunctionCoefficient curl_Ah_exact_coeff(sdim, curl_A_exact);

   {
      ConstantCoefficient zero(0.0);
      Vector zero_v(dim);
      zero_v = 0.0;
      VectorConstantCoefficient zero_vec(zero_v);

      double div_curl_A_error = div_curl_Ah.ComputeL2Error(zero);
      double div_A_error = div_Ah.ComputeL2Error(zero);
      double curl_Ah_h1_error = curl_Ah_h1.ComputeL2Error(curl_Ah_exact_coeff);
      double curl_Ah_error = curl_Ah.ComputeL2Error(curl_Ah_exact_coeff);
      double error = Ah.ComputeL2Error(A_coeff);
   

      if (myid == 0)
      {
         cout << "\n|| A_h - A ||_{L^2} = " << error << '\n' << endl;
         cout << "div(curl A) L2 norm (should be ~0): " << div_curl_A_error << endl;
         cout << "div(A) L2 norm (should be ~0): " << div_A_error << endl;
         cout << "curl Ah H1 L2 norm: " << curl_Ah_h1_error << endl;
         cout << "curl Ah L2 norm: " << curl_Ah_error << endl;
      }
   }


   // 18. Free the used memory.
   delete a;
   delete sigma;
   delete muinv;
   delete b;

}

void solve_scalar_potential( const ProjectorOps& ops,
                             const ParGridFunction &u_h1,
                             ParGridFunction &grad_phi_h1,
                             ParMesh *pmesh, bool pa)
{
   int myid = Mpi::WorldRank();

   // 1. Project u from H1 to Hdiv
   ParGridFunction u_hdiv(rt_fespace);
   ops.projectorH1ToHdiv.Apply(u_hdiv, u_h1);
   
   // 2. Divergence of u in L2 space
   ParGridFunction div_u_l2(l2_fespace_scalar);
   ops.projectorDivHdivToL2.Apply(div_u_l2, u_hdiv);
   
   // 2. Project div u from L2 to H1 for decomposition
   ParGridFunction div_u_h1(h1_fespace_scalar);
   ops.projectorL2ToH1Scalar.Apply(div_u_h1, div_u_l2);
   
   // 4. Solve Poisson problem \nabla^2 \phi = div(u)
   
   // Set up Laplacian operator in H1 space
   ParBilinearForm laplacian(h1_fespace_scalar);
   if (pa) { laplacian.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   laplacian.AddDomainIntegrator(new DiffusionIntegrator());
   laplacian.Assemble();
   if (!pa) { laplacian.Finalize(); }
   
   // Set up RHS using div_u_h1 (both already in same H1 space)
   GridFunctionCoefficient div_u_coeff(&div_u_h1);
   ParLinearForm rhs(h1_fespace_scalar);
   rhs.AddDomainIntegrator(new DomainLFIntegrator(div_u_coeff));
   rhs.Assemble();
   
   Vector RHS(h1_fespace_scalar->GetTrueVSize());
   Vector PHI(h1_fespace_scalar->GetTrueVSize());
   rhs.ParallelAssemble(RHS);
   RHS *= -1.0;
   PHI = 0.0;
   
   // Use OrthoSolver to handle null space (constant functions)
   Array<int> empty_ess_tdof;  // No essential BC for periodic problem
   
   if (pa)
   {
      OperatorPtr laplacian_op;
      laplacian.FormSystemMatrix(empty_ess_tdof, laplacian_op);
      
      // Create preconditioner
      OperatorJacobiSmoother jac(laplacian, empty_ess_tdof);
      
      // Set up base solver
      CGSolver base_solver(h1_fespace_scalar->GetComm());
      base_solver.SetRelTol(1e-12);
      base_solver.SetMaxIter(1000);
      base_solver.SetPrintLevel(0);  // Reduce output since OrthoSolver will print
      base_solver.SetOperator(*laplacian_op);
      base_solver.SetPreconditioner(jac);
      
      // Create OrthoSolver to handle null space
      OrthoSolver ortho_solver(h1_fespace_scalar->GetComm());
      ortho_solver.SetSolver(base_solver);
      ortho_solver.SetOperator(*laplacian_op);
      
      if (myid == 0) 
      {
         cout << "Using OrthoSolver for direct Poisson problem with null space" << endl;
      }
      
      ortho_solver.Mult(RHS, PHI);
   }
   else
   {
      std::unique_ptr<HypreParMatrix> A(laplacian.ParallelAssemble());
      
      // Create base preconditioner  
      HypreBoomerAMG amg(*A);
      amg.SetPrintLevel(0);
      
      // Set up base solver
      HyprePCG base_solver(*A);
      base_solver.SetTol(1e-12);
      base_solver.SetMaxIter(1000);
      base_solver.SetPrintLevel(0);  // Reduce output
      base_solver.SetPreconditioner(amg);
      
      // Create OrthoSolver to handle null space
      OrthoSolver ortho_solver(h1_fespace_scalar->GetComm());
      ortho_solver.SetSolver(base_solver);
      ortho_solver.SetOperator(*A);
      
      if (myid == 0) 
      {
         cout << "Using OrthoSolver for direct Poisson problem with null space" << endl;
      }
      
      ortho_solver.Mult(RHS, PHI);
   }
   
   // Set the solution
   ParGridFunction phi_scalar(h1_fespace_scalar);
   phi_scalar = 0.0;
   
   phi_scalar.SetFromTrueDofs(PHI);

   if (myid == 0)
   {
      cout << "Solved direct Poisson problem -∇²φ = div(u) for scalar potential" << endl;
   }

   // 5. Compute compressive part of velocify field
   ParGridFunction grad_phi(nd_fespace);
   ops.projectorComputeGradientH1ScalarToHcurl.Apply(grad_phi, phi_scalar);
   
   // 6. Compute curl of grad_phi for verification for later
   ParGridFunction curl_grad_phi(rt_fespace);
   ops.projectorCurlHcurlToHdiv.Apply(curl_grad_phi, grad_phi);

   // 7. Project grad phi from from Hcurl to H1
   ParGridFunction grad_phi_hdiv(rt_fespace);
   ops.projectorHdivToHcurl.Apply(grad_phi_hdiv,grad_phi);

   ParGridFunction grad_phi_l2(l2_fespace_vector);

   ops.projectorHdivToL2.Apply(grad_phi_l2, grad_phi_hdiv);

   ops.projectorL2ToH1.Apply(grad_phi_h1, grad_phi_l2);

   VectorFunctionCoefficient grad_phi_coeff(sdim, grad_phi_exact); // nabla \phi

   {

      ConstantCoefficient zero(0.0);

      Vector zero_v(dim);
      zero_v = 0.0;
      VectorConstantCoefficient zero_vec(zero_v);

      double curl_grad_phi_computed_error = curl_grad_phi.ComputeL2Error(zero_vec);
      double grad_phi_error = grad_phi.ComputeL2Error(grad_phi_coeff);
      double grad_phi_error_h1 = grad_phi_h1.ComputeL2Error(grad_phi_coeff);
   

      if (myid == 0)
      {
         cout << "curl(grad phi) L2 error (should be ~0): " << curl_grad_phi_computed_error << endl;
         cout << "grad_phi L2 norm: " << grad_phi_error << endl;
         cout << "grad_phi H1 L2 norm: " << grad_phi_error_h1 << endl;
      }
   }

   // 18. Free the used memory.
}

H1ToL2OrHdivProjector::H1ToL2OrHdivProjector(ParFiniteElementSpace *test_space, 
                   bool partial_assembly) 
    : test_fes(test_space), pa(partial_assembly)
{
    SetUpOperator();
}

void H1ToL2OrHdivProjector::SetUpOperator()
{
    // Create mass matrix on ND space (this is the expensive part)
    mass_form = std::make_unique<ParBilinearForm>(test_fes);
    if (pa) { mass_form->SetAssemblyLevel(AssemblyLevel::PARTIAL); }

    if (test_fes->GetVDim() == 1){
        mass_form->AddDomainIntegrator(new MassIntegrator());
    } else {
        mass_form->AddDomainIntegrator(new VectorMassIntegrator());
    }

    mass_form->Assemble();
    if (!pa) { mass_form->Finalize(); }
    
    // Setup solver for the mass matrix
    if (pa)
    {
        Array<int> ess_tdof_list; // empty for L2 projection
        mass_form->FormSystemMatrix(ess_tdof_list, mass_op);
        
        jacobi_prec = std::make_unique<OperatorJacobiSmoother>(*mass_form, ess_tdof_list);
        cg_solver = std::make_unique<CGSolver>(test_fes->GetComm());
        cg_solver->SetRelTol(1e-12);
        cg_solver->SetMaxIter(500);
        cg_solver->SetPrintLevel(0);
        cg_solver->SetOperator(*mass_op);
        cg_solver->SetPreconditioner(*jacobi_prec);
    }
    else
    {
        mass_matrix = std::unique_ptr<HypreParMatrix>(mass_form->ParallelAssemble());
        hypre_prec = std::make_unique<HypreDiagScale>(*mass_matrix);
        hypre_solver = std::make_unique<HyprePCG>(*mass_matrix);
        hypre_solver->SetTol(1e-12);
        hypre_solver->SetMaxIter(500);
        hypre_solver->SetPrintLevel(0);
        hypre_solver->SetPreconditioner(*hypre_prec);
    }
}

void H1ToL2OrHdivProjector::Apply(ParGridFunction &result, const ParGridFunction &u_h1) const
{
    ParLinearForm b(test_fes);

    if (test_fes->GetVDim() == 1){
        GridFunctionCoefficient ucoeff(&u_h1);
        b.AddDomainIntegrator(new DomainLFIntegrator(ucoeff));
    } else {
        VectorGridFunctionCoefficient ucoeff(&u_h1);
        b.AddDomainIntegrator(new VectorDomainLFIntegrator(ucoeff));
    }

    b.Assemble();
    
    Vector B(test_fes->GetTrueVSize()), X(test_fes->GetTrueVSize());
    b.ParallelAssemble(B);
    X = 0.0;
    
    if (pa)
    {
        cg_solver->Mult(B, X);
    }
    else
    {
        hypre_solver->Mult(B, X);
    }
    
    result = 0.0;
    result.SetFromTrueDofs(X);
}

 H1ToHdivOrHcurlProjector::H1ToHdivOrHcurlProjector(ParFiniteElementSpace *test_space, 
                       bool partial_assembly) 
        : test_fes(test_space), pa(partial_assembly)
 {
    SetUpOperator();
 }

 void H1ToHdivOrHcurlProjector::SetUpOperator()
{
    // Create mass matrix on ND space (this is the expensive part)
    mass_form = std::make_unique<ParBilinearForm>(test_fes);
    if (pa) { mass_form->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
    mass_form->AddDomainIntegrator(new VectorFEMassIntegrator());
    mass_form->Assemble();
    if (!pa) { mass_form->Finalize(); }
    
    // Setup solver for the mass matrix
    if (pa)
    {
        Array<int> ess_tdof_list; // empty for L2 projection
        mass_form->FormSystemMatrix(ess_tdof_list, mass_op);
        
        jacobi_prec = std::make_unique<OperatorJacobiSmoother>(*mass_form, ess_tdof_list);
        cg_solver = std::make_unique<CGSolver>(test_fes->GetComm());
        cg_solver->SetRelTol(1e-12);
        cg_solver->SetMaxIter(500);
        cg_solver->SetPrintLevel(0);
        cg_solver->SetOperator(*mass_op);
        cg_solver->SetPreconditioner(*jacobi_prec);
    }
    else
    {
        mass_matrix = std::unique_ptr<HypreParMatrix>(mass_form->ParallelAssemble());
        hypre_prec = std::make_unique<HypreDiagScale>(*mass_matrix);
        hypre_solver = std::make_unique<HyprePCG>(*mass_matrix);
        hypre_solver->SetTol(1e-12);
        hypre_solver->SetMaxIter(500);
        hypre_solver->SetPrintLevel(0);
        hypre_solver->SetPreconditioner(*hypre_prec);
    }
}

void H1ToHdivOrHcurlProjector::Apply(ParGridFunction &result, const ParGridFunction &u_h1) const
{
    VectorGridFunctionCoefficient ucoeff(&u_h1);
    ParLinearForm b(test_fes);
    b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(ucoeff));
    b.Assemble();
    
    Vector B(test_fes->GetTrueVSize()), X(test_fes->GetTrueVSize());
    b.ParallelAssemble(B);  // This B depends on u_h1!
    X = 0.0;
    
    if (pa)
    {
        cg_solver->Mult(B, X);  // Solve M*X = B
    }
    else
    {
        hypre_solver->Mult(B, X);  // Solve M*X = B
    }
    
    result = 0.0;
    result.SetFromTrueDofs(X);
}

 HcurlHdivProjector::HcurlHdivProjector(ParFiniteElementSpace *trial_space, 
                       ParFiniteElementSpace *test_space, 
                       bool partial_assembly) 
        : trial_fes(trial_space), test_fes(test_space), pa(partial_assembly)
 {
     SetUpOperator();
 }

 void HcurlHdivProjector::SetUpOperator()
 {
     mass_form = std::make_unique<ParBilinearForm>(test_fes);
     if (pa) { mass_form->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
     mass_form->AddDomainIntegrator(new VectorFEMassIntegrator());
     mass_form->Assemble();
     if (!pa) { mass_form->Finalize(); }

     mixed_form = std::make_unique<ParMixedBilinearForm>(trial_fes, test_fes);
     if (pa) { mixed_form->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
     mixed_form->AddDomainIntegrator(new VectorFEMassIntegrator());
     mixed_form->Assemble();
     if (!pa) { mixed_form->Finalize(); }
     
     // Setup solver for the mass matrix
     if (pa)
     {
         Array<int> ess_tdof_list; // empty for L2 projection
         mass_form->FormSystemMatrix(ess_tdof_list, mass_op);

         jacobi_prec = std::make_unique<OperatorJacobiSmoother>(*mass_form, ess_tdof_list);
         cg_solver = std::make_unique<CGSolver>(test_fes->GetComm());
         cg_solver->SetRelTol(1e-12);
         cg_solver->SetMaxIter(500);
         cg_solver->SetPrintLevel(0);
         cg_solver->SetOperator(*mass_op);
         cg_solver->SetPreconditioner(*jacobi_prec);
     }
     else
     {
         mixed_matrix = std::unique_ptr<HypreParMatrix>(mixed_form->ParallelAssemble());
         mass_matrix = std::unique_ptr<HypreParMatrix>(mass_form->ParallelAssemble());
         hypre_prec = std::make_unique<HypreDiagScale>(*mass_matrix);
         hypre_solver = std::make_unique<HyprePCG>(*mass_matrix);
         hypre_solver->SetTol(1e-12);
         hypre_solver->SetMaxIter(500);
         hypre_solver->SetPrintLevel(0);
         hypre_solver->SetPreconditioner(*hypre_prec);
     }
 }

void HcurlHdivProjector::Apply(ParGridFunction &result, const ParGridFunction &gftrial) const
{
    Vector B(test_fes->GetTrueVSize());
    Vector X(test_fes->GetTrueVSize());

    if (pa)
    {
       ParLinearForm b(test_fes); // used as a vector
       mixed_form->Mult(gftrial, b); // process-local multiplication
       b.ParallelAssemble(B);
    }
    else
    {

       Vector P(trial_fes->GetTrueVSize());
       gftrial.GetTrueDofs(P);
       mixed_matrix->Mult(P,B);
    }

    X = 0.0;
    if(pa)
    {
     cg_solver->Mult(B,X);
    }else{
     hypre_solver->Mult(B,X);
    }
    result.SetFromTrueDofs(X);
}

ComputeDivergenceHdivToL2::ComputeDivergenceHdivToL2(ParFiniteElementSpace *trial_space, 
                      ParFiniteElementSpace *test_space, 
                      bool partial_assembly) 
       : trial_fes(trial_space), test_fes(test_space), pa(partial_assembly)
{
    SetUpOperator();
}

void ComputeDivergenceHdivToL2::SetUpOperator()
{
    mass_form = std::make_unique<ParBilinearForm>(test_fes);
    if (pa) { mass_form->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
    mass_form->AddDomainIntegrator(new MassIntegrator());
    mass_form->Assemble();
    if (!pa) { mass_form->Finalize(); }

    mixed_form = std::make_unique<ParMixedBilinearForm>(trial_fes, test_fes);
    if (pa) { mixed_form->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
    mixed_form->AddDomainIntegrator(new VectorFEDivergenceIntegrator());
    mixed_form->Assemble();
    if (!pa) { mixed_form->Finalize(); }
    
    // Setup solver for the mass matrix
    if (pa)
    {
        Array<int> ess_tdof_list; // empty for L2 projection
        mass_form->FormSystemMatrix(ess_tdof_list, mass_op);

        jacobi_prec = std::make_unique<OperatorJacobiSmoother>(*mass_form, ess_tdof_list);
        cg_solver = std::make_unique<CGSolver>(test_fes->GetComm());
        cg_solver->SetRelTol(1e-12);
        cg_solver->SetMaxIter(500);
        cg_solver->SetPrintLevel(0);
        cg_solver->SetOperator(*mass_op);
        cg_solver->SetPreconditioner(*jacobi_prec);
    }
    else
    {
        mixed_matrix = std::unique_ptr<HypreParMatrix>(mixed_form->ParallelAssemble());
        mass_matrix = std::unique_ptr<HypreParMatrix>(mass_form->ParallelAssemble());
        hypre_prec = std::make_unique<HypreDiagScale>(*mass_matrix);
        hypre_solver = std::make_unique<HyprePCG>(*mass_matrix);
        hypre_solver->SetTol(1e-12);
        hypre_solver->SetMaxIter(500);
        hypre_solver->SetPrintLevel(0);
        hypre_solver->SetPreconditioner(*hypre_prec);
    }
}
void ComputeDivergenceHdivToL2::Apply(ParGridFunction &result, const ParGridFunction &gftrial) const
{
    Vector B(test_fes->GetTrueVSize());
    Vector X(test_fes->GetTrueVSize());

    if (pa)
    {
       ParLinearForm b(test_fes); // used as a vector
       mixed_form->Mult(gftrial, b); // process-local multiplication
       b.ParallelAssemble(B);
    }
    else
    {

       Vector P(trial_fes->GetTrueVSize());
       gftrial.GetTrueDofs(P);
       mixed_matrix->Mult(P,B);
    }

    X = 0.0;
    if(pa)
    {
     cg_solver->Mult(B,X);
    }else{
     hypre_solver->Mult(B,X);
    }
    result.SetFromTrueDofs(X);
}

ComputeCurlHcurlToHdiv::ComputeCurlHcurlToHdiv(ParFiniteElementSpace *trial_space, 
                       ParFiniteElementSpace *test_space, 
                       bool partial_assembly) 
        : trial_fes(trial_space), test_fes(test_space), pa(partial_assembly)
{
    SetUpOperator();
}

void ComputeCurlHcurlToHdiv::SetUpOperator()
{
    mass_form = std::make_unique<ParBilinearForm>(test_fes);
    if (pa) { mass_form->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
    mass_form->AddDomainIntegrator(new VectorFEMassIntegrator());
    mass_form->Assemble();
    if (!pa) { mass_form->Finalize(); }

    mixed_form = std::make_unique<ParMixedBilinearForm>(trial_fes, test_fes);
    if (pa) { mixed_form->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
    mixed_form->AddDomainIntegrator(new MixedVectorCurlIntegrator());
    mixed_form->Assemble();
    if (!pa) { mixed_form->Finalize(); }
    
    // Setup solver for the mass matrix
    if (pa)
    {
        Array<int> ess_tdof_list; // empty for L2 projection
        mass_form->FormSystemMatrix(ess_tdof_list, mass_op);

        jacobi_prec = std::make_unique<OperatorJacobiSmoother>(*mass_form, ess_tdof_list);
        cg_solver = std::make_unique<CGSolver>(test_fes->GetComm());
        cg_solver->SetRelTol(1e-12);
        cg_solver->SetMaxIter(500);
        cg_solver->SetPrintLevel(0);
        cg_solver->SetOperator(*mass_op);
        cg_solver->SetPreconditioner(*jacobi_prec);
    }
    else
    {
        mixed_matrix = std::unique_ptr<HypreParMatrix>(mixed_form->ParallelAssemble());
        mass_matrix = std::unique_ptr<HypreParMatrix>(mass_form->ParallelAssemble());
        hypre_prec = std::make_unique<HypreDiagScale>(*mass_matrix);
        hypre_solver = std::make_unique<HyprePCG>(*mass_matrix);
        hypre_solver->SetTol(1e-12);
        hypre_solver->SetMaxIter(500);
        hypre_solver->SetPrintLevel(0);
        hypre_solver->SetPreconditioner(*hypre_prec);
    }
}
    
void ComputeCurlHcurlToHdiv::Apply(ParGridFunction &result, const ParGridFunction &gftrial) const
{
    Vector B(test_fes->GetTrueVSize());
    Vector X(test_fes->GetTrueVSize());

    if (pa)
    {
       ParLinearForm b(test_fes); // used as a vector
       mixed_form->Mult(gftrial, b); // process-local multiplication
       b.ParallelAssemble(B);
    }
    else
    {

       Vector P(trial_fes->GetTrueVSize());
       gftrial.GetTrueDofs(P);
       mixed_matrix->Mult(P,B);
    }

    X = 0.0;
    if(pa)
    {
     cg_solver->Mult(B,X);
    }else{
     hypre_solver->Mult(B,X);
    }
    result.SetFromTrueDofs(X);
}

ComputeGradientH1ScalarToHcurl::ComputeGradientH1ScalarToHcurl(ParFiniteElementSpace *trial_space, 
                       ParFiniteElementSpace *test_space, 
                       bool partial_assembly) 
        : trial_fes(trial_space), test_fes(test_space), pa(partial_assembly)
{
    SetUpOperator();
}
    
void ComputeGradientH1ScalarToHcurl::SetUpOperator()
{
    mass_form = std::make_unique<ParBilinearForm>(test_fes);
    if (pa) { mass_form->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
    mass_form->AddDomainIntegrator(new VectorFEMassIntegrator());
    mass_form->Assemble();
    if (!pa) { mass_form->Finalize(); }

    mixed_form = std::make_unique<ParMixedBilinearForm>(trial_fes, test_fes);
    if (pa) { mixed_form->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
    mixed_form->AddDomainIntegrator(new MixedVectorGradientIntegrator());
    mixed_form->Assemble();
    if (!pa) { mixed_form->Finalize(); }
    
    // Setup solver for the mass matrix
    if (pa)
    {
        Array<int> ess_tdof_list; // empty for L2 projection
        mass_form->FormSystemMatrix(ess_tdof_list, mass_op);

        jacobi_prec = std::make_unique<OperatorJacobiSmoother>(*mass_form, ess_tdof_list);
        cg_solver = std::make_unique<CGSolver>(test_fes->GetComm());
        cg_solver->SetRelTol(1e-12);
        cg_solver->SetMaxIter(500);
        cg_solver->SetPrintLevel(0);
        cg_solver->SetOperator(*mass_op);
        cg_solver->SetPreconditioner(*jacobi_prec);
    }
    else
    {
        mixed_matrix = std::unique_ptr<HypreParMatrix>(mixed_form->ParallelAssemble());
        mass_matrix = std::unique_ptr<HypreParMatrix>(mass_form->ParallelAssemble());
        hypre_prec = std::make_unique<HypreDiagScale>(*mass_matrix);
        hypre_solver = std::make_unique<HyprePCG>(*mass_matrix);
        hypre_solver->SetTol(1e-12);
        hypre_solver->SetMaxIter(500);
        hypre_solver->SetPrintLevel(0);
        hypre_solver->SetPreconditioner(*hypre_prec);
    }
}
    
void ComputeGradientH1ScalarToHcurl::Apply(ParGridFunction &result, const ParGridFunction &gftrial) const
{
    Vector B(test_fes->GetTrueVSize());
    Vector X(test_fes->GetTrueVSize());

    if (pa)
    {
       ParLinearForm b(test_fes); // used as a vector
       mixed_form->Mult(gftrial, b); // process-local multiplication
       b.ParallelAssemble(B);
    }
    else
    {

       Vector P(trial_fes->GetTrueVSize());
       gftrial.GetTrueDofs(P);
       mixed_matrix->Mult(P,B);
    }

    X = 0.0;
    if(pa)
    {
     cg_solver->Mult(B,X);
    }else{
     hypre_solver->Mult(B,X);
    }
    result.SetFromTrueDofs(X);
}

void verify_vector_potential(const ProjectorOps& ops,
                             const ParGridFunction &curl_Ah_h1,
                             ParMesh *pmesh, bool pa)
{
   int myid = Mpi::WorldRank();

   // 2. Peform the needed projections
   // Project u in H1 to Hcurl
   ParGridFunction curl_Ah_hdiv(rt_fespace);
   ops.projectorH1ToHdiv.Apply(curl_Ah_hdiv, curl_Ah_h1);

   // Set \nabla \cdot (\nabla \times Ah) to be in L2
   // Compute \nabla \cdot (\nabla \times Ah) in H(div)
   ParGridFunction div_curl_Ah(l2_fespace_scalar);

   ops.projectorDivHdivToL2.Apply(div_curl_Ah, curl_Ah_hdiv);

   VectorFunctionCoefficient curl_Ah_exact_coeff(sdim, curl_A_exact);

   {
      ConstantCoefficient zero(0.0);
      Vector zero_v(dim);
      zero_v = 0.0;
      VectorConstantCoefficient zero_vec(zero_v);

      double div_curl_A_error = div_curl_Ah.ComputeL2Error(zero);
      double curl_Ah_h1_error = curl_Ah_h1.ComputeL2Error(curl_Ah_exact_coeff);

      if (myid == 0)
      {
         cout << "Verify curl A from grad phi: " << std::endl;
         cout << "div(curl A) L2 norm (should be ~0): " << div_curl_A_error << endl;
         cout << "curl Ah H1 L2 norm: " << curl_Ah_h1_error << endl;
      }
   }

}