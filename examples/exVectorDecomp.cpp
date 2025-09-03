#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Exact solution, E, and r.h.s., f. See below for implementation.
void A_exact(const Vector &x, Vector &A);
void curl_A_exact(const Vector &x, Vector &Acurl);
void w_exact(const Vector &x, Vector &f);
void u_exact(const Vector &x, Vector &A);
void grad_phi(const Vector &x, Vector &u);
void project_Hdiv_to_L2(ParGridFunction &result,
                        ParGridFunction &u_hdiv,
                        ParFiniteElementSpace *test_fes,   // vector L2(DG) target
                        bool pa);
void compute_Curl_Hcurl_to_Hdiv(ParGridFunction &result, ParGridFunction &gftrial, ParFiniteElementSpace *trial_fes,  
                         ParFiniteElementSpace *test_fes, bool pa);
void compute_div_Hdiv_to_L2(ParGridFunction &result, ParGridFunction &gftrial, ParFiniteElementSpace *trial_fes,  
                        ParFiniteElementSpace *test_fes, bool pa);
void project_H1_to_Hcurl(ParGridFunction &result,          // in ND space (output)
                         ParGridFunction &u_h1,            // in H1 vector space (input)
                         ParFiniteElementSpace *fes_nd,    // ND test/target
                         bool pa);
void project_H1_to_L2(ParGridFunction &result,          // in L2 space (output)
                      ParGridFunction &u_h1,            // in H1 vector space (input)
                      ParFiniteElementSpace *fes_l2,    // L2 test/target space
                      bool pa);

void solve_scalar_potential(ParGridFunction &phi,
                           ParGridFunction &u_vector,
                           ParFiniteElementSpace *h1_fes_scalar,
                           ParFiniteElementSpace *vector_fes,
                           ParFiniteElementSpace *l2_fes_scalar,  // Need this for div computation
                           bool pa);
// Project H1 (vector) → H(div) in L2-sense
void project_H1_to_Hdiv(ParGridFunction &result,          // in H(div) space (output)
                        ParGridFunction &u_h1,            // in H1 vector space (input)
                        ParFiniteElementSpace *fes_h1,    // H1 vector space
                        ParFiniteElementSpace *fes_hdiv,  // H(div) target
                        bool pa);

void solve_scalar_potential_direct(ParGridFunction &phi,
                                  ParGridFunction &div_u_h1,      // divergence already in H1 scalar space
                                  ParFiniteElementSpace *h1_fes_scalar,
                                  bool pa);

void project_L2_to_H1_scalar(ParGridFunction &result,          // in H1 space (output)
                             ParGridFunction &u_l2,            // in L2 scalar space (input)
                             ParFiniteElementSpace *fes_l2,    // L2 scalar space
                             ParFiniteElementSpace *fes_h1,    // H1 scalar target
                             bool pa);
void compute_gradient_H1_to_Hcurl(ParGridFunction &result,       // in H(curl) space (output)
                                  ParGridFunction &phi_h1,        // in H1 scalar space (input) 
                                  ParFiniteElementSpace *h1_fes,  // H1 scalar trial space
                                  ParFiniteElementSpace *nd_fes,  // H(curl) test space
                                  bool pa);

real_t freq = 1.0, kappa;
int dim;

void project_Hcurl_Hdiv(ParGridFunction &result, ParGridFunction &gftrial, ParFiniteElementSpace *trial_fes, 
                        ParFiniteElementSpace *test_fes, bool pa);

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   int order = 1;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = true;
   double length = 1.0;
   int num_el = 8;
   real_t delta_const = 1e-4;
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
   int sdim = mesh->SpaceDimension();

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

   // 7. Define a parallel finite element space on the parallel mesh. Here we
   //    use the Nedelec finite elements of the specified order.
   FiniteElementCollection *fec    = new ND_FECollection(order, dim);
   FiniteElementCollection *nd_fec = new ND_FECollection(order, dim);   // H(curl)
   FiniteElementCollection *rt_fec = new RT_FECollection(order-1, dim); // H(div)
   FiniteElementCollection *l2_fec = new L2_FECollection(order-1, dim);
   FiniteElementCollection *h1_fec = new H1_FECollection(order, dim);

   ParFiniteElementSpace *l2_fespace_scalar = new ParFiniteElementSpace(pmesh, l2_fec);
   ParFiniteElementSpace *l2_fespace_vector = new ParFiniteElementSpace(pmesh, l2_fec, dim);

   ParFiniteElementSpace *nd_fespace = new ParFiniteElementSpace(pmesh, nd_fec);
   ParFiniteElementSpace *rt_fespace = new ParFiniteElementSpace(pmesh, rt_fec);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   ParFiniteElementSpace *h1_fespace_scalar = new ParFiniteElementSpace(pmesh, h1_fec);
   ParFiniteElementSpace *h1_fespace_vector = new ParFiniteElementSpace(pmesh, h1_fec, dim);

   HYPRE_BigInt size = nd_fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   // nabla \phi
   VectorFunctionCoefficient grad_phi_coeff(sdim, grad_phi);
   ParGridFunction grad_phi_hcurl(nd_fespace);
   grad_phi_hcurl.ProjectCoefficient(grad_phi_coeff);

   ParGridFunction grad_phi_exact_h1(h1_fespace_vector);
   grad_phi_exact_h1.ProjectCoefficient(grad_phi_coeff);

   // \nabla \times Ah in H(div)
   VectorFunctionCoefficient curl_A_exact_coeff(sdim, curl_A_exact);
   ParGridFunction curl_Ah_exact_hdiv(rt_fespace);
   curl_Ah_exact_hdiv.ProjectCoefficient(curl_A_exact_coeff);

   VectorFunctionCoefficient u_coeff(sdim, u_exact);

   // Define u in H(curl)
   ParGridFunction u_hcurl(nd_fespace);
   u_hcurl.ProjectCoefficient(u_coeff);

   // Define u in H1
   ParGridFunction u_h1(h1_fespace_vector);
   u_h1.ProjectCoefficient(u_coeff);

   // Define u in L2
   ParGridFunction u_l2(l2_fespace_vector);
   u_l2 = 0.0;
   project_H1_to_L2(u_l2, u_h1, l2_fespace_vector, pa);          // in L2 space (output)
   // u_l2.ProjectCoefficient(u_coeff);

   // Project u1 to Hcurl
   ParGridFunction u_hcurl_l2_project(nd_fespace);
   project_H1_to_Hcurl(u_hcurl_l2_project, u_h1, nd_fespace, pa);

   // Compute the curl of u
   ParGridFunction curl_u_hdiv(rt_fespace);
   curl_u_hdiv = 0.0;
   compute_Curl_Hcurl_to_Hdiv(curl_u_hdiv, u_hcurl_l2_project, nd_fespace, rt_fespace, pa);

   // This is one way of moving from one space to another 
   VectorGridFunctionCoefficient curl_u_coeff(&curl_u_hdiv);
   ParGridFunction curl_u_hcurl(nd_fespace);
   curl_u_hcurl.ProjectCoefficient(curl_u_coeff);
   
   // We can also solve a linear system to move form one space to another
   ParGridFunction curl_u_hcurl_l2_project(nd_fespace);
   if (myid == 0){
   cout << "\nPerforming L2 projection from H(div) to (H(curl))" << "\n";
   }

   // The test space which is being projected to is
   // H(curl) from the trial space H(div)
   // Note that the trial space needs to not be empyt ie.
   // be projected to
   project_Hcurl_Hdiv(curl_u_hcurl_l2_project, curl_u_hdiv, rt_fespace, nd_fespace, pa);

   // Project the exact space for comparison later
   VectorFunctionCoefficient curl_u_coeff_exact(dim, w_exact);
   ParGridFunction curl_u_exact(nd_fespace);
   curl_u_exact.ProjectCoefficient(curl_u_coeff_exact);

   // 1) Everybody calls these MPI-collectives:
   real_t l2_err     = curl_u_hcurl.ComputeL2Error(curl_u_coeff_exact);
   real_t l2_err_same_space     = curl_u_exact.ComputeL2Error(curl_u_coeff_exact);
   real_t l2_err_sys     = curl_u_hcurl_l2_project.ComputeL2Error(curl_u_coeff_exact);
   real_t hcurl_err  = u_hcurl.ComputeHCurlError(&u_coeff, &curl_u_coeff_exact);
   
   // 2) Only rank 0 prints:
   if (myid == 0)
   {
      cout << "\nTwo ways of measuring the same error:\n";
      cout << "  curl L2 error      = " << l2_err    << "\n";
      cout << "  curl L2 same space      = " << l2_err_same_space    << "\n";
      cout << "  H(curl) norm error = " << hcurl_err << "\n";
      cout << "  H(curl) norm error lin sys = " << l2_err_sys << "\n\n";
   }

   if (myid == 0) { cout << "\nStarting Option A workflow for Poisson solve" << endl; }
   
   // 1. Project u from H1 → H(div)
   ParGridFunction u_hdiv_for_poisson(rt_fespace);
   project_H1_to_Hdiv(u_hdiv_for_poisson, u_h1, h1_fespace_vector, rt_fespace, pa);
   
   // 2. Divergence: H(div) → L2
   ParGridFunction div_u_l2_poisson(l2_fespace_scalar);
   compute_div_Hdiv_to_L2(div_u_l2_poisson, u_hdiv_for_poisson, rt_fespace, l2_fespace_scalar, pa);
   
   // 3. Project L2 → H1
   ParGridFunction div_u_h1_poisson(h1_fespace_scalar);
   project_L2_to_H1_scalar(div_u_h1_poisson, div_u_l2_poisson, l2_fespace_scalar, h1_fespace_scalar, pa);
   
   // 4. Solve Poisson: -∇²φ = div(u) where both are now in H1
   ParGridFunction phi_scalar(h1_fespace_scalar);
   solve_scalar_potential_direct(phi_scalar, div_u_h1_poisson, h1_fespace_scalar, pa);

   // After solving for phi_scalar, compute its gradient for comparison
   ParGridFunction grad_phi_computed(nd_fespace);
   compute_gradient_H1_to_Hcurl(grad_phi_computed, phi_scalar, h1_fespace_scalar, nd_fespace, pa);
   
   // Compute curl of grad_phi (should be zero)
   ParGridFunction curl_grad_phi_computed(rt_fespace);
   compute_Curl_Hcurl_to_Hdiv(curl_grad_phi_computed, grad_phi_computed, nd_fespace, rt_fespace, pa);
   
   // Now compare with your exact gradient
   double grad_phi_poisson_error = grad_phi_computed.ComputeL2Error(grad_phi_coeff);
   
   // Check if curl(grad φ) ≈ 0
   Vector zero_v(dim);
   zero_v = 0.0;
   VectorConstantCoefficient zero_vec(zero_v);
   double curl_grad_phi_computed_error = curl_grad_phi_computed.ComputeL2Error(zero_vec);
   
   if (myid == 0)
   {
      cout << "Gradient of computed phi L2 error: " << grad_phi_poisson_error << endl;
      cout << "curl(grad phi) L2 error (should be ~0): " << curl_grad_phi_computed_error << endl;
   }


   // 8. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   Array<int> ess_bdr;

   if (pmesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr = 0;
      // nd_fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 9. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (f,phi_i) where f is given by the function f_exact and phi_i are the
   //    basis functions in the finite element fespace.
   VectorGridFunctionCoefficient f(&curl_u_hcurl_l2_project);
   ParLinearForm *b = new ParLinearForm(nd_fespace);
   b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
   b->Assemble();

   // 10. Define the solution vector x as a parallel finite element grid function
   //     corresponding to fespace. Initialize x by projecting the exact
   //     solution. Note that only values from the boundary edges will be used
   //     when eliminating the non-homogeneous boundary condition to modify the
   //     r.h.s. vector b.
   ParGridFunction x(nd_fespace);
   VectorFunctionCoefficient A_coeff(sdim, A_exact);
   // x.ProjectCoefficient(A_coeff);
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
      MatrixFreeAMS ams(*a, *A, *fespace, muinv, sigma, NULL, ess_bdr, useAmgX);
#else
      MatrixFreeAMS ams(*a, *A, *nd_fespace, muinv, sigma, NULL, ess_bdr);
#endif
      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(1000);
      cg.SetPrintLevel(1);
      cg.SetOperator(*A);
      cg.SetPreconditioner(ams);
      cg.Mult(B, X);
   }
   else
   {
      if (myid == 0)
      {
         cout << "Size of linear system: "
              << A.As<HypreParMatrix>()->GetGlobalNumRows() << endl;
      }

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

   // Compute curl of Ah in H(div)
   ParGridFunction curl_Ah_hdiv(rt_fespace);
   compute_Curl_Hcurl_to_Hdiv(curl_Ah_hdiv, x, nd_fespace, rt_fespace, pa);

   ParGridFunction div_Ah_hdiv(rt_fespace);
   ParGridFunction Ah_hdiv(rt_fespace);
   ParGridFunction Ah_hcurl(nd_fespace);

   // The test space which is being projected to is
   // H(div) from the trial space H(curl)
   project_Hcurl_Hdiv(Ah_hdiv, x, nd_fespace, rt_fespace, pa);

   // Set \nabla \cdot (\nabla \times Ah) to be in L2
   // Compute \nabla \cdot (\nabla \times Ah) in H(div)
   ParGridFunction div_curl_Ah_l2(l2_fespace_scalar);

   ParGridFunction div_Ah_l2(l2_fespace_scalar);

   compute_div_Hdiv_to_L2(div_curl_Ah_l2, curl_Ah_hdiv, rt_fespace,l2_fespace_scalar,pa);
   compute_div_Hdiv_to_L2(div_Ah_l2, Ah_hdiv, rt_fespace,l2_fespace_scalar,pa);

   // Project curl Ah to L2 space
   ParGridFunction grad_phi_l2(l2_fespace_vector);
   grad_phi_l2 = 0.0;
   // grad_phi_l2.ProjectCoefficient(grad_phi_coeff);

   // VectorGridFunctionCoefficient curl_Ah_l2_coeff(&curl_Ah_hdiv);
   ParGridFunction curl_Ah_l2(l2_fespace_vector);
   project_Hdiv_to_L2(curl_Ah_l2, curl_Ah_hdiv, l2_fespace_vector, pa);

   grad_phi_l2 = u_l2;
   grad_phi_l2 -= curl_Ah_l2;

   // Project to grad phi_l2 and curl_Ah_l2 to H1
   ParGridFunction grad_phi_h1(h1_fespace_vector);
   ParGridFunction curl_Ah_h1(h1_fespace_vector);
   
   // Create coefficients from the L2 grid functions
   VectorGridFunctionCoefficient grad_phi_l2_coeff(&grad_phi_l2);
   VectorGridFunctionCoefficient curl_Ah_l2_coeff(&curl_Ah_l2);
   
   // Use ProjectDiscCoefficient for averaging-based projection from L2 to H1
   grad_phi_h1.ProjectDiscCoefficient(grad_phi_l2_coeff);
   curl_Ah_h1.ProjectDiscCoefficient(curl_Ah_l2_coeff);

   ParGridFunction curl_Ah_hcurl_l2_project(nd_fespace);
   project_H1_to_Hcurl(curl_Ah_hcurl_l2_project, curl_Ah_h1, nd_fespace, pa);

   grad_phi_hcurl = 0.0;
   // When grad phi exact is used here, the curl holds fine.
   // project_H1_to_Hcurl(grad_phi_hcurl, grad_phi_exact_h1, nd_fespace,pa);
   project_H1_to_Hcurl(grad_phi_hcurl, grad_phi_h1, nd_fespace,pa);

   ParGridFunction curl_grad_phi_hdiv(rt_fespace);
   curl_grad_phi_hdiv = 0.0;
   compute_Curl_Hcurl_to_Hdiv(curl_grad_phi_hdiv,grad_phi_hcurl,nd_fespace,rt_fespace,pa);

   // 15. Compute and print the L^2 norm of the error.
   {
      real_t error = x.ComputeL2Error(A_coeff);
      // Discretely this operation should be zero
      ConstantCoefficient zero(0.0);

      Vector zero_v(dim);
      zero_v = 0.0;
      VectorConstantCoefficient zero_vec(zero_v);

      double div_curl_A_error = div_curl_Ah_l2.ComputeL2Error(zero);
      double div_A_error = div_Ah_l2.ComputeL2Error(zero);
      double grad_phi_error = grad_phi_l2.ComputeL2Error(grad_phi_coeff);
      double curl_Ah_l2_error = curl_Ah_l2.ComputeL2Error(curl_A_exact_coeff);
      double grad_phi_error_h1 = grad_phi_h1.ComputeL2Error(grad_phi_coeff);
      double grad_phi_error_hcurl = grad_phi_hcurl.ComputeL2Error(grad_phi_coeff);
      double curl_Ah_h1_error = curl_Ah_h1.ComputeL2Error(curl_A_exact_coeff);
      double curl_grad_phi_hdiv_error = curl_grad_phi_hdiv.ComputeL2Error(zero_vec);

      if (myid == 0)
      {
         cout << "\n|| E_h - E ||_{L^2} = " << error << '\n' << endl;
         cout << "div(curl A) L2 norm: " << div_curl_A_error << endl;
         cout << "div(A) L2 norm: " << div_A_error << endl;
         cout << "grad_phi L2 norm: " << grad_phi_error << endl;
         cout << "curl Ah L2 norm: " << curl_Ah_l2_error << endl;
         cout << "grad_phi H1 L2 norm: " << grad_phi_error_h1 << endl;
         cout << "grad_phi Hcurl L2 norm: " << grad_phi_error_hcurl << endl;
         cout << "curl Ah H1 L2 norm: " << curl_Ah_h1_error << endl;
         cout << "curl grad phi Hdiv L2 norm: " << curl_grad_phi_hdiv_error << endl;
      }
   }


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
   x.Save(sol_ofs);

   // curl(u_h) in H(curl)
   ostringstream curl_name;
   curl_name << "curl_u_hcurl." << setfill('0') << setw(6) << myid;
   ofstream curl_ofs(curl_name.str());
   curl_ofs.precision(8);
   curl_u_hcurl.Save(curl_ofs);

   // curl(u_h) in H(curl)
   ostringstream curl_lin_sys_name;
   curl_lin_sys_name << "curl_u_lin_sys." << setfill('0') << setw(6) << myid;
   ofstream curl_lin_sys_ofs(curl_lin_sys_name.str());
   curl_lin_sys_ofs.precision(8);
   curl_u_hcurl_l2_project.Save(curl_lin_sys_ofs);

   // exact curl for comparison
   ostringstream curl_ex_name;
   curl_ex_name << "curl_u_exact_hcurl." << setfill('0') << setw(6) << myid;
   ofstream curl_ex_ofs(curl_ex_name.str());
   curl_ex_ofs.precision(8);
   curl_u_exact.Save(curl_ex_ofs);

   }

    ParGridFunction Agf_exact(nd_fespace);
    Agf_exact.ProjectCoefficient(A_coeff);

    VisItDataCollection dc("VelocityDecomposition", pmesh);
    dc.SetFormat(DataCollection::PARALLEL_FORMAT);
    dc.SetCycle(0);
    dc.SetTime(0.0);
    
    dc.RegisterField("Ah", &x);
    dc.RegisterField("Ah_exact", &Agf_exact);

    dc.RegisterField("curl_Ah_exact_hdiv", &curl_Ah_exact_hdiv);
    dc.RegisterField("curl_Ah_l2", &curl_Ah_l2);
    dc.RegisterField("curl_Ah_hdiv", &curl_Ah_hdiv);
    dc.RegisterField("curl_Ah_h1", &curl_Ah_h1);
    dc.RegisterField("curl_Ah_hcurl", &curl_Ah_hcurl_l2_project);
    dc.RegisterField("div_curl_Ah_l2", &div_curl_Ah_l2);

    dc.RegisterField("curl_u_computed", &curl_u_hcurl);
    dc.RegisterField("curl_u_hdiv", &curl_u_hdiv);
    dc.RegisterField("curl_u_hcurl_l2", &curl_u_hcurl_l2_project);
    dc.RegisterField("curl_u_exact",    &curl_u_exact);

    dc.RegisterField("grad_phi_exact_hcurl",   &grad_phi_hcurl);
    dc.RegisterField("grad_phi_exact_h1",   &grad_phi_exact_h1);
    dc.RegisterField("grad_phi_l2",    &grad_phi_l2);
    dc.RegisterField("grad_phi_h1",    &grad_phi_h1);
    dc.RegisterField("grad_phi_computed",    &grad_phi_computed);
    dc.RegisterField("curl_grad_phi_hdiv", &curl_grad_phi_hdiv);
    dc.RegisterField("curl_grad_phi_computed", &curl_grad_phi_computed);

    dc.Save();

   // 17. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << x << flush;
   }

   // 18. Free the used memory.
   delete a;
   delete sigma;
   delete muinv;
   delete b;
   delete nd_fespace;
   delete rt_fespace;
   delete fec;
   delete nd_fec;
   delete rt_fec;
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
   // real_t xi = 2*M_PI*x(0);
   // real_t yi = 2*M_PI*x(1);
   // real_t zi = 2*M_PI*x(2);
 
   // A(0) = sin(xi) * cos(yi) * cos(zi);
   // A(1) = -cos(xi) * sin(yi) * cos(zi);
   // A(2) = 0.0;
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

// The test space is what you are projecting to and the trial space is where you are projecting from
void project_Hcurl_Hdiv(ParGridFunction &result, ParGridFunction &gftrial, ParFiniteElementSpace *trial_fes,  
                        ParFiniteElementSpace *test_fes, bool pa)
{
   ParBilinearForm *a = new ParBilinearForm(test_fes);
   if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a->AddDomainIntegrator(new VectorFEMassIntegrator());
   ParMixedBilinearForm *a_mixed = new ParMixedBilinearForm(trial_fes, test_fes);
   if (pa) {a_mixed->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a_mixed->AddDomainIntegrator(new VectorFEMassIntegrator());

   // a_mixed->AddDomainIntegrator(new MixedVectorMassIntegrator());  // More explicit

   a->Assemble();
   if(!pa){a->Finalize();}

   a_mixed->Assemble();
   if(!pa){a_mixed->Finalize();}

   Vector B(test_fes->GetTrueVSize());
   Vector X(test_fes->GetTrueVSize());

   if (pa)
   {
      ParLinearForm b(test_fes); // used as a vector
      a_mixed->Mult(gftrial, b); // process-local multiplication
      b.ParallelAssemble(B);
   }
   else
   {
      HypreParMatrix *mixed = a_mixed->ParallelAssemble();

      Vector P(trial_fes->GetTrueVSize());
      gftrial.GetTrueDofs(P);

      mixed->Mult(P,B);

      delete mixed;
   }

    // 11. Define and apply a parallel PCG solver for AX=B with Jacobi
   //     preconditioner.
   if (pa)
   {
      Array<int> ess_tdof_list; // empty

      OperatorPtr A;
      a->FormSystemMatrix(ess_tdof_list, A);

      OperatorJacobiSmoother Jacobi(*a, ess_tdof_list);

      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(1000);
      cg.SetPrintLevel(1);
      cg.SetOperator(*A);
      cg.SetPreconditioner(Jacobi);
      X = 0.0;
      cg.Mult(B, X);
   }
   else
   {
      HypreParMatrix *Amat = a->ParallelAssemble();
      HypreDiagScale Jacobi(*Amat);
      HyprePCG pcg(*Amat);
      pcg.SetTol(1e-12);
      pcg.SetMaxIter(1000);
      pcg.SetPrintLevel(2);
      pcg.SetPreconditioner(Jacobi);
      X = 0.0;
      pcg.Mult(B, X);

      delete Amat;
   }

   result.SetFromTrueDofs(X);
}

// The test space is what you are projecting to and the trial space is where you are projecting from
void compute_Curl_Hcurl_to_Hdiv(ParGridFunction &result, ParGridFunction &gftrial, ParFiniteElementSpace *trial_fes,  
                        ParFiniteElementSpace *test_fes, bool pa)
{
   ParBilinearForm *a = new ParBilinearForm(test_fes);
   if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a->AddDomainIntegrator(new VectorFEMassIntegrator());
   ParMixedBilinearForm *a_mixed = new ParMixedBilinearForm(trial_fes, test_fes);
   if (pa) {a_mixed->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a_mixed->AddDomainIntegrator(new MixedVectorCurlIntegrator());

   a->Assemble();
   if(!pa){a->Finalize();}

   a_mixed->Assemble();
   if(!pa){a_mixed->Finalize();}

   Vector B(test_fes->GetTrueVSize());
   Vector X(test_fes->GetTrueVSize());

   if (pa)
   {
      ParLinearForm b(test_fes); // used as a vector
      a_mixed->Mult(gftrial, b); // process-local multiplication
      b.ParallelAssemble(B);
   }
   else
   {
      HypreParMatrix *mixed = a_mixed->ParallelAssemble();

      Vector P(trial_fes->GetTrueVSize());
      gftrial.GetTrueDofs(P);

      mixed->Mult(P,B);

      delete mixed;
   }

    // 11. Define and apply a parallel PCG solver for AX=B with Jacobi
   //     preconditioner.
   if (pa)
   {
      Array<int> ess_tdof_list; // empty

      OperatorPtr A;
      a->FormSystemMatrix(ess_tdof_list, A);

      OperatorJacobiSmoother Jacobi(*a, ess_tdof_list);

      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(1000);
      cg.SetPrintLevel(1);
      cg.SetOperator(*A);
      cg.SetPreconditioner(Jacobi);
      X = 0.0;
      cg.Mult(B, X);
   }
   else
   {
      HypreParMatrix *Amat = a->ParallelAssemble();
      HypreDiagScale Jacobi(*Amat);
      HyprePCG pcg(*Amat);
      pcg.SetTol(1e-12);
      pcg.SetMaxIter(1000);
      pcg.SetPrintLevel(2);
      pcg.SetPreconditioner(Jacobi);
      X = 0.0;
      pcg.Mult(B, X);

      delete Amat;
   }

   result.SetFromTrueDofs(X);
}

// Project H(div) field (u_hdiv) into vector L2(DG) (result) in the true L2 sense: M y = b.
// test_fes must be a vector L2/DG space with vdim = mesh dim.
void project_Hdiv_to_L2(ParGridFunction &result,
                        ParGridFunction &u_hdiv,
                        ParFiniteElementSpace *test_fes,   // vector L2 target
                        bool pa)
{
   // 1) L2 mass operator on target
   ParBilinearForm a(test_fes);
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a.AddDomainIntegrator(new VectorMassIntegrator());
   a.Assemble();
   if (!pa) { a.Finalize(); }

   // 2) RHS: b_i = (u_hdiv, phi_i)
   VectorGridFunctionCoefficient ucoeff(&u_hdiv);
   ParLinearForm b(test_fes);
   b.AddDomainIntegrator(new VectorDomainLFIntegrator(ucoeff));
   b.Assemble();

   // true-dof vectors
   Vector B(test_fes->GetTrueVSize());
   Vector X(test_fes->GetTrueVSize());
   b.ParallelAssemble(B);
   X = 0.0;

   if (pa)
   {
      Array<int> ess_tdof_list;
      OperatorPtr Aop;
      a.FormSystemMatrix(ess_tdof_list, Aop);

      OperatorJacobiSmoother Jacobi(a, ess_tdof_list); 
      CGSolver cg(test_fes->GetComm());
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(200);
      cg.SetPrintLevel(0);
      cg.SetOperator(*Aop);
      cg.SetPreconditioner(Jacobi);
      cg.Mult(B, X);
   }
   else
   {
      // Fully assembled Hypre path
      std::unique_ptr<HypreParMatrix> A(a.ParallelAssemble());
      HypreDiagScale Jacobi(*A);
      HyprePCG pcg(*A);
      pcg.SetTol(1e-12);
      pcg.SetMaxIter(200);
      pcg.SetPrintLevel(2);
      pcg.SetPreconditioner(Jacobi);
      pcg.Mult(B, X);
   }

   // 3) Scatter to result in L2(DG)
   result = 0.0;
   result.SetFromTrueDofs(X);
}

// Project H1 (vector) → H(curl) (ND) in L2-sense.
void project_H1_to_Hcurl(ParGridFunction &result,          // in ND space (output)
                         ParGridFunction &u_h1,            // in H1 vector space (input)
                         ParFiniteElementSpace *fes_nd,    // ND test/target
                         bool pa)
{
   // Mass matrix on the ND space
   ParBilinearForm M(fes_nd);
   if (pa) { M.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   M.AddDomainIntegrator(new VectorFEMassIntegrator()); // <- ND/RT mass
   M.Assemble();
   if (!pa) { M.Finalize(); }

   // RHS: b_i = (u_h1, w_i) with w_i in ND
   VectorGridFunctionCoefficient ucoeff(&u_h1);
   ParLinearForm b(fes_nd);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(ucoeff)); // <- ND/RT RHS
   b.Assemble();

   Vector B(fes_nd->GetTrueVSize()), X(fes_nd->GetTrueVSize());
   b.ParallelAssemble(B);
   X = 0.0;

   if (pa)
   {
      Array<int> ess_tdof_list; // none for pure L2 projection
      OperatorPtr Mop;
      M.FormSystemMatrix(ess_tdof_list, Mop);
      OperatorJacobiSmoother Jacobi(M, ess_tdof_list);
      CGSolver cg(fes_nd->GetComm());
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(500);
      cg.SetPrintLevel(0);
      cg.SetOperator(*Mop);
      cg.SetPreconditioner(Jacobi);
      cg.Mult(B, X);
   }
   else
   {
      std::unique_ptr<HypreParMatrix> Mpar(M.ParallelAssemble());
      HypreDiagScale Jacobi(*Mpar);
      HyprePCG pcg(*Mpar);
      pcg.SetTol(1e-12);
      pcg.SetMaxIter(500);
      pcg.SetPrintLevel(2);
      pcg.SetPreconditioner(Jacobi);
      pcg.Mult(B, X);
   }

   result = 0.0;
   result.SetFromTrueDofs(X);
}

// The test space is what you are projecting to and the trial space is where you are projecting from
void compute_div_Hdiv_to_L2(ParGridFunction &result, ParGridFunction &gftrial, ParFiniteElementSpace *trial_fes,  
                        ParFiniteElementSpace *test_fes, bool pa)
{
   ParBilinearForm *a = new ParBilinearForm(test_fes);
   if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a->AddDomainIntegrator(new MassIntegrator());
   ParMixedBilinearForm *a_mixed = new ParMixedBilinearForm(trial_fes, test_fes);
   if (pa) {a_mixed->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a_mixed->AddDomainIntegrator(new VectorFEDivergenceIntegrator());

   a->Assemble();
   if(!pa){a->Finalize();}

   a_mixed->Assemble();
   if(!pa){a_mixed->Finalize();}

   Vector B(test_fes->GetTrueVSize());
   Vector X(test_fes->GetTrueVSize());

   if (pa)
   {
      ParLinearForm b(test_fes); // used as a vector
      a_mixed->Mult(gftrial, b); // process-local multiplication
      b.ParallelAssemble(B);
   }
   else
   {
      HypreParMatrix *mixed = a_mixed->ParallelAssemble();

      Vector P(trial_fes->GetTrueVSize());
      gftrial.GetTrueDofs(P);

      mixed->Mult(P,B);

      delete mixed;
   }

    // 11. Define and apply a parallel PCG solver for AX=B with Jacobi
   //     preconditioner.
   if (pa)
   {
      Array<int> ess_tdof_list; // empty

      OperatorPtr A;
      a->FormSystemMatrix(ess_tdof_list, A);

      OperatorJacobiSmoother Jacobi(*a, ess_tdof_list);

      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(1000);
      cg.SetPrintLevel(1);
      cg.SetOperator(*A);
      cg.SetPreconditioner(Jacobi);
      X = 0.0;
      cg.Mult(B, X);
   }
   else
   {
      HypreParMatrix *Amat = a->ParallelAssemble();
      HypreDiagScale Jacobi(*Amat);
      HyprePCG pcg(*Amat);
      pcg.SetTol(1e-12);
      pcg.SetMaxIter(1000);
      pcg.SetPrintLevel(2);
      pcg.SetPreconditioner(Jacobi);
      X = 0.0;
      pcg.Mult(B, X);

      delete Amat;
   }

   result.SetFromTrueDofs(X);
}



// Compressive part (grad(phi))
void grad_phi(const Vector &x, Vector &u)
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

// Project H1 (vector) → L2 (vector) in the true L2 sense:
// Find y ∈ L2^d such that (y, v) = (u_h1, v)  ∀ v ∈ L2^d.
// That is:  M_L2 * y = b,  with  b_i = (u_h1, φ_i) on the L2 space.
void project_H1_to_L2(ParGridFunction &result,              // in L2 space (output)
                      ParGridFunction &u_h1,                // in H1 vector space (input)
                      ParFiniteElementSpace *fes_l2,        // L2^d target (vdim = mesh dim)
                      bool pa)
{
   // 1) L2^d mass matrix on the target space
   ParBilinearForm M_L2(fes_l2);
   if (pa) { M_L2.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   M_L2.AddDomainIntegrator(new VectorMassIntegrator());  // correct for L2/H1 vector spaces
   M_L2.Assemble();
   if (!pa) { M_L2.Finalize(); }

   // 2) RHS b = ∫ u_h1 · v_l2  (build it as a LinearForm on the L2 space)
   VectorGridFunctionCoefficient ucoeff(&u_h1);
   ParLinearForm b(fes_l2);
   b.AddDomainIntegrator(new VectorDomainLFIntegrator(ucoeff)); // no mixed operator needed
   b.Assemble();

   // 3) Solve M_L2 X = B
   Vector B(fes_l2->GetTrueVSize()), X(fes_l2->GetTrueVSize());
   b.ParallelAssemble(B);
   X = 0.0;

   if (pa)
   {
      Array<int> ess;                      // none for pure L2 projection
      OperatorPtr Mop;
      M_L2.FormSystemMatrix(ess, Mop);     // PA path OK for VectorMassIntegrator on L2
      OperatorJacobiSmoother J(M_L2, ess); // simple diagonal smoother works well for mass
      CGSolver cg(fes_l2->GetComm());
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(300);
      cg.SetPrintLevel(0);
      cg.SetOperator(*Mop);
      cg.SetPreconditioner(J);
      cg.Mult(B, X);
   }
   else
   {
      std::unique_ptr<HypreParMatrix> Mpar(M_L2.ParallelAssemble());
      HypreDiagScale J(*Mpar);
      HyprePCG pcg(*Mpar);
      pcg.SetTol(1e-12);
      pcg.SetMaxIter(300);
      pcg.SetPrintLevel(0);
      pcg.SetPreconditioner(J);
      pcg.Mult(B, X);
   }

   result = 0.0;
   result.SetFromTrueDofs(X);
}

// Solve Poisson problem: -∇²φ = div_u_h1 where div_u_h1 is already in H1 space
// Uses OrthoSolver to handle null space in periodic domains
void solve_scalar_potential_direct(ParGridFunction &phi,
                                  ParGridFunction &div_u_h1,      // divergence already in H1 scalar space
                                  ParFiniteElementSpace *h1_fes_scalar,
                                  bool pa)
{
   int myid = Mpi::WorldRank();
   
   // Set up Laplacian operator in H1 space
   ParBilinearForm laplacian(h1_fes_scalar);
   if (pa) { laplacian.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   laplacian.AddDomainIntegrator(new DiffusionIntegrator());
   laplacian.Assemble();
   if (!pa) { laplacian.Finalize(); }
   
   // Set up RHS using div_u_h1 (both already in same H1 space)
   GridFunctionCoefficient div_u_coeff(&div_u_h1);
   ParLinearForm rhs(h1_fes_scalar);
   rhs.AddDomainIntegrator(new DomainLFIntegrator(div_u_coeff));
   rhs.Assemble();
   
   Vector RHS(h1_fes_scalar->GetTrueVSize());
   Vector PHI(h1_fes_scalar->GetTrueVSize());
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
      CGSolver base_solver(h1_fes_scalar->GetComm());
      base_solver.SetRelTol(1e-12);
      base_solver.SetMaxIter(1000);
      base_solver.SetPrintLevel(0);  // Reduce output since OrthoSolver will print
      base_solver.SetOperator(*laplacian_op);
      base_solver.SetPreconditioner(jac);
      
      // Create OrthoSolver to handle null space
      OrthoSolver ortho_solver(h1_fes_scalar->GetComm());
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
      OrthoSolver ortho_solver(h1_fes_scalar->GetComm());
      ortho_solver.SetSolver(base_solver);
      ortho_solver.SetOperator(*A);
      
      if (myid == 0) 
      {
         cout << "Using OrthoSolver for direct Poisson problem with null space" << endl;
      }
      
      ortho_solver.Mult(RHS, PHI);
   }
   
   // Set the solution
   phi = 0.0;
   
   phi.SetFromTrueDofs(PHI);

   if (myid == 0)
   {
      cout << "Solved direct Poisson problem -∇²φ = div(u) for scalar potential" << endl;
   }
}

// You'll also need this projection function (scalar version)
void project_L2_to_H1_scalar(ParGridFunction &result,          // in H1 space (output)
                             ParGridFunction &u_l2,            // in L2 scalar space (input)
                             ParFiniteElementSpace *fes_l2,    // L2 scalar space
                             ParFiniteElementSpace *fes_h1,    // H1 scalar target
                             bool pa)
{
   // Mass matrix on the H1 space
   ParBilinearForm M(fes_h1);
   if (pa) { M.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   M.AddDomainIntegrator(new MassIntegrator()); // H1 scalar mass
   M.Assemble();
   if (!pa) { M.Finalize(); }

   // RHS: b_i = (u_l2, w_i) with w_i in H1
   GridFunctionCoefficient ucoeff(&u_l2);
   ParLinearForm b(fes_h1);
   b.AddDomainIntegrator(new DomainLFIntegrator(ucoeff)); // H1 scalar RHS
   b.Assemble();

   Vector B(fes_h1->GetTrueVSize()), X(fes_h1->GetTrueVSize());
   b.ParallelAssemble(B);
   X = 0.0;

   if (pa)
   {
      Array<int> ess_tdof_list; // none for pure L2 projection
      OperatorPtr Mop;
      M.FormSystemMatrix(ess_tdof_list, Mop);
      OperatorJacobiSmoother Jacobi(M, ess_tdof_list);
      CGSolver cg(fes_h1->GetComm());
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(500);
      cg.SetPrintLevel(0);
      cg.SetOperator(*Mop);
      cg.SetPreconditioner(Jacobi);
      cg.Mult(B, X);
   }
   else
   {
      std::unique_ptr<HypreParMatrix> Mpar(M.ParallelAssemble());
      HypreDiagScale Jacobi(*Mpar);
      HyprePCG pcg(*Mpar);
      pcg.SetTol(1e-12);
      pcg.SetMaxIter(500);
      pcg.SetPrintLevel(0);
      pcg.SetPreconditioner(Jacobi);
      pcg.Mult(B, X);
   }

   result = 0.0;
   result.SetFromTrueDofs(X);
}
// Project H1 (vector) → H(div) in L2-sense
void project_H1_to_Hdiv(ParGridFunction &result,          // in H(div) space (output)
                        ParGridFunction &u_h1,            // in H1 vector space (input)
                        ParFiniteElementSpace *fes_h1,    // H1 vector space
                        ParFiniteElementSpace *fes_hdiv,  // H(div) target
                        bool pa)
{
   // Mass matrix on the H(div) space
   ParBilinearForm M(fes_hdiv);
   if (pa) { M.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   M.AddDomainIntegrator(new VectorFEMassIntegrator()); // H(div) mass
   M.Assemble();
   if (!pa) { M.Finalize(); }

   // RHS: b_i = (u_h1, w_i) with w_i in H(div)
   VectorGridFunctionCoefficient ucoeff(&u_h1);
   ParLinearForm b(fes_hdiv);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(ucoeff)); // H(div) RHS
   b.Assemble();

   Vector B(fes_hdiv->GetTrueVSize()), X(fes_hdiv->GetTrueVSize());
   b.ParallelAssemble(B);
   X = 0.0;

   if (pa)
   {
      Array<int> ess_tdof_list;
      OperatorPtr Mop;
      M.FormSystemMatrix(ess_tdof_list, Mop);
      OperatorJacobiSmoother Jacobi(M, ess_tdof_list);
      CGSolver cg(fes_hdiv->GetComm());
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(500);
      cg.SetPrintLevel(0);
      cg.SetOperator(*Mop);
      cg.SetPreconditioner(Jacobi);
      cg.Mult(B, X);
   }
   else
   {
      std::unique_ptr<HypreParMatrix> Mpar(M.ParallelAssemble());
      HypreDiagScale Jacobi(*Mpar);
      HyprePCG pcg(*Mpar);
      pcg.SetTol(1e-12);
      pcg.SetMaxIter(500);
      pcg.SetPrintLevel(0);
      pcg.SetPreconditioner(Jacobi);
      pcg.Mult(B, X);
   }

   result = 0.0;
   result.SetFromTrueDofs(X);
}
// Compute gradient of H1 scalar field and project to H(curl) space
// Maps H1 scalar → H(curl) using gradient operator
void compute_gradient_H1_to_Hcurl(ParGridFunction &result,       // in H(curl) space (output)
                                  ParGridFunction &phi_h1,        // in H1 scalar space (input) 
                                  ParFiniteElementSpace *h1_fes,  // H1 scalar trial space
                                  ParFiniteElementSpace *nd_fes,  // H(curl) test space
                                  bool pa)
{
   // H(curl) mass matrix on the target space
   ParBilinearForm *a = new ParBilinearForm(nd_fes);
   if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a->AddDomainIntegrator(new VectorFEMassIntegrator()); // H(curl) mass
   
   // Mixed gradient operator: H1 scalar → H(curl)
   ParMixedBilinearForm *a_mixed = new ParMixedBilinearForm(h1_fes, nd_fes);
   if (pa) { a_mixed->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a_mixed->AddDomainIntegrator(new MixedVectorGradientIntegrator()); // grad operator
   
   a->Assemble();
   if (!pa) { a->Finalize(); }
   
   a_mixed->Assemble();
   if (!pa) { a_mixed->Finalize(); }
   
   Vector B(nd_fes->GetTrueVSize());
   Vector X(nd_fes->GetTrueVSize());
   
   if (pa)
   {
      ParLinearForm b(nd_fes); // used as a vector
      a_mixed->Mult(phi_h1, b); // process-local multiplication
      b.ParallelAssemble(B);
   }
   else
   {
      HypreParMatrix *mixed = a_mixed->ParallelAssemble();
      
      Vector P(h1_fes->GetTrueVSize());
      phi_h1.GetTrueDofs(P);
      
      mixed->Mult(P, B);
      
      delete mixed;
   }
   
   // Solve the linear system
   if (pa)
   {
      Array<int> ess_tdof_list; // empty
      
      OperatorPtr A;
      a->FormSystemMatrix(ess_tdof_list, A);
      
      OperatorJacobiSmoother Jacobi(*a, ess_tdof_list);
      
      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(1000);
      cg.SetPrintLevel(0);
      cg.SetOperator(*A);
      cg.SetPreconditioner(Jacobi);
      X = 0.0;
      cg.Mult(B, X);
   }
   else
   {
      HypreParMatrix *Amat = a->ParallelAssemble();
      HypreDiagScale Jacobi(*Amat);
      HyprePCG pcg(*Amat);
      pcg.SetTol(1e-12);
      pcg.SetMaxIter(1000);
      pcg.SetPrintLevel(0);
      pcg.SetPreconditioner(Jacobi);
      X = 0.0;
      pcg.Mult(B, X);
      
      delete Amat;
   }
   
   result.SetFromTrueDofs(X);
   
   // Clean up
   delete a;
   delete a_mixed;
}
void RemoveMean(ParGridFunction &gf, ParFiniteElementSpace *fes)
{
   int myid = Mpi::WorldRank();
   
   // Method 1: Use inner product properly
   Vector gf_vec(fes->GetTrueVSize());
   gf.GetTrueDofs(gf_vec);
   
   // Create constant function = 1 everywhere
   ParGridFunction one_gf(fes);
   ConstantCoefficient one_coeff(1.0);
   one_gf.ProjectCoefficient(one_coeff);
   
   Vector one_vec(fes->GetTrueVSize());
   one_gf.GetTrueDofs(one_vec);
   
   // Compute ∫gf dx / ∫1 dx = mean
   double numerator = gf_vec * one_vec;    // ∫gf dx
   double denominator = one_vec * one_vec;  // ∫1 dx = volume
   double mean = numerator / denominator;
   
   // Subtract mean
   gf_vec.Add(-mean, one_vec);
   gf.SetFromTrueDofs(gf_vec);
   
   if (myid == 0)
   {
      cout << "Removed mean value: " << mean << " from field" << endl;
   }
}