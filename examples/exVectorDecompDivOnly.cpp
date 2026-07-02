//                 Simplified "divergence-only" Helmholtz decomposition
//
// Compile with: make exVectorDecompDivOnly
//
// Sample run:   mpirun -np 4 exVectorDecompDivOnly -o 2 -n 4 -ref 4
//
// Description:
//
//   The Helmholtz decomposition of a (here, periodic) velocity field is
//
//         u = u_c + u_v = grad(phi) + curl(A),
//
//   where u_c = grad(phi) is the compressible / irrotational part and
//   u_v = curl(A) is the vortical / solenoidal part.
//
//   The full example (exVectorDecompConvergence.cpp) solves BOTH a scalar
//   Poisson problem for phi AND a vector curl-curl problem for A.  This
//   standalone experiment tries the cheaper route: only solve for the
//   compressible part, then recover the vortical part by subtraction.
//
//   Taking the divergence of u and using div(curl(A)) = 0 gives a single
//   scalar Poisson problem for the potential phi:
//
//         div(u) = div(grad(phi)) = Laplacian(phi).
//
//   On a periodic domain (no boundary terms) the weak form is: find phi in
//   H1 (defined up to a constant) such that
//
//         (grad(phi), grad(psi)) = (u, grad(psi))   for all psi in H1.
//
//   Note the right-hand side (u, grad(psi)) is the *weak* divergence of u:
//   we never differentiate u, we only integrate against grad(psi).  The
//   constant null space (periodic mesh) is handled with an OrthoSolver.
//
//   Once phi is known:
//
//         u_c = grad(phi)      (compressible part)
//         u_v = u - u_c        (vortical part, by subtraction)
//
//   To test correctness we use a manufactured field with a known exact
//   split (the same one used by exVectorDecompConvergence.cpp):
//
//         u_exact = grad_phi_exact + curl_A_exact,
//
//   and report, under mesh refinement,
//
//         ||u_c - grad_phi_exact||   compressible error   -> should converge
//         ||u_v - curl_A_exact||     vortical error       -> should converge
//         ||u  - (u_c + u_v)||       reconstruction       -> ~ machine zero
//         ||div(u_v)||               solenoidality of u_v -> should be small
//         ||(u_v, grad psi)||        weak divergence      -> solver tolerance

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>

using namespace std;
using namespace mfem;

// Spatial dimension (this experiment is 3D and periodic).
static const int dim = 3;

// Manufactured exact fields -------------------------------------------------

// Vortical field selector:
//   0 = each component independent of its own coordinate, so div vanishes
//       term by term.  Tensor-product nodal interpolation preserves this,
//       making the interpolant *exactly* divergence-free element-wise (the
//       element-wise div check sits at round-off).
//   1 = "generic" cyclic field whose divergence vanishes only by cancellation
//       between terms; its interpolant has an O(h^k) element-wise divergence,
//       so that check becomes a real convergence quantity while the
//       weak-divergence identity should remain at solver tolerance.
//   2 = Taylor-Green vortex.  Exactly divergence free, so the exact split is
//       u_c = 0, u_v = u: the compressible "error" measures the spurious
//       dilatational field the discrete decomposition invents.
static int vortical_field = 0;

// Compressible part: u_c = grad(phi) with phi = -1/(2pi) * sum cos(2 pi x_i).
// The Taylor-Green field (2) is purely solenoidal: grad(phi) = 0.
void grad_phi_exact(const Vector &x, Vector &u)
{
   if (vortical_field == 2) { u = 0.0; return; }
   u(0) = sin(2*M_PI*x(0));
   u(1) = sin(2*M_PI*x(1));
   u(2) = sin(2*M_PI*x(2));
}

// Vortical part: u_v = curl(A).  Divergence free by construction.
void curl_A_exact(const Vector &x, Vector &w)
{
   if (vortical_field == 0)
   {
      w(0) = sin(4*M_PI*x(1)) + sin(6*M_PI*x(2));
      w(1) = sin(6*M_PI*x(0)) + sin(4*M_PI*x(2));
      w(2) = sin(4*M_PI*x(0)) + sin(6*M_PI*x(1));
   }
   else if (vortical_field == 1)
   {
      // w = curl(-A/(2*pi)) with A = (s2*s3, s3*s1, s1*s2);
      // div(w) = 2*pi*(c1*c2 - c1*c3 + c2*c3 - c2*c1 + c3*c1 - c3*c2) = 0
      // only by cancellation across components.
      const double s1 = sin(2*M_PI*x(0)), c1 = cos(2*M_PI*x(0));
      const double s2 = sin(2*M_PI*x(1)), c2 = cos(2*M_PI*x(1));
      const double s3 = sin(2*M_PI*x(2)), c3 = cos(2*M_PI*x(2));
      w(0) = s1*(c2 - c3);
      w(1) = s2*(c3 - c1);
      w(2) = s3*(c1 - c2);
   }
   else
   {
      // Taylor-Green vortex: div(w) = 2*pi*(c1*c2*c3 - c1*c2*c3) = 0, again
      // only by cancellation, so its interpolant is not exactly div free.
      const double s1 = sin(2*M_PI*x(0)), c1 = cos(2*M_PI*x(0));
      const double s2 = sin(2*M_PI*x(1)), c2 = cos(2*M_PI*x(1));
      const double s3 = sin(2*M_PI*x(2)), c3 = cos(2*M_PI*x(2));
      w(0) =  s1*c2*c3;
      w(1) = -c1*s2*c3;
      w(2) = 0.0;
   }
}

// Full field: u = grad(phi) + curl(A).
void u_exact(const Vector &x, Vector &u)
{
   Vector gc(dim), wc(dim);
   grad_phi_exact(x, gc);
   curl_A_exact(x, wc);
   u(0) = gc(0) + wc(0);
   u(1) = gc(1) + wc(1);
   u(2) = gc(2) + wc(2);
}

// Metrics gathered per refinement level.
struct LevelMetrics
{
   double h_min      = 0.0;
   long long dofs    = 0;
   double err_uc_l2  = 0.0;   // ||u_c - grad_phi_exact||_L2
   double err_uc_l1  = 0.0;   // ||u_c - grad_phi_exact||_L1
   double err_uv_l2  = 0.0;   // ||u_v - curl_A_exact||_L2
   double err_uv_l1  = 0.0;   // ||u_v - curl_A_exact||_L1
   double err_recon  = 0.0;   // ||u - (u_c + u_v)||
   double curl_uc    = 0.0;   // ||curl(u_c)||  (discrete de Rham identity)
   double div_uv     = 0.0;   // ||div(u_v)||_L2  (element-wise div, projected
                              // into the discontinuous L2 space)
   double weak_div_uv = 0.0;  // ||(u_v, grad psi)|| / ||(u, grad psi)||  (identity)
   double ke_total   = 0.0;   // 1/2 (u, u)
   double ke_c       = 0.0;   // 1/2 (u_c, u_c)
   double ke_v       = 0.0;   // 1/2 (u_v, u_v)
   double inner_cv   = 0.0;   // (u_c, u_v)   -> ~0 if orthogonal
};

// Build a periodic unit-cube hex mesh with num_pts cells per direction.
Mesh MakePeriodicCube(int num_pts)
{
   Mesh base = Mesh::MakeCartesian3D(num_pts, num_pts, num_pts,
                                     Element::HEXAHEDRON, 1.0, 1.0, 1.0, false);
   Vector tx({1.0, 0.0, 0.0}), ty({0.0, 1.0, 0.0}), tz({0.0, 0.0, 1.0});
   std::vector<Vector> translations = {tx, ty, tz};
   return Mesh::MakePeriodic(base,
                             base.CreatePeriodicVertexMapping(translations));
}

// Solve one refinement level and fill in the metrics.  If do_visit is true,
// the computed and exact fields are written to a VisIt data collection.
void RunLevel(int num_pts, int order, int par_ref, LevelMetrics &m,
              bool pa = false, bool do_visit = false)
{
   // 1. Periodic mesh, refined par_ref times in parallel.
   Mesh serial_mesh = MakePeriodicCube(num_pts);
   ParMesh pmesh(MPI_COMM_WORLD, serial_mesh);
   pmesh.Finalize(true);
   for (int l = 0; l < par_ref; ++l) { pmesh.UniformRefinement(); }

   // 2. Finite element spaces along the de Rham complex
   //        H1 --grad--> ND(H(curl)) --curl--> RT(H(div))
   //    - phi lives in the scalar H1 space.
   //    - u, u_c, u_v live in the ND (H(curl)) space.  Putting grad(phi) in ND
   //      via the discrete gradient makes curl(grad(phi)) = 0 hold *exactly*.
   //    - curl(u_c) is measured in the RT (H(div)) space.
   H1_FECollection h1_fec(order, dim);
   ND_FECollection nd_fec(order, dim);
   RT_FECollection rt_fec(order-1, dim);
   L2_FECollection l2_fec(order, dim);
   ParFiniteElementSpace h1_scalar(&pmesh, &h1_fec);
   ParFiniteElementSpace h1_vector(&pmesh, &h1_fec, dim);
   ParFiniteElementSpace nd_space(&pmesh, &nd_fec);
   ParFiniteElementSpace rt_space(&pmesh, &rt_fec);
   ParFiniteElementSpace l2_vector(&pmesh, &l2_fec, dim);   // vortical part lives here
   ParFiniteElementSpace l2_scalar(&pmesh, &l2_fec);        // for ||div(u_v)|| checks

   // 3. The velocity is born in the vector H1 space (as in the Navier solver),
   //    then projected into ND (H(curl)).  We use a nodal (interpolation)
   //    projection: sample u_h1's values at the ND degrees of freedom.  This
   //    is cheap and standalone (no mass solve); an L2/Galerkin projection is
   //    the accuracy-preserving alternative used by exVectorDecompConvergence.
   ParGridFunction u_h1(&h1_vector);
   VectorFunctionCoefficient u_coeff(dim, u_exact);
   u_h1.ProjectCoefficient(u_coeff);

   ParGridFunction u_gf(&nd_space);
   VectorGridFunctionCoefficient u_h1_coeff(&u_h1);
   u_gf.ProjectCoefficient(u_h1_coeff);   // nodal H1 -> ND projection

   // 4. Assemble the Poisson operator  a(phi,psi) = (grad phi, grad psi).
   //    With -pa the operator is matrix-free (tensor-product kernels, the
   //    device-friendly path); otherwise a sparse matrix is assembled.
   ParBilinearForm a(&h1_scalar);
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a.AddDomainIntegrator(new DiffusionIntegrator());
   a.Assemble();

   // 5. Assemble the weak-divergence right-hand side  b(psi) = (u, grad psi).
   //    DomainLFGradIntegrator(Q) assembles exactly (Q, grad psi), with Q the
   //    velocity coefficient, so we never take a derivative of u.
   VectorGridFunctionCoefficient u_gf_coeff(&u_gf);
   ParLinearForm b(&h1_scalar);
   b.AddDomainIntegrator(new DomainLFGradIntegrator(u_gf_coeff));
   b.Assemble();

   Array<int> empty_tdofs;   // periodic: no essential BCs
   OperatorPtr A;
   a.FormSystemMatrix(empty_tdofs, A);

   Vector B(h1_scalar.GetTrueVSize()), PHI(h1_scalar.GetTrueVSize());
   b.ParallelAssemble(B);
   PHI = 0.0;

   // 6. Solve the singular (periodic => constant null space) system.  The
   //    OrthoSolver wraps the *preconditioner* (as in MFEM's navier solver),
   //    so every CG iterate is projected off the constant null space; running
   //    Hypre PCG directly on the singular system can stall or break down
   //    with a NaN residual.  BoomerAMG needs an assembled matrix, so the
   //    matrix-free (-pa) path uses a Jacobi smoother built from the PA
   //    diagonal instead; CG then needs more iterations (no multigrid).
   std::unique_ptr<Solver> prec;
   if (pa)
   {
      prec = std::make_unique<OperatorJacobiSmoother>(a, empty_tdofs);
   }
   else
   {
      auto amg = std::make_unique<HypreBoomerAMG>(*A.As<HypreParMatrix>());
      amg->SetPrintLevel(0);
      prec = std::move(amg);
   }

   OrthoSolver ortho_prec(MPI_COMM_WORLD);
   ortho_prec.SetSolver(*prec);

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetAbsTol(0.0);
   cg.SetMaxIter(pa ? 10000 : 1000);
   cg.SetPrintLevel(0);
   cg.SetPreconditioner(ortho_prec);
   cg.SetOperator(*A);
   cg.Mult(B, PHI);

   ParGridFunction phi(&h1_scalar);
   phi.SetFromTrueDofs(PHI);

   // 7. Compressible part u_c = grad(phi) in the ND (H(curl)) space, formed
   //    with the DISCRETE gradient operator G (a topological matrix, not a
   //    solve): G maps H1 dofs to ND dofs so that G*phi represents grad(phi_h)
   //    *exactly* in ND.  Because the discrete de Rham complex satisfies
   //    curl(grad(.)) = 0 at the matrix level, curl(u_c) will be zero to
   //    round-off.
   ParDiscreteLinearOperator grad_op(&h1_scalar, &nd_space);
   grad_op.AddDomainInterpolator(new GradientInterpolator());
   grad_op.Assemble();
   grad_op.Finalize();

   ParGridFunction u_c(&nd_space);
   grad_op.Mult(phi, u_c);

   // 8. Vortical part in the L2 vector space.  We project u and u_c into L2 by
   //    NODAL interpolation and subtract there: u_v = Proj_L2(u) - Proj_L2(u_c).
   //    (By linearity this equals Proj_L2(u - u_c).)  u_l2 and uc_l2 are kept
   //    for the L2 energy / orthogonality checks below.
   ParGridFunction u_l2(&l2_vector), uc_l2(&l2_vector);
   VectorGridFunctionCoefficient u_gf_as_coeff(&u_gf);
   VectorGridFunctionCoefficient u_c_as_coeff(&u_c);
   u_l2.ProjectCoefficient(u_gf_as_coeff);    // nodal ND -> L2
   uc_l2.ProjectCoefficient(u_c_as_coeff);    // nodal ND -> L2

   ParGridFunction u_v(&l2_vector);
   u_v = u_l2;
   u_v -= uc_l2;

   // 8b. Quick-and-dirty solenoidality check: the element-wise
   //     divergence of the L2 vortical field.  DivergenceGridFunctionCoefficient
   //     evaluates div(u_v) inside each element; we project it into a scalar L2
   //     space and take its L2 norm (ComputeL2Error vs zero).  Since u_v
   //     approximates the divergence-free curl(A), this should decrease under
   //     refinement and increasing order.
   {
      ConstantCoefficient zero_scalar(0.0);
      DivergenceGridFunctionCoefficient div_uv_coeff(&u_v);
      ParGridFunction div_uv_gf(&l2_scalar);
      div_uv_gf.ProjectCoefficient(div_uv_coeff);
      m.div_uv = div_uv_gf.ComputeL2Error(zero_scalar);
   }

   // 8c. Weak-divergence check: the Galerkin condition of the Poisson solve
   //     makes u_v L2-orthogonal to all discrete gradients,
   //
   //         (u_v, grad psi) = 0   for all psi in H1,
   //
   //     which is the discrete statement "u_v is weakly divergence-free".
   //     Unlike the element-wise divergence above (a convergence quantity;
   //     it is also blind to normal jumps across faces), this is
   //     an identity that should hold at solver tolerance on every level.  We
   //     assemble the same linear form used for the Poisson RHS, but with u_v
   //     as the coefficient, and report the l2 norm of the resulting dual
   //     vector relative to that of the RHS b(psi) = (u, grad psi).
   {
      VectorGridFunctionCoefficient u_v_coeff(&u_v);
      ParLinearForm bv(&h1_scalar);
      bv.AddDomainIntegrator(new DomainLFGradIntegrator(u_v_coeff));
      bv.Assemble();
      Vector BV(h1_scalar.GetTrueVSize());
      bv.ParallelAssemble(BV);
      double bv2 = InnerProduct(MPI_COMM_WORLD, BV, BV);
      double b2  = InnerProduct(MPI_COMM_WORLD, B, B);
      m.weak_div_uv = sqrt(bv2) / sqrt(b2);
   }

   // 9. Errors against the known exact split.
   VectorFunctionCoefficient grad_phi_ex(dim, grad_phi_exact);
   VectorFunctionCoefficient curl_A_ex(dim, curl_A_exact);
   m.err_uc_l2 = u_c.ComputeL2Error(grad_phi_ex);
   m.err_uc_l1 = u_c.ComputeL1Error(grad_phi_ex);
   m.err_uv_l2 = u_v.ComputeL2Error(curl_A_ex);
   m.err_uv_l1 = u_v.ComputeL1Error(curl_A_ex);

   // Reconstruction: Proj_L2(u) - (u_c + u_v), measured in L2.  Zero by
   // construction since u_v = u_l2 - uc_l2 and u_c projects to uc_l2.
   ParGridFunction recon(&l2_vector);
   recon = uc_l2;
   recon += u_v;
   recon -= u_l2;
   Vector zerov(dim); zerov = 0.0;
   VectorConstantCoefficient zerovec(zerov);
   m.err_recon = recon.ComputeL2Error(zerovec);

   // 10. Discrete de Rham check: curl(u_c) in RT via the DISCRETE curl
   //     operator.  Since u_c = G*phi and curl(grad) = 0 discretely, this is
   //     the identity that holds exactly in this space (down to round-off).
   ParDiscreteLinearOperator curl_op(&nd_space, &rt_space);
   curl_op.AddDomainInterpolator(new CurlInterpolator());
   curl_op.Assemble();
   curl_op.Finalize();

   ParGridFunction curl_uc(&rt_space);
   curl_op.Mult(u_c, curl_uc);
   m.curl_uc = curl_uc.ComputeL2Error(zerovec);

   // 10b. Kinetic energies and the u_c . u_v inner product, evaluated in the L2
   //      vector space with one consistent inner product (a, b) = a^T M_L2 b,
   //      using the L2 representations u_l2, uc_l2, u_v.  Since u_v = u_l2 -
   //      uc_l2 exactly, KE_total = KE_c + KE_v + (u_c, u_v).  The Galerkin
   //      condition makes u_v L2-orthogonal to discrete gradients, so (u_c,u_v)
   //      should stay near zero (up to the nodal-projection consistency).
   ParBilinearForm mass_l2(&l2_vector);
   mass_l2.AddDomainIntegrator(new VectorMassIntegrator());
   mass_l2.Assemble();
   mass_l2.Finalize();
   HypreParMatrix *Ml2 = mass_l2.ParallelAssemble();

   Vector U, UC, UV;
   u_l2.GetTrueDofs(U);
   uc_l2.GetTrueDofs(UC);
   u_v.GetTrueDofs(UV);

   auto mass_inner = [&](const Vector &a, const Vector &b) -> double
   {
      Vector Mb(b.Size());
      Ml2->Mult(b, Mb);
      return InnerProduct(MPI_COMM_WORLD, a, Mb);
   };

   m.ke_total = 0.5 * mass_inner(U,  U);
   m.ke_c     = 0.5 * mass_inner(UC, UC);
   m.ke_v     = 0.5 * mass_inner(UV, UV);
   m.inner_cv = mass_inner(UC, UV);

   delete Ml2;

   // 11. Optional VisIt dump: computed fields alongside the exact fields.
   //     u_c is in ND (compared to grad_phi_exact in ND); u_v is in L2
   //     (compared to curl_A_exact in L2).
   if (do_visit)
   {
      ParGridFunction grad_phi_exact_gf(&nd_space);
      ParGridFunction curl_A_exact_gf(&l2_vector);
      grad_phi_exact_gf.ProjectCoefficient(grad_phi_ex);
      curl_A_exact_gf.ProjectCoefficient(curl_A_ex);

      // Pointwise error fields (computed - exact) for quick visual inspection.
      ParGridFunction uc_err(&nd_space), uv_err(&l2_vector);
      uc_err = u_c;  uc_err -= grad_phi_exact_gf;
      uv_err = u_v;  uv_err -= curl_A_exact_gf;

      VisItDataCollection dc("exVectorDecompDivOnly", &pmesh);
      dc.SetPrefixPath("visit_output");
      dc.RegisterField("u",              &u_gf);
      dc.RegisterField("phi",            &phi);
      dc.RegisterField("u_c",            &u_c);
      dc.RegisterField("u_v",            &u_v);
      dc.RegisterField("grad_phi_exact", &grad_phi_exact_gf);
      dc.RegisterField("curl_A_exact",   &curl_A_exact_gf);
      dc.RegisterField("u_c_error",      &uc_err);
      dc.RegisterField("u_v_error",      &uv_err);
      dc.RegisterField("curl_u_c",       &curl_uc);
      dc.SetLevelsOfDetail(order);
      dc.SetCycle(par_ref);
      dc.SetTime(par_ref);
      dc.Save();
      if (Mpi::Root())
      {
         mfem::out << "  [visit] wrote data collection "
                   << "'visit_output/exVectorDecompDivOnly' "
                   << "(cycle " << par_ref << ")" << endl;
      }
   }

   // 12. Mesh size and problem size.
   double h_max, kmin, kmax;
   pmesh.GetCharacteristics(m.h_min, h_max, kmin, kmax);
   m.dofs = nd_space.GlobalTrueVSize();
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   int order    = 2;
   int num_pts  = 4;
   int max_ref  = 4;
   bool visit   = false;
   bool pa      = false;
   const char *device_config = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&order,   "-o",   "--order",       "Finite element order.");
   args.AddOption(&num_pts, "-n",   "--num-pts",     "Base cells per direction.");
   args.AddOption(&max_ref, "-ref", "--max-ref",     "Number of refinement levels.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly",
                  "Matrix-free Poisson operator with Jacobi-preconditioned CG "
                  "(the device-friendly path) instead of assembled matrix + AMG.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&vortical_field, "-f", "--field",
                  "Vortical field: 0 = interpolation-exactly div-free, "
                  "1 = generic (interpolant has O(h^k) element-wise divergence), "
                  "2 = Taylor-Green vortex (purely solenoidal, exact u_c = 0).");
   args.AddOption(&visit, "-vis", "--visit", "-no-vis", "--no-visit",
                  "Dump fields (computed + exact) to VisIt on the finest level.");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root()) { args.PrintUsage(mfem::out); }
      Mpi::Finalize();
      return 1;
   }
   if (Mpi::Root()) { args.PrintOptions(mfem::out); }

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   std::vector<LevelMetrics> results;
   for (int ref = 0; ref < max_ref; ++ref)
   {
      LevelMetrics m;
      bool dump = visit && (ref == max_ref - 1);   // dump on the finest level
      RunLevel(num_pts, order, ref, m, pa, dump);
      results.push_back(m);
      if (Mpi::Root())
      {
         mfem::out << "[level " << ref+1 << "/" << max_ref
                   << "] dofs=" << m.dofs
                   << "  h_min=" << m.h_min << endl;
      }
   }

   if (Mpi::Root())
   {
      auto rate = [](double e, double e0, double h, double h0) -> double
      {
         if (e > 1e-16 && e0 > 1e-16 && h < h0 && h0 > 0.0)
         {
            return log(e/e0)/log(h/h0);
         }
         return 0.0;
      };

      auto table = [&](const string &title,
                       double LevelMetrics::*field,
                       const string &err_label = "error")
      {
         mfem::out << "\n# " << title << "\n"
                   << setw(6)  << "level"
                   << setw(14) << "DOFs"
                   << setw(14) << "h_min"
                   << setw(16) << err_label
                   << setw(10) << "rate" << "\n"
                   << string(60, '-') << "\n";
         for (size_t i = 0; i < results.size(); ++i)
         {
            double r = 0.0;
            if (i > 0)
            {
               r = rate(results[i].*field, results[i-1].*field,
                        results[i].h_min,   results[i-1].h_min);
            }
            mfem::out << setw(6)  << (int)i+1
                      << setw(14) << results[i].dofs
                      << setw(14) << results[i].h_min
                      << setw(16) << results[i].*field
                      << setw(10) << r << "\n";
         }
      };

      table("compressible error  ||u_c - grad_phi_exact||_L2", &LevelMetrics::err_uc_l2, "L2 error");
      table("compressible error  ||u_c - grad_phi_exact||_L1", &LevelMetrics::err_uc_l1, "L1 error");
      table("vortical error      ||u_v - curl_A_exact||_L2",   &LevelMetrics::err_uv_l2, "L2 error");
      table("vortical error      ||u_v - curl_A_exact||_L1",   &LevelMetrics::err_uv_l1, "L1 error");
      table("reconstruction      ||u - (u_c + u_v)||",         &LevelMetrics::err_recon, "L2 error");
      table("irrotationality     ||curl(u_c)||",               &LevelMetrics::curl_uc,   "L2 error");
      table("solenoidality       ||div(u_v)||  (element-wise, disc. L2 proj.)",  &LevelMetrics::div_uv,    "L2 error");
      table("weak divergence     ||(u_v, grad psi)|| (rel.)",  &LevelMetrics::weak_div_uv, "rel. norm");

      // Energy split and orthogonality of the decomposition.
      //   KE_total should equal KE_c + KE_v when (u_c, u_v) = 0.
      mfem::out << "\n# kinetic energy split and orthogonality  "
                   "(KE = 1/2 (.,.),  ideally (u_c,u_v) = 0)\n"
                << setw(6)  << "level"
                << setw(14) << "KE_total"
                << setw(14) << "KE_c"
                << setw(14) << "KE_v"
                << setw(14) << "KE_c+KE_v"
                << setw(16) << "(u_c,u_v)" << "\n"
                << string(78, '-') << "\n";
      for (size_t i = 0; i < results.size(); ++i)
      {
         mfem::out << setw(6)  << (int)i+1
                   << setw(14) << results[i].ke_total
                   << setw(14) << results[i].ke_c
                   << setw(14) << results[i].ke_v
                   << setw(14) << (results[i].ke_c + results[i].ke_v)
                   << setw(16) << results[i].inner_cv << "\n";
      }
   }

   Mpi::Finalize();
   return 0;
}
