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
//   3 = "shocklet": the compressible part is a periodic tanh compression
//       sheet u_c = (tanh(sin(2 pi x)/delta), 0, 0) of width ~delta, plus a
//       smooth vortical part with ZERO x-component.  Since the vortical
//       x-component vanishes and rho = rho(x), div(rho u_v) = 0 as well, so
//       the rho-weighted and unweighted Helmholtz splits have the SAME exact
//       answer and the error tables stay meaningful with -rs.  Combined with
//       the density diagnostics (10c) this reproduces, without any time
//       integration, the "KE drift once the shock forms" seen when kinetic
//       energy is defined through an L2-projected density (Laghos
//       ComputeDensity) instead of the quadrature-point density.
static int vortical_field = 0;

// Field 3 parameters: tanh layer width and relative density jump.  The layer
// center is shifted to a generic point: centered on a mesh vertex (or element
// midpoint) of the uniform grid, the odd projection error times the even
// |u|^2 profile integrates to zero by symmetry and the KE-definition gap
// cancels exactly -- an accident real shocks do not enjoy.
static double layer_delta = 0.05;
static double layer_drho  = 0.5;
static const double layer_x0 = 0.1327;

// Density-weighted Poisson solve: (rho grad phi, grad psi) = (rho u, grad psi).
static bool rho_weighted_solve = false;

// Periodic tanh layer: smooth for delta > 0, approaching a square wave with
// interfaces at x = 0 and x = 1/2 as delta -> 0.  f(s + 1/2) = -f(s), so it
// is exactly mean zero and its antiderivative (the potential phi) is periodic.
double tanh_layer(double s) { return tanh(sin(2*M_PI*s)/layer_delta); }

// Density for the shocklet study: jumps collocated with the velocity layers,
// as across a real shock.  tanh_layer is ANTIPERIODIC with period 1/2 (two
// opposite-sign fronts per period) while |u|^2 has period 1/2, so with a
// plain tanh density every rho-antisymmetric error (projection residual,
// quadrature error) cancels EXACTLY between the two fronts.  The cosine
// modulation gives the two fronts different jump strengths and breaks that
// hidden cancellation.
double rho_exact(const Vector &x)
{
   return 1.0 + layer_drho*tanh_layer(x(0) - layer_x0)
              *(0.75 + 0.25*cos(2*M_PI*x(0)));
}

// Compressible part: u_c = grad(phi) with phi = -1/(2pi) * sum cos(2 pi x_i).
// The Taylor-Green field (2) is purely solenoidal: grad(phi) = 0.  The
// shocklet field (3) has the tanh compression sheet as its potential part.
void grad_phi_exact(const Vector &x, Vector &u)
{
   if (vortical_field == 2) { u = 0.0; return; }
   if (vortical_field == 3)
   {
      u = 0.0;
      u(0) = tanh_layer(x(0) - layer_x0);
      return;
   }
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
   else if (vortical_field == 3)
   {
      // Vortical companion for the shocklet: a SHEAR layer collocated with
      // the compression sheet (a tangential velocity jump, as across an
      // oblique shock) plus a smooth 3D part.  div = 0 term by term (each
      // component independent of its own coordinate) and w_x = 0, so
      // div(rho(x) w) = 0 too -- the weighted split shares the exact answer.
      // The shear layer matters for the density diagnostics: the ND space
      // truncates the x-degree of u_x to k-1, making u_x^2 orthogonal to the
      // density projection residual; u_y keeps x-degree k, so the shear
      // carries the |u|^2 layer content that correlates with the rho error.
      w(0) = 0.0;
      w(1) = tanh_layer(x(0) - layer_x0) + sin(2*M_PI*(x(0) + x(2)));
      w(2) = sin(2*M_PI*(x(0) + x(1)));
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
   // Field-3 density diagnostics (see step 10c).
   double ke_rho_q   = 0.0;   // 1/2 (rho u, u), analytic rho, hi-order quad
   double ke_rho_p   = 0.0;   // same, rho = its element-wise L2 projection
   double ke_rho_qd  = 0.0;   // analytic rho with default quadrature rule
   double cross_rho  = 0.0;   // (rho u_c, u_v), analytic rho
   double id_gap_nd  = 0.0;   // KE(u_h1) - [KE_c + KE_v + 2(rho u_c,u_v)]:
                              // energy-split identity when the TOTAL is taken
                              // on the H1 velocity but the split comes from
                              // the ND-projected chain (production pattern)
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

   // 3. The velocity is born in the vector H1 space (as in the Navier solver)
   //    and is used DIRECTLY everywhere downstream: in the Poisson RHS and as
   //    the field that gets split.  An ND copy is kept only for visualization.
   //    (Routing the pipeline through ND -- as earlier versions did -- is
   //    lossy: the ND x-component only carries degree k-1 in x, an O(1)
   //    pointwise loss at under-resolved fronts.  Then u_c + u_v reconstructs
   //    the ND image of u rather than u itself, and the energy identity
   //    KE = KE_c + KE_v + 2(rho u_c, u_v) fails against a KE computed on the
   //    H1 velocity, growing as fronts sharpen.)
   ParGridFunction u_h1(&h1_vector);
   VectorFunctionCoefficient u_coeff(dim, u_exact);
   u_h1.ProjectCoefficient(u_coeff);
   VectorGridFunctionCoefficient u_h1_coeff(&u_h1);

   ParGridFunction u_gf(&nd_space);
   u_gf.ProjectCoefficient(u_h1_coeff);   // reference/visualization only

   // 4. Assemble the Poisson operator  a(phi,psi) = (grad phi, grad psi),
   //    or its density-weighted variant (rho grad phi, grad psi) with -rs.
   //    With -pa the operator is matrix-free (tensor-product kernels, the
   //    device-friendly path); otherwise a sparse matrix is assembled.
   //    rho is not polynomial, so every rho-weighted form below uses ONE
   //    explicit high-order rule: mixing default rules would make each form
   //    a slightly different discrete inner product and silently break the
   //    algebraic identities (weak divergence, energy split).
   FunctionCoefficient rho_coeff(rho_exact);
   const IntegrationRule &ir_hi = IntRules.Get(Geometry::CUBE, 3*order + 6);

   ParBilinearForm a(&h1_scalar);
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   if (rho_weighted_solve)
   {
      auto *di = new DiffusionIntegrator(rho_coeff);
      di->SetIntRule(&ir_hi);
      a.AddDomainIntegrator(di);
   }
   else { a.AddDomainIntegrator(new DiffusionIntegrator()); }
   a.Assemble();

   // 5. Assemble the weak-divergence right-hand side  b(psi) = (u, grad psi).
   //    DomainLFGradIntegrator(Q) assembles exactly (Q, grad psi), with Q the
   //    velocity coefficient, so we never take a derivative of u.
   ScalarVectorProductCoefficient rho_u_coeff(rho_coeff, u_h1_coeff);
   ParLinearForm b(&h1_scalar);
   if (rho_weighted_solve)
   {
      auto *lfi = new DomainLFGradIntegrator(rho_u_coeff);
      lfi->SetIntRule(&ir_hi);
      b.AddDomainIntegrator(lfi);
   }
   else { b.AddDomainIntegrator(new DomainLFGradIntegrator(u_h1_coeff)); }
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
   //    Both images are EXACT on hexes: continuous Q_k and ND_k are subspaces
   //    of the broken (Q_k)^d space, so u_c + u_v == u_h1 at the dof level.
   ParGridFunction u_l2(&l2_vector), uc_l2(&l2_vector);
   VectorGridFunctionCoefficient u_c_as_coeff(&u_c);
   u_l2.ProjectCoefficient(u_h1_coeff);       // exact H1 -> L2 image
   uc_l2.ProjectCoefficient(u_c_as_coeff);    // exact ND -> L2 image

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
   //     With -rs the same check is done in the rho-weighted inner product,
   //     (rho u_v, grad psi), which is what that solve enforces.
   {
      VectorGridFunctionCoefficient u_v_coeff(&u_v);
      ScalarVectorProductCoefficient rho_uv_coeff(rho_coeff, u_v_coeff);
      ParLinearForm bv(&h1_scalar);
      if (rho_weighted_solve)
      {
         auto *lfi = new DomainLFGradIntegrator(rho_uv_coeff);
         lfi->SetIntRule(&ir_hi);
         bv.AddDomainIntegrator(lfi);
      }
      else { bv.AddDomainIntegrator(new DomainLFGradIntegrator(u_v_coeff)); }
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

   // 10c. Density diagnostics for the shocklet field: the kinetic energy
   //      1/2 (rho u, u) computed with three versions of the SAME density.
   //        rho_q : analytic rho evaluated at the points of a high-order
   //                quadrature rule (the Laghos qdata analogue -- "truth"),
   //        rho_p : its element-wise L2 projection into the order-k L2 space
   //                (the Laghos ComputeDensity analogue), same rule,
   //        rho_q with the mass integrator's default rule (rho is not a
   //                polynomial, so the rule choice changes the answer).
   //      While the layer is resolved all three agree; once delta < h the
   //      projected density is O(1) wrong inside the layer elements and the
   //      KE definitions split apart -- no time integration involved.
   if (vortical_field == 3)
   {
      // Element-wise L2 projection of rho (block-diagonal mass solve).
      ParGridFunction rho_l2(&l2_scalar);
      {
         ParBilinearForm mrho(&l2_scalar);
         auto *mi = new MassIntegrator();
         mi->SetIntRule(&ir_hi);
         mrho.AddDomainIntegrator(mi);
         mrho.Assemble();
         mrho.Finalize();
         HypreParMatrix *Mr = mrho.ParallelAssemble();
         ParLinearForm rl(&l2_scalar);
         auto *dlf = new DomainLFIntegrator(rho_coeff);
         dlf->SetIntRule(&ir_hi);
         rl.AddDomainIntegrator(dlf);
         rl.Assemble();
         Vector R(l2_scalar.GetTrueVSize()), RHO(l2_scalar.GetTrueVSize());
         rl.ParallelAssemble(R);
         RHO = 0.0;
         CGSolver mcg(MPI_COMM_WORLD);
         mcg.SetOperator(*Mr);
         mcg.SetRelTol(1e-14);
         mcg.SetMaxIter(200);
         mcg.SetPrintLevel(0);
         mcg.Mult(R, RHO);
         rho_l2.SetFromTrueDofs(RHO);
         delete Mr;
      }
      GridFunctionCoefficient rho_l2_coeff(&rho_l2);

      auto weighted_mass = [&](Coefficient &rc,
                               const IntegrationRule *ir) -> HypreParMatrix*
      {
         ParBilinearForm mw(&l2_vector);
         auto *vmi = new VectorMassIntegrator(rc);
         if (ir) { vmi->SetIntRule(ir); }
         mw.AddDomainIntegrator(vmi);
         mw.Assemble();
         mw.Finalize();
         return mw.ParallelAssemble();
      };
      HypreParMatrix *M_q  = weighted_mass(rho_coeff,    &ir_hi);
      HypreParMatrix *M_p  = weighted_mass(rho_l2_coeff, &ir_hi);
      HypreParMatrix *M_qd = weighted_mass(rho_coeff,    nullptr);

      auto minner = [&](HypreParMatrix *M, const Vector &va, const Vector &vb)
      {
         Vector Mb(vb.Size());
         M->Mult(vb, Mb);
         return InnerProduct(MPI_COMM_WORLD, va, Mb);
      };
      m.ke_rho_q  = 0.5*minner(M_q,  U, U);
      m.ke_rho_p  = 0.5*minner(M_p,  U, U);
      m.ke_rho_qd = 0.5*minner(M_qd, U, U);
      m.cross_rho = minner(M_q, UC, UV);

      // Energy-split identity, production-style: total KE on the ORIGINAL H1
      // velocity, split terms from the ND-projected chain.  The H1 -> L2
      // nodal projection is exact (Q_k continuous is a subset of Q_k broken),
      // so UH represents u_h1 itself; the gap below is therefore exactly the
      // energy the H1 -> ND hop loses at under-resolved fronts.  Defining
      // u_v := u_h1 - u_c in the common L2 space (and driving the Poisson
      // RHS with u_h1 directly) closes this identity to round-off.
      {
         ParGridFunction uh1_l2(&l2_vector);
         VectorGridFunctionCoefficient u_h1_c(&u_h1);
         uh1_l2.ProjectCoefficient(u_h1_c);
         Vector UH;
         uh1_l2.GetTrueDofs(UH);
         const double ke_h1 = 0.5*minner(M_q, UH, UH);
         m.id_gap_nd = ke_h1 - (0.5*minner(M_q, UC, UC)
                                + 0.5*minner(M_q, UV, UV) + m.cross_rho);
      }

      delete M_q;
      delete M_p;
      delete M_qd;
   }

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
                  "2 = Taylor-Green vortex (purely solenoidal, exact u_c = 0), "
                  "3 = tanh shocklet + density diagnostics.");
   args.AddOption(&layer_delta, "-delta", "--layer-width",
                  "Width of the tanh layer for field 3.");
   args.AddOption(&layer_drho, "-drho", "--density-jump",
                  "Relative density jump across the field-3 layer.");
   args.AddOption(&rho_weighted_solve, "-rs", "--rho-solve", "-no-rs",
                  "--no-rho-solve",
                  "Density-weighted Poisson solve: (rho grad phi, grad psi) = "
                  "(rho u, grad psi), with the weak-divergence check done in "
                  "the same weighted inner product.");
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

      // Density diagnostics for the shocklet field: one kinetic energy, three
      // density representations (see step 10c in RunLevel).
      if (vortical_field == 3)
      {
         mfem::out << "\n# density-weighted KE:  rho_q = analytic rho at hi-order quadrature (qdata analogue),\n"
                      "#                        rho_p = element-wise L2-projected rho (ComputeDensity analogue)\n"
                   << setw(6)  << "level"
                   << setw(12) << "h_min"
                   << setw(15) << "KE(rho_q)"
                   << setw(15) << "dKE(project)"
                   << setw(15) << "dKE(quadrule)"
                   << setw(16) << "(rho u_c,u_v)"
                   << setw(15) << "id-gap(ND)" << "\n"
                   << string(94, '-') << "\n";
         for (size_t i = 0; i < results.size(); ++i)
         {
            mfem::out << setw(6)  << (int)i+1
                      << setw(12) << results[i].h_min
                      << setw(15) << results[i].ke_rho_q
                      << setw(15) << (results[i].ke_rho_p  - results[i].ke_rho_q)
                      << setw(15) << (results[i].ke_rho_qd - results[i].ke_rho_q)
                      << setw(16) << results[i].cross_rho
                      << setw(15) << results[i].id_gap_nd << "\n";
         }
      }
   }

   Mpi::Finalize();
   return 0;
}
