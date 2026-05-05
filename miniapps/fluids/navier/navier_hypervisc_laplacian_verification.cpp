// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// Standalone verification of the scalar discrete Laplacian / hyper-Laplacian
// used by the hyperviscosity patch. This mirrors the paper's Section 6.1 test:
//   u(x,y)   = -(1 / (2*pi^2)) sin(pi x) cos(pi y)
//   Delta u  =  sin(pi x) cos(pi y)
//   Delta^2u = -2*pi^2 sin(pi x) cos(pi y)
// on [0,1]^2 with Cartesian quadrilateral meshes under h-refinement.
//
// Important implementation note:
// We use the primal-to-primal discrete Laplacian described in the paper,
//
//   Delta_h ~= M^{-1} S_hat,   S_hat = S_bc - S,
//
// where S is the usual H1 stiffness matrix and S_bc is the boundary term
// induced by integration by parts. For the repeated-application verification,
// the boundary term must be included; omitting it causes the operator to fail
// the paper's convergence study.

#include "mfem.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace mfem;

struct LapCtx
{
   int order_min = 2;
   int order_max = 8;
   int ref_min = 2;
   int ref_max = 6;
   bool paper_cases = false;
   bool pa = false;
   bool include_boundary = true;
   bool flip_boundary_normal = false;
   bool analytic_boundary = false;
   bool print_mass_solve = false;
   int cg_max_it = 500;
   real_t cg_rtol = 1e-12;
   real_t cg_atol = 0.0;
   bool visit = false;
   std::string csv = "hypervisc_laplacian_verification.csv";
} ctx;

class ScalarUCoeff : public Coefficient
{
public:
   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      T.SetIntPoint(&ip);
      Vector x;
      T.Transform(ip, x);
      return -(1.0 / (2.0 * M_PI * M_PI)) * std::sin(M_PI * x(0)) *
             std::cos(M_PI * x(1));
   }
};

class ScalarLapUCoeff : public Coefficient
{
public:
   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      T.SetIntPoint(&ip);
      Vector x;
      T.Transform(ip, x);
      return std::sin(M_PI * x(0)) * std::cos(M_PI * x(1));
   }
};

class ScalarLap2UCoeff : public Coefficient
{
public:
   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      T.SetIntPoint(&ip);
      Vector x;
      T.Transform(ip, x);
      return -2.0 * M_PI * M_PI * std::sin(M_PI * x(0)) *
             std::cos(M_PI * x(1));
   }
};

class ScalarGradUCoeff : public VectorCoefficient
{
public:
   ScalarGradUCoeff() : VectorCoefficient(2) { }

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      T.SetIntPoint(&ip);
      Vector x;
      T.Transform(ip, x);
      V.SetSize(2);
      V(0) = -(1.0 / (2.0 * M_PI)) * std::cos(M_PI * x(0)) *
             std::cos(M_PI * x(1));
      V(1) =  (1.0 / (2.0 * M_PI)) * std::sin(M_PI * x(0)) *
             std::sin(M_PI * x(1));
   }
};

class ScalarGradLapUCoeff : public VectorCoefficient
{
public:
   ScalarGradLapUCoeff() : VectorCoefficient(2) { }

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      T.SetIntPoint(&ip);
      Vector x;
      T.Transform(ip, x);
      V.SetSize(2);
      V(0) = M_PI * std::cos(M_PI * x(0)) * std::cos(M_PI * x(1));
      V(1) = -M_PI * std::sin(M_PI * x(0)) * std::sin(M_PI * x(1));
   }
};

class ScaledVectorCoefficient : public VectorCoefficient
{
private:
   VectorCoefficient &base;
   real_t scale;

public:
   ScaledVectorCoefficient(real_t scale_, VectorCoefficient &base_)
      : VectorCoefficient(base_.GetVDim()), base(base_), scale(scale_) { }

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      base.Eval(V, T, ip);
      V *= scale;
   }
};

/// Boundary linear-form integrator for int_{dOmega} (d_n u_h) v_h.
class BoundaryNormalDerivativeLFIntegrator : public LinearFormIntegrator
{
private:
   GradientGridFunctionCoefficient grad_u;
   const real_t normal_sign;
   mutable Vector shape, grad_val, nor;

public:
   explicit BoundaryNormalDerivativeLFIntegrator(const GridFunction *gf,
                                                 real_t normal_sign_ = 1.0)
      : grad_u(gf), normal_sign(normal_sign_) { }

   using LinearFormIntegrator::AssembleRHSElementVect;

   void AssembleRHSElementVect(const FiniteElement &el,
                               ElementTransformation &Tr,
                               Vector &elvect) override
   {
      elvect.SetSize(el.GetDof());
      elvect = 0.0;
   }

   void AssembleRHSElementVect(const FiniteElement &el,
                               FaceElementTransformations &Tr,
                               Vector &elvect) override
   {
      const int dof = el.GetDof();
      const int dim = Tr.Elem1->GetSpaceDim();

      shape.SetSize(dof);
      grad_val.SetSize(dim);
      nor.SetSize(dim);
      elvect.SetSize(dof);
      elvect = 0.0;

      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
      {
         // On a face, grad(u_h) has degree p-1 and the scalar test function has
         // degree p, so degree 2p-1 is sufficient on affine faces.
         ir = &IntRules.Get(Tr.FaceGeom, 2 * el.GetOrder());
      }

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         Tr.SetAllIntPoints(&ip);
         const IntegrationPoint &eip = Tr.GetElement1IntPoint();

         grad_u.Eval(grad_val, *Tr.Elem1, eip);
         if (dim == 1)
         {
            nor(0) = 2.0 * eip.x - 1.0;
         }
         else
         {
            CalcOrtho(Tr.Face->Jacobian(), nor);
         }

         const real_t dudn = normal_sign * (grad_val * nor);
         el.CalcShape(eip, shape);
         add(elvect, ip.weight * dudn, shape, elvect);
      }
   }
};

struct ScalarLaplacianOperator
{
   ParFiniteElementSpace *fes = nullptr;
   ParBilinearForm *m_form = nullptr;
   ParBilinearForm *k_form = nullptr;
   OperatorHandle M_handle, K_handle;
   CGSolver *M_inv = nullptr;
   bool include_boundary = true;
   bool flip_boundary_normal = false;
   bool print_mass_solve = false;

   mutable ParGridFunction x_gf;

   ScalarLaplacianOperator(ParFiniteElementSpace *pfes,
                           bool include_boundary_,
                           bool flip_boundary_normal_,
                           bool print_mass_solve_,
                           bool use_pa,
                           int cg_max_it,
                           real_t cg_rtol,
                           real_t cg_atol)
      : fes(pfes),
        include_boundary(include_boundary_),
        flip_boundary_normal(flip_boundary_normal_),
        print_mass_solve(print_mass_solve_),
        x_gf(fes)
   {
      m_form = new ParBilinearForm(fes);
      k_form = new ParBilinearForm(fes);

      if (use_pa)
      {
         m_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
         k_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      }

      m_form->AddDomainIntegrator(new MassIntegrator());
      k_form->AddDomainIntegrator(new DiffusionIntegrator());

      m_form->Assemble();
      k_form->Assemble();
      if (!use_pa)
      {
         m_form->Finalize();
         k_form->Finalize();
      }

      Array<int> ess_tdof_list;
      m_form->FormSystemMatrix(ess_tdof_list, M_handle);
      k_form->FormSystemMatrix(ess_tdof_list, K_handle);

      M_inv = new CGSolver(MPI_COMM_WORLD);
      M_inv->iterative_mode = false;
      M_inv->SetPrintLevel(0);
      M_inv->SetMaxIter(cg_max_it);
      M_inv->SetRelTol(cg_rtol);
      M_inv->SetAbsTol(cg_atol);
      M_inv->SetOperator(*M_handle.Ptr());
   }

   ~ScalarLaplacianOperator()
   {
      delete M_inv;
      delete m_form;
      delete k_form;
   }

   void Apply(const Vector &x, Vector &y,
              VectorCoefficient *analytic_bdr_grad = nullptr,
              const char *label = nullptr) const
   {
      // Interior contribution: -S x.
      Vector rhs;
      rhs.SetSize(x.Size());
      K_handle.Ptr()->Mult(x, rhs);
      rhs *= -1.0;

      if (include_boundary)
      {
         ParLinearForm b_form(fes);
         const real_t normal_sign = flip_boundary_normal ? -1.0 : 1.0;
         if (analytic_bdr_grad)
         {
            ScaledVectorCoefficient flux_coeff(normal_sign, *analytic_bdr_grad);
            b_form.AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(flux_coeff));
         }
         else
         {
            // Boundary contribution: +S_bc x.
            x_gf = 0.0;
            x_gf.SetFromTrueDofs(x);
            b_form.AddBdrFaceIntegrator(
               new BoundaryNormalDerivativeLFIntegrator(&x_gf, normal_sign));
         }
         b_form.Assemble();

         Vector b_rhs;
         b_form.ParallelAssemble(b_rhs);
         rhs += b_rhs;
      }

      // Solve M y = rhs.
      M_inv->Mult(rhs, y);
      if (print_mass_solve && Mpi::Root())
      {
         out << "Mass solve";
         if (label) { out << " [" << label << "]"; }
         out << ": iters = " << M_inv->GetNumIterations()
             << ", final norm = " << M_inv->GetFinalNorm() << "\n";
      }
   }
};

static real_t Rate(const real_t e_prev, const real_t e_cur)
{
   if (e_prev <= 0.0 || e_cur <= 0.0)
   {
      return std::numeric_limits<real_t>::quiet_NaN();
   }
   return std::log(e_prev / e_cur) / std::log(2.0);
}

static int PaperRefMaxForOrder(const int order)
{
   if (order <= 5) { return 6; }
   if (order == 6) { return 5; }
   if (order == 7) { return 4; }
   return 3;
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   OptionsParser args(argc, argv);
   args.AddOption(&ctx.order_min, "-omin", "--order-min",
                  "Minimum scalar polynomial degree.");
   args.AddOption(&ctx.order_max, "-omax", "--order-max",
                  "Maximum scalar polynomial degree.");
   args.AddOption(&ctx.ref_min, "-rmin", "--refine-min",
                  "Minimum refinement exponent; nx = ny = 2^rmin.");
   args.AddOption(&ctx.ref_max, "-rmax", "--refine-max",
                  "Maximum refinement exponent; nx = ny = 2^rmax.");
   args.AddOption(&ctx.paper_cases,
                  "-paper", "--paper-cases",
                  "-no-paper", "--no-paper-cases",
                  "Use the Section 6.1 paper sweep: orders 2..8 and refinements 2..6.");
   args.AddOption(&ctx.pa,
                  "-pa", "--enable-pa",
                  "-no-pa", "--disable-pa",
                  "Enable partial assembly for the scalar mass and stiffness operators.");
   args.AddOption(&ctx.include_boundary,
                  "-bc", "--include-boundary",
                  "-no-bc", "--interior-only",
                  "Include or omit the boundary term S_bc in Delta_h = M^{-1}(S_bc-S).");
   args.AddOption(&ctx.flip_boundary_normal,
                  "-flip-bdr-norm", "--flip-boundary-normal",
                  "-no-flip-bdr-norm", "--no-flip-boundary-normal",
                  "Flip the sign of the boundary normal in the boundary term.");
   args.AddOption(&ctx.analytic_boundary,
                  "-analytic-bdr", "--analytic-boundary",
                  "-no-analytic-bdr", "--no-analytic-boundary",
                  "Use analytic normal derivatives in the boundary term where available.");
   args.AddOption(&ctx.print_mass_solve,
                  "-print-mass-solve", "--print-mass-solve",
                  "-no-print-mass-solve", "--no-print-mass-solve",
                  "Print CG iterations and final residual for each mass-inverse apply.");
   args.AddOption(&ctx.cg_max_it, "-mi", "--max-it",
                  "Maximum CG iterations for the scalar mass inverse.");
   args.AddOption(&ctx.cg_rtol, "-rtol", "--cg-rtol",
                  "Relative tolerance for the scalar mass inverse CG solve.");
   args.AddOption(&ctx.cg_atol, "-atol", "--cg-atol",
                  "Absolute tolerance for the scalar mass inverse CG solve.");
   args.AddOption(&ctx.visit,
                  "-visit", "--visit-output",
                  "-no-visit", "--no-visit-output",
                  "Write VisIt data for the finest mesh of each order.");
   args.AddOption(&ctx.csv, "-csv", "--csv-file",
                  "CSV file for convergence data.");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root()) { args.PrintUsage(out); }
      return 1;
   }

   if (ctx.paper_cases)
   {
      ctx.order_min = 2;
      ctx.order_max = 8;
      ctx.ref_min = 2;
      ctx.ref_max = 6;
   }

   if (Mpi::Root()) { args.PrintOptions(out); }

   if (ctx.order_max < ctx.order_min || ctx.ref_max < ctx.ref_min)
   {
      if (Mpi::Root())
      {
         mfem::err << "Invalid range for order or refinement." << std::endl;
      }
      return 2;
   }

   std::ofstream csv;
   if (Mpi::Root())
   {
      csv.open(ctx.csv);
      csv << "order,refinement,nx,ny,elements,scalar_dofs,assembly,err_lap1,"
             "rate_lap1,err_lap2_repeated,rate_lap2_repeated,"
             "err_lap2_from_exact_lap1,rate_lap2_from_exact_lap1\n";
      out << "\nScalar hypervisc Laplacian verification\n"
          << "-------------------------------------\n";
      out << "u(x,y)   = -(1/(2*pi^2)) sin(pi x) cos(pi y)\n"
          << "Delta u  =  sin(pi x) cos(pi y)\n"
          << "Delta^2u = -2*pi^2 sin(pi x) cos(pi y)\n\n";
   }

   ScalarUCoeff u_coeff;
   ScalarLapUCoeff lap1_coeff;
   ScalarLap2UCoeff lap2_coeff;
   ScalarGradUCoeff grad_u_coeff;
   ScalarGradLapUCoeff grad_lap1_coeff;

   for (int order = ctx.order_min; order <= ctx.order_max; order++)
   {
      const int ref_max = ctx.paper_cases ?
                          std::min(ctx.ref_max, PaperRefMaxForOrder(order)) :
                          ctx.ref_max;

      std::vector<real_t> err1_list, err2_rep_list, err2_exactlap1_list;
      if (Mpi::Root())
      {
         out << "Order N = " << order << "\n";
         out << std::setw(6) << "r"
             << std::setw(10) << "nx"
             << std::setw(16) << "err_lap1"
             << std::setw(12) << "rate1"
             << std::setw(20) << "err_lap2_rep"
             << std::setw(12) << "rate2r"
             << std::setw(20) << "err_lap2_exact"
             << std::setw(12) << "rate2e" << "\n";
      }

      for (int r = ctx.ref_min; r <= ref_max; r++)
      {
         const int n = 1 << r;

         Mesh serial_mesh = Mesh::MakeCartesian2D(n, n,
                                                  Element::QUADRILATERAL,
                                                  true,
                                                  1.0,
                                                  1.0);
         ParMesh pmesh(MPI_COMM_WORLD, serial_mesh);

         H1_FECollection fec(order, 2);
         ParFiniteElementSpace fes(&pmesh, &fec);

         ParGridFunction u_gf(&fes), lap1_gf(&fes), lap2_gf(&fes),
                         lap1_exact_gf(&fes), lap2_from_exact_lap1_gf(&fes);
         u_gf = 0.0;
         lap1_gf = 0.0;
         lap2_gf = 0.0;
         lap1_exact_gf = 0.0;
         lap2_from_exact_lap1_gf = 0.0;
         u_gf.ProjectCoefficient(u_coeff);
         lap1_exact_gf.ProjectCoefficient(lap1_coeff);

         ScalarLaplacianOperator lap_op(&fes, ctx.include_boundary,
                                        ctx.flip_boundary_normal,
                                        ctx.print_mass_solve,
                                        ctx.pa, ctx.cg_max_it,
                                        ctx.cg_rtol, ctx.cg_atol);

         Vector x, y, z, y_exact, z_from_exact_lap1;
         u_gf.GetTrueDofs(x);
         y.SetSize(x.Size());
         z.SetSize(x.Size());
         lap_op.Apply(x, y,
                      ctx.analytic_boundary ? &grad_u_coeff : nullptr,
                      "lap1");
         lap_op.Apply(y, z, nullptr, "lap2_repeated");
         lap1_gf.SetFromTrueDofs(y);
         lap2_gf.SetFromTrueDofs(z);

         lap1_exact_gf.GetTrueDofs(y_exact);
         z_from_exact_lap1.SetSize(y_exact.Size());
         lap_op.Apply(y_exact, z_from_exact_lap1,
                      ctx.analytic_boundary ? &grad_lap1_coeff : nullptr,
                      "lap2_exact_lap1");
         lap2_from_exact_lap1_gf.SetFromTrueDofs(z_from_exact_lap1);

         const real_t err1 = lap1_gf.ComputeL2Error(lap1_coeff);
         const real_t err2_rep = lap2_gf.ComputeL2Error(lap2_coeff);
         const real_t err2_exact_lap1 =
            lap2_from_exact_lap1_gf.ComputeL2Error(lap2_coeff);
         err1_list.push_back(err1);
         err2_rep_list.push_back(err2_rep);
         err2_exactlap1_list.push_back(err2_exact_lap1);

         const real_t rate1 = (err1_list.size() > 1) ?
            Rate(err1_list[err1_list.size() - 2], err1) :
            std::numeric_limits<real_t>::quiet_NaN();
         const real_t rate2_rep = (err2_rep_list.size() > 1) ?
            Rate(err2_rep_list[err2_rep_list.size() - 2], err2_rep) :
            std::numeric_limits<real_t>::quiet_NaN();
         const real_t rate2_exact_lap1 = (err2_exactlap1_list.size() > 1) ?
            Rate(err2_exactlap1_list[err2_exactlap1_list.size() - 2],
                 err2_exact_lap1) :
            std::numeric_limits<real_t>::quiet_NaN();

         if (Mpi::Root())
         {
            out << std::setw(6) << r
                << std::setw(10) << n
                << std::setw(16) << std::setprecision(8) << std::scientific
                << err1
                << std::setw(12) << rate1
                << std::setw(20) << err2_rep
                << std::setw(12) << rate2_rep
                << std::setw(20) << err2_exact_lap1
                << std::setw(12) << rate2_exact_lap1 << "\n";

            csv << order << ',' << r << ',' << n << ',' << n << ','
                << pmesh.GetNE() << ',' << fes.GlobalTrueVSize() << ','
                << (ctx.pa ? "PA" : "FA") << ','
                << std::setprecision(16) << err1 << ',' << rate1 << ','
                << err2_rep << ',' << rate2_rep << ','
                << err2_exact_lap1 << ',' << rate2_exact_lap1 << '\n';
         }

         if (ctx.visit && r == ref_max)
         {
            VisItDataCollection visit_dc("hypervisc_lap_verify_o" +
                                            std::to_string(order),
                                         &pmesh);
            visit_dc.RegisterField("u_proj", &u_gf);
            visit_dc.RegisterField("lap1_h", &lap1_gf);
            visit_dc.RegisterField("lap2_h", &lap2_gf);
            visit_dc.SetCycle(order);
            visit_dc.SetTime(static_cast<real_t>(order));
            visit_dc.Save();
         }
      }

      if (Mpi::Root()) { out << "\n"; }
   }

   return 0;
}
