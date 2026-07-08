/*
 * vgt_mfem_periodic.cpp
 *
 * Single-run VGT diagnostic on a periodic ABC (Arnold-Beltrami-Childress)
 * velocity field:
 *
 *   u_1(x,y,z) = a sin(2 pi z) + c cos(2 pi y)
 *   u_2(x,y,z) = b sin(2 pi x) + a cos(2 pi z)
 *   u_3(x,y,z) = c sin(2 pi y) + b cos(2 pi x)
 *
 * with incommensurate amplitudes  a = 1, b = sqrt(2), c = sqrt(3).
 *
 * Properties:
 *   - Periodic on [0,1]^3.
 *   - Divergence-free (incompressible): trace(A) == 0 everywhere.
 *   - Beltrami (curl u proportional to u).
 *   - No constant offset matrix B (impossible while remaining periodic),
 *     so the degenerate-eigenvalue set is a codim-1 surface rather than
 *     empty. The cubic-discriminant filter handles it.
 *
 * Mesh: N x N x N hex on [0,1]^3, made fully periodic via
 *       Mesh::MakePeriodic with axis-aligned translations.
 *
 * Diagnostics (mirror vgt_mfem_example.cpp exactly):
 *   Tier 1 -- gradient L^2 error:        all quadrature points
 *   Tier 2 -- partition identity max:    all quadrature points (~eps_mach)
 *   Tier 3 -- component RMS error:       regular points only
 *
 * Backend:
 *   This file uses the LAPACK backend from vgt_lapack.hpp for both
 *   decomposition methods (EIG and Schur).
 *
 * Build:
 *   make MFEM_CXX=/usr/local/bin/mpicxx vgt_mfem_periodic
 *
 * Run (serial, one MPI rank):
 *   ./vgt_mfem_periodic
 *   ./vgt_mfem_periodic -o 2 -n 8
 *
 * Run (parallel):
 *   mpirun -n 4 ./vgt_mfem_periodic -o 2 -n 16
 *
 * Options:
 *   -o <int>   polynomial order (default 2)
 *   -n <int>   elements per side, N x N x N mesh (default 16)
 *
 * Output:
 *   visit_output/vgt_mfem_periodic/vgt_mfem_periodic_vis_000000.mfem_root
 *   <- open this file in VisIt
 *
 * Exit: 0 = partition identity holds for both pathways, 1 = failure.
 */

#include "mfem.hpp"
#include "vgt_lapack.hpp"
#include "vgt_mfem_vis_space.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

static constexpr double PI = M_PI;
static const double ABC_A = 1.0;
static const double ABC_B = std::sqrt(2.0);
static const double ABC_C = std::sqrt(3.0);

static void u_abc_func(const mfem::Vector& x, mfem::Vector& v)
{
    const double sx = std::sin(2 * PI * x[0]);
    const double cx = std::cos(2 * PI * x[0]);
    const double sy = std::sin(2 * PI * x[1]);
    const double cy = std::cos(2 * PI * x[1]);
    const double sz = std::sin(2 * PI * x[2]);
    const double cz = std::cos(2 * PI * x[2]);

    v[0] = ABC_A * sz + ABC_C * cy;
    v[1] = ABC_B * sx + ABC_A * cz;
    v[2] = ABC_C * sy + ABC_B * cx;
}

static vgt_lapack::Mat3L A_exact_at(const mfem::Vector& x)
{
    const double sx = std::sin(2 * PI * x[0]);
    const double cx = std::cos(2 * PI * x[0]);
    const double sy = std::sin(2 * PI * x[1]);
    const double cy = std::cos(2 * PI * x[1]);
    const double sz = std::sin(2 * PI * x[2]);
    const double cz = std::cos(2 * PI * x[2]);
    const double k = 2 * PI;

    vgt_lapack::Mat3L M;
    M(0, 0) = 0.0;            M(0, 1) = -ABC_C * k * sy; M(0, 2) =  ABC_A * k * cz;
    M(1, 0) =  ABC_B * k * cx; M(1, 1) = 0.0;            M(1, 2) = -ABC_A * k * sz;
    M(2, 0) = -ABC_B * k * sx; M(2, 1) =  ABC_C * k * cy; M(2, 2) = 0.0;
    return M;
}

static double disc_norm(const vgt_lapack::Mat3L& A)
{
    const double I1 = A(0, 0) + A(1, 1) + A(2, 2);
    double trA2 = 0.0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            trA2 += A(i, j) * A(j, i);
    const double I2 = 0.5 * (I1 * I1 - trA2);
    const double I3 = A(0, 0) * (A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1))
                    - A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0))
                    + A(0, 2) * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0));
    const double D = I1 * I1 * I2 * I2 - 4.0 * I2 * I2 * I2
                   - 4.0 * I1 * I1 * I1 * I3 - 27.0 * I3 * I3 + 18.0 * I1 * I2 * I3;
    const double sc = std::max(1.0, std::sqrt(A.squaredNorm()));
    return D / std::pow(sc, 6);
}

// ── VisIt coefficients ───────────────────────────────────────────────────────
// Each class below is an mfem::Coefficient: a callable that MFEM evaluates at a
// point (via Eval) to fill a grid function during ProjectCoefficient. They turn
// the velocity field / analytic formula into the scalar fields we plot.

// Frobenius norm of the *discrete* VGT:  ||A_h||_F = sqrt( sum_ij (du_i/dx_j)^2 ).
class VGTFroNormCoeff : public mfem::Coefficient
{
    mfem::ParGridFunction& vel;

public:
    explicit VGTFroNormCoeff(mfem::ParGridFunction& v) : vel(v) {}

    double Eval(mfem::ElementTransformation& T,
                const mfem::IntegrationPoint& ip) override
    {
        mfem::DenseMatrix gv(3, 3);
        vel.GetVectorGradient(T, gv);   // differentiate velocity -> 3x3 gradient
        return gv.FNorm();              // built-in Frobenius norm of that matrix
    }
};

// Frobenius norm of the *exact* analytic VGT:  ||A_exact||_F. Ground-truth
// reference for the field above.
class VGTFroNormExactCoeff : public mfem::Coefficient
{
public:
    double Eval(mfem::ElementTransformation& T,
                const mfem::IntegrationPoint& ip) override
    {
        mfem::Vector xp(3);
        T.Transform(ip, xp);                          // physical coords of this point
        return std::sqrt(A_exact_at(xp).squaredNorm());
    }
};

// Pointwise gradient error:  ||A_h - A_exact||_F. Shows where the discrete
// gradient deviates from the analytic one.
class GradErrorCoeff : public mfem::Coefficient
{
    mfem::ParGridFunction& vel;

public:
    explicit GradErrorCoeff(mfem::ParGridFunction& v) : vel(v) {}

    double Eval(mfem::ElementTransformation& T,
                const mfem::IntegrationPoint& ip) override
    {
        mfem::DenseMatrix gv(3, 3);
        vel.GetVectorGradient(T, gv);                 // discrete gradient A_h
        mfem::Vector xp(3);
        T.Transform(ip, xp);
        const vgt_lapack::Mat3L Ae = A_exact_at(xp);  // exact gradient A_exact

        double err2 = 0.0;                            // accumulate squared entry diffs
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
                const double d = gv(i, j) - Ae(i, j);
                err2 += d * d;
            }
        return std::sqrt(err2);                       // = ||A_h - A_exact||_F
    }
};

// Normalized cubic discriminant of the exact VGT. Near zero => the eigenvalues
// are (nearly) degenerate; used to flag points the decomposition should skip.
class DeltaNormCoeff : public mfem::Coefficient
{
public:
    double Eval(mfem::ElementTransformation& T,
                const mfem::IntegrationPoint& ip) override
    {
        mfem::Vector xp(3);
        T.Transform(ip, xp);
        return std::abs(disc_norm(A_exact_at(xp)));
    }
};

// All four Schur partition components of the *discrete* VGT, as one vector
// field [A2_ax, A2_sh, A2_rr, A2_sr].
//
// This is a vdim-4 VectorCoefficient (it returns 4 numbers per point) rather
// than four separate scalar coefficients. The reason: the Schur decomposition
// produces all four components in a single call, so computing them together
// runs part_vgt_batch_schur ONCE per evaluation point instead of four times.
// The caller splits the resulting 4-component field back into four named
// scalar fields for VisIt.
class PartVecCoeff : public mfem::VectorCoefficient
{
    mfem::ParGridFunction& vel;

public:
    // VectorCoefficient(4) tells MFEM this coefficient has 4 components.
    explicit PartVecCoeff(mfem::ParGridFunction& v)
        : mfem::VectorCoefficient(4), vel(v) {}

    // Called by ProjectCoefficient at each node: evaluate the discrete VGT
    // here, decompose it once, and write all four components into V.
    void Eval(mfem::Vector& V, mfem::ElementTransformation& T,
              const mfem::IntegrationPoint& ip) override
    {
        mfem::DenseMatrix gv(3, 3);
        vel.GetVectorGradient(T, gv);              // discrete VGT at this point
        vgt_lapack::Mat3L Ah;
        for (int i = 0; i < 3; i++)                // repack row-major -> column-major
            for (int j = 0; j < 3; j++)
                Ah.d[i + 3 * j] = gv(i, j);
        const auto res = vgt_lapack::part_vgt_batch_schur({Ah});  // one decomposition
        V.SetSize(4);
        V[0] = res.A2_ax[0];   // axial / biaxial stretching
        V[1] = res.A2_sh[0];   // shearing
        V[2] = res.A2_rr[0];   // rigid rotation (Rortex/Liutex)
        V[3] = res.A2_sr[0];   // shear-rotation coupling
    }
};

// Same four components, but for the *exact* analytic VGT instead of the
// discrete field. Used as the ground-truth reference in VisIt.
class PartVecExactCoeff : public mfem::VectorCoefficient
{
public:
    PartVecExactCoeff() : mfem::VectorCoefficient(4) {}

    void Eval(mfem::Vector& V, mfem::ElementTransformation& T,
              const mfem::IntegrationPoint& ip) override
    {
        mfem::Vector xp(3);
        T.Transform(ip, xp);                       // physical coordinates of this point
        const vgt_lapack::Mat3L Ae = A_exact_at(xp);
        const auto res = vgt_lapack::part_vgt_batch_schur({Ae});  // one decomposition
        V.SetSize(4);
        V[0] = res.A2_ax[0];
        V[1] = res.A2_sh[0];
        V[2] = res.A2_rr[0];
        V[3] = res.A2_sr[0];
    }
};

static mfem::Mesh MakePeriodicCartesian3D(int N)
{
    mfem::Mesh base = mfem::Mesh::MakeCartesian3D(
        N, N, N, mfem::Element::HEXAHEDRON, 1.0, 1.0, 1.0, false);
    mfem::Vector tx({1.0, 0.0, 0.0});
    mfem::Vector ty({0.0, 1.0, 0.0});
    mfem::Vector tz({0.0, 0.0, 1.0});
    std::vector<mfem::Vector> translations = {tx, ty, tz};
    return mfem::Mesh::MakePeriodic(base, base.CreatePeriodicVertexMapping(translations));
}

static void hline(char c = '-', int w = 70)
{
    std::cout << std::string(w, c) << "\n";
}

// ── Explicit nodal projection (reference implementation) ─────────────────────
// This spells out what gf.ProjectCoefficient(coeff) does internally for an
// interpolatory space like L2: it visits every nodal point of every element,
// evaluates the coefficient there, and stores the result in the matching dof.
//
// The point set is fe.GetNodes() — the element's *nodal* points (in reference
// coordinates) — which is the visualization counterpart of the *quadrature*
// rule (IntRules.Get(...)) used by the error loop in main(). Both feed the same
// coefficient/GetVectorGradient machinery; only the sample points differ.
//
// NOTE: this is a teaching reference, equivalent to ProjectCoefficient only for
// interpolatory bases (the default L2 GaussLegendre basis qualifies). For a
// non-nodal basis MFEM's ProjectCoefficient instead does a local L2 solve, so
// prefer the built-in in production code. Used below on one field as a demo;
// the other fields keep the built-in ProjectCoefficient.
static void ProjectAtNodes(mfem::ParGridFunction& gf, mfem::Coefficient& coeff)
{
    mfem::ParFiniteElementSpace& fes = *gf.ParFESpace();
    mfem::Array<int> dofs;

    for (int e = 0; e < fes.GetNE(); e++)
    {
        const mfem::FiniteElement&   fe = *fes.GetFE(e);
        mfem::ElementTransformation& T  = *fes.GetElementTransformation(e);

        const mfem::IntegrationRule& nodes = fe.GetNodes();  // the nodal points
        fes.GetElementDofs(e, dofs);                         // global dof per node

        for (int i = 0; i < fe.GetDof(); i++)
        {
            const mfem::IntegrationPoint& ip = nodes.IntPoint(i);  // i-th node
            T.SetIntPoint(&ip);                  // evaluate the field at this node
            gf(dofs[i]) = coeff.Eval(T, ip);     // store the nodal value
        }
    }
}

int main(int argc, char* argv[])
{
    mfem::MPI_Session mpi(argc, argv);
    const int rank = mpi.WorldRank();
    const int size = mpi.WorldSize();

    int order = 2;
    int N = 16;
    std::string vis_space_name = "l2";
    VisualizationSpace vis_space = VisualizationSpace::l2;

    mfem::OptionsParser args(argc, argv);
    args.AddOption(&order, "-o", "--order", "Polynomial order (1, 2, 4, ...).");
    args.AddOption(&N, "-n", "--mesh-n", "Elements per side (N x N x N mesh).");
    args.AddOption(&vis_space_name, "-vs", "--vis-space",
                   "Visualization field space: 'l2' or 'h1'.");
    args.Parse();
    if (!args.Good()) {
        if (rank == 0) { args.PrintUsage(std::cout); }
        return 1;
    }
    if (!parse_visualization_space(vis_space_name, vis_space)) {
        if (rank == 0) {
            std::cerr << "Unknown visualization space '" << vis_space_name
                      << "'. Expected 'l2' or 'h1'\n";
        }
        return 1;
    }

    mfem::Mesh serial = MakePeriodicCartesian3D(N);
    mfem::ParMesh pmesh(MPI_COMM_WORLD, serial);
    serial.Clear();

    mfem::H1_FECollection fec(order, 3);
    mfem::ParFiniteElementSpace fes(&pmesh, &fec, 3);
    const HYPRE_BigInt ndofs = fes.GlobalTrueVSize();

    mfem::ParGridFunction vel(&fes);
    mfem::VectorFunctionCoefficient vcoeff(3, u_abc_func);
    vel.ProjectCoefficient(vcoeff);

    const mfem::IntegrationRule& ir =
        mfem::IntRules.Get(mfem::Geometry::CUBE, 2 * order + 1);
    const int nqp = ir.GetNPoints();

    int N_elem_local = pmesh.GetNE();
    int N_elem_global = 0;
    MPI_Allreduce(&N_elem_local, &N_elem_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    const long long N_qpts_global = static_cast<long long>(N_elem_global) * nqp;

    if (rank == 0) {
        hline('=');
        std::cout << "  VGT MFEM Periodic ABC Example  [MPI x" << size << "]\n";
        hline('=');
        std::cout << "  Field   : ABC, a=1, b=sqrt(2), c=sqrt(3), k=2pi\n";
        std::cout << "  Domain  : [0,1]^3 with full periodicity (x,y,z)\n";
        std::cout << "  Mesh    : " << N << "^3 hex"
                  << "  (h = " << std::fixed << std::setprecision(6) << 1.0 / N << ")\n";
        std::cout << "  Order   : P" << order << "\n";
        std::cout << "  Vis     : " << visualization_space_label(vis_space) << " fields\n";
        std::cout << "  DOFs    : " << ndofs << "  (global, all components)\n";
        std::cout << "  Qpts    : " << nqp << " per element  ("
                  << N_qpts_global << " total)\n";
        std::cout << "  Backend : LAPACK (vgt_lapack.hpp)\n";
        std::cout << "  Methods : EIG (Rortex/Liutex), Schur\n";
        hline();
    }

    double grad_err2 = 0.0;
    double eig_ax2 = 0.0, eig_sh2 = 0.0, eig_rr2 = 0.0, eig_sr2 = 0.0;
    double sch_ax2 = 0.0, sch_sh2 = 0.0, sch_rr2 = 0.0, sch_sr2 = 0.0;
    double ex_ax_sum = 0.0, ex_sh_sum = 0.0, ex_rr_sum = 0.0, ex_sr_sum = 0.0;
    double eig_pmax = 0.0, sch_pmax = 0.0;
    double reg_vol = 0.0, total_vol = 0.0;
    double min_dn = 1e300, max_dn = -1e300;
    long long skipped = 0, regular = 0;

    std::vector<vgt_lapack::Mat3L> Ah_vec(nqp), Ae_vec(nqp);
    std::vector<double> wts(nqp);
    std::vector<bool> reg(nqp);

    for (int e = 0; e < pmesh.GetNE(); e++)
    {
        mfem::ElementTransformation* T = pmesh.GetElementTransformation(e);

        for (int q = 0; q < nqp; q++)
        {
            const mfem::IntegrationPoint& ip = ir.IntPoint(q);
            T->SetIntPoint(&ip);
            const double wt = ip.weight * T->Weight();
            wts[q] = wt;
            total_vol += wt;

            mfem::Vector xp(3);
            T->Transform(ip, xp);

            mfem::DenseMatrix gv(3, 3);
            vel.GetVectorGradient(*T, gv);
            vgt_lapack::Mat3L Ah;
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    Ah.d[i + 3 * j] = gv(i, j);
            Ah_vec[q] = Ah;
            Ae_vec[q] = A_exact_at(xp);

            double d2 = 0.0;
            for (int k = 0; k < 9; k++) {
                const double d = Ah.d[k] - Ae_vec[q].d[k];
                d2 += d * d;
            }
            grad_err2 += wt * d2;

            const double dn = disc_norm(Ae_vec[q]);
            min_dn = std::min(min_dn, std::abs(dn));
            max_dn = std::max(max_dn, std::abs(dn));
            reg[q] = (std::abs(dn) > 1e-10);
            if (reg[q]) { ++regular; } else { ++skipped; }
        }

        const auto eig_h = vgt_lapack::part_vgt_batch_eig(Ah_vec);
        const auto sch_h = vgt_lapack::part_vgt_batch_schur(Ah_vec);
        const auto eig_ex = vgt_lapack::part_vgt_batch_eig(Ae_vec);
        const auto sch_ex = vgt_lapack::part_vgt_batch_schur(Ae_vec);

        for (int q = 0; q < nqp; q++)
        {
            const double wt = wts[q];
            const double An2 = Ah_vec[q].squaredNorm();

            eig_pmax = std::max(
                eig_pmax,
                std::abs(An2 - eig_h.A2_ax[q] - eig_h.A2_sh[q]
                               - eig_h.A2_rr[q] - eig_h.A2_sr[q]));
            sch_pmax = std::max(
                sch_pmax,
                std::abs(An2 - sch_h.A2_ax[q] - sch_h.A2_sh[q]
                               - sch_h.A2_rr[q] - sch_h.A2_sr[q]));

            if (reg[q]) {
                reg_vol += wt;
                auto sq = [](double v) { return v * v; };

                eig_ax2 += wt * sq(eig_h.A2_ax[q] - eig_ex.A2_ax[q]);
                eig_sh2 += wt * sq(eig_h.A2_sh[q] - eig_ex.A2_sh[q]);
                eig_rr2 += wt * sq(eig_h.A2_rr[q] - eig_ex.A2_rr[q]);
                eig_sr2 += wt * sq(eig_h.A2_sr[q] - eig_ex.A2_sr[q]);

                sch_ax2 += wt * sq(sch_h.A2_ax[q] - sch_ex.A2_ax[q]);
                sch_sh2 += wt * sq(sch_h.A2_sh[q] - sch_ex.A2_sh[q]);
                sch_rr2 += wt * sq(sch_h.A2_rr[q] - sch_ex.A2_rr[q]);
                sch_sr2 += wt * sq(sch_h.A2_sr[q] - sch_ex.A2_sr[q]);

                ex_ax_sum += wt * eig_ex.A2_ax[q];
                ex_sh_sum += wt * eig_ex.A2_sh[q];
                ex_rr_sum += wt * eig_ex.A2_rr[q];
                ex_sr_sum += wt * eig_ex.A2_sr[q];
            }
        }
    }

    auto rsum = [&](double v) { double g; MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); return g; };
    auto rmax = [&](double v) { double g; MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); return g; };
    auto rmin = [&](double v) { double g; MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD); return g; };
    auto rsuml = [&](long long v) { long long g; MPI_Allreduce(&v, &g, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD); return g; };

    const double g_grad = rsum(grad_err2);
    const double g_tvol = rsum(total_vol);
    const double g_rvol = rsum(reg_vol);
    const double g_ea2 = rsum(eig_ax2), g_es2 = rsum(eig_sh2);
    const double g_er2 = rsum(eig_rr2), g_ew2 = rsum(eig_sr2);
    const double g_sa2 = rsum(sch_ax2), g_ss2 = rsum(sch_sh2);
    const double g_sr2 = rsum(sch_rr2), g_sw2 = rsum(sch_sr2);
    const double g_xax = rsum(ex_ax_sum), g_xsh = rsum(ex_sh_sum);
    const double g_xrr = rsum(ex_rr_sum), g_xsr = rsum(ex_sr_sum);
    const double g_ep = rmax(eig_pmax), g_sp = rmax(sch_pmax);
    const double g_dnmin = rmin(min_dn), g_dnmax = rmax(max_dn);
    const long long g_sk = rsuml(skipped), g_rg = rsuml(regular);

    const double rv = std::max(g_rvol, 1e-300);
    auto rms = [&](double s2) { return std::sqrt(s2 / rv); };

    // ── Build the VisIt fields ───────────────────────────────────────────────
    // Scalar visualization space (viss_fes), plus a 4-component visualization
    // space (visv_fes) used to hold all four partition components at once.
    // byNODES ordering means the
    // 4-component data is laid out component-by-component, so each component is
    // a contiguous block we can copy straight into a scalar field below.
    auto vis_fec = make_visualization_fec(vis_space, order, 3);
    mfem::ParFiniteElementSpace viss_fes(&pmesh, vis_fec.get());
    mfem::ParFiniteElementSpace visv_fes(&pmesh, vis_fec.get(), 4, mfem::Ordering::byNODES);

    mfem::ParGridFunction vgt_fro_gf(&viss_fes);
    mfem::ParGridFunction vgt_fro_exact_gf(&viss_fes);
    mfem::ParGridFunction grad_err_gf(&viss_fes);
    mfem::ParGridFunction a2_ax_gf(&viss_fes);
    mfem::ParGridFunction a2_sh_gf(&viss_fes);
    mfem::ParGridFunction a2_rr_gf(&viss_fes);
    mfem::ParGridFunction a2_sr_gf(&viss_fes);
    mfem::ParGridFunction a2_ax_ex_gf(&viss_fes);
    mfem::ParGridFunction a2_sh_ex_gf(&viss_fes);
    mfem::ParGridFunction a2_rr_ex_gf(&viss_fes);
    mfem::ParGridFunction a2_sr_ex_gf(&viss_fes);
    mfem::ParGridFunction delta_norm_gf(&viss_fes);

    // Project the simple scalar fields (one decomposition-free pass each).
    VGTFroNormCoeff vgt_fro_coeff(vel);
    VGTFroNormExactCoeff vgt_fro_exact_coeff;
    GradErrorCoeff grad_err_coeff(vel);
    DeltaNormCoeff delta_coeff;
    if (vis_space == VisualizationSpace::l2)
    {
        ProjectAtNodes(vgt_fro_gf, vgt_fro_coeff);   // demo: explicit nodal projection
                                                     // (same result as ProjectCoefficient)
    }
    else
    {
        project_to_visualization_space(vgt_fro_gf, vgt_fro_coeff, vis_space);
    }
    project_to_visualization_space(vgt_fro_exact_gf, vgt_fro_exact_coeff, vis_space);
    project_to_visualization_space(grad_err_gf, grad_err_coeff, vis_space);
    project_to_visualization_space(delta_norm_gf, delta_coeff, vis_space);

    // Project the four partition components in ONE pass each (discrete + exact),
    // running the Schur decomposition once per node instead of four times, then
    // slice the 4-component result into the individual scalar fields.
    PartVecCoeff      part_coeff(vel);
    PartVecExactCoeff part_exact_coeff;
    mfem::ParGridFunction part_gf(&visv_fes);
    mfem::ParGridFunction part_exact_gf(&visv_fes);
    project_to_visualization_space(part_gf, part_coeff, vis_space);
    project_to_visualization_space(part_exact_gf, part_exact_coeff, vis_space);

    // Copy each contiguous component block [k*n, (k+1)*n) into its scalar field.
    // n = number of scalar dofs; component order is [ax, sh, rr, sr] (see Eval).
    const int n = viss_fes.GetVSize();
    std::copy_n(part_gf.GetData()       + 0 * n, n, a2_ax_gf.GetData());
    std::copy_n(part_gf.GetData()       + 1 * n, n, a2_sh_gf.GetData());
    std::copy_n(part_gf.GetData()       + 2 * n, n, a2_rr_gf.GetData());
    std::copy_n(part_gf.GetData()       + 3 * n, n, a2_sr_gf.GetData());
    std::copy_n(part_exact_gf.GetData() + 0 * n, n, a2_ax_ex_gf.GetData());
    std::copy_n(part_exact_gf.GetData() + 1 * n, n, a2_sh_ex_gf.GetData());
    std::copy_n(part_exact_gf.GetData() + 2 * n, n, a2_rr_ex_gf.GetData());
    std::copy_n(part_exact_gf.GetData() + 3 * n, n, a2_sr_ex_gf.GetData());

    mfem::VisItDataCollection dc("vgt_mfem_periodic_vis", &pmesh);
    dc.SetPrefixPath("visit_output/vgt_mfem_periodic");
    dc.RegisterField("vel_h", &vel);
    dc.RegisterField("vgt_fro_norm", &vgt_fro_gf);
    dc.RegisterField("vgt_fro_norm_exact", &vgt_fro_exact_gf);
    dc.RegisterField("grad_error", &grad_err_gf);
    dc.RegisterField("A2_ax", &a2_ax_gf);
    dc.RegisterField("A2_sh", &a2_sh_gf);
    dc.RegisterField("A2_rr", &a2_rr_gf);
    dc.RegisterField("A2_sr", &a2_sr_gf);
    dc.RegisterField("A2_ax_exact", &a2_ax_ex_gf);
    dc.RegisterField("A2_sh_exact", &a2_sh_ex_gf);
    dc.RegisterField("A2_rr_exact", &a2_rr_ex_gf);
    dc.RegisterField("A2_sr_exact", &a2_sr_ex_gf);
    dc.RegisterField("delta_norm", &delta_norm_gf);
    dc.SetCycle(0);
    dc.SetTime(0.0);
    dc.Save();

    if (rank == 0) {
        const double tol_part = 1e-11;
        const bool eig_ok = (g_ep < tol_part);
        const bool sch_ok = (g_sp < tol_part);

        std::cout << std::scientific << std::setprecision(4);

        hline('-');
        std::cout << "  TIER 1  Gradient L2 error  (all " << (g_rg + g_sk) << " quadrature points)\n";
        hline('-');
        std::cout << "  ||A_h - A_exact||_L2  =  " << std::sqrt(g_grad) << "\n";
        std::cout << "  RMS entry-wise error   =  " << std::sqrt(g_grad / (g_tvol * 9.0)) << "\n\n";

        hline('-');
        std::cout << "  TIER 2  Partition identity max residual  (all points)\n";
        hline('-');
        std::cout << "  EIG   [LAPACK] max |||A_h||^2 - sum A^2_k|  =  " << g_ep
                  << "  " << (eig_ok ? "PASS" : "FAIL") << "\n";
        std::cout << "  Schur [LAPACK] max |||A_h||^2 - sum A^2_k|  =  " << g_sp
                  << "  " << (sch_ok ? "PASS" : "FAIL") << "\n\n";

        hline('-');
        std::cout << "  Eigenvalue degeneracy check\n";
        hline('-');
        std::cout << "  min |Delta_norm(A_exact)|  =  " << g_dnmin
                  << "  (threshold 1e-10: "
                  << (g_dnmin > 1e-10 ? "no degenerate points"
                                       : "degenerate-set surface present (expected for periodic)")
                  << ")\n";
        std::cout << "  max |Delta_norm(A_exact)|  =  " << g_dnmax << "\n";
        std::cout << "  Regular pts  " << g_rg << " / " << (g_rg + g_sk)
                  << "   Skipped  " << g_sk
                  << "   (skip fraction = " << std::fixed << std::setprecision(4)
                  << double(g_sk) / double(g_rg + g_sk) * 100.0 << " %)\n\n";

        std::cout << std::scientific << std::setprecision(4);

        hline('-');
        std::cout << "  TIER 3  Component RMS errors vs exact  (" << g_rg << " regular points)\n";
        hline('-');

        const int w0 = 12, w1 = 16, w2 = 18;
        std::cout << "\n  "
                  << std::left << std::setw(w0) << "Component"
                  << std::right << std::setw(w1) << "Mean A2_exact"
                  << std::setw(w2) << "E_rms (EIG/L)"
                  << std::setw(w2) << "rel err"
                  << std::setw(w2) << "E_rms (Schur/L)"
                  << std::setw(w2) << "rel err"
                  << "\n  " << std::string(w0 + w1 + 4 * w2, '-') << "\n";

        auto row = [&](const char* name, double xsum,
                       double eig_e2, double sch_e2) {
            const double mean = xsum / rv;
            const double e_rms = rms(eig_e2);
            const double s_rms = rms(sch_e2);
            const double e_rel = (mean > 0.0) ? e_rms / mean : 0.0;
            const double s_rel = (mean > 0.0) ? s_rms / mean : 0.0;
            std::cout << "  " << std::left << std::setw(w0) << name
                      << std::right
                      << std::setw(w1) << mean
                      << std::setw(w2) << e_rms
                      << std::setw(w2) << e_rel
                      << std::setw(w2) << s_rms
                      << std::setw(w2) << s_rel
                      << "\n";
        };

        row("A2_ax", g_xax, g_ea2, g_sa2);
        row("A2_sh", g_xsh, g_es2, g_ss2);
        row("A2_rr", g_xrr, g_er2, g_sr2);
        row("A2_sr", g_xsr, g_ew2, g_sw2);

        std::cout << "\n  E_rms   = sqrt( sum_q w_q (A2_k,h - A2_k,exact)^2 / vol_regular )\n";
        std::cout << "  rel err = E_rms / mean(A2_k,exact)\n";
        std::cout << "  note    = compare against vgt_mfem_periodic_eigen for the Eigen backend\n\n";

        std::cout << "  VisIt output saved:\n";
        std::cout << "    File > Open > "
                  << "visit_output/vgt_mfem_periodic/"
                  << "vgt_mfem_periodic_vis_000000.mfem_root\n\n";

        hline('=');
        std::cout << "  " << ((eig_ok && sch_ok) ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << "\n";
        hline('=');
    }

    return (g_ep < 1e-11 && g_sp < 1e-11) ? 0 : 1;
}
