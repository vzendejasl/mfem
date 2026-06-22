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
 *     empty.  The cubic-discriminant filter handles it.
 *
 * Mesh: N x N x N hex on [0,1]^3, made fully periodic via
 *       Mesh::MakePeriodic with axis-aligned translations.
 *
 * Diagnostics (mirror vgt_mfem_example.cpp exactly):
 *   Tier 1 -- gradient L^2 error:        all quadrature points
 *   Tier 2 -- partition identity max:    all quadrature points (~eps_mach)
 *   Tier 3 -- component RMS error:       regular points only
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
 *
 * Output:
 *   visit_output/vgt_mfem_periodic/vgt_mfem_periodic_vis_000000.mfem_root
 *   <- open this file in VisIt
 *
 * Exit: 0 = partition identity holds for both pathways, 1 = failure.
 */

#include "mfem.hpp"
#include "vgt_lapack.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace mfem;
using namespace vgt_lapack;

// ── Constants ─────────────────────────────────────────────────────────────────
static constexpr double PI = M_PI;
static const     double ABC_A = 1.0;
static const     double ABC_B = std::sqrt(2.0);
static const     double ABC_C = std::sqrt(3.0);

// ── Periodic ABC velocity field ──────────────────────────────────────────────
static void u_abc_func(const Vector& x, Vector& v)
{
    const double sx = std::sin(2*PI*x[0]), cx = std::cos(2*PI*x[0]);
    const double sy = std::sin(2*PI*x[1]), cy = std::cos(2*PI*x[1]);
    const double sz = std::sin(2*PI*x[2]), cz = std::cos(2*PI*x[2]);

    v[0] = ABC_A * sz + ABC_C * cy;
    v[1] = ABC_B * sx + ABC_A * cz;
    v[2] = ABC_C * sy + ABC_B * cx;
}

// ── Analytical VGT A = grad(u), column-major d[i+3j] = du_i/dx_j ─────────────
static Mat3L A_exact_at(const Vector& x)
{
    const double sx = std::sin(2*PI*x[0]), cx = std::cos(2*PI*x[0]);
    const double sy = std::sin(2*PI*x[1]), cy = std::cos(2*PI*x[1]);
    const double sz = std::sin(2*PI*x[2]), cz = std::cos(2*PI*x[2]);
    const double k = 2*PI;

    Mat3L M;
    M(0,0) = 0.0;             M(0,1) = -ABC_C * k * sy; M(0,2) =  ABC_A * k * cz;
    M(1,0) =  ABC_B * k * cx; M(1,1) = 0.0;             M(1,2) = -ABC_A * k * sz;
    M(2,0) = -ABC_B * k * sx; M(2,1) =  ABC_C * k * cy; M(2,2) = 0.0;
    return M;
}

// ── Cubic discriminant (normalized by ||A||_F^6) ─────────────────────────────
static double disc_norm(const Mat3L& A)
{
    const double I1 = A(0,0)+A(1,1)+A(2,2);
    double trA2 = 0.0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            trA2 += A(i,j)*A(j,i);
    const double I2 = 0.5*(I1*I1 - trA2);
    const double I3 = A(0,0)*(A(1,1)*A(2,2)-A(1,2)*A(2,1))
                    - A(0,1)*(A(1,0)*A(2,2)-A(1,2)*A(2,0))
                    + A(0,2)*(A(1,0)*A(2,1)-A(1,1)*A(2,0));
    const double D  = I1*I1*I2*I2 - 4.0*I2*I2*I2
                    - 4.0*I1*I1*I1*I3 - 27.0*I3*I3 + 18.0*I1*I2*I3;
    const double sc = std::max(1.0, std::sqrt(A.squaredNorm()));
    return D / std::pow(sc, 6);
}

// ── Coefficients for VisIt output ────────────────────────────────────────────

class VGTHCoeff : public VectorCoefficient
{
    ParGridFunction& vel;
public:
    explicit VGTHCoeff(ParGridFunction& v) : VectorCoefficient(9), vel(v) {}
    void Eval(Vector& V, ElementTransformation& T,
              const IntegrationPoint& ip) override
    {
        DenseMatrix grad(3, 3);
        vel.GetVectorGradient(T, grad);
        V.SetSize(9);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                V[3*i + j] = grad(i, j);
    }
};

class VGTExactCoeff : public VectorCoefficient
{
public:
    VGTExactCoeff() : VectorCoefficient(9) {}
    void Eval(Vector& V, ElementTransformation& T,
              const IntegrationPoint& ip) override
    {
        Vector xp(3);
        T.Transform(ip, xp);
        const Mat3L Ae = A_exact_at(xp);
        V.SetSize(9);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                V[3*i + j] = Ae(i, j);
    }
};

class GradErrorCoeff : public Coefficient
{
    ParGridFunction& vel;
public:
    explicit GradErrorCoeff(ParGridFunction& v) : vel(v) {}
    double Eval(ElementTransformation& T,
                const IntegrationPoint& ip) override
    {
        DenseMatrix gv(3, 3);
        vel.GetVectorGradient(T, gv);
        Vector xp(3);
        T.Transform(ip, xp);
        const Mat3L Ae = A_exact_at(xp);
        double err2 = 0.0;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
                const double d = gv(i, j) - Ae(i, j);
                err2 += d*d;
            }
        return std::sqrt(err2);
    }
};

class DeltaNormCoeff : public Coefficient
{
public:
    double Eval(ElementTransformation& T,
                const IntegrationPoint& ip) override
    {
        Vector xp(3);
        T.Transform(ip, xp);
        return std::abs(disc_norm(A_exact_at(xp)));
    }
};

class PartCoeff : public Coefficient
{
    ParGridFunction& vel;
    int comp;
public:
    PartCoeff(ParGridFunction& v, int c) : vel(v), comp(c) {}
    double Eval(ElementTransformation& T,
                const IntegrationPoint& ip) override
    {
        DenseMatrix gv(3, 3);
        vel.GetVectorGradient(T, gv);
        Mat3L Ah;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Ah.d[i + 3*j] = gv(i, j);
        const auto res = part_vgt_batch_schur({Ah});
        switch (comp) {
            case 0: return res.A2_ax[0];
            case 1: return res.A2_sh[0];
            case 2: return res.A2_rr[0];
            default: return res.A2_sr[0];
        }
    }
};

class PartExactCoeff : public Coefficient
{
    int comp;
public:
    explicit PartExactCoeff(int c) : comp(c) {}
    double Eval(ElementTransformation& T,
                const IntegrationPoint& ip) override
    {
        Vector xp(3);
        T.Transform(ip, xp);
        const Mat3L Ae = A_exact_at(xp);
        const auto res = part_vgt_batch_schur({Ae});
        switch (comp) {
            case 0: return res.A2_ax[0];
            case 1: return res.A2_sh[0];
            case 2: return res.A2_rr[0];
            default: return res.A2_sr[0];
        }
    }
};

// ── Build a fully periodic N x N x N hex mesh on [0,1]^3 ─────────────────────
static Mesh MakePeriodicCartesian3D(int N)
{
    Mesh base = Mesh::MakeCartesian3D(N, N, N, Element::HEXAHEDRON,
                                      1.0, 1.0, 1.0, false);
    Vector tx({1.0, 0.0, 0.0});
    Vector ty({0.0, 1.0, 0.0});
    Vector tz({0.0, 0.0, 1.0});
    std::vector<Vector> translations = {tx, ty, tz};
    return Mesh::MakePeriodic(base, base.CreatePeriodicVertexMapping(translations));
}

// ── ASCII separator helpers ───────────────────────────────────────────────────
static void hline(char c = '-', int w = 70)
{ std::cout << std::string(w, c) << "\n"; }

// ── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    mfem::MPI_Session mpi(argc, argv);
    const int rank = mpi.WorldRank();
    const int size = mpi.WorldSize();

    int order = 2;
    int N     = 16;

    OptionsParser args(argc, argv);
    args.AddOption(&order, "-o", "--order",  "Polynomial order (1, 2, 4, ...).");
    args.AddOption(&N,     "-n", "--mesh-n", "Elements per side (N x N x N mesh).");
    args.Parse();
    if (!args.Good()) {
        if (rank == 0) args.PrintUsage(std::cout);
        return 1;
    }

    // ── Periodic mesh ────────────────────────────────────────────────────────
    Mesh serial = MakePeriodicCartesian3D(N);
    ParMesh pmesh(MPI_COMM_WORLD, serial);
    serial.Clear();

    H1_FECollection fec(order, 3);
    ParFiniteElementSpace fes(&pmesh, &fec, 3);
    const HYPRE_BigInt ndofs = fes.GlobalTrueVSize();

    ParGridFunction vel(&fes);
    VectorFunctionCoefficient vcoeff(3, u_abc_func);
    vel.ProjectCoefficient(vcoeff);

    const IntegrationRule& ir = IntRules.Get(Geometry::CUBE, 2*order+1);
    const int nqp = ir.GetNPoints();

    int N_elem_local = pmesh.GetNE();
    int N_elem_global = 0;
    MPI_Allreduce(&N_elem_local, &N_elem_global, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    const long long N_qpts_global = (long long)N_elem_global * nqp;

    if (rank == 0) {
        hline('=');
        std::cout << "  VGT MFEM Periodic ABC Example  [MPI x" << size << "]\n";
        hline('=');
        std::cout << "  Field   : ABC, a=1, b=sqrt(2), c=sqrt(3), k=2pi\n";
        std::cout << "  Domain  : [0,1]^3 with full periodicity (x,y,z)\n";
        std::cout << "  Mesh    : " << N << "^3 hex"
                  << "  (h = " << std::fixed << std::setprecision(6) << 1.0/N << ")\n";
        std::cout << "  Order   : P" << order << "\n";
        std::cout << "  DOFs    : " << ndofs << "  (global, all components)\n";
        std::cout << "  Qpts    : " << nqp << " per element  ("
                  << N_qpts_global << " total)\n";
        hline();
    }

    // ── Quadrature loop ──────────────────────────────────────────────────────
    double grad_err2   = 0.0;
    double eig_ax2=0,  eig_sh2=0,  eig_rr2=0,  eig_sr2=0;
    double sch_ax2=0,  sch_sh2=0,  sch_rr2=0,  sch_sr2=0;
    double ex_ax_sum=0,ex_sh_sum=0,ex_rr_sum=0,ex_sr_sum=0;
    double eig_pmax=0, sch_pmax=0;
    double reg_vol=0,  total_vol=0;
    double min_dn=1e300, max_dn=-1e300;
    long long skipped=0, regular=0;

    std::vector<Mat3L> Ah_vec(nqp), Ae_vec(nqp);
    std::vector<double> wts(nqp);
    std::vector<bool>   reg(nqp);

    for (int e = 0; e < pmesh.GetNE(); e++)
    {
        ElementTransformation* T = pmesh.GetElementTransformation(e);

        for (int q = 0; q < nqp; q++)
        {
            const IntegrationPoint& ip = ir.IntPoint(q);
            T->SetIntPoint(&ip);
            const double wt = ip.weight * T->Weight();
            wts[q]     = wt;
            total_vol += wt;

            Vector xp(3);
            T->Transform(ip, xp);

            DenseMatrix gv(3, 3);
            vel.GetVectorGradient(*T, gv);
            Mat3L Ah;
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    Ah.d[i + 3*j] = gv(i, j);
            Ah_vec[q] = Ah;
            Ae_vec[q] = A_exact_at(xp);

            double d2 = 0.0;
            for (int k = 0; k < 9; k++) {
                double d = Ah.d[k] - Ae_vec[q].d[k];
                d2 += d*d;
            }
            grad_err2 += wt * d2;

            const double dn = disc_norm(Ae_vec[q]);
            min_dn = std::min(min_dn, std::abs(dn));
            max_dn = std::max(max_dn, std::abs(dn));
            reg[q] = (std::abs(dn) > 1e-10);
            if (reg[q]) ++regular; else ++skipped;
        }

        const auto eig_h  = part_vgt_batch_eig  (Ah_vec);
        const auto sch_h  = part_vgt_batch_schur (Ah_vec);
        const auto eig_ex = part_vgt_batch_eig  (Ae_vec);
        const auto sch_ex = part_vgt_batch_schur (Ae_vec);

        for (int q = 0; q < nqp; q++)
        {
            const double wt  = wts[q];
            const double An2 = Ah_vec[q].squaredNorm();

            eig_pmax = std::max(eig_pmax,
                std::abs(An2 - eig_h.A2_ax[q] - eig_h.A2_sh[q]
                             - eig_h.A2_rr[q] - eig_h.A2_sr[q]));
            sch_pmax = std::max(sch_pmax,
                std::abs(An2 - sch_h.A2_ax[q] - sch_h.A2_sh[q]
                             - sch_h.A2_rr[q] - sch_h.A2_sr[q]));

            if (reg[q]) {
                reg_vol += wt;
                auto sq = [](double v){ return v*v; };

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

    // ── MPI reduce ───────────────────────────────────────────────────────────
    auto rsum  = [&](double v)    { double g;    MPI_Allreduce(&v,&g,1,MPI_DOUBLE,   MPI_SUM,MPI_COMM_WORLD); return g; };
    auto rmax  = [&](double v)    { double g;    MPI_Allreduce(&v,&g,1,MPI_DOUBLE,   MPI_MAX,MPI_COMM_WORLD); return g; };
    auto rmin  = [&](double v)    { double g;    MPI_Allreduce(&v,&g,1,MPI_DOUBLE,   MPI_MIN,MPI_COMM_WORLD); return g; };
    auto rsuml = [&](long long v) { long long g; MPI_Allreduce(&v,&g,1,MPI_LONG_LONG,MPI_SUM,MPI_COMM_WORLD); return g; };

    const double g_grad   = rsum(grad_err2);
    const double g_tvol   = rsum(total_vol);
    const double g_rvol   = rsum(reg_vol);
    const double g_ea2    = rsum(eig_ax2),   g_es2 = rsum(eig_sh2);
    const double g_er2    = rsum(eig_rr2),   g_ew2 = rsum(eig_sr2);
    const double g_sa2    = rsum(sch_ax2),   g_ss2 = rsum(sch_sh2);
    const double g_sr2    = rsum(sch_rr2),   g_sw2 = rsum(sch_sr2);
    const double g_xax    = rsum(ex_ax_sum), g_xsh = rsum(ex_sh_sum);
    const double g_xrr    = rsum(ex_rr_sum), g_xsr = rsum(ex_sr_sum);
    const double g_ep     = rmax(eig_pmax),  g_sp  = rmax(sch_pmax);
    const double g_dnmin  = rmin(min_dn),    g_dnmax = rmax(max_dn);
    const long long g_sk  = rsuml(skipped),  g_rg  = rsuml(regular);

    const double rv  = std::max(g_rvol, 1e-300);
    auto rms = [&](double s2){ return std::sqrt(s2 / rv); };

    // ── Save VisIt data collection automatically ────────────────────────────
    L2_FECollection l2_fec(order, 3);
    ParFiniteElementSpace l2s_fes(&pmesh, &l2_fec);
    ParFiniteElementSpace l2t_fes(&pmesh, &l2_fec, 9, Ordering::byVDIM);

    ParGridFunction vgt_h_gf(&l2t_fes);
    ParGridFunction vgt_exact_gf(&l2t_fes);
    ParGridFunction grad_err_gf(&l2s_fes);
    ParGridFunction a2_ax_gf(&l2s_fes);
    ParGridFunction a2_sh_gf(&l2s_fes);
    ParGridFunction a2_rr_gf(&l2s_fes);
    ParGridFunction a2_sr_gf(&l2s_fes);
    ParGridFunction a2_ax_ex_gf(&l2s_fes);
    ParGridFunction a2_sh_ex_gf(&l2s_fes);
    ParGridFunction a2_rr_ex_gf(&l2s_fes);
    ParGridFunction a2_sr_ex_gf(&l2s_fes);
    ParGridFunction delta_norm_gf(&l2s_fes);

    VGTHCoeff vgt_h_coeff(vel);
    VGTExactCoeff vgt_exact_coeff;
    GradErrorCoeff grad_err_coeff(vel);
    DeltaNormCoeff delta_coeff;
    PartCoeff ax_coeff(vel, 0), sh_coeff(vel, 1), rr_coeff(vel, 2), sr_coeff(vel, 3);
    PartExactCoeff ax_ex_coeff(0), sh_ex_coeff(1), rr_ex_coeff(2), sr_ex_coeff(3);

    vgt_h_gf.ProjectCoefficient(vgt_h_coeff);
    vgt_exact_gf.ProjectCoefficient(vgt_exact_coeff);
    grad_err_gf.ProjectCoefficient(grad_err_coeff);
    delta_norm_gf.ProjectCoefficient(delta_coeff);
    a2_ax_gf.ProjectCoefficient(ax_coeff);
    a2_sh_gf.ProjectCoefficient(sh_coeff);
    a2_rr_gf.ProjectCoefficient(rr_coeff);
    a2_sr_gf.ProjectCoefficient(sr_coeff);
    a2_ax_ex_gf.ProjectCoefficient(ax_ex_coeff);
    a2_sh_ex_gf.ProjectCoefficient(sh_ex_coeff);
    a2_rr_ex_gf.ProjectCoefficient(rr_ex_coeff);
    a2_sr_ex_gf.ProjectCoefficient(sr_ex_coeff);

    // The raw tensors are exported as 9-component L2 fields (vgt_h, vgt_exact),
    // one component per entry A(i,j) in by-VDIM order. The derived A2_* and
    // grad_error outputs are scalar fields for direct plotting in VisIt.
    VisItDataCollection dc("vgt_mfem_periodic_vis", &pmesh);
    dc.SetPrefixPath("visit_output/vgt_mfem_periodic");
    dc.RegisterField("vel_h", &vel);
    dc.RegisterField("vgt_h", &vgt_h_gf);
    dc.RegisterField("vgt_exact", &vgt_exact_gf);
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

        // Tier 1
        hline('-');
        std::cout << "  TIER 1  Gradient L2 error  (all " << (g_rg+g_sk) << " quadrature points)\n";
        hline('-');
        std::cout << "  ||A_h - A_exact||_L2  =  " << std::sqrt(g_grad) << "\n";
        std::cout << "  RMS entry-wise error   =  " << std::sqrt(g_grad / (g_tvol * 9.0)) << "\n\n";

        // Tier 2
        hline('-');
        std::cout << "  TIER 2  Partition identity max residual  (all points)\n";
        hline('-');
        std::cout << "  EIG   max |||A_h||^2 - sum A^2_k|  =  " << g_ep
                  << "  " << (eig_ok ? "PASS" : "FAIL") << "\n";
        std::cout << "  Schur max |||A_h||^2 - sum A^2_k|  =  " << g_sp
                  << "  " << (sch_ok ? "PASS" : "FAIL") << "\n\n";

        // Degeneracy diagnosis (this is where ABC differs from the Bx+u0 field)
        hline('-');
        std::cout << "  Eigenvalue degeneracy check\n";
        hline('-');
        std::cout << "  min |Delta_norm(A_exact)|  =  " << g_dnmin
                  << "  (threshold 1e-10: "
                  << (g_dnmin > 1e-10 ? "no degenerate points"
                                       : "degenerate-set surface present (expected for periodic)")
                  << ")\n";
        std::cout << "  max |Delta_norm(A_exact)|  =  " << g_dnmax << "\n";
        std::cout << "  Regular pts  " << g_rg << " / " << (g_rg+g_sk)
                  << "   Skipped  " << g_sk
                  << "   (skip fraction = " << std::fixed << std::setprecision(4)
                  << double(g_sk)/double(g_rg+g_sk) * 100.0 << " %)\n\n";

        std::cout << std::scientific << std::setprecision(4);

        // Tier 3
        hline('-');
        std::cout << "  TIER 3  Component RMS errors vs exact  (" << g_rg << " regular points)\n";
        hline('-');

        const int w0=12, w1=16, w2=18;
        std::cout << "\n  "
                  << std::left  << std::setw(w0) << "Component"
                  << std::right << std::setw(w1) << "Mean A2_exact"
                  << std::setw(w2) << "E_rms (EIG)"
                  << std::setw(w2) << "rel err (EIG)"
                  << std::setw(w2) << "E_rms (Schur)"
                  << std::setw(w2) << "rel err (Schur)"
                  << "\n  " << std::string(w0+w1+4*w2, '-') << "\n";

        std::cout << std::right << std::scientific << std::setprecision(4);

        auto row = [&](const char* name, double xsum,
                       double eig_e2, double sch_e2) {
            const double mean  = xsum / rv;
            const double e_rms = rms(eig_e2);
            const double s_rms = rms(sch_e2);
            const double e_rel = (mean > 0) ? e_rms / mean : 0.0;
            const double s_rel = (mean > 0) ? s_rms / mean : 0.0;
            std::cout << "  " << std::left  << std::setw(w0) << name
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
        std::cout << "  rel err = E_rms / mean(A2_k,exact)\n\n";

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
