/*
 * vgt_mfem_periodic_convergence.cpp
 *
 * Convergence study: project the periodic ABC velocity field
 *
 *   u_1 = a sin(2pi z) + c cos(2pi y)
 *   u_2 = b sin(2pi x) + a cos(2pi z)
 *   u_3 = c sin(2pi y) + b cos(2pi x)
 *
 * with a = 1, b = sqrt(2), c = sqrt(3), onto H1 FE spaces of order p = 1, 2, 4
 * on uniform Cartesian hex meshes of [0,1]^3 made fully periodic, extract the
 * VGT via GetVectorGradient, and measure convergence of the Kronberg-Hoffman
 * decomposition against the analytically known exact VGT.
 *
 * Three-tier diagnostics (identical to vgt_mfem_convergence.cpp):
 *   Tier 1 -- gradient L2 error:           all quadrature points
 *   Tier 2 -- partition identity residual: all quadrature points (~eps_mach)
 *   Tier 3 -- component RMS error:         regular points only
 *
 * The degenerate-eigenvalue set of A_exact is a codim-1 surface for this
 * field (it can't be empty for a periodic divergence-free field), so a small
 * fraction of points is excluded from Tier 3.  Skip counts are reported.
 *
 * Output: two tables per polynomial order
 *   Table A -- EIG (Rortex/Liutex) method, LAPACK backend
 *   Table B -- Schur method, LAPACK backend
 *
 * Build:
 *   make MFEM_CXX=/usr/local/bin/mpicxx vgt_mfem_periodic_convergence
 *
 * Run (serial, one MPI rank):
 *   ./vgt_mfem_periodic_convergence
 *
 * Run (parallel):
 *   mpirun -n 4 ./vgt_mfem_periodic_convergence
 */

#include "mfem.hpp"
#include "vgt_lapack.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>


// ── Constants ─────────────────────────────────────────────────────────────────
static constexpr double PI = M_PI;
static const     double ABC_A = 1.0;
static const     double ABC_B = std::sqrt(2.0);
static const     double ABC_C = std::sqrt(3.0);

// ── Periodic ABC velocity field ──────────────────────────────────────────────
static void u_abc_func(const mfem::Vector& x, mfem::Vector& v)
{
    const double cx = std::cos(2*PI*x[0]);
    const double sx = std::sin(2*PI*x[0]);
    const double cy = std::cos(2*PI*x[1]);
    const double sy = std::sin(2*PI*x[1]);
    const double cz = std::cos(2*PI*x[2]);
    const double sz = std::sin(2*PI*x[2]);

    v[0] = ABC_A * sz + ABC_C * cy;
    v[1] = ABC_B * sx + ABC_A * cz;
    v[2] = ABC_C * sy + ABC_B * cx;
}

// ── Analytical VGT, column-major d[i+3j] = du_i/dx_j ─────────────────────────
static vgt_lapack::Mat3L A_exact_at(const mfem::Vector& x)
{
    const double cx = std::cos(2*PI*x[0]);
    const double sx = std::sin(2*PI*x[0]);
    const double cy = std::cos(2*PI*x[1]);
    const double sy = std::sin(2*PI*x[1]);
    const double cz = std::cos(2*PI*x[2]);
    const double sz = std::sin(2*PI*x[2]);
    const double k = 2*PI;

    vgt_lapack::Mat3L M;
    M(0,0) = 0.0;             M(0,1) = -ABC_C * k * sy; M(0,2) =  ABC_A * k * cz;
    M(1,0) =  ABC_B * k * cx; M(1,1) = 0.0;             M(1,2) = -ABC_A * k * sz;
    M(2,0) = -ABC_B * k * sx; M(2,1) =  ABC_C * k * cy; M(2,2) = 0.0;
    return M;
}

// ── Cubic discriminant (normalized) ──────────────────────────────────────────
static double disc_norm(const vgt_lapack::Mat3L& A)
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

// ── Build a fully periodic N x N x N hex mesh on [0,1]^3 ─────────────────────
static mfem::Mesh MakePeriodicCartesian3D(int N)
{
    mfem::Mesh base = mfem::Mesh::MakeCartesian3D(N, N, N, mfem::Element::HEXAHEDRON,
                                      1.0, 1.0, 1.0, false);
    mfem::Vector tx({1.0, 0.0, 0.0});
    mfem::Vector ty({0.0, 1.0, 0.0});
    mfem::Vector tz({0.0, 0.0, 1.0});
    std::vector<mfem::Vector> translations = {tx, ty, tz};
    return mfem::Mesh::MakePeriodic(base, base.CreatePeriodicVertexMapping(translations));
}

// ── Per-(order,N) result ──────────────────────────────────────────────────────
struct RunResult {
    int N; double h;
    double E_grad;
    double eig_part_max, sch_part_max;
    double eig_ax, eig_sh, eig_rr, eig_sr;
    double sch_ax, sch_sh, sch_rr, sch_sr;
    double min_Dnorm;
    long long skipped, regular;
};

static RunResult run_case(int order, int N, MPI_Comm comm)
{
    mfem::Mesh serial = MakePeriodicCartesian3D(N);
    mfem::ParMesh pmesh(comm, serial);
    serial.Clear();

    mfem::H1_FECollection fec(order, 3);
    mfem::ParFiniteElementSpace fes(&pmesh, &fec, 3);

    mfem::ParGridFunction vel(&fes);
    mfem::VectorFunctionCoefficient vcoeff(3, u_abc_func);
    vel.ProjectCoefficient(vcoeff);

    const mfem::IntegrationRule& ir = mfem::IntRules.Get(mfem::Geometry::CUBE, 2*order+1);
    const int nqp = ir.GetNPoints();

    double grad_err2   = 0.0;
    double eig_ax2=0,  eig_sh2=0,  eig_rr2=0,  eig_sr2=0;
    double sch_ax2=0,  sch_sh2=0,  sch_rr2=0,  sch_sr2=0;
    double eig_pmax=0, sch_pmax=0;
    double reg_vol=0,  min_dn=1e300;
    long long skipped=0, regular=0;

    std::vector<vgt_lapack::Mat3L> Ah_vec(nqp), Ae_vec(nqp);
    std::vector<double> wts(nqp);
    std::vector<bool>   reg(nqp);

    for (int e = 0; e < pmesh.GetNE(); e++)
    {
        mfem::ElementTransformation* T = pmesh.GetElementTransformation(e);

        for (int q = 0; q < nqp; q++)
        {
            const mfem::IntegrationPoint& ip = ir.IntPoint(q);
            T->SetIntPoint(&ip);

            const double wt = ip.weight * T->Weight();
            wts[q] = wt;

            mfem::Vector xp(3);
            T->Transform(ip, xp);

            mfem::DenseMatrix gv(3, 3);
            vel.GetVectorGradient(*T, gv);
            vgt_lapack::Mat3L Ah;
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
            reg[q] = (std::abs(dn) > 1e-10);
            if (reg[q]) { ++regular; } else { ++skipped; }
        }

        const auto eig_h  = vgt_lapack::part_vgt_batch_eig  (Ah_vec);
        const auto sch_h  = vgt_lapack::part_vgt_batch_schur (Ah_vec);
        const auto eig_ex = vgt_lapack::part_vgt_batch_eig  (Ae_vec);
        const auto sch_ex = vgt_lapack::part_vgt_batch_schur (Ae_vec);

        for (int q = 0; q < nqp; q++)
        {
            const double wt = wts[q];
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
            }
        }
    }

    auto rsum = [&](double v){ double g; MPI_Allreduce(&v,&g,1,MPI_DOUBLE,MPI_SUM,comm); return g; };
    auto rmax = [&](double v){ double g; MPI_Allreduce(&v,&g,1,MPI_DOUBLE,MPI_MAX,comm); return g; };
    auto rmin = [&](double v){ double g; MPI_Allreduce(&v,&g,1,MPI_DOUBLE,MPI_MIN,comm); return g; };
    auto rsuml= [&](long long v){ long long g; MPI_Allreduce(&v,&g,1,MPI_LONG_LONG,MPI_SUM,comm); return g; };

    const double g_grad   = rsum(grad_err2);
    const double g_regvol = rsum(reg_vol);
    const double g_ea2    = rsum(eig_ax2),  g_es2 = rsum(eig_sh2);
    const double g_er2    = rsum(eig_rr2),  g_ew2 = rsum(eig_sr2);
    const double g_sa2    = rsum(sch_ax2),  g_ss2 = rsum(sch_sh2);
    const double g_sr2    = rsum(sch_rr2),  g_sw2 = rsum(sch_sr2);
    const double g_ep     = rmax(eig_pmax), g_sp  = rmax(sch_pmax);
    const double g_dn     = rmin(min_dn);
    const long long g_sk  = rsuml(skipped), g_rg  = rsuml(regular);

    const double vol = std::max(g_regvol, 1e-300);
    auto rms = [&](double s2){ return std::sqrt(s2 / vol); };

    RunResult r;
    r.N            = N;     r.h          = 1.0/N;
    r.E_grad       = std::sqrt(g_grad);
    r.eig_part_max = g_ep;  r.sch_part_max = g_sp;
    r.eig_ax = rms(g_ea2);  r.eig_sh = rms(g_es2);
    r.eig_rr = rms(g_er2);  r.eig_sr = rms(g_ew2);
    r.sch_ax = rms(g_sa2);  r.sch_sh = rms(g_ss2);
    r.sch_rr = rms(g_sr2);  r.sch_sr = rms(g_sw2);
    r.min_Dnorm    = g_dn;
    r.skipped      = g_sk;  r.regular    = g_rg;
    return r;
}

static std::string rate_str(double prev, double curr)
{
    if (prev <= 0.0 || curr <= 0.0 || curr >= prev) return "  —  ";
    std::ostringstream s;
    s << std::fixed << std::setprecision(2) << std::log2(prev/curr);
    return s.str();
}

static void print_table(const char* tag, int order,
                        const std::vector<RunResult>& R, bool schur)
{
    std::cout << "\n" << std::string(129, '-') << "\n";
    std::cout << "  P" << order << "  |  method=" << tag << ", backend=LAPACK\n";
    std::cout << std::string(129, '-') << "\n";
    std::cout << std::setw(5)  << "N"
              << std::setw(10) << "h"
              << std::setw(13) << "E_grad"
              << std::setw(7)  << "rate"
              << std::setw(13) << "E_ax(RMS)"
              << std::setw(13) << "E_sh(RMS)"
              << std::setw(13) << "E_rr(RMS)"
              << std::setw(13) << "E_sr(RMS)"
              << std::setw(12) << "partErr"
              << std::setw(8)  << "skip"
              << "\n";
    std::cout << std::string(129, '-') << "\n";

    std::cout << std::scientific << std::setprecision(3);
    for (int i = 0; i < (int)R.size(); i++) {
        const auto& r = R[i];
        const std::string rt = (i > 0) ? rate_str(R[i-1].E_grad, r.E_grad) : "  —  ";
        const double ax = schur ? r.sch_ax : r.eig_ax;
        const double sh = schur ? r.sch_sh : r.eig_sh;
        const double rr = schur ? r.sch_rr : r.eig_rr;
        const double sr = schur ? r.sch_sr : r.eig_sr;
        const double pm = schur ? r.sch_part_max : r.eig_part_max;
        std::cout << std::setw(5)  << r.N
                  << std::setw(10) << r.h
                  << std::setw(13) << r.E_grad
                  << std::setw(7)  << rt
                  << std::setw(13) << ax
                  << std::setw(13) << sh
                  << std::setw(13) << rr
                  << std::setw(13) << sr
                  << std::setw(12) << pm
                  << std::setw(8)  << r.skipped
                  << "\n";
    }
}

int main(int argc, char* argv[])
{
    mfem::MPI_Session mpi(argc, argv);
    const int rank = mpi.WorldRank();
    const int size = mpi.WorldSize();

    if (rank == 0) {
        std::cout << "==================================================================\n";
        std::cout << "  VGT MFEM Periodic ABC Convergence Study  [MPI x" << size << "]\n";
        std::cout << "  Field: ABC,  a=1, b=sqrt(2), c=sqrt(3), k=2pi\n";
        std::cout << "  Domain: [0,1]^3 with full periodicity (x,y,z)\n";
        std::cout << "  Backend: LAPACK (vgt_lapack.hpp)\n";
        std::cout << "  Methods: EIG (Rortex/Liutex), Schur\n";
        std::cout << "==================================================================\n";
    }

    const std::vector<std::pair<int,std::vector<int>>> runs = {
        {1, {8, 16, 32, 64}},
        {2, {8, 16, 32, 64}},
        {4, {8, 16, 32   }},
    };

    for (const auto& [order, Nvec] : runs)
    {
        if (rank == 0) {
            std::cout << "\n\n==================================================================\n";
            std::cout << "  ORDER  p = " << order << "\n";
            std::cout << "==================================================================\n";
        }

        std::vector<RunResult> results;
        for (int N : Nvec) {
            if (rank == 0)
                std::cout << "  Running N=" << N << " ... " << std::flush;

            RunResult r = run_case(order, N, MPI_COMM_WORLD);
            results.push_back(r);

            if (rank == 0)
                std::cout << "done  (E_grad=" << std::scientific << std::setprecision(2)
                          << r.E_grad << ", skip=" << r.skipped
                          << ", regular=" << r.regular << ")\n";
        }

        if (rank == 0) {
            print_table("EIG (Rortex/Liutex)", order, results, false);
            print_table("Schur",               order, results, true);

            const auto& finest = results.back();
            const double skip_frac = double(finest.skipped) /
                                     double(finest.skipped + finest.regular);
            std::cout << "\n  min |Delta_norm| (p=" << order
                      << ", N=" << finest.N << "): "
                      << std::scientific << std::setprecision(3)
                      << finest.min_Dnorm << "    skip fraction: "
                      << std::fixed << std::setprecision(4)
                      << skip_frac*100.0 << " %\n";
        }
    }

    if (rank == 0)
        std::cout << "\n==================================================================\n"
                     "  Done.\n"
                     "==================================================================\n";

    return 0;
}
