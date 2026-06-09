/*
 * vgt_mfem_convergence.cpp
 *
 * Convergence study: project the modified manufactured velocity field
 *
 *   u_ε(x) = Bx + ε u₀(x),   ε = 0.05,
 *   B = [[1,-10,0],[10,1,0],[0,0,-2]],  eigenvalues 1±10i, -2
 *
 * onto H1 FE spaces of order p = 1, 2, 4 on uniform Cartesian hex meshes,
 * extract the VGT via GetVectorGradient, and measure convergence of the
 * Kronberg–Hoffman decomposition components against the analytically known
 * exact VGT A_ε = B + ε A₀.
 *
 * Three-tier diagnostics:
 *   Tier 1  — gradient L² error:         all quadrature points
 *   Tier 2  — partition identity residual: all quadrature points (~ε_mach)
 *   Tier 3  — component RMS error:        regular points only (|Δ_norm| > 1e-10)
 *
 * Output: two separate tables per polynomial order
 *   Table A — EIG (Rortex/Liutex) pathway
 *   Table B — Schur pathway
 *
 * Build:
 *   make MFEM_CXX=/usr/local/bin/mpicxx vgt_mfem_convergence
 *
 * Run:
 *   mpirun -n 1 ./vgt_mfem_convergence
 *   mpirun -n 4 ./vgt_mfem_convergence
 *
 * Exit: 0 always (inspect tables for rate).
 */

#include "mfem.hpp"
#include "vgt_lapack.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

using namespace mfem;
using namespace vgt_lapack;

// ── Constants ─────────────────────────────────────────────────────────────────
static constexpr double PI  = M_PI;
static constexpr double EPS = 0.05;

// ── Modified manufactured velocity u_ε = Bx + ε u₀ ──────────────────────────
static void u_eps_func(const Vector& x, Vector& v)
{
    v[0] =  x[0] - 10.0*x[1]
           + EPS*(std::sin(2*PI*x[0]) + std::sin(4*PI*x[1]) + std::sin(6*PI*x[2]));
    v[1] = 10.0*x[0] + x[1]
           + EPS*(std::sin(6*PI*x[0]) + std::sin(2*PI*x[1]) + std::sin(4*PI*x[2]));
    v[2] = -2.0*x[2]
           + EPS*(std::sin(4*PI*x[0]) + std::sin(6*PI*x[1]) + std::sin(2*PI*x[2]));
}

// ── Analytical VGT: A_ε = B + ε A₀, column-major d[i+3j] = ∂uᵢ/∂xⱼ ─────────
static Mat3L A_exact_at(const Vector& x)
{
    Mat3L M;
    M(0,0) =  1.0 + EPS * 2*PI * std::cos(2*PI*x[0]);
    M(0,1) = -10.0 + EPS * 4*PI * std::cos(4*PI*x[1]);
    M(0,2) =  0.0  + EPS * 6*PI * std::cos(6*PI*x[2]);
    M(1,0) =  10.0 + EPS * 6*PI * std::cos(6*PI*x[0]);
    M(1,1) =  1.0  + EPS * 2*PI * std::cos(2*PI*x[1]);
    M(1,2) =  0.0  + EPS * 4*PI * std::cos(4*PI*x[2]);
    M(2,0) =  0.0  + EPS * 4*PI * std::cos(4*PI*x[0]);
    M(2,1) =  0.0  + EPS * 6*PI * std::cos(6*PI*x[1]);
    M(2,2) = -2.0  + EPS * 2*PI * std::cos(2*PI*x[2]);
    return M;
}

// ── Cubic discriminant (normalized) ──────────────────────────────────────────
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

// ── Per-(order,N) result ──────────────────────────────────────────────────────
struct RunResult {
    int N; double h;
    // Tier 1
    double E_grad;
    // Tier 2: partition identity max residuals
    double eig_part_max, sch_part_max;
    // Tier 3: component RMS errors (regular points only)
    double eig_ax, eig_sh, eig_rr, eig_sr;
    double sch_ax, sch_sh, sch_rr, sch_sr;
    // diagnostics
    double min_Dnorm;
    long long skipped, regular;
};

// ── Run one (order, N) case ───────────────────────────────────────────────────
static RunResult run_case(int order, int N, MPI_Comm comm)
{
    Mesh serial = Mesh::MakeCartesian3D(N, N, N, Element::HEXAHEDRON);
    ParMesh pmesh(comm, serial);
    serial.Clear();

    H1_FECollection fec(order, 3);
    ParFiniteElementSpace fes(&pmesh, &fec, 3);

    ParGridFunction vel(&fes);
    VectorFunctionCoefficient vcoeff(3, u_eps_func);
    vel.ProjectCoefficient(vcoeff);

    const IntegrationRule& ir = IntRules.Get(Geometry::CUBE, 2*order+1);
    const int nqp = ir.GetNPoints();

    // Local accumulators
    double grad_err2   = 0.0;
    double eig_ax2=0,  eig_sh2=0,  eig_rr2=0,  eig_sr2=0;
    double sch_ax2=0,  sch_sh2=0,  sch_rr2=0,  sch_sr2=0;
    double eig_pmax=0, sch_pmax=0;
    double reg_vol=0,  min_dn=1e300;
    long long skipped=0, regular=0;

    // Per-element workspace
    std::vector<Mat3L> Ah_vec(nqp), Ae_vec(nqp);
    std::vector<double> wts(nqp);
    std::vector<bool>   reg(nqp);

    for (int e = 0; e < pmesh.GetNE(); e++)
    {
        ElementTransformation* T = pmesh.GetElementTransformation(e);

        // ── Collect VGTs for this element ─────────────────────────────────────
        for (int q = 0; q < nqp; q++)
        {
            const IntegrationPoint& ip = ir.IntPoint(q);
            T->SetIntPoint(&ip);

            const double wt = ip.weight * T->Weight();
            wts[q] = wt;

            // Physical coordinates at this quadrature point
            Vector xp(3);
            T->Transform(ip, xp);

            // Approximate VGT from FE projection
            DenseMatrix gv(3, 3);
            vel.GetVectorGradient(*T, gv);
            Mat3L Ah;
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    Ah.d[i + 3*j] = gv(i, j);
            Ah_vec[q] = Ah;

            // Exact VGT
            Ae_vec[q] = A_exact_at(xp);

            // Tier 1: gradient L² error (all pts)
            double d2 = 0.0;
            for (int k = 0; k < 9; k++) {
                double d = Ah.d[k] - Ae_vec[q].d[k];
                d2 += d*d;
            }
            grad_err2 += wt * d2;

            // Discriminant of exact VGT — classify point
            const double dn = disc_norm(Ae_vec[q]);
            min_dn = std::min(min_dn, std::abs(dn));
            reg[q] = (std::abs(dn) > 1e-10);
            if (reg[q]) { ++regular; } else { ++skipped; }
        }

        // ── Batch decompose (nqp VGTs per call) ──────────────────────────────
        const auto eig_h  = part_vgt_batch_eig  (Ah_vec);
        const auto sch_h  = part_vgt_batch_schur (Ah_vec);
        const auto eig_ex = part_vgt_batch_eig  (Ae_vec);
        const auto sch_ex = part_vgt_batch_schur (Ae_vec);

        // ── Accumulate Tier 2 and Tier 3 ─────────────────────────────────────
        for (int q = 0; q < nqp; q++)
        {
            const double wt = wts[q];
            const double An2 = Ah_vec[q].squaredNorm();

            // Tier 2: partition identity residual for Ah (all pts)
            eig_pmax = std::max(eig_pmax,
                std::abs(An2 - eig_h.A2_ax[q] - eig_h.A2_sh[q]
                             - eig_h.A2_rr[q] - eig_h.A2_sr[q]));
            sch_pmax = std::max(sch_pmax,
                std::abs(An2 - sch_h.A2_ax[q] - sch_h.A2_sh[q]
                             - sch_h.A2_rr[q] - sch_h.A2_sr[q]));

            // Tier 3: component RMS errors (regular pts only)
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

    // ── MPI reduce ────────────────────────────────────────────────────────────
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
    r.N          = N;    r.h          = 1.0/N;
    r.E_grad     = std::sqrt(g_grad);
    r.eig_part_max = g_ep; r.sch_part_max = g_sp;
    r.eig_ax = rms(g_ea2); r.eig_sh = rms(g_es2);
    r.eig_rr = rms(g_er2); r.eig_sr = rms(g_ew2);
    r.sch_ax = rms(g_sa2); r.sch_sh = rms(g_ss2);
    r.sch_rr = rms(g_sr2); r.sch_sr = rms(g_sw2);
    r.min_Dnorm  = g_dn;
    r.skipped    = g_sk;  r.regular    = g_rg;
    return r;
}

// ── Table printing ────────────────────────────────────────────────────────────
static std::string rate_str(double prev, double curr)
{
    if (prev <= 0.0 || curr <= 0.0 || curr >= prev) return "  —  ";
    std::ostringstream s;
    s << std::fixed << std::setprecision(2) << std::log2(prev/curr);
    return s.str();
}

static void print_eig_table(int order, const std::vector<RunResult>& R)
{
    std::cout << "\n─────────────────────────────────────────────────────────────────"
                 "──────────────────────────────────────────────────────────────────\n";
    std::cout << "  P" << order << "  |  EIG (Rortex/Liutex) pathway\n";
    std::cout << "─────────────────────────────────────────────────────────────────"
                 "──────────────────────────────────────────────────────────────────\n";
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
    std::cout << std::string(111, '-') << "\n";

    std::cout << std::scientific << std::setprecision(3);
    for (int i = 0; i < (int)R.size(); i++) {
        const auto& r = R[i];
        const std::string rt = (i > 0) ? rate_str(R[i-1].E_grad, r.E_grad) : "  —  ";
        std::cout << std::setw(5)  << r.N
                  << std::setw(10) << r.h
                  << std::setw(13) << r.E_grad
                  << std::setw(7)  << rt
                  << std::setw(13) << r.eig_ax
                  << std::setw(13) << r.eig_sh
                  << std::setw(13) << r.eig_rr
                  << std::setw(13) << r.eig_sr
                  << std::setw(12) << r.eig_part_max
                  << std::setw(8)  << r.skipped
                  << "\n";
    }
}

static void print_sch_table(int order, const std::vector<RunResult>& R)
{
    std::cout << "\n─────────────────────────────────────────────────────────────────"
                 "──────────────────────────────────────────────────────────────────\n";
    std::cout << "  P" << order << "  |  Schur pathway\n";
    std::cout << "─────────────────────────────────────────────────────────────────"
                 "──────────────────────────────────────────────────────────────────\n";
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
    std::cout << std::string(111, '-') << "\n";

    std::cout << std::scientific << std::setprecision(3);
    for (int i = 0; i < (int)R.size(); i++) {
        const auto& r = R[i];
        const std::string rt = (i > 0) ? rate_str(R[i-1].E_grad, r.E_grad) : "  —  ";
        std::cout << std::setw(5)  << r.N
                  << std::setw(10) << r.h
                  << std::setw(13) << r.E_grad
                  << std::setw(7)  << rt
                  << std::setw(13) << r.sch_ax
                  << std::setw(13) << r.sch_sh
                  << std::setw(13) << r.sch_rr
                  << std::setw(13) << r.sch_sr
                  << std::setw(12) << r.sch_part_max
                  << std::setw(8)  << r.skipped
                  << "\n";
    }
}

// ── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    mfem::MPI_Session mpi(argc, argv);
    const int rank = mpi.WorldRank();
    const int size = mpi.WorldSize();

    if (rank == 0) {
        std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║  VGT MFEM Convergence Study  [MPI ×" << std::setw(2) << size << "]                      ║\n";
        std::cout << "║  Field: u_ε = Bx + ε u₀,  ε = 0.05,  B eigs: 1±10i, -2    ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    }

    // ── Orders and mesh sequences ─────────────────────────────────────────────
    // P1: 4 levels,  P2: 4 levels,  P4: 3 levels
    const std::vector<std::pair<int,std::vector<int>>> runs = {
        {1, {8, 16, 32, 64}},
        {2, {8, 16, 32, 64}},
        {4, {8, 16, 32   }},
    };

    for (const auto& [order, Nvec] : runs)
    {
        if (rank == 0) {
            std::cout << "\n\n══════════════════════════════════════════════════════════════════\n";
            std::cout << "  ORDER  p = " << order << "\n";
            std::cout << "══════════════════════════════════════════════════════════════════\n";
        }

        std::vector<RunResult> results;
        for (int N : Nvec) {
            if (rank == 0)
                std::cout << "  Running N=" << N << " ... " << std::flush;

            RunResult r = run_case(order, N, MPI_COMM_WORLD);
            results.push_back(r);

            if (rank == 0)
                std::cout << "done  (E_grad=" << std::scientific << std::setprecision(2)
                          << r.E_grad << ", skip=" << r.skipped << ")\n";
        }

        if (rank == 0) {
            print_eig_table(order, results);
            print_sch_table(order, results);

            // Summary line: min discriminant across finest mesh
            const auto& finest = results.back();
            std::cout << "\n  min |Δ_norm| (p=" << order
                      << ", N=" << finest.N << "): "
                      << std::scientific << std::setprecision(3)
                      << finest.min_Dnorm << "  (> 0 → no degenerate points)\n";
        }
    }

    if (rank == 0)
        std::cout << "\n══════════════════════════════════════════════════════════════════\n"
                     "  Done.\n"
                     "══════════════════════════════════════════════════════════════════\n";

    return 0;
}
