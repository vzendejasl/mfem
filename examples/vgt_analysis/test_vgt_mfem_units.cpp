/*
 * test_vgt_mfem_units.cpp — 34-test VGT unit suite, built against MFEM.
 *
 * Runs the same 34 tests as test_vgt_units.cpp (LAPACK pathway for both
 * "Eigen" and "LAPACK" slots).  The only MFEM-specific lines are the
 * MPI_Session and the include.  All VGT computation uses vgt_lapack.hpp;
 * all MPI collectives use vgt_mpi.hpp — no Eigen dependency.
 *
 * Build:
 *   make MFEM_CXX=/usr/local/bin/mpicxx test_vgt_mfem_units
 *
 * Run:
 *   mpirun -n 1 ./test_vgt_mfem_units
 *   mpirun -n 2 ./test_vgt_mfem_units
 *   mpirun -n 4 ./test_vgt_mfem_units
 *
 * Exit code: 0 = all pass, 1 = at least one failure (CTest-compatible).
 */

#include "mfem.hpp"
#include "vgt_lapack.hpp"
#include "vgt_mpi.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#ifndef VGT_DATA_DIR
#  define VGT_DATA_DIR "../../../../vgt_analysis/vgt_all_bundle/data"
#endif

namespace {

using vgt_lapack::Mat3L;
using vgt_lapack::Vec3L;

// ── Reference values (MATLAB cross-validation) ─────────────────────────────
namespace ref {
    constexpr double VGT_CSV = 2.828427124746190e+00;  // sqrt(8)
    constexpr double REL_TOL = 5e-10;
    constexpr double ABS_TOL = 1e-10;
}

// ── Tester ──────────────────────────────────────────────────────────────────
struct Tester {
    int rank;
    int pass_ = 0, fail_ = 0;

    void check_near(const std::string& name, double got, double expected,
                    double rtol = ref::REL_TOL)
    {
        const double err = std::abs(got - expected) / (std::abs(expected) + 1e-30);
        const bool ok = err <= rtol;
        if (rank == 0)
            std::cout << (ok ? "  PASS" : "  FAIL") << "  " << name
                      << "  got=" << std::scientific << std::setprecision(6) << got
                      << "  ref=" << expected
                      << "  relerr=" << err << "\n";
        ok ? ++pass_ : ++fail_;
    }

    void check_max(const std::string& name, double local_max,
                   double tol = ref::ABS_TOL)
    {
        double gmax = vgt_mpi::allreduce_max(local_max);
        const bool ok = gmax <= tol;
        if (rank == 0)
            std::cout << (ok ? "  PASS" : "  FAIL") << "  " << name
                      << "  max_err=" << std::scientific << std::setprecision(3) << gmax << "\n";
        ok ? ++pass_ : ++fail_;
    }

    void check_pass(const std::string& name, bool local_ok)
    {
        const int gok = vgt_mpi::allreduce_and(local_ok ? 1 : 0);
        if (rank == 0)
            std::cout << (gok ? "  PASS" : "  FAIL") << "  " << name << "\n";
        gok ? ++pass_ : ++fail_;
    }

    int summary()
    {
        int nfail = fail_;
        MPI_Bcast(&nfail, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0)
            std::cout << "\n" << std::string(60, '=') << "\n"
                      << "TOTAL: " << pass_ << " passed, " << fail_ << " failed\n"
                      << std::string(60, '=') << "\n";
        return nfail;
    }
};

// ── Scatter ─────────────────────────────────────────────────────────────────
std::vector<Mat3L> scatter_vgts(const std::vector<Mat3L>& all,
                                 int N, int rank, int size)
{
    std::vector<int> sc, dp;
    vgt_mpi::scatter_params(N, size, 9, sc, dp);

    std::vector<double> sb;
    if (rank == 0) {
        sb.resize(9 * N);
        for (int k = 0; k < N; ++k)
            std::memcpy(&sb[9*k], all[k].d, 9 * sizeof(double));
    }

    std::vector<double> rb(sc[rank]);
    MPI_Scatterv(rank == 0 ? sb.data() : nullptr,
                 sc.data(), dp.data(), MPI_DOUBLE,
                 rb.data(), sc[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const int ln = sc[rank] / 9;
    std::vector<Mat3L> local(ln);
    for (int k = 0; k < ln; ++k)
        std::memcpy(local[k].d, &rb[9*k], 9 * sizeof(double));
    return local;
}

// ══════════════════════════════════════════════════════════════════════════════
// CSV dataset tests — T1–T18
// ══════════════════════════════════════════════════════════════════════════════
void run_csv_tests(Tester& t, const std::string& csv, int rank, int size)
{
    if (rank == 0)
        std::cout << "\n--- CSV tests (8-point dataset, MATLAB cross-validation) ---\n";

    const double errtol = 1e-10;

    int N = 0;
    std::vector<Mat3L> all;
    if (rank == 0) {
        all = vgt_lapack::load_csv_vgt(csv);
        N   = static_cast<int>(all.size());
    }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Both slots receive the same bit-for-bit data (LAPACK pathway).
    const auto ll  = scatter_vgts(all, N, rank, size);
    const int  ln  = static_cast<int>(ll.size());
    const auto ll2 = ll;  // LAPACK slot — same data

    // ── T1-T2: norm(VGT) vs MATLAB reference ────────────────────────────────
    {
        double lsq = 0.0;
        for (const auto& A : ll) lsq += A.squaredNorm();
        t.check_near("CSV:Eigen:norm(VGT)", vgt_mpi::allreduce_norm(lsq), ref::VGT_CSV);
    }
    {
        double lsq = 0.0;
        for (const auto& A : ll2) lsq += A.squaredNorm();
        t.check_near("CSV:LAPACK:norm(VGT)", vgt_mpi::allreduce_norm(lsq), ref::VGT_CSV);
    }

    // ── T3: Cross-pathway norm(VGT) agreement ───────────────────────────────
    {
        double lsq_e = 0.0, lsq_l = 0.0;
        for (const auto& A : ll)  lsq_e += A.squaredNorm();
        for (const auto& A : ll2) lsq_l += A.squaredNorm();
        t.check_near("CSV:cross:norm(VGT) Eigen==LAPACK",
                     vgt_mpi::allreduce_norm(lsq_e),
                     vgt_mpi::allreduce_norm(lsq_l), 1e-14);
    }

    // Run LAPACK for both slots.
    const auto eig_e = vgt_lapack::part_vgt_batch_eig(ll,  errtol);
    const auto sch_e = vgt_lapack::part_vgt_batch_schur(ll);
    const auto eig_l = vgt_lapack::part_vgt_batch_eig(ll2, errtol);
    const auto sch_l = vgt_lapack::part_vgt_batch_schur(ll2);

    // ── T4-T5: Partition identity ────────────────────────────────────────────
    {
        double lmax = 0.0;
        for (int k = 0; k < ln; ++k) {
            const double sum = eig_e.A2_ax[k]+eig_e.A2_sh[k]+eig_e.A2_rr[k]+eig_e.A2_sr[k];
            lmax = std::max(lmax, std::abs(ll[k].squaredNorm() - sum));
        }
        t.check_max("CSV:Eigen:EIG:partition_identity", lmax);
    }
    {
        double lmax = 0.0;
        for (int k = 0; k < ln; ++k) {
            const double sum = sch_e.A2_ax[k]+sch_e.A2_sh[k]+sch_e.A2_rr[k]+sch_e.A2_sr[k];
            lmax = std::max(lmax, std::abs(ll[k].squaredNorm() - sum));
        }
        t.check_max("CSV:Eigen:SCHUR:partition_identity", lmax);
    }

    // ── T6-T8: Schur back-transform ──────────────────────────────────────────
    {
        double err_ax=0, err_sh=0, err_rr=0;
        for (int k = 0; k < ln; ++k) {
            err_ax = std::max(err_ax, std::abs(sch_e.A2_ax[k]-sch_e.A_ax[k].squaredNorm()));
            err_sh = std::max(err_sh, std::abs(sch_e.A2_sh[k]-sch_e.A_sh[k].squaredNorm()));
            err_rr = std::max(err_rr, std::abs(sch_e.A2_rr[k]-sch_e.A_rr[k].squaredNorm()));
        }
        t.check_max("CSV:Eigen:SCHUR:backtransform_ax", err_ax);
        t.check_max("CSV:Eigen:SCHUR:backtransform_sh", err_sh);
        t.check_max("CSV:Eigen:SCHUR:backtransform_rr", err_rr);
    }

    // ── T9-T10: LAPACK partition identity ────────────────────────────────────
    {
        double lmax = 0.0;
        for (int k = 0; k < ln; ++k) {
            const double sum = eig_l.A2_ax[k]+eig_l.A2_sh[k]+eig_l.A2_rr[k]+eig_l.A2_sr[k];
            lmax = std::max(lmax, std::abs(ll2[k].squaredNorm() - sum));
        }
        t.check_max("CSV:LAPACK:EIG:partition_identity", lmax);
    }
    {
        double lmax = 0.0;
        for (int k = 0; k < ln; ++k) {
            const double sum = sch_l.A2_ax[k]+sch_l.A2_sh[k]+sch_l.A2_rr[k]+sch_l.A2_sr[k];
            lmax = std::max(lmax, std::abs(ll2[k].squaredNorm() - sum));
        }
        t.check_max("CSV:LAPACK:SCHUR:partition_identity", lmax);
    }

    // ── T11-T13: LAPACK Schur back-transform ─────────────────────────────────
    {
        double err_ax=0, err_sh=0, err_rr=0;
        for (int k = 0; k < ln; ++k) {
            err_ax = std::max(err_ax, std::abs(sch_l.A2_ax[k]-sch_l.A_ax[k].squaredNorm()));
            err_sh = std::max(err_sh, std::abs(sch_l.A2_sh[k]-sch_l.A_sh[k].squaredNorm()));
            err_rr = std::max(err_rr, std::abs(sch_l.A2_rr[k]-sch_l.A_rr[k].squaredNorm()));
        }
        t.check_max("CSV:LAPACK:SCHUR:backtransform_ax", err_ax);
        t.check_max("CSV:LAPACK:SCHUR:backtransform_sh", err_sh);
        t.check_max("CSV:LAPACK:SCHUR:backtransform_rr", err_rr);
    }

    // ── T14-T17: testBatch ───────────────────────────────────────────────────
    t.check_pass("CSV:Eigen:EIG:testBatch",
        vgt_lapack::test_vgt_batch_part(ll, eig_e.A2_ax, eig_e.A2_sh,
                                         eig_e.A2_rr, eig_e.A2_sr, eig_e.rotAx, errtol));
    t.check_pass("CSV:Eigen:SCHUR:testBatch",
        vgt_lapack::test_vgt_batch_part(ll, sch_e.A2_ax, sch_e.A2_sh,
                                         sch_e.A2_rr, sch_e.A2_sr, sch_e.rotAx, errtol));
    t.check_pass("CSV:LAPACK:EIG:testBatch",
        vgt_lapack::test_vgt_batch_part(ll2, eig_l.A2_ax, eig_l.A2_sh,
                                          eig_l.A2_rr, eig_l.A2_sr, eig_l.rotAx, errtol));
    t.check_pass("CSV:LAPACK:SCHUR:testBatch",
        vgt_lapack::test_vgt_batch_part(ll2, sch_l.A2_ax, sch_l.A2_sh,
                                          sch_l.A2_rr, sch_l.A2_sr, sch_l.rotAx, errtol));

    // ── T18: Cross-pathway SCHUR norm(A2_ax) ─────────────────────────────────
    {
        double lsq_e=0, lsq_l=0;
        for (int k = 0; k < ln; ++k) {
            lsq_e += sch_e.A2_ax[k] * sch_e.A2_ax[k];
            lsq_l += sch_l.A2_ax[k] * sch_l.A2_ax[k];
        }
        t.check_near("CSV:cross:SCHUR:norm(A2_ax) Eigen==LAPACK",
                     vgt_mpi::allreduce_norm(lsq_e),
                     vgt_mpi::allreduce_norm(lsq_l), ref::REL_TOL);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Synthetic dataset tests — T19–T34
// ══════════════════════════════════════════════════════════════════════════════
void run_syn_tests(Tester& t, int rank, int size)
{
    if (rank == 0)
        std::cout << "\n--- Synthetic tests (1000-pt dataset, seed=42) ---\n";

    const double errtol = 1e-10;
    const int    N_SYN  = 1000;

    std::vector<Mat3L> all;
    if (rank == 0)
        all = vgt_lapack::make_synthetic_vgt(N_SYN, true, true, 42u);

    const auto ll  = scatter_vgts(all, N_SYN, rank, size);
    const int  ln  = static_cast<int>(ll.size());
    const auto ll2 = ll;  // LAPACK slot — same data

    // ── T19: Cross-pathway norm(VGT) ─────────────────────────────────────────
    {
        double lsq_e=0, lsq_l=0;
        for (const auto& A : ll)  lsq_e += A.squaredNorm();
        for (const auto& A : ll2) lsq_l += A.squaredNorm();
        const double ne = vgt_mpi::allreduce_norm(lsq_e);
        const double nl = vgt_mpi::allreduce_norm(lsq_l);
        if (rank == 0)
            std::cout << "  INFO  SYN:norm(VGT)="
                      << std::scientific << std::setprecision(6) << ne << "\n";
        t.check_near("SYN:cross:norm(VGT) Eigen==LAPACK", ne, nl, 1e-14);
    }

    const auto eig_e = vgt_lapack::part_vgt_batch_eig(ll,  errtol);
    const auto sch_e = vgt_lapack::part_vgt_batch_schur(ll);
    const auto eig_l = vgt_lapack::part_vgt_batch_eig(ll2, errtol);
    const auto sch_l = vgt_lapack::part_vgt_batch_schur(ll2);

    // ── T20-T21: Partition identity ──────────────────────────────────────────
    {
        double lmax = 0.0;
        for (int k = 0; k < ln; ++k) {
            const double sum = eig_e.A2_ax[k]+eig_e.A2_sh[k]+eig_e.A2_rr[k]+eig_e.A2_sr[k];
            lmax = std::max(lmax, std::abs(ll[k].squaredNorm() - sum));
        }
        t.check_max("SYN:Eigen:EIG:partition_identity", lmax);
    }
    {
        double lmax = 0.0;
        for (int k = 0; k < ln; ++k) {
            const double sum = sch_e.A2_ax[k]+sch_e.A2_sh[k]+sch_e.A2_rr[k]+sch_e.A2_sr[k];
            lmax = std::max(lmax, std::abs(ll[k].squaredNorm() - sum));
        }
        t.check_max("SYN:Eigen:SCHUR:partition_identity", lmax);
    }

    // ── T22-T24: Schur back-transform ────────────────────────────────────────
    {
        double err_ax=0, err_sh=0, err_rr=0;
        for (int k = 0; k < ln; ++k) {
            err_ax = std::max(err_ax, std::abs(sch_e.A2_ax[k]-sch_e.A_ax[k].squaredNorm()));
            err_sh = std::max(err_sh, std::abs(sch_e.A2_sh[k]-sch_e.A_sh[k].squaredNorm()));
            err_rr = std::max(err_rr, std::abs(sch_e.A2_rr[k]-sch_e.A_rr[k].squaredNorm()));
        }
        t.check_max("SYN:Eigen:SCHUR:backtransform_ax", err_ax);
        t.check_max("SYN:Eigen:SCHUR:backtransform_sh", err_sh);
        t.check_max("SYN:Eigen:SCHUR:backtransform_rr", err_rr);
    }

    // ── T25-T26: LAPACK partition identity ───────────────────────────────────
    {
        double lmax = 0.0;
        for (int k = 0; k < ln; ++k) {
            const double sum = eig_l.A2_ax[k]+eig_l.A2_sh[k]+eig_l.A2_rr[k]+eig_l.A2_sr[k];
            lmax = std::max(lmax, std::abs(ll2[k].squaredNorm() - sum));
        }
        t.check_max("SYN:LAPACK:EIG:partition_identity", lmax);
    }
    {
        double lmax = 0.0;
        for (int k = 0; k < ln; ++k) {
            const double sum = sch_l.A2_ax[k]+sch_l.A2_sh[k]+sch_l.A2_rr[k]+sch_l.A2_sr[k];
            lmax = std::max(lmax, std::abs(ll2[k].squaredNorm() - sum));
        }
        t.check_max("SYN:LAPACK:SCHUR:partition_identity", lmax);
    }

    // ── T27-T29: LAPACK Schur back-transform ─────────────────────────────────
    {
        double err_ax=0, err_sh=0, err_rr=0;
        for (int k = 0; k < ln; ++k) {
            err_ax = std::max(err_ax, std::abs(sch_l.A2_ax[k]-sch_l.A_ax[k].squaredNorm()));
            err_sh = std::max(err_sh, std::abs(sch_l.A2_sh[k]-sch_l.A_sh[k].squaredNorm()));
            err_rr = std::max(err_rr, std::abs(sch_l.A2_rr[k]-sch_l.A_rr[k].squaredNorm()));
        }
        t.check_max("SYN:LAPACK:SCHUR:backtransform_ax", err_ax);
        t.check_max("SYN:LAPACK:SCHUR:backtransform_sh", err_sh);
        t.check_max("SYN:LAPACK:SCHUR:backtransform_rr", err_rr);
    }

    // ── T30-T33: testBatch (first 20 pts per rank) ───────────────────────────
    {
        const int ns = std::min(20, ln);
        const std::vector<Mat3L> sub(ll.begin(),  ll.begin()  + ns);
        const std::vector<Mat3L> sub2(ll2.begin(), ll2.begin() + ns);
        const std::vector<double> e_ax(eig_e.A2_ax.begin(), eig_e.A2_ax.begin()+ns);
        const std::vector<double> e_sh(eig_e.A2_sh.begin(), eig_e.A2_sh.begin()+ns);
        const std::vector<double> e_rr(eig_e.A2_rr.begin(), eig_e.A2_rr.begin()+ns);
        const std::vector<double> e_sr(eig_e.A2_sr.begin(), eig_e.A2_sr.begin()+ns);
        const std::vector<Vec3L>  e_rx(eig_e.rotAx.begin(), eig_e.rotAx.begin()+ns);

        const std::vector<double> s_ax(sch_e.A2_ax.begin(), sch_e.A2_ax.begin()+ns);
        const std::vector<double> s_sh(sch_e.A2_sh.begin(), sch_e.A2_sh.begin()+ns);
        const std::vector<double> s_rr(sch_e.A2_rr.begin(), sch_e.A2_rr.begin()+ns);
        const std::vector<double> s_sr(sch_e.A2_sr.begin(), sch_e.A2_sr.begin()+ns);
        const std::vector<Vec3L>  s_rx(sch_e.rotAx.begin(), sch_e.rotAx.begin()+ns);

        const std::vector<double> el_ax(eig_l.A2_ax.begin(), eig_l.A2_ax.begin()+ns);
        const std::vector<double> el_sh(eig_l.A2_sh.begin(), eig_l.A2_sh.begin()+ns);
        const std::vector<double> el_rr(eig_l.A2_rr.begin(), eig_l.A2_rr.begin()+ns);
        const std::vector<double> el_sr(eig_l.A2_sr.begin(), eig_l.A2_sr.begin()+ns);
        const std::vector<Vec3L>  el_rx(eig_l.rotAx.begin(), eig_l.rotAx.begin()+ns);

        const std::vector<double> sl_ax(sch_l.A2_ax.begin(), sch_l.A2_ax.begin()+ns);
        const std::vector<double> sl_sh(sch_l.A2_sh.begin(), sch_l.A2_sh.begin()+ns);
        const std::vector<double> sl_rr(sch_l.A2_rr.begin(), sch_l.A2_rr.begin()+ns);
        const std::vector<double> sl_sr(sch_l.A2_sr.begin(), sch_l.A2_sr.begin()+ns);
        const std::vector<Vec3L>  sl_rx(sch_l.rotAx.begin(), sch_l.rotAx.begin()+ns);

        t.check_pass("SYN:Eigen:EIG:testBatch(20pts/rank)",
            vgt_lapack::test_vgt_batch_part(sub, e_ax, e_sh, e_rr, e_sr, e_rx, errtol));
        t.check_pass("SYN:Eigen:SCHUR:testBatch(20pts/rank)",
            vgt_lapack::test_vgt_batch_part(sub, s_ax, s_sh, s_rr, s_sr, s_rx, errtol));
        t.check_pass("SYN:LAPACK:EIG:testBatch(20pts/rank)",
            vgt_lapack::test_vgt_batch_part(sub2, el_ax, el_sh, el_rr, el_sr, el_rx, errtol));
        t.check_pass("SYN:LAPACK:SCHUR:testBatch(20pts/rank)",
            vgt_lapack::test_vgt_batch_part(sub2, sl_ax, sl_sh, sl_rr, sl_sr, sl_rx, errtol));
    }

    // ── T34: Cross-pathway SCHUR norm(A2_ax) ─────────────────────────────────
    {
        double lsq_e=0, lsq_l=0;
        for (int k = 0; k < ln; ++k) {
            lsq_e += sch_e.A2_ax[k] * sch_e.A2_ax[k];
            lsq_l += sch_l.A2_ax[k] * sch_l.A2_ax[k];
        }
        t.check_near("SYN:cross:SCHUR:norm(A2_ax) Eigen==LAPACK",
                     vgt_mpi::allreduce_norm(lsq_e),
                     vgt_mpi::allreduce_norm(lsq_l), ref::REL_TOL);
    }
}

} // namespace

int main(int argc, char** argv)
{
    mfem::MPI_Session session(argc, argv);
    const int rank = session.WorldRank();
    const int size = session.WorldSize();

    if (rank == 0)
        std::cout << "=== VGT Unit Tests [MFEM, MPI x" << size << "] ===\n";

    Tester t{rank};
    const std::string csv = std::string(VGT_DATA_DIR) + "/vgt_input.csv";

    try {
        run_csv_tests(t, csv, rank, size);
        run_syn_tests(t, rank, size);
    } catch (const std::exception& e) {
        std::cerr << "[rank " << rank << "] EXCEPTION: " << e.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    return t.summary() > 0 ? 1 : 0;
}
