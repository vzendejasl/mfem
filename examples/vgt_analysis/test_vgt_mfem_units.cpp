/*
 * test_vgt_mfem_units.cpp — MPI unit tests for Eigen and LAPACK VGT pathways.
 *
 * This mirrors the bundle-level test_vgt_units.cpp, but uses MFEM's
 * MPI_Session for startup/shutdown so it drops cleanly into examples/vgt_analysis.
 * The data path is injected by the local Makefile via VGT_DATA_DIR.
 *
 * Build:
 *   make test_vgt_mfem_units
 *   make MFEM_CXX=/path/to/mpicxx VGT_DIR=/path/to/vgt_all_bundle test_vgt_mfem_units
 *
 * Run (serial, one MPI rank):
 *   ./test_vgt_mfem_units
 *
 * Run (parallel):
 *   mpirun -n 2 ./test_vgt_mfem_units
 *   mpirun -n 4 ./test_vgt_mfem_units
 *
 * Exit code: 0 = all pass, 1 = at least one failure.
 */

#include "mfem.hpp"
#include "vgt_partition.hpp"
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
#  error "VGT_DATA_DIR must point to vgt_all_bundle/data (set by Makefile or compiler flags)"
#endif

namespace {

namespace ref {
constexpr double VGT_CSV = 2.828427124746190e+00;
constexpr double REL_TOL = 5e-10;
constexpr double ABS_TOL = 1e-10;
}

struct Tester {
    int rank;
    int pass_ = 0;
    int fail_ = 0;

    void check_near(const std::string& name, double got, double expected,
                    double rtol = ref::REL_TOL)
    {
        const double err = std::abs(got - expected) / (std::abs(expected) + 1e-30);
        const bool ok = err <= rtol;
        if (rank == 0) {
            std::cout << (ok ? "  PASS" : "  FAIL") << "  " << name
                      << "  got=" << std::scientific << std::setprecision(6) << got
                      << "  ref=" << expected
                      << "  relerr=" << err << "\n";
        }
        ok ? ++pass_ : ++fail_;
    }

    void check_max(const std::string& name, double local_max,
                   double tol = ref::ABS_TOL)
    {
        const double gmax = vgt_mpi::allreduce_max(local_max);
        const bool ok = gmax <= tol;
        if (rank == 0) {
            std::cout << (ok ? "  PASS" : "  FAIL") << "  " << name
                      << "  max_err=" << std::scientific << std::setprecision(3) << gmax << "\n";
        }
        ok ? ++pass_ : ++fail_;
    }

    void check_pass(const std::string& name, bool local_ok)
    {
        const int gok = vgt_mpi::allreduce_and(local_ok ? 1 : 0);
        if (rank == 0) {
            std::cout << (gok ? "  PASS" : "  FAIL") << "  " << name << "\n";
        }
        gok ? ++pass_ : ++fail_;
    }

    int summary()
    {
        int nfail = fail_;
        MPI_Bcast(&nfail, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "\n" << std::string(60, '=') << "\n"
                      << "TOTAL: " << pass_ << " passed, " << fail_ << " failed\n"
                      << std::string(60, '=') << "\n";
        }
        return nfail;
    }
};

using vgt::Mat3;
using vgt_lapack::Mat3L;
using vgt_lapack::Vec3L;

std::vector<double> pack_eigen(const std::vector<Mat3>& v)
{
    std::vector<double> buf(9 * v.size());
    for (std::size_t k = 0; k < v.size(); ++k) {
        std::memcpy(&buf[9 * k], v[k].data(), 9 * sizeof(double));
    }
    return buf;
}

std::vector<Mat3> unpack_eigen(const std::vector<double>& buf, int n)
{
    std::vector<Mat3> v(n);
    for (int k = 0; k < n; ++k) {
        std::memcpy(v[k].data(), &buf[9 * k], 9 * sizeof(double));
    }
    return v;
}

std::vector<Mat3> scatter_eigen(const std::vector<Mat3>& all,
                                int N, int rank, int size)
{
    std::vector<int> sc, dp;
    vgt_mpi::scatter_params(N, size, 9, sc, dp);

    std::vector<double> sb;
    if (rank == 0) {
        sb = pack_eigen(all);
    }

    std::vector<double> rb(sc[rank]);
    MPI_Scatterv(rank == 0 ? sb.data() : nullptr,
                 sc.data(), dp.data(), MPI_DOUBLE,
                 rb.data(), sc[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return unpack_eigen(rb, sc[rank] / 9);
}

std::vector<Mat3L> unpack_lapack(const std::vector<double>& buf, int n)
{
    std::vector<Mat3L> v(n);
    for (int k = 0; k < n; ++k) {
        std::memcpy(v[k].d, &buf[9 * k], 9 * sizeof(double));
    }
    return v;
}

std::vector<Mat3L> scatter_lapack(const std::vector<double>& sendbuf,
                                  int N, int rank, int size)
{
    std::vector<int> sc, dp;
    vgt_mpi::scatter_params(N, size, 9, sc, dp);

    std::vector<double> rb(sc[rank]);
    MPI_Scatterv(rank == 0 ? sendbuf.data() : nullptr,
                 sc.data(), dp.data(), MPI_DOUBLE,
                 rb.data(), sc[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return unpack_lapack(rb, sc[rank] / 9);
}

void run_csv_tests(Tester& t, const std::string& csv, int rank, int size)
{
    if (rank == 0) {
        std::cout << "\n--- CSV tests (8-point dataset, MATLAB cross-validation) ---\n";
    }

    const double errtol = 1e-10;

    int N = 0;
    std::vector<Mat3> all_eigen;
    std::vector<double> flat;
    if (rank == 0) {
        all_eigen = vgt::load_csv_vgt(csv);
        N = static_cast<int>(all_eigen.size());
        flat = pack_eigen(all_eigen);
    }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const auto le = scatter_eigen(all_eigen, N, rank, size);
    const int ln = static_cast<int>(le.size());
    const auto ll = scatter_lapack(flat, N, rank, size);

    {
        double lsq = 0.0;
        for (const auto& A : le) { lsq += A.squaredNorm(); }
        t.check_near("CSV:Eigen:norm(VGT)", vgt_mpi::allreduce_norm(lsq), ref::VGT_CSV);
    }
    {
        double lsq = 0.0;
        for (const auto& A : ll) { lsq += A.squaredNorm(); }
        t.check_near("CSV:LAPACK:norm(VGT)", vgt_mpi::allreduce_norm(lsq), ref::VGT_CSV);
    }
    {
        double lsq_e = 0.0;
        double lsq_l = 0.0;
        for (const auto& A : le) { lsq_e += A.squaredNorm(); }
        for (const auto& A : ll) { lsq_l += A.squaredNorm(); }
        t.check_near("CSV:cross:norm(VGT) Eigen==LAPACK",
                     vgt_mpi::allreduce_norm(lsq_e),
                     vgt_mpi::allreduce_norm(lsq_l), 1e-14);
    }

    const auto eig_e = vgt::part_vgt_batch_eig(le, errtol);
    const auto sch_e = vgt::part_vgt_batch_schur(le);
    const auto eig_l = vgt_lapack::part_vgt_batch_eig(ll, errtol);
    const auto sch_l = vgt_lapack::part_vgt_batch_schur(ll);

    {
        double lmax = 0.0;
        for (int k = 0; k < ln; ++k) {
            const double sum = eig_e.A2_ax(k) + eig_e.A2_sh(k)
                             + eig_e.A2_rr(k) + eig_e.A2_sr(k);
            lmax = std::max(lmax, std::abs(le[k].squaredNorm() - sum));
        }
        t.check_max("CSV:Eigen:EIG:partition_identity", lmax);
    }
    {
        double lmax = 0.0;
        for (int k = 0; k < ln; ++k) {
            const double sum = sch_e.A2_ax(k) + sch_e.A2_sh(k)
                             + sch_e.A2_rr(k) + sch_e.A2_sr(k);
            lmax = std::max(lmax, std::abs(le[k].squaredNorm() - sum));
        }
        t.check_max("CSV:Eigen:SCHUR:partition_identity", lmax);
    }
    {
        double err_ax = 0.0;
        double err_sh = 0.0;
        double err_rr = 0.0;
        for (int k = 0; k < ln; ++k) {
            err_ax = std::max(err_ax, std::abs(sch_e.A2_ax(k) - sch_e.A_ax[k].squaredNorm()));
            err_sh = std::max(err_sh, std::abs(sch_e.A2_sh(k) - sch_e.A_sh[k].squaredNorm()));
            err_rr = std::max(err_rr, std::abs(sch_e.A2_rr(k) - sch_e.A_rr[k].squaredNorm()));
        }
        t.check_max("CSV:Eigen:SCHUR:backtransform_ax", err_ax);
        t.check_max("CSV:Eigen:SCHUR:backtransform_sh", err_sh);
        t.check_max("CSV:Eigen:SCHUR:backtransform_rr", err_rr);
    }

    {
        double lmax = 0.0;
        for (int k = 0; k < ln; ++k) {
            const double sum = eig_l.A2_ax[k] + eig_l.A2_sh[k]
                             + eig_l.A2_rr[k] + eig_l.A2_sr[k];
            lmax = std::max(lmax, std::abs(ll[k].squaredNorm() - sum));
        }
        t.check_max("CSV:LAPACK:EIG:partition_identity", lmax);
    }
    {
        double lmax = 0.0;
        for (int k = 0; k < ln; ++k) {
            const double sum = sch_l.A2_ax[k] + sch_l.A2_sh[k]
                             + sch_l.A2_rr[k] + sch_l.A2_sr[k];
            lmax = std::max(lmax, std::abs(ll[k].squaredNorm() - sum));
        }
        t.check_max("CSV:LAPACK:SCHUR:partition_identity", lmax);
    }
    {
        double err_ax = 0.0;
        double err_sh = 0.0;
        double err_rr = 0.0;
        for (int k = 0; k < ln; ++k) {
            err_ax = std::max(err_ax, std::abs(sch_l.A2_ax[k] - sch_l.A_ax[k].squaredNorm()));
            err_sh = std::max(err_sh, std::abs(sch_l.A2_sh[k] - sch_l.A_sh[k].squaredNorm()));
            err_rr = std::max(err_rr, std::abs(sch_l.A2_rr[k] - sch_l.A_rr[k].squaredNorm()));
        }
        t.check_max("CSV:LAPACK:SCHUR:backtransform_ax", err_ax);
        t.check_max("CSV:LAPACK:SCHUR:backtransform_sh", err_sh);
        t.check_max("CSV:LAPACK:SCHUR:backtransform_rr", err_rr);
    }

    {
        const bool ok = vgt::test_vgt_batch_part(
            le, eig_e.A2_ax, eig_e.A2_sh, eig_e.A2_rr, eig_e.A2_sr,
            eig_e.rotAx, errtol);
        t.check_pass("CSV:Eigen:EIG:testBatch", ok);
    }
    {
        const bool ok = vgt::test_vgt_batch_part(
            le, sch_e.A2_ax, sch_e.A2_sh, sch_e.A2_rr, sch_e.A2_sr,
            sch_e.rotAx, errtol);
        t.check_pass("CSV:Eigen:SCHUR:testBatch", ok);
    }
    {
        const bool ok = vgt_lapack::test_vgt_batch_part(
            ll, eig_l.A2_ax, eig_l.A2_sh, eig_l.A2_rr, eig_l.A2_sr,
            eig_l.rotAx, errtol);
        t.check_pass("CSV:LAPACK:EIG:testBatch", ok);
    }
    {
        const bool ok = vgt_lapack::test_vgt_batch_part(
            ll, sch_l.A2_ax, sch_l.A2_sh, sch_l.A2_rr, sch_l.A2_sr,
            sch_l.rotAx, errtol);
        t.check_pass("CSV:LAPACK:SCHUR:testBatch", ok);
    }

    {
        double lsq_e = 0.0;
        double lsq_l = 0.0;
        for (int k = 0; k < ln; ++k) {
            lsq_e += sch_e.A2_ax(k) * sch_e.A2_ax(k);
            lsq_l += sch_l.A2_ax[k] * sch_l.A2_ax[k];
        }
        t.check_near("CSV:cross:SCHUR:norm(A2_ax) Eigen==LAPACK",
                     vgt_mpi::allreduce_norm(lsq_e),
                     vgt_mpi::allreduce_norm(lsq_l), ref::REL_TOL);
    }
}

void run_syn_tests(Tester& t, int rank, int size)
{
    if (rank == 0) {
        std::cout << "\n--- Synthetic tests (1000-pt dataset, seed=42) ---\n";
    }

    const double errtol = 1e-10;
    const int N_SYN = 1000;

    std::vector<Mat3> all_eigen;
    std::vector<double> flat;
    if (rank == 0) {
        all_eigen = vgt::make_synthetic_vgt(N_SYN, true, true, 42u);
        flat = pack_eigen(all_eigen);
    }

    const auto le = scatter_eigen(all_eigen, N_SYN, rank, size);
    const int ln = static_cast<int>(le.size());
    const auto ll = scatter_lapack(flat, N_SYN, rank, size);

    {
        double lsq_e = 0.0;
        double lsq_l = 0.0;
        for (const auto& A : le) { lsq_e += A.squaredNorm(); }
        for (const auto& A : ll) { lsq_l += A.squaredNorm(); }
        const double ne = vgt_mpi::allreduce_norm(lsq_e);
        const double nl = vgt_mpi::allreduce_norm(lsq_l);
        if (rank == 0) {
            std::cout << "  INFO  SYN:norm(VGT)="
                      << std::scientific << std::setprecision(6) << ne << "\n";
        }
        t.check_near("SYN:cross:norm(VGT) Eigen==LAPACK", ne, nl, 1e-14);
    }

    const auto eig_e = vgt::part_vgt_batch_eig(le, errtol);
    const auto sch_e = vgt::part_vgt_batch_schur(le);
    const auto eig_l = vgt_lapack::part_vgt_batch_eig(ll, errtol);
    const auto sch_l = vgt_lapack::part_vgt_batch_schur(ll);

    {
        double lmax = 0.0;
        for (int k = 0; k < ln; ++k) {
            const double sum = eig_e.A2_ax(k) + eig_e.A2_sh(k)
                             + eig_e.A2_rr(k) + eig_e.A2_sr(k);
            lmax = std::max(lmax, std::abs(le[k].squaredNorm() - sum));
        }
        t.check_max("SYN:Eigen:EIG:partition_identity", lmax);
    }
    {
        double lmax = 0.0;
        for (int k = 0; k < ln; ++k) {
            const double sum = sch_e.A2_ax(k) + sch_e.A2_sh(k)
                             + sch_e.A2_rr(k) + sch_e.A2_sr(k);
            lmax = std::max(lmax, std::abs(le[k].squaredNorm() - sum));
        }
        t.check_max("SYN:Eigen:SCHUR:partition_identity", lmax);
    }
    {
        double err_ax = 0.0;
        double err_sh = 0.0;
        double err_rr = 0.0;
        for (int k = 0; k < ln; ++k) {
            err_ax = std::max(err_ax, std::abs(sch_e.A2_ax(k) - sch_e.A_ax[k].squaredNorm()));
            err_sh = std::max(err_sh, std::abs(sch_e.A2_sh(k) - sch_e.A_sh[k].squaredNorm()));
            err_rr = std::max(err_rr, std::abs(sch_e.A2_rr(k) - sch_e.A_rr[k].squaredNorm()));
        }
        t.check_max("SYN:Eigen:SCHUR:backtransform_ax", err_ax);
        t.check_max("SYN:Eigen:SCHUR:backtransform_sh", err_sh);
        t.check_max("SYN:Eigen:SCHUR:backtransform_rr", err_rr);
    }

    {
        double lmax = 0.0;
        for (int k = 0; k < ln; ++k) {
            const double sum = eig_l.A2_ax[k] + eig_l.A2_sh[k]
                             + eig_l.A2_rr[k] + eig_l.A2_sr[k];
            lmax = std::max(lmax, std::abs(ll[k].squaredNorm() - sum));
        }
        t.check_max("SYN:LAPACK:EIG:partition_identity", lmax);
    }
    {
        double lmax = 0.0;
        for (int k = 0; k < ln; ++k) {
            const double sum = sch_l.A2_ax[k] + sch_l.A2_sh[k]
                             + sch_l.A2_rr[k] + sch_l.A2_sr[k];
            lmax = std::max(lmax, std::abs(ll[k].squaredNorm() - sum));
        }
        t.check_max("SYN:LAPACK:SCHUR:partition_identity", lmax);
    }
    {
        double err_ax = 0.0;
        double err_sh = 0.0;
        double err_rr = 0.0;
        for (int k = 0; k < ln; ++k) {
            err_ax = std::max(err_ax, std::abs(sch_l.A2_ax[k] - sch_l.A_ax[k].squaredNorm()));
            err_sh = std::max(err_sh, std::abs(sch_l.A2_sh[k] - sch_l.A_sh[k].squaredNorm()));
            err_rr = std::max(err_rr, std::abs(sch_l.A2_rr[k] - sch_l.A_rr[k].squaredNorm()));
        }
        t.check_max("SYN:LAPACK:SCHUR:backtransform_ax", err_ax);
        t.check_max("SYN:LAPACK:SCHUR:backtransform_sh", err_sh);
        t.check_max("SYN:LAPACK:SCHUR:backtransform_rr", err_rr);
    }

    {
        const int ns = std::min(20, ln);
        std::vector<Mat3> sub(le.begin(), le.begin() + ns);
        const bool ok = vgt::test_vgt_batch_part(
            sub,
            eig_e.A2_ax.head(ns), eig_e.A2_sh.head(ns),
            eig_e.A2_rr.head(ns), eig_e.A2_sr.head(ns),
            eig_e.rotAx.leftCols(ns), errtol);
        t.check_pass("SYN:Eigen:EIG:testBatch(20pts/rank)", ok);
    }
    {
        const int ns = std::min(20, ln);
        std::vector<Mat3> sub(le.begin(), le.begin() + ns);
        const bool ok = vgt::test_vgt_batch_part(
            sub,
            sch_e.A2_ax.head(ns), sch_e.A2_sh.head(ns),
            sch_e.A2_rr.head(ns), sch_e.A2_sr.head(ns),
            sch_e.rotAx.leftCols(ns), errtol);
        t.check_pass("SYN:Eigen:SCHUR:testBatch(20pts/rank)", ok);
    }
    {
        const int ns = std::min(20, ln);
        std::vector<Mat3L> sub(ll.begin(), ll.begin() + ns);
        const std::vector<double> e_ax(eig_l.A2_ax.begin(), eig_l.A2_ax.begin() + ns);
        const std::vector<double> e_sh(eig_l.A2_sh.begin(), eig_l.A2_sh.begin() + ns);
        const std::vector<double> e_rr(eig_l.A2_rr.begin(), eig_l.A2_rr.begin() + ns);
        const std::vector<double> e_sr(eig_l.A2_sr.begin(), eig_l.A2_sr.begin() + ns);
        const std::vector<Vec3L> e_rx(eig_l.rotAx.begin(), eig_l.rotAx.begin() + ns);
        const bool ok = vgt_lapack::test_vgt_batch_part(sub, e_ax, e_sh, e_rr, e_sr, e_rx, errtol);
        t.check_pass("SYN:LAPACK:EIG:testBatch(20pts/rank)", ok);
    }
    {
        const int ns = std::min(20, ln);
        std::vector<Mat3L> sub(ll.begin(), ll.begin() + ns);
        const std::vector<double> s_ax(sch_l.A2_ax.begin(), sch_l.A2_ax.begin() + ns);
        const std::vector<double> s_sh(sch_l.A2_sh.begin(), sch_l.A2_sh.begin() + ns);
        const std::vector<double> s_rr(sch_l.A2_rr.begin(), sch_l.A2_rr.begin() + ns);
        const std::vector<double> s_sr(sch_l.A2_sr.begin(), sch_l.A2_sr.begin() + ns);
        const std::vector<Vec3L> s_rx(sch_l.rotAx.begin(), sch_l.rotAx.begin() + ns);
        const bool ok = vgt_lapack::test_vgt_batch_part(sub, s_ax, s_sh, s_rr, s_sr, s_rx, errtol);
        t.check_pass("SYN:LAPACK:SCHUR:testBatch(20pts/rank)", ok);
    }

    {
        double lsq_e = 0.0;
        double lsq_l = 0.0;
        for (int k = 0; k < ln; ++k) {
            lsq_e += sch_e.A2_ax(k) * sch_e.A2_ax(k);
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

    if (rank == 0) {
        std::cout << "=== VGT Unit Tests [MFEM, MPI x" << size << "] ===\n";
    }

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
