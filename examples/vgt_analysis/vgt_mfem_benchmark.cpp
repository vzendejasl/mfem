/*
 * vgt_mfem_benchmark.cpp
 *
 * Benchmarks backend impact directly: Eigen vs LAPACK VGT batch decomposition
 * on representative datasets used by the MFEM examples in this directory.
 *
 * Cases:
 *   - csv          : 8-point reference dataset from vgt_input.csv
 *   - synthetic    : 1000-point synthetic batch, seed=42
 *   - mesh_exact   : VGTs extracted from the exact linear mesh test
 *   - manufactured : VGTs extracted from the manufactured MFEM example
 *
 * For each case, the benchmark measures:
 *   - EIG method time on the Eigen backend
 *   - EIG method time on the LAPACK backend
 *   - Schur method time on the Eigen backend
 *   - Schur method time on the LAPACK backend
 *
 * Timings compare the Eigen-backed implementation against the LAPACK-backed
 * implementation on the same local tensor batches. Reported times are the
 * median of repeated runs, using the maximum wall time across MPI ranks for
 * each repetition.
 *
 * Build:
 *   make MFEM_CXX=/usr/local/bin/mpicxx vgt_mfem_benchmark
 *
 * Run (serial, one MPI rank):
 *   ./vgt_mfem_benchmark
 *
 * Run (parallel):
 *   mpirun -n 4 ./vgt_mfem_benchmark
 */

#include "mfem.hpp"
#include "vgt_partition.hpp"
#include "vgt_lapack.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#ifndef VGT_DATA_DIR
#  error "VGT_DATA_DIR must point to the shared VGT data directory (set by Makefile or compiler flags)"
#endif

using namespace mfem;

namespace {

using vgt::Mat3;
using vgt_lapack::Mat3L;

static constexpr double PI  = M_PI;
static constexpr double EPS = 0.05;
static constexpr double ERR_TOL = 1e-10;
static volatile double g_sink = 0.0;

struct CaseData {
    std::string name;
    std::string description;
    int global_n = 0;
    std::vector<Mat3> eigen_local;
    std::vector<Mat3L> lapack_local;
};

struct BenchPair {
    double eigen_ms = 0.0;
    double lapack_ms = 0.0;
    double lapack_over_eigen = 0.0;
    double eigen_speedup_pct = 0.0;
};

struct BenchResult {
    BenchPair eig;
    BenchPair schur;
};

double allreduce_max(double x, MPI_Comm comm)
{
    double g = 0.0;
    MPI_Allreduce(&x, &g, 1, MPI_DOUBLE, MPI_MAX, comm);
    return g;
}

int allreduce_sum_int(int x, MPI_Comm comm)
{
    int g = 0;
    MPI_Allreduce(&x, &g, 1, MPI_INT, MPI_SUM, comm);
    return g;
}

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

std::vector<Mat3L> unpack_lapack(const std::vector<double>& buf, int n)
{
    std::vector<Mat3L> v(n);
    for (int k = 0; k < n; ++k) {
        std::memcpy(v[k].d, &buf[9 * k], 9 * sizeof(double));
    }
    return v;
}

std::vector<Mat3> scatter_eigen(const std::vector<Mat3>& all,
                                int N, int rank, int size)
{
    std::vector<int> counts(size), displs(size);
    const int base = N / size;
    const int rem  = N % size;
    int off = 0;
    for (int r = 0; r < size; ++r) {
        const int nr = base + (r < rem ? 1 : 0);
        counts[r] = 9 * nr;
        displs[r] = 9 * off;
        off += nr;
    }

    std::vector<double> sendbuf;
    if (rank == 0) { sendbuf = pack_eigen(all); }
    std::vector<double> recvbuf(counts[rank]);
    MPI_Scatterv(rank == 0 ? sendbuf.data() : nullptr,
                 counts.data(), displs.data(), MPI_DOUBLE,
                 recvbuf.data(), counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return unpack_eigen(recvbuf, counts[rank] / 9);
}

std::vector<Mat3L> scatter_lapack(const std::vector<double>& flat,
                                  int N, int rank, int size)
{
    std::vector<int> counts(size), displs(size);
    const int base = N / size;
    const int rem  = N % size;
    int off = 0;
    for (int r = 0; r < size; ++r) {
        const int nr = base + (r < rem ? 1 : 0);
        counts[r] = 9 * nr;
        displs[r] = 9 * off;
        off += nr;
    }

    std::vector<double> recvbuf(counts[rank]);
    MPI_Scatterv(rank == 0 ? flat.data() : nullptr,
                 counts.data(), displs.data(), MPI_DOUBLE,
                 recvbuf.data(), counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return unpack_lapack(recvbuf, counts[rank] / 9);
}

std::vector<Mat3> to_eigen(const std::vector<Mat3L>& vgts)
{
    std::vector<Mat3> out(vgts.size());
    for (std::size_t k = 0; k < vgts.size(); ++k) {
        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 3; ++i) {
                out[k](i, j) = vgts[k](i, j);
            }
        }
    }
    return out;
}

static const double A_REF[9] = {
    -0.121351798876562,  0.435956031200938,  0.224397487935023,
     0.0954279919744173,-0.332717968316346, -0.332741298629221,
    -0.427446193346677,  0.354186004220843,  0.454069767192908
};

void exact_vel_func(const Vector& x, Vector& v)
{
    for (int i = 0; i < 3; ++i) {
        v[i] = 0.0;
        for (int j = 0; j < 3; ++j) {
            v[i] += A_REF[i + 3 * j] * x[j];
        }
    }
}

void manufactured_vel_func(const Vector& x, Vector& v)
{
    v[0] =  x[0] - 10.0*x[1]
           + EPS*(std::sin(2*PI*x[0]) + std::sin(4*PI*x[1]) + std::sin(6*PI*x[2]));
    v[1] = 10.0*x[0] + x[1]
           + EPS*(std::sin(6*PI*x[0]) + std::sin(2*PI*x[1]) + std::sin(4*PI*x[2]));
    v[2] = -2.0*x[2]
           + EPS*(std::sin(4*PI*x[0]) + std::sin(6*PI*x[1]) + std::sin(2*PI*x[2]));
}

void periodic_abc_vel_func(const Vector& x, Vector& v)
{
    static const double ABC_A = 1.0;
    static const double ABC_B = std::sqrt(2.0);
    static const double ABC_C = std::sqrt(3.0);
    const double sx = std::sin(2*PI*x[0]), cx = std::cos(2*PI*x[0]);
    const double sy = std::sin(2*PI*x[1]), cy = std::cos(2*PI*x[1]);
    const double sz = std::sin(2*PI*x[2]), cz = std::cos(2*PI*x[2]);
    v[0] = ABC_A * sz + ABC_C * cy;
    v[1] = ABC_B * sx + ABC_A * cz;
    v[2] = ABC_C * sy + ABC_B * cx;
}

std::vector<Mat3L> collect_mfem_vgts(int N, int order,
                                     void (*vel_func)(const Vector&, Vector&),
                                     MPI_Comm comm,
                                     bool periodic = false)
{
    Mesh base = Mesh::MakeCartesian3D(N, N, N, Element::HEXAHEDRON,
                                      1.0, 1.0, 1.0, false);
    Mesh serial = periodic
        ? Mesh::MakePeriodic(base,
              base.CreatePeriodicVertexMapping(
                  std::vector<Vector>{Vector({1.0,0.0,0.0}),
                                      Vector({0.0,1.0,0.0}),
                                      Vector({0.0,0.0,1.0})}))
        : std::move(base);
    ParMesh pmesh(comm, serial);
    serial.Clear();

    H1_FECollection fec(order, 3);
    ParFiniteElementSpace fes(&pmesh, &fec, 3);

    ParGridFunction vel(&fes);
    VectorFunctionCoefficient coeff(3, vel_func);
    vel.ProjectCoefficient(coeff);

    const IntegrationRule& ir = IntRules.Get(Geometry::CUBE, 2*order+1);
    const int nqp = ir.GetNPoints();

    std::vector<Mat3L> local_vgts;
    local_vgts.reserve(pmesh.GetNE() * nqp);

    for (int e = 0; e < pmesh.GetNE(); ++e) {
        ElementTransformation* T = pmesh.GetElementTransformation(e);
        for (int q = 0; q < nqp; ++q) {
            const IntegrationPoint& ip = ir.IntPoint(q);
            T->SetIntPoint(&ip);
            DenseMatrix grad(3, 3);
            vel.GetVectorGradient(*T, grad);
            Mat3L M;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    M.d[i + 3*j] = grad(i, j);
                }
            }
            local_vgts.push_back(M);
        }
    }
    return local_vgts;
}

CaseData make_csv_case(int rank, int size)
{
    CaseData c;
    c.name = "csv";
    c.description = "8-point reference dataset from vgt_input.csv";

    std::vector<Mat3> all_eigen;
    std::vector<double> flat;
    if (rank == 0) {
        const std::string csv = std::string(VGT_DATA_DIR) + "/vgt_input.csv";
        all_eigen = vgt::load_csv_vgt(csv);
        c.global_n = static_cast<int>(all_eigen.size());
        flat = pack_eigen(all_eigen);
    }
    MPI_Bcast(&c.global_n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    c.eigen_local = scatter_eigen(all_eigen, c.global_n, rank, size);
    c.lapack_local = scatter_lapack(flat, c.global_n, rank, size);
    return c;
}

CaseData make_synthetic_case(int rank, int size)
{
    CaseData c;
    c.name = "synthetic";
    c.description = "1000-point synthetic batch, seed=42";
    std::vector<Mat3> all_eigen;
    std::vector<double> flat;
    if (rank == 0) {
        c.global_n = 1000;
        all_eigen = vgt::make_synthetic_vgt(c.global_n, true, true, 42u);
        flat = pack_eigen(all_eigen);
    }
    MPI_Bcast(&c.global_n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    c.eigen_local = scatter_eigen(all_eigen, c.global_n, rank, size);
    c.lapack_local = scatter_lapack(flat, c.global_n, rank, size);
    return c;
}

CaseData make_mesh_exact_case(MPI_Comm comm)
{
    CaseData c;
    c.name = "mesh_exact";
    c.description = "Exact linear MFEM mesh case (N=8, order=4)";
    c.lapack_local = collect_mfem_vgts(8, 4, exact_vel_func, comm);
    c.eigen_local = to_eigen(c.lapack_local);
    c.global_n = allreduce_sum_int(static_cast<int>(c.lapack_local.size()), comm);
    return c;
}

CaseData make_manufactured_case(MPI_Comm comm)
{
    CaseData c;
    c.name = "manufactured";
    c.description = "Manufactured MFEM example case (N=4, order=2)";
    c.lapack_local = collect_mfem_vgts(4, 2, manufactured_vel_func, comm);
    c.eigen_local = to_eigen(c.lapack_local);
    c.global_n = allreduce_sum_int(static_cast<int>(c.lapack_local.size()), comm);
    return c;
}

CaseData make_periodic_abc_case(MPI_Comm comm)
{
    CaseData c;
    c.name = "periodic_abc";
    c.description = "Periodic ABC field, fully periodic mesh (N=16, order=2)";
    c.lapack_local = collect_mfem_vgts(16, 2, periodic_abc_vel_func, comm, true);
    c.eigen_local = to_eigen(c.lapack_local);
    c.global_n = allreduce_sum_int(static_cast<int>(c.lapack_local.size()), comm);
    return c;
}

template <typename Fn>
double median_ms(Fn&& fn, int repeats, MPI_Comm comm)
{
    std::vector<double> samples;
    samples.reserve(repeats);
    for (int rep = 0; rep < repeats; ++rep) {
        MPI_Barrier(comm);
        const double t0 = MPI_Wtime();
        const double checksum = fn();
        const double local_dt = MPI_Wtime() - t0;
        const double global_dt = allreduce_max(local_dt, comm);
        samples.push_back(global_dt * 1000.0);
        g_sink += checksum;
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

BenchResult bench_case(const CaseData& c, MPI_Comm comm)
{
    BenchResult out;

    // Warmup both pathways once.
    {
        const auto eig_e = vgt::part_vgt_batch_eig(c.eigen_local, ERR_TOL);
        const auto eig_l = vgt_lapack::part_vgt_batch_eig(c.lapack_local, ERR_TOL);
        const auto sch_e = vgt::part_vgt_batch_schur(c.eigen_local);
        const auto sch_l = vgt_lapack::part_vgt_batch_schur(c.lapack_local);
        if (!c.eigen_local.empty()) {
            g_sink += eig_e.A2_ax(0) + sch_e.A2_ax(0);
        }
        if (!c.lapack_local.empty()) {
            g_sink += eig_l.A2_ax[0] + sch_l.A2_ax[0];
        }
    }

    const int repeats = 9;

    const double eig_eig_ms = median_ms([&]() -> double {
        const auto res = vgt::part_vgt_batch_eig(c.eigen_local, ERR_TOL);
        return c.eigen_local.empty() ? 0.0 :
               res.A2_ax(0) + res.A2_sh(0) + res.A2_rr(0) + res.A2_sr(0);
    }, repeats, comm);

    const double lap_eig_ms = median_ms([&]() -> double {
        const auto res = vgt_lapack::part_vgt_batch_eig(c.lapack_local, ERR_TOL);
        return c.lapack_local.empty() ? 0.0 :
               res.A2_ax[0] + res.A2_sh[0] + res.A2_rr[0] + res.A2_sr[0];
    }, repeats, comm);

    const double eig_sch_ms = median_ms([&]() -> double {
        const auto res = vgt::part_vgt_batch_schur(c.eigen_local);
        return c.eigen_local.empty() ? 0.0 :
               res.A2_ax(0) + res.A2_sh(0) + res.A2_rr(0) + res.A2_sr(0);
    }, repeats, comm);

    const double lap_sch_ms = median_ms([&]() -> double {
        const auto res = vgt_lapack::part_vgt_batch_schur(c.lapack_local);
        return c.lapack_local.empty() ? 0.0 :
               res.A2_ax[0] + res.A2_sh[0] + res.A2_rr[0] + res.A2_sr[0];
    }, repeats, comm);

    out.eig.eigen_ms = eig_eig_ms;
    out.eig.lapack_ms = lap_eig_ms;
    out.eig.lapack_over_eigen = (eig_eig_ms > 0.0) ? (lap_eig_ms / eig_eig_ms) : 0.0;
    out.eig.eigen_speedup_pct = (lap_eig_ms > 0.0)
        ? (100.0 * (lap_eig_ms - eig_eig_ms) / lap_eig_ms)
        : 0.0;
    out.schur.eigen_ms = eig_sch_ms;
    out.schur.lapack_ms = lap_sch_ms;
    out.schur.lapack_over_eigen = (eig_sch_ms > 0.0) ? (lap_sch_ms / eig_sch_ms) : 0.0;
    out.schur.eigen_speedup_pct = (lap_sch_ms > 0.0)
        ? (100.0 * (lap_sch_ms - eig_sch_ms) / lap_sch_ms)
        : 0.0;

    return out;
}

} // namespace

int main(int argc, char* argv[])
{
    mfem::MPI_Session mpi(argc, argv);
    const int rank = mpi.WorldRank();
    const int size = mpi.WorldSize();

    if (rank == 0) {
        std::cout << "=== VGT MFEM Benchmark [MPI x" << size << "] ===\n";
        std::cout << "# Distinction: method=eig|schur, backend=Eigen|LAPACK.\n";
        std::cout << "# Timings are median wall times in milliseconds.\n";
        std::cout << "# Each sample measures decomposition only; mesh construction and gradient extraction are excluded.\n";
        std::cout << "# Output fields: eigen_ms = Eigen backend, lapack_ms = LAPACK backend.\n";
    }

    std::vector<CaseData> cases;
    cases.push_back(make_csv_case(rank, size));
    cases.push_back(make_synthetic_case(rank, size));
    cases.push_back(make_mesh_exact_case(MPI_COMM_WORLD));
    cases.push_back(make_manufactured_case(MPI_COMM_WORLD));
    cases.push_back(make_periodic_abc_case(MPI_COMM_WORLD));

    for (const auto& c : cases) {
        if (rank == 0) {
            std::cout << "# case_desc[" << c.name << "]=" << c.description << "\n";
        }
        const BenchResult bench = bench_case(c, MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "BENCH"
                      << " case=" << c.name
                      << " kernel=eig"
                      << " global_n=" << c.global_n
                      << " local_n=" << c.eigen_local.size()
                      << " ranks=" << size
                      << " eigen_ms=" << std::fixed << std::setprecision(6) << bench.eig.eigen_ms
                      << " lapack_ms=" << bench.eig.lapack_ms
                      << " lapack_over_eigen=" << bench.eig.lapack_over_eigen
                      << " eigen_speedup_pct=" << bench.eig.eigen_speedup_pct
                      << "\n";
            std::cout << "BENCH"
                      << " case=" << c.name
                      << " kernel=schur"
                      << " global_n=" << c.global_n
                      << " local_n=" << c.eigen_local.size()
                      << " ranks=" << size
                      << " eigen_ms=" << bench.schur.eigen_ms
                      << " lapack_ms=" << bench.schur.lapack_ms
                      << " lapack_over_eigen=" << bench.schur.lapack_over_eigen
                      << " eigen_speedup_pct=" << bench.schur.eigen_speedup_pct
                      << "\n";
        }
    }

    if (rank == 0) {
        std::cout << "# sink=" << g_sink << "\n";
    }
    return 0;
}
