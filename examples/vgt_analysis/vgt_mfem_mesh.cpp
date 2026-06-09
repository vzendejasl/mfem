/*
 * vgt_mfem_mesh.cpp — VGT extraction from an MFEM parallel mesh.
 *
 * Constructs a 2×2×2 Cartesian hex mesh with a manufactured linear velocity
 * field v(x) = A·x, where A is the first VGT from vgt_input.csv.  Because
 * the field is linear it is represented exactly in any H1 space of order ≥ 1,
 * so GetVectorGradient returns A exactly at every quadrature point.
 *
 * Checks:
 *   1. Each extracted VGT matches A entry-wise to machine precision.
 *   2. EIG  partition identity holds at every quadrature point.
 *   3. Schur partition identity holds at every quadrature point.
 *   4. Global batch norm equals sqrt(N * ||A||²).
 *
 * Build:
 *   make MFEM_CXX=/usr/local/bin/mpicxx vgt_mfem_mesh
 *
 * Run:
 *   mpirun -n 1 ./vgt_mfem_mesh
 *   mpirun -n 2 ./vgt_mfem_mesh
 *   mpirun -n 4 ./vgt_mfem_mesh
 *
 * Exit code: 0 = all pass, 1 = any failure.
 */

#include "mfem.hpp"
#include "vgt_lapack.hpp"
#include "vgt_mpi.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace mfem;
using namespace vgt_lapack;

// First VGT from vgt_input.csv, column-major layout: d[i + j*3] = A[i][j]
// Convention: A[i][j] = du_i / dx_j
static const double A_REF[9] = {
    -0.121351798876562,  0.435956031200938,  0.224397487935023,  // col 0
     0.0954279919744173,-0.332717968316346, -0.332741298629221,  // col 1
    -0.427446193346677,  0.354186004220843,  0.454069767192908   // col 2
};

// Manufactured velocity: v[i] = sum_j A[i][j] * x[j]
static void vel_func(const Vector& x, Vector& v)
{
    for (int i = 0; i < 3; i++) {
        v[i] = 0.0;
        for (int j = 0; j < 3; j++)
            v[i] += A_REF[i + j*3] * x[j];
    }
}

int main(int argc, char* argv[])
{
    mfem::MPI_Session mpi(argc, argv);
    const int rank = mpi.WorldRank();
    const int size = mpi.WorldSize();

    if (rank == 0)
        std::cout << "=== VGT Mesh Test [MFEM, MPI x" << size << "] ===\n\n";

    // 8×8×8 hex mesh on [0,1]^3
    Mesh serial_mesh = Mesh::MakeCartesian3D(8, 8, 8, Element::HEXAHEDRON);
    ParMesh pmesh(MPI_COMM_WORLD, serial_mesh);
    serial_mesh.Clear();

    // H1 order 4, vdim=3
    const int order = 4;
    H1_FECollection fec(order, 3);
    ParFiniteElementSpace fes(&pmesh, &fec, 3);

    // Project manufactured velocity
    ParGridFunction vel(&fes);
    VectorFunctionCoefficient vcoeff(3, vel_func);
    vel.ProjectCoefficient(vcoeff);

    // Gauss rule: (order+1)^3 = 27 points per hex
    const IntegrationRule& ir = IntRules.Get(Geometry::CUBE, 2*order+1);
    const int nqp = ir.GetNPoints();

    // Extract VGTs at every quadrature point on local elements
    std::vector<Mat3L> local_vgts;
    local_vgts.reserve(pmesh.GetNE() * nqp);

    for (int e = 0; e < pmesh.GetNE(); e++) {
        ElementTransformation* T = pmesh.GetElementTransformation(e);
        for (int q = 0; q < nqp; q++) {
            const IntegrationPoint& ip = ir.IntPoint(q);
            T->SetIntPoint(&ip);
            DenseMatrix grad_v(3, 3);
            vel.GetVectorGradient(*T, grad_v);
            // grad_v(i,j) = dv_i/dx_j = A[i][j]; pack column-major into Mat3L
            Mat3L M;
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    M.d[i + j*3] = grad_v(i, j);
            local_vgts.push_back(M);
        }
    }

    const int N_local = static_cast<int>(local_vgts.size());
    int N_global = 0;
    MPI_Allreduce(&N_local, &N_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Decompose
    auto res_eig   = part_vgt_batch_eig(local_vgts);
    auto res_schur = part_vgt_batch_schur(local_vgts);

    // ── Check 1: extracted VGT matches A_REF entry-wise ───────────────────
    double local_max_entry = 0.0;
    for (const auto& M : local_vgts)
        for (int k = 0; k < 9; k++)
            local_max_entry = std::max(local_max_entry, std::abs(M.d[k] - A_REF[k]));

    double max_entry_err = 0.0;
    MPI_Allreduce(&local_max_entry, &max_entry_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // ── Check 2: EIG partition identity ────────────────────────────────────
    double local_eig_part = 0.0;
    for (int k = 0; k < N_local; k++) {
        double err = std::abs(local_vgts[k].squaredNorm() -
                              (res_eig.A2_ax[k] + res_eig.A2_sh[k] +
                               res_eig.A2_rr[k] + res_eig.A2_sr[k]));
        local_eig_part = std::max(local_eig_part, err);
    }
    double max_eig_part = 0.0;
    MPI_Allreduce(&local_eig_part, &max_eig_part, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // ── Check 3: Schur partition identity ──────────────────────────────────
    double local_schur_part = 0.0;
    for (int k = 0; k < N_local; k++) {
        double err = std::abs(local_vgts[k].squaredNorm() -
                              (res_schur.A2_ax[k] + res_schur.A2_sh[k] +
                               res_schur.A2_rr[k] + res_schur.A2_sr[k]));
        local_schur_part = std::max(local_schur_part, err);
    }
    double max_schur_part = 0.0;
    MPI_Allreduce(&local_schur_part, &max_schur_part, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // ── Check 4: batch norm == sqrt(N * ||A||²) ────────────────────────────
    double local_norm2 = 0.0;
    for (const auto& M : local_vgts) local_norm2 += M.squaredNorm();
    double global_norm2 = 0.0;
    MPI_Allreduce(&local_norm2, &global_norm2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    const double batch_norm = std::sqrt(global_norm2);

    double A_ref_norm2 = 0.0;
    for (int k = 0; k < 9; k++) A_ref_norm2 += A_REF[k] * A_REF[k];
    const double expected_norm = std::sqrt(static_cast<double>(N_global) * A_ref_norm2);
    const double norm_relerr   = std::abs(batch_norm - expected_norm) / expected_norm;

    // ── Report (rank 0 only) ───────────────────────────────────────────────
    const double tol_entry = 1e-12;
    const double tol_part  = 1e-13;

    bool pass = true;

    if (rank == 0) {
        std::cout << std::scientific << std::setprecision(6);

        std::cout << "Mesh   : 8×8×8 hex,  H1 order " << order << "\n";
        std::cout << "Qpts   : " << nqp << " per element,  "
                  << N_global << " total\n";
        std::cout << "norm(A): " << std::sqrt(A_ref_norm2) << "\n\n";

        auto check = [&](const char* label, double err, double tol) {
            bool ok = (err <= tol);
            if (!ok) pass = false;
            std::cout << (ok ? "  PASS  " : "  FAIL  ")
                      << label << "  max_err=" << err
                      << "  tol=" << tol << "\n";
        };

        check("VGT extraction (entry-wise vs A_REF)", max_entry_err,  tol_entry);
        check("EIG  partition identity",              max_eig_part,   tol_part);
        check("Schur partition identity",             max_schur_part, tol_part);
        check("batch norm vs sqrt(N*||A||²)",         norm_relerr,    tol_entry);

        std::cout << "\n  norm(VGT batch) = " << batch_norm
                  << "  (expected " << expected_norm << ")\n\n";
        std::cout << (pass ? "ALL CHECKS PASSED\n" : "SOME CHECKS FAILED\n");
    }

    return pass ? 0 : 1;
}
