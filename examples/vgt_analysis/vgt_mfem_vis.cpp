/*
 * vgt_mfem_vis.cpp
 *
 * Saves a VisIt data collection with all VGT fields for the modified
 * manufactured velocity field u_e = Bx + eps*u0, eps=0.05.
 *
 * Fields written:
 *
 *   vel_exact    — exact manufactured velocity u_e  (vector, 3-comp H1)
 *   vel_h        — H1 FE projection of u_e          (vector, 3-comp H1)
 *   vel_error    — vel_h - vel_exact                (vector, 3-comp H1)
 *   vgt_h        — FE velocity gradient A_h         (full tensor, 9-comp L2)
 *   vgt_exact    — analytical A_exact               (full tensor, 9-comp L2)
 *   grad_error   — ||A_h - A_exact||_F pointwise    (scalar L2)
 *   A2_ax        — axial straining strength  (Schur)  (scalar L2)
 *   A2_sh        — shear straining strength  (Schur)  (scalar L2)
 *   A2_rr        — rigid rotation  (Rortex)  (Schur)  (scalar L2)
 *   A2_sr        — shear-rotation coupling   (Schur)  (scalar L2)
 *   delta_norm   — |Delta_norm(A_exact)|    (degeneracy map)  (scalar L2)
 *
 * VisIt notes:
 *   - vgt_h and vgt_exact have vdim=9 → VisIt automatically classifies them
 *     as full asymmetric tensors.  Use "Tensor" plot for glyphs, or
 *     "Pseudocolor" to plot individual components (vgt_h_0 ... vgt_h_8).
 *   - A2_rr is the rotation (Rortex) strength — the most interesting field
 *     for identifying vortical structures.
 *   - Components are stored row-major: index k = 3*row + col.
 *
 * Build:
 *   make MFEM_CXX=/usr/local/bin/mpicxx vgt_mfem_vis
 *
 * Run:
 *   mpirun -n 1 ./vgt_mfem_vis
 *   mpirun -n 4 ./vgt_mfem_vis -o 2 -n 32
 *
 * Options:
 *   -o <int>   polynomial order (default 2)
 *   -n <int>   elements per side (default 16)
 *
 * Output:
 *   vgt_mfem_vis.mfem_root   <- open this file in VisIt
 */

#include "mfem.hpp"
#include "vgt_lapack.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>

using namespace mfem;
using namespace vgt_lapack;

// ── Constants ─────────────────────────────────────────────────────────────────
static constexpr double PI  = M_PI;
static constexpr double EPS = 0.05;

// ── Manufactured velocity u_e = Bx + eps*u0 ──────────────────────────────────
static void u_eps_func(const Vector& x, Vector& v)
{
    v[0] =  x[0] - 10.0*x[1]
           + EPS*(std::sin(2*PI*x[0]) + std::sin(4*PI*x[1]) + std::sin(6*PI*x[2]));
    v[1] = 10.0*x[0] + x[1]
           + EPS*(std::sin(6*PI*x[0]) + std::sin(2*PI*x[1]) + std::sin(4*PI*x[2]));
    v[2] = -2.0*x[2]
           + EPS*(std::sin(4*PI*x[0]) + std::sin(6*PI*x[1]) + std::sin(2*PI*x[2]));
}

// ── Analytical VGT A_e = B + eps*A0, column-major storage ────────────────────
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
        for (int j = 0; j < 3; j++) trA2 += A(i,j)*A(j,i);
    const double I2 = 0.5*(I1*I1 - trA2);
    const double I3 = A(0,0)*(A(1,1)*A(2,2)-A(1,2)*A(2,1))
                    - A(0,1)*(A(1,0)*A(2,2)-A(1,2)*A(2,0))
                    + A(0,2)*(A(1,0)*A(2,1)-A(1,1)*A(2,0));
    const double D  = I1*I1*I2*I2 - 4.0*I2*I2*I2
                    - 4.0*I1*I1*I1*I3 - 27.0*I3*I3 + 18.0*I1*I2*I3;
    return D / std::pow(std::max(1.0, std::sqrt(A.squaredNorm())), 6);
}

// ── Coefficients ──────────────────────────────────────────────────────────────

// 9-component FE VGT, row-major so VisIt reads it as a full tensor.
// VisIt tensor convention: component k = row k/3, col k%3.
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
                V[3*i + j] = grad(i, j);   // row-major
    }
};

// 9-component analytical VGT (same layout).
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
                V[3*i + j] = Ae(i, j);     // row-major
    }
};

// Pointwise gradient Frobenius error ||A_h - A_exact||_F.
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
                double d = gv(i, j) - Ae(i, j);
                err2 += d*d;
            }
        return std::sqrt(err2);
    }
};

// Pointwise |Delta_norm| of A_exact — identifies near-degenerate regions.
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

// Schur partition component (comp: 0=ax, 1=sh, 2=rr, 3=sr) of A_h.
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

// Schur partition component (comp: 0=ax, 1=sh, 2=rr, 3=sr) of A_exact.
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

// ── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    mfem::MPI_Session mpi(argc, argv);
    const int rank = mpi.WorldRank();
    const int size = mpi.WorldSize();

    // ── Options ───────────────────────────────────────────────────────────────
    int order = 2;
    int N     = 16;

    OptionsParser args(argc, argv);
    args.AddOption(&order, "-o", "--order",  "Polynomial order.");
    args.AddOption(&N,     "-n", "--mesh-n", "Elements per side (N^3 mesh).");
    args.Parse();
    if (!args.Good()) {
        if (rank == 0) args.PrintUsage(std::cout);
        return 1;
    }

    // ── Mesh + velocity projection ─────────────────────────────────────────
    Mesh serial = Mesh::MakeCartesian3D(N, N, N, Element::HEXAHEDRON);
    ParMesh pmesh(MPI_COMM_WORLD, serial);
    serial.Clear();

    H1_FECollection h1_fec(order, 3);
    ParFiniteElementSpace h1_fes(&pmesh, &h1_fec, 3);   // vector H1

    // H1 velocity fields
    ParGridFunction vel_h(&h1_fes);
    ParGridFunction vel_exact(&h1_fes);

    VectorFunctionCoefficient vcoeff(3, u_eps_func);
    vel_h.ProjectCoefficient(vcoeff);      // FE projection
    vel_exact.ProjectCoefficient(vcoeff);  // same field — exact representation

    // Velocity error: vel_h - vel_exact  (H1 space, same mesh)
    ParGridFunction vel_error(&h1_fes);
    vel_error  = vel_h;
    vel_error -= vel_exact;

    // ── L2 (DG) spaces for derived quantities ─────────────────────────────
    // Use same order as H1 so VisIt nodal DOF counts match.
    const int l2_order = order;
    L2_FECollection l2_fec(l2_order, 3);

    // Scalar L2 space
    ParFiniteElementSpace l2s_fes(&pmesh, &l2_fec);

    // 9-component L2 space for the full VGT tensor.
    // vdim=9 → VisIt classifies this as a full asymmetric 3x3 tensor.
    ParFiniteElementSpace l2t_fes(&pmesh, &l2_fec, 9, Ordering::byVDIM);

    // ── Derived GridFunctions ─────────────────────────────────────────────
    ParGridFunction vgt_h_gf    (&l2t_fes);   // FE VGT tensor
    ParGridFunction vgt_exact_gf(&l2t_fes);   // Exact VGT tensor
    ParGridFunction grad_err_gf (&l2s_fes);   // ||A_h - A_exact||_F
    ParGridFunction A2_ax_gf    (&l2s_fes);   // axial straining        (A_h)
    ParGridFunction A2_sh_gf    (&l2s_fes);   // shear straining        (A_h)
    ParGridFunction A2_rr_gf    (&l2s_fes);   // rigid rotation         (A_h)
    ParGridFunction A2_sr_gf    (&l2s_fes);   // shear-rotation         (A_h)
    ParGridFunction A2_ax_ex_gf (&l2s_fes);   // axial straining exact  (A_exact)
    ParGridFunction A2_sh_ex_gf (&l2s_fes);   // shear straining exact  (A_exact)
    ParGridFunction A2_rr_ex_gf (&l2s_fes);   // rigid rotation exact   (A_exact)
    ParGridFunction A2_sr_ex_gf (&l2s_fes);   // shear-rotation exact   (A_exact)
    ParGridFunction delta_norm_gf(&l2s_fes);  // degeneracy indicator

    if (rank == 0)
        std::cout << "Projecting fields onto L2 (order " << l2_order
                  << ") ... " << std::flush;

    VGTHCoeff      vgt_h_coeff   (vel_h);
    VGTExactCoeff  vgt_ex_coeff;
    GradErrorCoeff grad_err_coeff(vel_h);
    DeltaNormCoeff delta_coeff;
    PartCoeff      ax_coeff   (vel_h, 0), sh_coeff   (vel_h, 1),
                   rr_coeff   (vel_h, 2), sr_coeff   (vel_h, 3);
    PartExactCoeff ax_ex_coeff(0),        sh_ex_coeff(1),
                   rr_ex_coeff(2),        sr_ex_coeff(3);

    vgt_h_gf.ProjectCoefficient     (vgt_h_coeff);
    vgt_exact_gf.ProjectCoefficient (vgt_ex_coeff);
    grad_err_gf.ProjectCoefficient  (grad_err_coeff);
    delta_norm_gf.ProjectCoefficient(delta_coeff);
    A2_ax_gf.ProjectCoefficient     (ax_coeff);
    A2_sh_gf.ProjectCoefficient     (sh_coeff);
    A2_rr_gf.ProjectCoefficient     (rr_coeff);
    A2_sr_gf.ProjectCoefficient     (sr_coeff);
    A2_ax_ex_gf.ProjectCoefficient  (ax_ex_coeff);
    A2_sh_ex_gf.ProjectCoefficient  (sh_ex_coeff);
    A2_rr_ex_gf.ProjectCoefficient  (rr_ex_coeff);
    A2_sr_ex_gf.ProjectCoefficient  (sr_ex_coeff);

    if (rank == 0) std::cout << "done.\n";

    // ── Save VisIt data collection ─────────────────────────────────────────
    VisItDataCollection dc("vgt_mfem_vis", &pmesh);

    // Velocity fields (H1 — full resolution, VisIt sees as 3-vectors)
    dc.RegisterField("vel_exact",  &vel_exact);
    dc.RegisterField("vel_h",      &vel_h);
    dc.RegisterField("vel_error",  &vel_error);

    // Full VGT tensors (9-comp L2 — VisIt tensor plot)
    dc.RegisterField("vgt_h",      &vgt_h_gf);
    dc.RegisterField("vgt_exact",  &vgt_exact_gf);

    // Scalar diagnostic fields — FE (A_h)
    dc.RegisterField("grad_error",  &grad_err_gf);
    dc.RegisterField("A2_ax",       &A2_ax_gf);
    dc.RegisterField("A2_sh",       &A2_sh_gf);
    dc.RegisterField("A2_rr",       &A2_rr_gf);
    dc.RegisterField("A2_sr",       &A2_sr_gf);
    // Scalar diagnostic fields — exact (A_exact)
    dc.RegisterField("A2_ax_exact", &A2_ax_ex_gf);
    dc.RegisterField("A2_sh_exact", &A2_sh_ex_gf);
    dc.RegisterField("A2_rr_exact", &A2_rr_ex_gf);
    dc.RegisterField("A2_sr_exact", &A2_sr_ex_gf);
    dc.RegisterField("delta_norm",  &delta_norm_gf);

    dc.SetCycle(0);
    dc.SetTime(0.0);
    dc.Save();

    if (rank == 0) {
        std::cout << "\n";
        std::cout << "Saved: vgt_mfem_vis.mfem_root\n";
        std::cout << "\n";
        std::cout << "Open in VisIt:\n";
        std::cout << "  File > Open > vgt_mfem_vis.mfem_root\n";
        std::cout << "\n";
        std::cout << "Suggested plots:\n";
        std::cout << "  Pseudocolor -> A2_rr          : rotation strength (FE)\n";
        std::cout << "  Pseudocolor -> A2_rr_exact    : rotation strength (exact)\n";
        std::cout << "  Pseudocolor -> A2_ax          : axial straining (FE)\n";
        std::cout << "  Pseudocolor -> A2_ax_exact    : axial straining (exact)\n";
        std::cout << "  Pseudocolor -> A2_sh          : shear straining (FE)\n";
        std::cout << "  Pseudocolor -> A2_sh_exact    : shear straining (exact)\n";
        std::cout << "  Pseudocolor -> A2_sr          : shear-rotation (FE)\n";
        std::cout << "  Pseudocolor -> A2_sr_exact    : shear-rotation (exact)\n";
        std::cout << "  Pseudocolor -> grad_error     : ||A_h - A_exact||_F pointwise\n";
        std::cout << "  Pseudocolor -> delta_norm     : eigenvalue degeneracy map\n";
        std::cout << "  Vector      -> vel_h          : velocity glyphs / streamlines\n";
        std::cout << "  Tensor      -> vgt_h          : full VGT tensor glyphs (FE)\n";
        std::cout << "  Tensor      -> vgt_exact      : full VGT tensor glyphs (exact)\n";
        std::cout << "\n";
        std::cout << "Config:\n";
        std::cout << "  Mesh  : " << N << "^3 hex,  P" << order << "\n";
        std::cout << "  DOFs  : " << h1_fes.GlobalTrueVSize() << " (velocity)\n";
        std::cout << "  Ranks : " << size << "\n";
    }

    return 0;
}
