// victor_compute_error.cpp
#include <mfem.hpp>
#include <cmath>
#include <iostream>

using namespace mfem;

int main(int argc, char *argv[])
{
    // 1) User‐adjustable parameters
    int order = 2;         // polynomial degree
    int mx    = 10, my = 10; // mesh subdivisions in x and y
    double Lx = 1.0, Ly = 1.0; // domain [0,Lx] x [0,Ly]

    // 2) Build a 2D Cartesian mesh of QUADRILATERAL elements
    Mesh mesh = Mesh::MakeCartesian2D(mx, my,
                                      Element::QUADRILATERAL,
                                      /*generate_edges=*/true,
                                      Lx, Ly);

    // 3) Create an L2 (discontinuous) FE space of the given order
    L2_FECollection fec(order, mesh.Dimension());
    FiniteElementSpace fes(&mesh, &fec);

    // 4) Define the exact solution u_ex(x,y) = sin(2π x)
    FunctionCoefficient u_ex(
        [](const Vector &x){
            return std::sin(2.0 * M_PI * x[0]);
        }
    );

    // 5) Define its exact gradient ∇u_ex = [2π cos(2π x), 0]
    VectorFunctionCoefficient grad_ex(
        mesh.Dimension(),
        [](const Vector &x, Vector &g){
            g.SetSize(2);
            g[0] = 2.0 * M_PI * std::cos(2.0 * M_PI * x[0]);
            g[1] = 0.0;
        }
    );

    // 6) Project the exact solution into the L2 space
    GridFunction u_h(&fes);
    u_h.ProjectCoefficient(u_ex);

    // 7) Compute the L2‐norm of the gradient error
    //    (using MFEM’s built‐in default quadrature rules)
    double grad_err = u_h.ComputeGradError(&grad_ex);

    std::cout << "2D mesh (mx=" << mx << ", my=" << my << "),"
              << " L2‐space, order=" << order
              << " → ||∇u_ex − ∇u_h||_L2 = "
              << grad_err << std::endl;

    return 0;
}

