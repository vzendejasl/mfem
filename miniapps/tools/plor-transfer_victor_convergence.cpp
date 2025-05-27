#include "mfem.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>

using namespace std;
using namespace mfem;

// Exact velocity field for projection
void u_exact(const Vector &x, Vector &u)
{
    double xi = x(0), yi = x(1), zi = x(2);
    u(0) = sin(2*M_PI*xi) * cos(2*M_PI*yi) * cos(2*M_PI*zi);
    u(1) = -cos(2*M_PI*xi) * sin(2*M_PI*yi) * cos(2*M_PI*zi);
    u(2) = 0.0;
}

// Kinetic energy diagnostic
double ComputeKineticEnergy(ParGridFunction &v)
{
    auto *fes = v.ParFESpace();
    ParBilinearForm mass(fes);
    mass.AddDomainIntegrator(new VectorMassIntegrator());
    mass.Assemble();
    mass.Finalize();
    return 0.5 * mass.ParInnerProduct(v, v); // assuming unit volume
}

// Computes the error in kinetic energy for given lref
// Returns: error, sets DOFs for both meshes and LOR mesh size h_min by reference
double ComputeKineticEnergyErrorForLref(
    int order, int lorder, int lref, bool useH1, bool use_pointwise_transfer,
    bool use_ea, int nx, int ny, int nz, int dim,
    double &dofs_ho, double &dofs_lor, double &h_min_lor, double &h_min_ho,
    const char *device_config = "cpu")
{
    double x1 = 0.0, x2 = 1.0;
    Mesh *init_mesh = new Mesh(Mesh::MakeCartesian3D(nx, ny, nz,
                                                     Element::HEXAHEDRON,
                                                     x2-x1, x2-x1, x2-x1));
    Vector x_translation({x2 - x1, 0.0, 0.0});
    Vector y_translation({0.0, x2 - x1, 0.0});
    Vector z_translation({0.0, 0.0, x2 - x1});
    std::vector<Vector> translations = {x_translation, y_translation, z_translation};
    Mesh periodic_mesh = Mesh(Mesh::MakePeriodic(*init_mesh,
                                                 init_mesh->CreatePeriodicVertexMapping(translations)));
    delete init_mesh;
    ParMesh mesh(MPI_COMM_WORLD, periodic_mesh);

    FiniteElementCollection *fec, *fec_lor;
    if (useH1) {
        fec = new H1_FECollection(order, dim);
        fec_lor = new H1_FECollection(lorder, dim);
    } else {
        fec = new L2_FECollection(order, dim);
        fec_lor = new L2_FECollection(lorder, dim);
    }

    ParFiniteElementSpace fes_ho(&mesh, fec, dim);
    int basis_lor = BasisType::GaussLobatto;
    ParMesh mesh_lor = ParMesh::MakeRefined(mesh, lref, basis_lor);
    ParFiniteElementSpace fes_lor(&mesh_lor, fec_lor, dim);

    ParGridFunction u_ho(&fes_ho);
    VectorFunctionCoefficient u_exact_coeff(dim, u_exact);
    u_ho.ProjectCoefficient(u_exact_coeff);

    double ho_ke = ComputeKineticEnergy(u_ho);

    GridTransfer *gt = nullptr;
    if (useH1 || use_pointwise_transfer)
        gt = new InterpolationGridTransfer(fes_ho, fes_lor);
    else
        gt = new L2ProjectionGridTransfer(fes_ho, fes_lor);

    gt->UseEA(use_ea);

    const Operator &R = gt->ForwardOperator();
    ParGridFunction u_lor(&fes_lor);
    R.Mult(u_ho, u_lor);

    double lor_ke = ComputeKineticEnergy(u_lor);

    double abs_error = std::abs(ho_ke - lor_ke);

    // Get DOFs for both spaces, and h_min for LOR mesh
    dofs_ho = fes_ho.GlobalTrueVSize();
    dofs_lor = fes_lor.GlobalTrueVSize();
    double h_max, kappa_min, kappa_max;
    mesh_lor.GetCharacteristics(h_min_lor, h_max, kappa_min, kappa_max);
    mesh.GetCharacteristics(h_min_ho, h_max, kappa_min, kappa_max);

    delete fec;
    delete fec_lor;
    delete gt;

    return abs_error;
}

int main(int argc, char *argv[])
{
    Mpi::Init(argc, argv);
    Hypre::Init();

    int order = 2;
    int max_lref = 5;
    int lorder = 0;
    bool vis = false;
    bool useH1 = false;
    int visport = 19916;
    bool use_pointwise_transfer = false;
    const char *device_config = "cpu";
    bool use_ea = false;
    int nx = 5, ny = 5, nz = 5;
    int dim = 3;
    string space;

    OptionsParser args(argc, argv);
    args.AddOption(&order, "-o", "--order",
                   "Finite element order (polynomial degree) or -1 for isoparametric space.");
    args.AddOption(&max_lref, "-lref", "--max-lor-refinement", "Max LOR refinement level.");
    args.AddOption(&lorder, "-lo", "--lor-order",
                   "LOR space order (polynomial degree, default 1 for H1, 0 for L2).");
    args.AddOption(&useH1, "-h1", "--use-h1", "-l2", "--use-l2",
                   "Use H1 spaces instead of L2.");
    args.AddOption(&use_pointwise_transfer, "-t", "--use-pointwise-transfer",
                   "-no-t", "--dont-use-pointwise-transfer",
                   "Use pointwise transfer operators instead of L2 projection.");
    args.AddOption(&device_config, "-d", "--device",
                   "Device configuration string, see Device::Configure().");
    args.AddOption(&use_ea, "-ea", "--ea-version", "-no-ea",
                   "--no-ea-version", "Use element assembly version.");
    args.AddOption(&nx, "-nx", "--num-x", "Number of mesh elements in x.");
    args.AddOption(&ny, "-ny", "--num-y", "Number of mesh elements in y.");
    args.AddOption(&nz, "-nz", "--num-z", "Number of mesh elements in z.");
    args.ParseCheck();

    Device device(device_config);
    if (Mpi::Root()) { device.Print(); }

    if (useH1 && lorder == 0) lorder = 1; // H1 requires lorder >= 1
    space = useH1 ? "H1" : "L2";

    if (Mpi::Root()) {
        cout << "# plor_convergence_rate study\n";
        args.PrintOptions(cout);
        cout << setw(8)  << "lref"
             << setw(14) << "DOFs_HO"
             << setw(14) << "h_min_HO"
             << setw(14) << "DOFs_LOR"
             << setw(14) << "h_min_LOR"
             << setw(18) << "KE Error"
             << setw(12) << "Rate"
             << endl << string(80, '-') << endl;
    }

    vector<double> lref_list, dofs_ho_list, dofs_lor_list, h_list_ho, h_list_lor, error_list, rate_list;
    double prev_error = 0.0, prev_h = 0.0;

    for (int lref = 1; lref <= max_lref; ++lref)
    {
        double dofs_ho = 0.0, dofs_lor = 0.0, h_min_ho = 0.0, h_min_lor = 0.0;
        double error = ComputeKineticEnergyErrorForLref(
            order, lorder, lref, useH1, use_pointwise_transfer,
            use_ea, nx, ny, nz, dim, dofs_ho, dofs_lor, h_min_lor, h_min_ho, device_config);

        double rate = 0.0;
        if (lref > 1 && error > 1e-16 && prev_error > 1e-16 && h_min_lor < prev_h && prev_h > 0.0)
            rate = log(error/prev_error) / log(h_min_lor/prev_h);

        if (Mpi::Root()) {
            cout << setw(8)  << lref
                 << setw(14) << (long long)dofs_ho
                 << setw(14) << h_min_ho
                 << setw(14) << (long long)dofs_lor
                 << setw(14) << h_min_lor
                 << setw(18) << error
                 << setw(12) << rate << endl;
        }

        lref_list.push_back(lref);
        dofs_ho_list.push_back(dofs_ho);
        dofs_lor_list.push_back(dofs_lor);
        h_list_ho.push_back(h_min_ho);
        h_list_lor.push_back(h_min_lor);
        error_list.push_back(error);
        rate_list.push_back(rate);

        prev_error = error;
        prev_h = h_min_lor;
    }

    if (Mpi::Root())
    {
        std::ofstream ofs("plor_convergence_rate.txt");
        ofs << "# lref DOFs_HO DOFs_LOR h_min_LOR KE_Error Rate" << std::endl;
        for (size_t i = 0; i < lref_list.size(); ++i)
        {
            ofs << lref_list[i] << " "
                << dofs_ho_list[i] << " "
                << h_list_ho[i] << " "
                << dofs_lor_list[i] << " "
                << h_list_lor[i] << " "
                << error_list[i] << " "
                << rate_list[i] << std::endl;
        }
        ofs.close();
    }

    Mpi::Finalize();
    return 0;
}

