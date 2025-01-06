/*
 * test_mfem_conduit.cpp
 *
 * A comprehensive test program to verify MFEM's ConduitDataCollection integration.
 *
 * This program:
 * 1. Creates a simple MFEM mesh.
 * 2. Initializes ConduitDataCollection.
 * 4. Saves the mesh data to disk using Conduit.
 * 5. Loads the mesh data back from disk.
 * 6. Prints properties of the loaded mesh and solution.
 */

#include "mfem.hpp"

// Check to make sure mesh is periodic
template<typename T>
bool InArray(const T* begin, size_t sz, T i)
{
   const T *end = begin + sz;
   return std::find(begin, end, i) != end;
}

bool IndicesAreConnected(const mfem::Table &t, int i, int j)
{
   return InArray(t.GetRow(i), t.RowSize(i), j)
          && InArray(t.GetRow(j), t.RowSize(j), i);
}

void VerifyPeriodicMesh(mfem::Mesh *mesh);

// Initial condition
mfem::real_t u0_function(const mfem::Vector &x);

int num_pts = 4;

int main(int argc, char *argv[])
{
    // Initialize MPI
    mfem::Mpi::Init();

    // Initialize as mesh
    mfem::Mesh *init_mesh;
    mfem::Mesh *mesh;
    mfem::ParMesh *pmesh;
    mfem::Mesh *loaded_mesh;


    init_mesh = new mfem::Mesh(mfem::Mesh::MakeCartesian3D(num_pts,
                                               num_pts,
                                               num_pts,
                                               mfem::Element::HEXAHEDRON,
                                               2.0 * M_PI,
                                               2.0 * M_PI,
                                               2.0 * M_PI, false));

    mfem::Vector x_translation({2.0 * M_PI, 0.0, 0.0});
    mfem::Vector y_translation({0.0, 2.0 * M_PI, 0.0});
    mfem::Vector z_translation({0.0, 0.0, 2.0 * M_PI});

    std::vector<mfem::Vector> translations = {x_translation, y_translation, z_translation};

    mesh = new mfem::Mesh(mfem::Mesh::MakePeriodic(*init_mesh, init_mesh->CreatePeriodicVertexMapping(translations)));

    if (mfem::Mpi::Root())
    {
       VerifyPeriodicMesh(mesh);
    }

    // Define a translation function for the mesh nodes
    mfem::VectorFunctionCoefficient translate(mesh->Dimension(), 
                                              [&](const mfem::Vector &x_in, mfem::Vector &x_out)
                                        {
       double shift = -M_PI;

       x_out[0] = x_in[0] + shift; // Translate x-coordinate
       x_out[1] = x_in[1] + shift; // Translate y-coordinate
       if (mesh->Dimension() == 3){
         x_out[2] = x_in[2] + shift; // Translate z-coordinate
       } });

    mesh->Transform(translate);

    pmesh = new mfem::ParMesh(MPI_COMM_WORLD, *mesh);
    pmesh->Finalize(true);

    int par_ref_levels = 5;
    for (int lev = 0; lev < par_ref_levels; lev++)
    {
       pmesh->UniformRefinement();
    }

    if(mfem::Mpi::Root()){
      if (!pmesh)
      {
        mfem::out << "[ERROR] Failed to create MFEM mesh." << std::endl;
      }

    std::cout << "Created a simple MFEM mesh with " << std::endl;
    std::cout << pmesh->GetNE() << " elements and " << std::endl;
    std::cout << pmesh->GetNV() << " vertices." << std::endl;
    }

    mfem::FunctionCoefficient u0(u0_function);
    mfem::DG_FECollection fec(2, pmesh->Dimension(), mfem::BasisType::GaussLobatto);
    mfem::ParFiniteElementSpace *fes = new mfem::ParFiniteElementSpace(pmesh, &fec);
    mfem::ParGridFunction *u = new mfem::ParGridFunction(fes);
    u->ProjectCoefficient(u0);

    mfem::real_t u_inf_loc = u->Normlinf();
    mfem::real_t u_inf = mfem::GlobalLpNorm(mfem::infinity(), u_inf_loc, MPI_COMM_WORLD);
       
    if(mfem::Mpi::Root())
    {
        std::cout << "Computed L2 norm of the grid function: " << u_inf << std::endl;
    }

    std::string collection_name = "test_collection";

#ifdef MFEM_USE_CONDUIT
         // Create a parallel ConduitDataCollection
         mfem::ConduitDataCollection dc(MPI_COMM_WORLD, collection_name, pmesh);
#else
         MFEM_ABORT("Must build with MFEM_USE_CONDUIT=YES for binary output.");
#endif
    // Set the Conduit relay protocol (options include "hdf5", "json", "conduit_json", "conduit_bin")
    dc.SetProtocol("hdf5"); // Using "json" for human-readable output

    if(mfem::Mpi::Root()){
     std::cout << "Initialized ConduitDataCollection with protocol 'hdf5'." << std::endl;
    }

    {
        // Save the mesh and associated fields (if any)
        dc.RegisterField("solution", u);
        dc.SetPrecision(16);
        dc.SetCycle(0);
        dc.SetTime(0.0);
        dc.Save();

        if(mfem::Mpi::Root()){
          std::cout << "Mesh data saved using ConduitDataCollection." << std::endl;
        }
    }

    // Optional: Delete the original mesh to ensure that loading works correctly
    delete init_mesh;
    delete pmesh;

    // Create a parallel ConduitDataCollection for loading with mesh set to nullptr
    mfem::ConduitDataCollection dc_load(MPI_COMM_WORLD, collection_name, nullptr);

    // Set the Conduit relay protocol to match the one used during saving
    dc_load.SetProtocol("hdf5"); // Must match the protocol used during saving

    if(mfem::Mpi::Root()){
      std::cout << "Initialized ConduitDataCollection for loading with protocol 'hdf5'." << std::endl;
    }

      // Load the mesh and associated fields (if any)
      dc_load.Load();

      // Retrieve the loaded mesh
      loaded_mesh = dc_load.GetMesh();
      mfem::GridFunction *loaded_u = dc_load.GetField("solution");

      mfem::real_t u_inf_loc_loaded = u->Normlinf();
      mfem::real_t u_inf_loaded = mfem::GlobalLpNorm(mfem::infinity(), 
                                                    u_inf_loc_loaded, 
                                                    MPI_COMM_WORLD);
         
    if(mfem::Mpi::Root()){
      if (!loaded_mesh)
      {
        mfem::out << "[ERROR] Failed to create MFEM mesh." << std::endl;
      }

      std::cout << "Computed L2 norm of the loaded grid function: " << u_inf_loaded << std::endl;
      std::cout << "Mesh data loaded from ConduitDataCollection." << std::endl;
      std::cout << "\n--- Loaded Mesh Properties ---" << std::endl;
      std::cout << "Number of Elements: " << loaded_mesh->GetNE() << std::endl;
      std::cout << "Number of Vertices: " << loaded_mesh->GetNV() << std::endl;
      std::cout << "Dimension: " << loaded_mesh->Dimension() << std::endl;
    }

    if(mfem::Mpi::Root()){
      std::cout << "\nTest completed successfully." << std::endl;
    }

    return 0;
}

// Initial condition
double u0_function(const mfem::Vector &x)
{
   return sin(x[0]) * sin(x[1]) * cos(x[2]);
}

void VerifyPeriodicMesh(mfem::Mesh *mesh)
{
    int n = num_pts;
    const mfem::Table &e2e = mesh->ElementToElementTable();
    int n2 = n * n;

    std::cout << "Checking to see if mesh is periodic.." << std::endl;

    if (mesh->GetNV() == pow(n - 1, 3) + 3 * pow(n - 1, 2) + 3 * (n - 1) + 1) {
        std::cout << "Total number of vertices match a periodic mesh." << std::endl;
    } else {
        MFEM_ABORT("Mesh does not have the correct number of vertices for a periodic mesh.");
    }

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            // Check periodicity in z direction
            if (!IndicesAreConnected(e2e, i + j * n, i + j * n + n2 * (n - 1))) {
                MFEM_ABORT("Mesh is not periodic in the z direction.");
            }

            // Check periodicity in y direction
            if (!IndicesAreConnected(e2e, i + j * n2, i + j * n2 + n * (n - 1))) {
                MFEM_ABORT("Mesh is not periodic in the y direction.");
            }

            // Check periodicity in x direction
            if (!IndicesAreConnected(e2e, i * n + j * n2, i * n + j * n2 + n - 1)) {
                MFEM_ABORT("Mesh is not periodic in the x direction.");
            }
        }
    }
            
    std::cout << "Done checking... Periodic in all directions." << std::endl;
}


