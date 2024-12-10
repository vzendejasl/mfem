//                       MFEM 3D Averaging Operator Convergence Test
//
// Compile with: make averaging_convergence_test_3d
//
// Sample runs:
//    mpirun -np 4 averaging_convergence_test_3d -o 1 -r 4
//
// Description:  This script tests whether an averaging operator, such as computing
//               the solution at element centers, converges at the same rate as the
//               polynomial order in 3D. It uses a periodic mesh generated inline
//               and projects a known exact function onto the finite element space.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;
using namespace mfem;

// Exact solution function
double u_exact(const Vector &x);

// Function to compute the convergence rates
void ComputeAveragingConvergence(int order, int refinements);

// Function to compute errors at cell centers
void ComputeCellCenterErrors(ParGridFunction* sol, ParFiniteElementSpace* fes, ParMesh* pmesh,
                             double &global_error);

void ComputeElementCenterValues(ParGridFunction* sol, ParFiniteElementSpace* fes, ParMesh* pmesh);

// Global variables for mesh bounding box (used in your mesh generation code)
Vector bb_min, bb_max;

// Mesh parameters (as per your code)
int nx = 5, ny = 5, nz = 5;
double freq = 1.0;
double kappa;;

int main(int argc, char *argv[])
{
   // Initialize MPI.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();

   // Parse command-line options.
   int order = 1;
   int refinements = 4;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "Finite element order.");
   args.AddOption(&refinements, "-r", "--refinements", "Number of total refinements.");
   args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact"
                     " solution.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   kappa = freq * M_PI;

   // Perform convergence study
   ComputeAveragingConvergence(order, refinements);

   // Finalize MPI.
   Mpi::Finalize();

   return 0;
}

void ComputeAveragingConvergence(int order, int refinements)
{
   int myid = Mpi::WorldRank();
   int num_procs = Mpi::WorldSize();

   // Create list to append data to text file
   std::vector<HYPRE_Int> dofs_list;
   std::vector<double> h_list, error_list, rate_list;

   // Header for output
   if (myid == 0)
   {
      cout << setw(8) << "DOFs" << setw(16) << "h" << setw(16) << "Avg Error"
           << setw(16) << "Rate" << endl;
      cout << string(64, '-') << endl;
   }

   double h0 = 0.0;
   double error0 = 0.0;

   for (int level = 0; level <= refinements; ++level)
   {
      // Adjust mesh parameters based on refinement level
      int ref_factor = 1 << level;
      int current_nx = nx * ref_factor;
      int current_ny = ny * ref_factor;
      int current_nz = nz * ref_factor;

      // Mesh boundaries (unit cube)
      double x1 = 0.0, x2 = 1.0;
      double y1 = 0.0, y2 = 1.0;
      double z1 = 0.0, z2 = 1.0;

      Mesh *mesh;
      Mesh *init_mesh;

      // 3D Mesh
      if (current_nz > 1)
      {
            init_mesh = new Mesh(Mesh::MakeCartesian3D(current_nx,
                                                       current_ny,
                                                       current_nz,
                                                       Element::HEXAHEDRON,
                                                       x2 - x1,
                                                       y2 - y1,
                                                       z2 - z1));
         Vector x_translation({x2 - x1, 0.0, 0.0});
         Vector y_translation({0.0, y2 - y1, 0.0});
         Vector z_translation({0.0, 0.0, z2 - z1});

         std::vector<Vector> translations = {x_translation, y_translation, z_translation};

         mesh = new Mesh(Mesh::MakePeriodic(*init_mesh,
                                            init_mesh->CreatePeriodicVertexMapping(translations)));

         delete init_mesh;
      }
      else
      {
         // Handle 2D mesh if needed
      }

      // Define a translation function for the mesh nodes
      VectorFunctionCoefficient translate_set_mesh(mesh->Dimension(), [&](const Vector &x_in, Vector &x_out){

         x_out[0] = x_in[0] + x1; // Translate x-coordinate
         x_out[1] = x_in[1] + y1; // Translate y-coordinate
         if (mesh->Dimension() == 3)
         {
            x_out[2] = x_in[2] + z1; // Translate z-coordinate
         }
      });

      // Apply translation to the mesh
      mesh->Transform(translate_set_mesh);

      int dim = mesh->Dimension();


      // Refine the mesh in serial if needed
      int ser_ref_levels = 0; // No additional serial refinements
      for (int lev = 0; lev < ser_ref_levels; lev++)
      {
         mesh->UniformRefinement();
      }
      if (mesh->NURBSext)
      {
         mesh->SetCurvature(max(order, 1));
      }

      // Create the parallel mesh
      ParMesh pmesh(MPI_COMM_WORLD, *mesh);
      delete mesh;

      int par_ref_levels = 0; // No additional parallel refinements
      for (int lev = 0; lev < par_ref_levels; lev++)
      {
         pmesh.UniformRefinement();
      }

      pmesh.GetBoundingBox(bb_min, bb_max, max(order, 1));

      // Finite element space
      H1_FECollection fec(order, dim);
      ParFiniteElementSpace fespace(&pmesh, &fec);

      // Project the exact solution onto the finite element space
      ParGridFunction u(&fespace);
      FunctionCoefficient u_ex_coeff(u_exact);
      u.ProjectCoefficient(u_ex_coeff);

      // Compute the error at cell centers
      double global_error;
      ComputeCellCenterErrors(&u, &fespace, &pmesh, global_error);

      ComputeElementCenterValues(&u, &fespace, &pmesh);

      // Mesh size h
      double h_min, h_max, kappa_min, kappa_max;
      pmesh.GetCharacteristics(h_min, h_max, kappa_min, kappa_max);

      // Compute convergence rate
      double rate = 0.0;
      if (level > 0 && global_error > 1e-16)
      {
         rate = log(global_error/error0) / log(h_min/h0);
      }

      // Compute global DOFs and global NE on all processes
      // (These call do an MPI reduce internally)
      HYPRE_Int global_true_vsize = fespace.GlobalTrueVSize();
      HYPRE_Int global_ne = pmesh.GetGlobalNE();

      // Output results
      if (myid == 0)
      {
          cout << setw(8) << global_true_vsize << setw(16) << h_min
               << setw(16) << global_error << setw(16) << rate << endl;
      }

      dofs_list.push_back(global_true_vsize);
      h_list.push_back(h_min);
      error_list.push_back(global_error);
      rate_list.push_back(rate);

      // Update previous values
      h0 = h_min;
      error0 = global_error;
   }

   // After the loop, write data to a text file from process 0
   if (myid == 0)
   {
      std::ofstream ofs("convergence_data.txt");
      ofs << "DOFs h Avg_Error Rate" << std::endl;
      for (size_t i = 0; i < dofs_list.size(); ++i)
      {
         ofs << dofs_list[i] << " " << h_list[i] << " " << error_list[i] << " " << rate_list[i] << std::endl;
      }
      ofs.close();
   }

}

// Modified exact solution to include higher frequencies
double u_exact(const Vector &x)
{
   double val = 3.0 * kappa * kappa * (sin(kappa * x[0]) * sin(kappa * x[1]) * sin(
                                    kappa * x[2]));
   return val;
   
}

// // Modified exact solution to include higher frequencies
// double u_exact(const Vector &x)
// {
//    double val = sin(kappa * x[0]);
//    return val;
//    
// }


// double u_exact(const Vector &x)
// {
//    // Shifted anti-symmetric function that is non-zero at element centers
//    double val = sin(M_PI * (x[0] - 0.3)) * sin(M_PI * (x[1] - 0.3)) * sin(M_PI * (x[2] - 0.3));
//    return val;
// }

// double u_exact(const Vector &x)
// {
//    double val = sin(2.5 * M_PI * x[0]) + sin(3.5 * M_PI * x[1]) + sin(4.5 * M_PI * x[2]);
//    return val;
// }



// Compute errors at cell centers
void ComputeCellCenterErrors(ParGridFunction* sol, ParFiniteElementSpace* fes, ParMesh* pmesh,
                             double &global_error)
{
   int dim = pmesh->Dimension();
   int NE = pmesh->GetNE();
   double local_error = 0.0;
   // double local_exact_value = 0.0;
   // double global_exact_value_sum = 0.0;

   // Set the integration point to the center of the reference element
   IntegrationPoint ip;
   ip.Set3(0.5001, 0.5, 0.5); // Center for 3D element

   // Loop over local elements
   for (int i = 0; i < NE; i++)
   {
       // Get the element transformation
       ElementTransformation *Trans = pmesh->GetElementTransformation(i);

       // Evaluate the solution at the element center
       Trans->SetIntPoint(&ip);
       double value = sol->GetValue(*Trans, ip);

       // Transform the reference point to physical coordinates
       Vector phys_coords(dim);
       Trans->Transform(ip, phys_coords);

       // Compute exact solution at this point
       double exact_value = u_exact(phys_coords);

       // Compute error
       double error = value - exact_value;

       // Accumulate squared error
       local_error += error * error;

       // Accumulate the exact solution
       // local_exact_value += exact_value;
   }

   // Sum the local errors over all processors
   MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
   // MPI_Allreduce(&local_exact_value, &global_exact_value_sum, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
   global_error = sqrt(global_error / pmesh->GetGlobalNE()); // Average error per element
   // global_error = sqrt(global_error / global_exact_value_sum); // Average error per element
}

void ComputeElementCenterValues(ParGridFunction* sol, ParFiniteElementSpace* fes, ParMesh* pmesh)
{
    // Local arrays to store the data
    std::vector<double> local_x, local_y, local_z, local_value;

    // Set the integration point to the center of the reference element
    IntegrationPoint ip;
    ip.Set3(0.5, 0.5, 0.5);  // Center of the reference element

    // Loop over local elements
    for (int i = 0; i < pmesh->GetNE(); i++)
    {
        // Get the element transformation
        ElementTransformation *Trans = pmesh->GetElementTransformation(i);

        // Evaluate the solution at the element center
        Trans->SetIntPoint(&ip);
        double value = sol->GetValue(*Trans, ip);

        // Transform the reference point to physical coordinates
        Vector phys_coords(3);
        Trans->Transform(ip, phys_coords);

        double x_center = phys_coords[0];
        double y_center = phys_coords[1];
        double z_center = phys_coords[2];

        // Store the data
        local_x.push_back(x_center);
        local_y.push_back(y_center);
        local_z.push_back(z_center);
        local_value.push_back(value);
    }

    // MPI setup
    MPI_Comm comm = pmesh->GetComm();
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Gather all data on rank 0
    std::vector<double> all_x, all_y, all_z, all_value;
    int local_num_elements = local_x.size();
    std::vector<int> all_num_elements(size);
    std::vector<int> displs(size);

    MPI_Gather(&local_num_elements, 1, MPI_INT, 
        all_num_elements.data(), 1, MPI_INT, 0, comm);

    if (rank == 0)
    {
        int total_elements = 0;
        displs[0] = 0;
        for (int i = 0; i < size; ++i)
        {
            total_elements += all_num_elements[i];
            if (i > 0)
            {
                displs[i] = displs[i - 1] + all_num_elements[i - 1];
            }
        }

        all_x.resize(total_elements);
        all_y.resize(total_elements);
        all_z.resize(total_elements);
        all_value.resize(total_elements);
    }

    MPI_Gatherv(local_x.data(), local_num_elements, MPI_DOUBLE, 
        all_x.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(local_y.data(), local_num_elements, MPI_DOUBLE, 
        all_y.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(local_z.data(), local_num_elements, MPI_DOUBLE, 
        all_z.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(local_value.data(), local_num_elements, MPI_DOUBLE, 
        all_value.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);

    // Write the data to a file in a human-readable format on rank 0
    if (rank == 0)
    {
        std::ofstream ofs("element_centers.txt");
        for (size_t i = 0; i < all_x.size(); ++i)
        {
            ofs << all_x[i] << " " << all_y[i] << " " << all_z[i] << " " << all_value[i] << std::endl;
        }
        ofs.close();
    }
    
}

/*
// Compute errors at cell centers
void ComputeCellCenterErrors(ParGridFunction* sol, ParFiniteElementSpace* fes, ParMesh* pmesh,
                             double &global_error)
{
    // Local arrays to store the data
    std::vector<double> local_x, local_y, local_z, local_value;

    // Set the integration point to the center of the reference element
    IntegrationPoint ip;
    ip.Set3(0.50001, 0.5, 0.5);  // Center of the reference element

    // Loop over local elements
    for (int i = 0; i < pmesh->GetNE(); i++)
    {
        // Get the element transformation
        ElementTransformation *Trans = pmesh->GetElementTransformation(i);

        // Evaluate the solution at the element center
        Trans->SetIntPoint(&ip);
        double value = sol->GetValue(*Trans, ip);

        // Transform the reference point to physical coordinates
        Vector phys_coords(3);
        Trans->Transform(ip, phys_coords);

        double x_center = phys_coords[0];
        double y_center = phys_coords[1];
        double z_center = phys_coords[2];

        // Store the data
        local_x.push_back(x_center);
        local_y.push_back(y_center);
        local_z.push_back(z_center);
        local_value.push_back(value);
    }

    // MPI setup
    MPI_Comm comm = pmesh->GetComm();
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Gather all data on rank 0
    std::vector<double> all_x, all_y, all_z, all_value;
    int local_num_elements = local_x.size();
    std::vector<int> all_num_elements(size);
    std::vector<int> displs(size);

    MPI_Gather(&local_num_elements, 1, MPI_INT, 
        all_num_elements.data(), 1, MPI_INT, 0, comm);

    if (rank == 0)
    {
        int total_elements = 0;
        displs[0] = 0;
        for (int i = 0; i < size; ++i)
        {
            total_elements += all_num_elements[i];
            if (i > 0)
            {
                displs[i] = displs[i - 1] + all_num_elements[i - 1];
            }
        }

        all_x.resize(total_elements);
        all_y.resize(total_elements);
        all_z.resize(total_elements);
        all_value.resize(total_elements);
    }

    MPI_Gatherv(local_x.data(), local_num_elements, MPI_DOUBLE, 
        all_x.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(local_y.data(), local_num_elements, MPI_DOUBLE, 
        all_y.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(local_z.data(), local_num_elements, MPI_DOUBLE, 
        all_z.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(local_value.data(), local_num_elements, MPI_DOUBLE, 
        all_value.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);

    double local_error = 0.0;
    int counter = 0;
    // Compute averages on rank 0
    if (rank == 0)
    {
        std::map<double, std::pair<double, int>> x_value_count_map;

        // Accumulate sums and counts
        for (size_t i = 0; i < all_x.size(); ++i)
        {
            double x = all_x[i];
            double value = all_value[i];

            if (x_value_count_map.find(x) == x_value_count_map.end())
            {
                x_value_count_map[x] = std::make_pair(value, 1);
            }
            else
            {
                x_value_count_map[x].first += value;
                x_value_count_map[x].second += 1;
            }
        }

        // Compute averages
        std::ofstream ofs("averaged_values.txt");
        for (const auto& entry : x_value_count_map)
        {
            double x = entry.first;
            double sum_values = entry.second.first;
            int count = entry.second.second;
            double average = sum_values / count;

            ofs << x << " " << average << std::endl;

            // Transform the reference point to physical coordinates
            Vector phys_coords(1);
            phys_coords[0] = x;

            // Compute exact solution at this point
            double exact_value = u_exact(phys_coords);

            // Compute error
            double error = average - exact_value;

            // Accumulate squared error
            local_error += error * error;
            counter += 1;
    

        }
        ofs.close();
        global_error = sqrt(local_error / counter); // Average error per element
    }


}
*/
