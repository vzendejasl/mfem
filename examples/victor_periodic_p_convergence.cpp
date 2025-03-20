//                       MFEM 3D Averaging Operator Convergence Test
//
// Compile with: make averaging_convergence_test_3d
//
// Sample runs:
//    srun -np 4 averaging_convergence_test_3d -o 1 -r 4
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

// Exact solution function
double u_exact_vec(const Vector &x, Vector &u);

// Function to compute the convergence rates
void ComputeAveragingConvergence(int order, int refinements);

void ComputeAveragingConvergenceVectorField(int order, int refinements);

// Function to compute errors at cell centers
void ComputeCellCenterErrors(ParGridFunction* sol, ParFiniteElementSpace* fes, ParMesh* pmesh,
                             double &global_error);

void ComputeCellCenterErrorsVector(ParGridFunction* sol, ParFiniteElementSpace* fes, ParMesh* pmesh,
                             double &global_error);

void ComputeElementCenterValues(ParGridFunction* sol, ParFiniteElementSpace* fes, ParMesh* pmesh);

void ComputeConvergenceVectorFieldEvenlySpacedValues(int order, int refinements);

void ComputeElementSpacedValues(ParGridFunction* sol, ParMesh* pmesh, int order, double &global_error);

void ComputeConvergenceVectorFieldEvenlySpacedValuesWithBoundary(int order, int refinements);

void ComputeElementSpacedValuesWithBoundary(ParGridFunction* sol, ParMesh* pmesh, int order, double &global_error);

// Global variables for mesh bounding box (used in your mesh generation code)
Vector bb_min, bb_max;

// Mesh parameters (as per your code)
int nx = 5, ny = 5, nz = 5;
double freq = 2.0;
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
   if (myid == 0) 
     std::cout << " Order of convergence study for scalar field evaulated at cell center" << std::endl;
   ComputeAveragingConvergence(order, refinements);

   if (myid == 0) 
     std::cout << " Order of convergence study for vector field evaulated at cell center" << std::endl;
   ComputeAveragingConvergenceVectorField(order, refinements);

   if (myid == 0) 
     std::cout << " Order of convergence study for vector field evaulated at uniformly based on order + 1" << std::endl;
   ComputeConvergenceVectorFieldEvenlySpacedValues(order, refinements);

   if (myid == 0) 
     std::cout << " Order of convergence study for vector field evaulated at uniformly based on order + 1 (including both boundaries)" << std::endl;
   ComputeConvergenceVectorFieldEvenlySpacedValuesWithBoundary(order, refinements);

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

void ComputeAveragingConvergenceVectorField(int order, int refinements)
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
      ParFiniteElementSpace fespace(&pmesh, &fec, dim);

      // Project the exact solution onto the finite element space
      ParGridFunction u(&fespace);
      VectorFunctionCoefficient u_ex_coeff(dim, u_exact_vec);
      u.ProjectCoefficient(u_ex_coeff);

      // Compute the error at cell centers
      double global_error;
      ComputeCellCenterErrorsVector(&u, &fespace, &pmesh, global_error);

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
      std::ofstream ofs("convergence_data_vec.txt");
      ofs << "DOFs h Avg_Error Rate" << std::endl;
      for (size_t i = 0; i < dofs_list.size(); ++i)
      {
         ofs << dofs_list[i] << " " << h_list[i] << " " << error_list[i] << " " << rate_list[i] << std::endl;
      }
      ofs.close();
   }

}

void ComputeConvergenceVectorFieldEvenlySpacedValues(int order, int refinements)
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
      ParFiniteElementSpace fespace(&pmesh, &fec, dim);

      // Project the exact solution onto the finite element space
      ParGridFunction u(&fespace);
      VectorFunctionCoefficient u_ex_coeff(dim, u_exact_vec);
      u.ProjectCoefficient(u_ex_coeff);

      // Compute the error at cell centers
      double global_error;
      ComputeElementSpacedValues(&u, &pmesh, order, global_error);

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
      std::ofstream ofs("convergence_data_vec_evenly_sampled.txt");
      ofs << "DOFs h Avg_Error Rate" << std::endl;
      for (size_t i = 0; i < dofs_list.size(); ++i)
      {
         ofs << dofs_list[i] << " " << h_list[i] << " " << error_list[i] << " " << rate_list[i] << std::endl;
      }
      ofs.close();
   }

}

void ComputeConvergenceVectorFieldEvenlySpacedValuesWithBoundary(int order, int refinements)
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
      ParFiniteElementSpace fespace(&pmesh, &fec, dim);

      // Project the exact solution onto the finite element space
      ParGridFunction u(&fespace);
      VectorFunctionCoefficient u_ex_coeff(dim, u_exact_vec);
      u.ProjectCoefficient(u_ex_coeff);

      // Compute the error at cell centers
      double global_error;
      ComputeElementSpacedValues(&u, &pmesh, order, global_error);

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
      std::ofstream ofs("convergence_data_vec_evenly_sampledWithBoundary.txt");
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
   // double val = 3.0 * kappa * kappa * (sin(kappa*x[0]) * 
   //                                     sin(kappa*x[1]) * 
   //                                     sin(kappa*x[2]));
   double val = 2 + sin(kappa*(x[0]))*
                    cos(kappa*(x[1]))*
                    cos(kappa*(x[2]));
   return val;
   
}



// Modified exact solution to include higher frequencies
double u_exact_vec(const Vector &x, Vector &u)
{
    real_t xi = x(0);
    real_t yi = x(1);
    real_t zi = x(2);

    u(0) =  sin(kappa*xi+0.01189) * 
            cos(kappa*yi+0.1189) * 
            cos(kappa*zi+0.31189);
    u(1) = -cos(kappa*xi+0.21189) *
            sin(kappa*yi+0.1189) *
            cos(kappa*zi+2.31189);
    u(2) = 0.0;
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

// Compute errors at cell centers
void ComputeCellCenterErrorsVector(ParGridFunction* sol, ParFiniteElementSpace* fes, ParMesh* pmesh,
                             double &global_error)
{
    // MPI setup
    MPI_Comm comm = pmesh->GetComm();
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);


    double global_error_x, global_error_y, global_error_z;
    double local_error_x, local_error_y, local_error_z;

    local_error_x = 0.0;
    local_error_y = 0.0;
    local_error_z = 0.0;

    // Set the integration point to the center of the reference element
    IntegrationPoint ip;
    ip.Set3(0.5, 0.5, 0.5); // Center of element

    // Loop over local elements
    for (int i = 0; i < pmesh->GetNE(); i++)
    {
        // Get element transformation
        ElementTransformation *Trans = pmesh->GetElementTransformation(i);

        // Evaluate at center
        Trans->SetIntPoint(&ip);

        // Get vector dimension (should be 3 for velocity)
        int vdim = fes->GetVDim();
        Vector u_val(vdim);
        
        // Get vector value at ip
        sol->GetVectorValue(*Trans, ip, u_val);

        double u_x = u_val(0);
        double u_y = u_val(1);
        double u_z = u_val(2);
        
        // Physical coordinates of element center
        Vector phys_coords(3);
        Trans->Transform(ip, phys_coords);

        Vector u_exact(3); 
        u_exact_vec(phys_coords,u_exact);

        // Compute error
        double error_x = u_x - u_exact(0);
        double error_y = u_y - u_exact(1);
        double error_z = u_z - u_exact(2);

        // Accumulate squared error
        local_error_x += error_x * error_x;
        local_error_y += error_y * error_y;
        local_error_z += error_z * error_z;
    }

   // Sum the local errors over all processors
   MPI_Allreduce(&local_error_x, &global_error_x, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
   MPI_Allreduce(&local_error_y, &global_error_y, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
   MPI_Allreduce(&local_error_z, &global_error_z, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());

   global_error = std::max(std::max(global_error_x, global_error_y), global_error_z);
   global_error = sqrt(global_error/pmesh->GetGlobalNE());
}

void ComputeElementSpacedValues(ParGridFunction* sol,
                                ParMesh* pmesh, int order, double &global_error)
{
   // MPI setup
   MPI_Comm comm = pmesh->GetComm();
   int rank, size;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &size);

   double global_error_x, global_error_y, global_error_z;
   double local_error_x, local_error_y, local_error_z;

   local_error_x = 0.0;
   local_error_y = 0.0;
   local_error_z = 0.0;


   // Instead of one integration point (the element center), we will sample each element
   // on an N x N x N grid, where N = ctx.order + 1.
   int npts = order + 1;  // number of sample points per coordinate direction

   FiniteElementSpace *fes = sol->FESpace();
   int vdim = fes->GetVDim();

   // Loop over local elements
   for (int e = 0; e < pmesh->GetNE(); e++)
   {
      ElementTransformation *Trans = pmesh->GetElementTransformation(e);
      
      // For each element, loop over a uniform grid of points in the reference element [0,1]^d.
      for (int iz = 0; iz < npts; iz++)
      {
         double z_ref = static_cast<double>(iz) / npts;
         for (int iy = 0; iy < npts; iy++)
         {
            double y_ref = static_cast<double>(iy) / npts;
            for (int ix = 0; ix < npts; ix++)
            {
               double x_ref = static_cast<double>(ix) / npts;

               IntegrationPoint ip;
               ip.Set3(x_ref, y_ref, z_ref); // sample point in reference element

               // Get the physical coordinates for this sample point
               Vector phys_coords(Trans->GetSpaceDim());
               Trans->Transform(ip, phys_coords);

               double x_physical = phys_coords(0);
               double y_physical = phys_coords(1);
               double z_physical = phys_coords(2);

               // Evaluate the solution at the sample point
               Vector u_val(vdim);
               sol->GetVectorValue(*Trans, ip, u_val);

               double u_x = u_val(0);
               double u_y = u_val(1);
               double u_z = u_val(2);

               // Reference position (in the reference element)
               int ref_dim = fes->GetVDim(); // Dimension of the reference element
               Vector ref_pos(3);

               ref_pos(0) = ip.x; // x-coordinate
               ref_pos(1) = ip.y; // y-coordinate
               ref_pos(2) = ip.z; // z-coordinate

               // Physical position (mapped to the physical element)
               Vector phys_pos(Trans->GetSpaceDim()); // Physical space dimension
               Trans->Transform(ip, phys_pos); // Maps reference -> physical

               Vector u_exact(3); 
               u_exact_vec(phys_coords, u_exact);

               // Compute error
               double error_x = u_x - u_exact(0);
               double error_y = u_y - u_exact(1);
               double error_z = u_z - u_exact(2);

               // Accumulate squared error
               local_error_x += error_x * error_x;
               local_error_y += error_y * error_y;
               local_error_z += error_z * error_z;

            } // ix
         } // iy
      } // iz
   } // for each local element

   // Sum the local errors over all processors
   MPI_Allreduce(&local_error_x, &global_error_x, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
   MPI_Allreduce(&local_error_y, &global_error_y, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
   MPI_Allreduce(&local_error_z, &global_error_z, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());

   // global_error = std::max(std::max(global_error_x, global_error_y), global_error_z);
   global_error = global_error_x + global_error_y + global_error_z;
   global_error = sqrt(global_error/(pmesh->GetGlobalNE()*npts));
}

void ComputeElementSpacedValuesWithBoundary(ParGridFunction* sol,
                                ParMesh* pmesh, int order, double &global_error)
{
   // MPI setup
   MPI_Comm comm = pmesh->GetComm();
   int rank, size;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &size);

   double global_error_x, global_error_y, global_error_z;
   double local_error_x, local_error_y, local_error_z;

   local_error_x = 0.0;
   local_error_y = 0.0;
   local_error_z = 0.0;

   // Instead of one integration point (the element center), we will sample each element
   // on an N x N x N grid, where N = ctx.order + 1.
   int npts = order + 1;  // number of sample points per coordinate direction

   FiniteElementSpace *fes = sol->FESpace();
   int vdim = fes->GetVDim();

   // Loop over local elements
   for (int e = 0; e < pmesh->GetNE(); e++)
   {
      ElementTransformation *Trans = pmesh->GetElementTransformation(e);
      // For each element, loop over a uniform grid of points in the reference element [0,1]^d.
      for (int iz = 0; iz < npts; iz++)
      {
      
         double z_ref = (npts == 1) ? 0.5 : static_cast<double>(iz) / (npts - 1);
         for (int iy = 0; iy < npts; iy++)
         {
            double y_ref = (npts == 1) ? 0.5 : static_cast<double>(iy) / (npts - 1);
            for (int ix = 0; ix < npts; ix++)
            {
               double x_ref = (npts == 1) ? 0.5 : static_cast<double>(ix) / (npts - 1);

               IntegrationPoint ip;
               ip.Set3(x_ref, y_ref, z_ref); // sample point in reference element

               // Get the physical coordinates for this sample point
               Vector phys_coords(Trans->GetSpaceDim());
               Trans->Transform(ip, phys_coords);

               double x_physical = phys_coords(0);
               double y_physical = phys_coords(1);
               double z_physical = phys_coords(2);

               // Evaluate the solution at the sample point
               Vector u_val(vdim);
               sol->GetVectorValue(*Trans, ip, u_val);

               double u_x = u_val(0);
               double u_y = u_val(1);
               double u_z = u_val(2);

               // Reference position (in the reference element)
               int ref_dim = fes->GetVDim(); // Dimension of the reference element
               Vector ref_pos(3);

               ref_pos(0) = ip.x; // x-coordinate
               ref_pos(1) = ip.y; // y-coordinate
               ref_pos(2) = ip.z; // z-coordinate

               // Physical position (mapped to the physical element)
               Vector phys_pos(Trans->GetSpaceDim()); // Physical space dimension
               Trans->Transform(ip, phys_pos); // Maps reference -> physical

               Vector u_exact(3); 
               u_exact_vec(phys_coords, u_exact);

               // Compute error
               double error_x = u_x - u_exact(0);
               double error_y = u_y - u_exact(1);
               double error_z = u_z - u_exact(2);

               // Accumulate squared error
               local_error_x += error_x * error_x;
               local_error_y += error_y * error_y;
               local_error_z += error_z * error_z;

            } // ix
         } // iy
      } // iz
   } // for each local element

   // Sum the local errors over all processors
   MPI_Allreduce(&local_error_x, &global_error_x, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
   MPI_Allreduce(&local_error_y, &global_error_y, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
   MPI_Allreduce(&local_error_z, &global_error_z, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());

   // global_error = std::max(std::max(global_error_x, global_error_y), global_error_z);
   global_error = global_error_x + global_error_y + global_error_z;
   global_error = sqrt(global_error/(pmesh->GetGlobalNE()*npts));
}
