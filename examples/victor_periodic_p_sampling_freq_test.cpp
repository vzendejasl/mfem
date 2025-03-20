#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <cstdlib>

using namespace std;
using namespace mfem;

// Global mesh parameters
Vector bb_min, bb_max;
int nx = 5, ny = 5, nz = 5;
double freq = 2.0;  // Frequency for the analytic solution

// -----------------------------------------------------------------------------
// The analytic vector function used for projection.
// Although in this example we compute the analytic value directly, this
// function is available if you wish to project it into a ParGridFunction.
void u_exact_vec(const Vector &x, Vector &u)
{
   u.SetSize(3);
   u(0) = cos(2 * M_PI * x(0) * freq);
   u(1) = 0.0;
   u(2) = 0.0;
}

// -----------------------------------------------------------------------------
// This function samples each element using interior-only sampling.
// For each element, a uniform grid of (order+1)^3 points is used where the
// reference coordinate is computed as i/npts. The physical coordinates and
// analytic vector field values are computed, then gathered and written to a
// text file ("sample_points_interior.txt") for post-processing.
void OutputSamplePointsInterior(ParMesh *pmesh, int order, double time)
{
   int npts = order + 1;
   vector<double> local_x, local_y, local_z;
   vector<double> local_u0, local_u1, local_u2;

   for (int e = 0; e < pmesh->GetNE(); e++)
   {
      ElementTransformation *Trans = pmesh->GetElementTransformation(e);
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
               ip.Set3(x_ref, y_ref, z_ref);

               Vector phys_coords(Trans->GetSpaceDim());
               Trans->Transform(ip, phys_coords);
               double x_phys = phys_coords(0);
               double y_phys = phys_coords(1);
               double z_phys = phys_coords(2);

               // Compute the analytic vector field.
               double u0 = cos(2 * M_PI * x_phys * freq);
               double u1 = 0.0;
               double u2 = 0.0;

               local_x.push_back(x_phys);
               local_y.push_back(y_phys);
               local_z.push_back(z_phys);
               local_u0.push_back(u0);
               local_u1.push_back(u1);
               local_u2.push_back(u2);
            }
         }
      }
   }

   // Gather data from all processors.
   MPI_Comm comm = pmesh->GetComm();
   int rank, size;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &size);
   int local_num = local_x.size();
   vector<int> all_num_elements(size);
   vector<int> displs(size);

   MPI_Gather(&local_num, 1, MPI_INT, all_num_elements.data(), 1, MPI_INT, 0, comm);

   vector<double> all_x, all_y, all_z, all_u0, all_u1, all_u2;
   if (rank == 0)
   {
      int total = 0;
      displs[0] = 0;
      for (int i = 0; i < size; i++)
      {
         total += all_num_elements[i];
         if (i > 0)
         {
            displs[i] = displs[i - 1] + all_num_elements[i - 1];
         }
      }
      all_x.resize(total);
      all_y.resize(total);
      all_z.resize(total);
      all_u0.resize(total);
      all_u1.resize(total);
      all_u2.resize(total);
   }

   MPI_Gatherv(local_x.data(), local_num, MPI_DOUBLE,
               all_x.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
   MPI_Gatherv(local_y.data(), local_num, MPI_DOUBLE,
               all_y.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
   MPI_Gatherv(local_z.data(), local_num, MPI_DOUBLE,
               all_z.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
   MPI_Gatherv(local_u0.data(), local_num, MPI_DOUBLE,
               all_u0.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
   MPI_Gatherv(local_u1.data(), local_num, MPI_DOUBLE,
               all_u1.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
   MPI_Gatherv(local_u2.data(), local_num, MPI_DOUBLE,
               all_u2.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);

   // Rank 0 writes the gathered data to a text file.
   if (rank == 0)
   {
      ofstream ofs("sample_points_interior.txt");
      ofs << "# Sample Points (Interior Sampling)\n";
      ofs << "# Order = " << order << ", Frequency = " << freq << ", Time = " << time << "\n";
      ofs << "# x y z u0 u1 u2\n";
      for (size_t i = 0; i < all_x.size(); i++)
      {
         ofs << all_x[i] << " " << all_y[i] << " " << all_z[i] << " "
             << all_u0[i] << " " << all_u1[i] << " " << all_u2[i] << "\n";
      }
      ofs.close();
      cout << "Output sample points (interior) saved: sample_points_interior.txt" << endl;
   }
}

// -----------------------------------------------------------------------------
// This function samples each element using boundary–inclusive sampling.
// The reference coordinates are computed as i/(npts–1) (with a special case when npts==1),
// so that the sample points include the boundaries of the reference element.
// The physical coordinates and analytic vector field values are gathered and written
// to a text file ("sample_points_boundary.txt").
void OutputSamplePointsWithBoundary(ParMesh *pmesh, int order, double time)
{
   int npts = order + 1;
   vector<double> local_x, local_y, local_z;
   vector<double> local_u0, local_u1, local_u2;

   for (int e = 0; e < pmesh->GetNE(); e++)
   {
      ElementTransformation *Trans = pmesh->GetElementTransformation(e);
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
               ip.Set3(x_ref, y_ref, z_ref);

               Vector phys_coords(Trans->GetSpaceDim());
               Trans->Transform(ip, phys_coords);
               double x_phys = phys_coords(0);
               double y_phys = phys_coords(1);
               double z_phys = phys_coords(2);

               double u0 = cos(2 * M_PI * x_phys * freq);
               double u1 = 0.0;
               double u2 = 0.0;

               local_x.push_back(x_phys);
               local_y.push_back(y_phys);
               local_z.push_back(z_phys);
               local_u0.push_back(u0);
               local_u1.push_back(u1);
               local_u2.push_back(u2);
            }
         }
      }
   }

   MPI_Comm comm = pmesh->GetComm();
   int rank, size;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &size);
   int local_num = local_x.size();
   vector<int> all_num_elements(size);
   vector<int> displs(size);

   MPI_Gather(&local_num, 1, MPI_INT, all_num_elements.data(), 1, MPI_INT, 0, comm);

   vector<double> all_x, all_y, all_z, all_u0, all_u1, all_u2;
   if (rank == 0)
   {
      int total = 0;
      displs[0] = 0;
      for (int i = 0; i < size; i++)
      {
         total += all_num_elements[i];
         if (i > 0)
         {
            displs[i] = displs[i - 1] + all_num_elements[i - 1];
         }
      }
      all_x.resize(total);
      all_y.resize(total);
      all_z.resize(total);
      all_u0.resize(total);
      all_u1.resize(total);
      all_u2.resize(total);
   }

   MPI_Gatherv(local_x.data(), local_num, MPI_DOUBLE,
               all_x.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
   MPI_Gatherv(local_y.data(), local_num, MPI_DOUBLE,
               all_y.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
   MPI_Gatherv(local_z.data(), local_num, MPI_DOUBLE,
               all_z.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
   MPI_Gatherv(local_u0.data(), local_num, MPI_DOUBLE,
               all_u0.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
   MPI_Gatherv(local_u1.data(), local_num, MPI_DOUBLE,
               all_u1.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
   MPI_Gatherv(local_u2.data(), local_num, MPI_DOUBLE,
               all_u2.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);

   if (rank == 0)
   {
      ofstream ofs("sample_points_boundary.txt");
      ofs << "# Sample Points (Boundary-Inclusive Sampling)\n";
      ofs << "# Order = " << order << ", Frequency = " << freq << ", Time = " << time << "\n";
      ofs << "# x y z u0 u1 u2\n";
      for (size_t i = 0; i < all_x.size(); i++)
      {
         ofs << all_x[i] << " " << all_y[i] << " " << all_z[i] << " "
             << all_u0[i] << " " << all_u1[i] << " " << all_u2[i] << "\n";
      }
      ofs.close();
      cout << "Output sample points (boundary) saved: sample_points_boundary.txt" << endl;
   }
}

// -----------------------------------------------------------------------------
// Main: creates one mesh with one order and produces two text files for
// post-processing.
int main(int argc, char *argv[])
{
   // Initialize MPI.
   Mpi::Init(argc, argv);
   int myid = Mpi::WorldRank();

   // Parse command-line options.
   int order = 1;
   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "Finite element order.");
   args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the analytic solution.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
         args.PrintUsage(cout);
      return 1;
   }
   if (myid == 0)
      args.PrintOptions(cout);

   // Build a 3D unit-cube mesh with periodic boundaries.
   double x1 = 0.0, x2 = 1.0;
   double y1 = 0.0, y2 = 1.0;
   double z1 = 0.0, z2 = 1.0;

   Mesh *init_mesh = new Mesh(Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON, x2 - x1, y2 - y1, z2 - z1));

   // Create a periodic version of the mesh.
   Vector x_translation({x2 - x1, 0.0, 0.0});
   Vector y_translation({0.0, y2 - y1, 0.0});
   Vector z_translation({0.0, 0.0, z2 - z1});
   vector<Vector> translations = {x_translation, y_translation, z_translation};
   Mesh *periodic_mesh = new Mesh(Mesh::MakePeriodic(*init_mesh, init_mesh->CreatePeriodicVertexMapping(translations)));
   delete init_mesh;


   // Create the parallel mesh.
   ParMesh pmesh(MPI_COMM_WORLD, *periodic_mesh);
   delete periodic_mesh;
   pmesh.GetBoundingBox(bb_min, bb_max, order);

   double time = 0.0; // Set the simulation time if needed.

   // Output two sets of sample-point data.
   OutputSamplePointsInterior(&pmesh, order, time);
   OutputSamplePointsWithBoundary(&pmesh, order, time);

   // Finalize MPI.
   Mpi::Finalize();
   return 0;
}

