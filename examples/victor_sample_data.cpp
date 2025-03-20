// sample_checkpoint_debug.cpp
//
// A simplified program that loads a checkpoint state using VisItDataCollection,
// makes a deep copy of the "velocity" field, and then uniformly samples the
// velocity field using the provided SampledDataUniform function.
// Major debug print statements are printed only on the root process.
// 
// Usage: mpirun -n <num_procs> ./sample_checkpoint_debug <visit_dir> <cycle_number>
//
// Compile with your MFEM build settings.

#include "mfem.hpp"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>

using namespace mfem;
using namespace std;

//------------------------------------------------------------------------------
// Minimal context structure to supply parameters for the sampling function.
struct s_NavierContext
{
   int order = 2;
};

// Global context variable.
s_NavierContext ctx;

//------------------------------------------------------------------------------
// Uniform sampling function (prints major messages only on root).
void SampledDataUniform(ParGridFunction* sol,
                                ParMesh* pmesh,
                                int step,
                                double time,
                                const std::string &suffix)
{
   int rank, size;
   MPI_Comm comm = pmesh->GetComm();
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &size);

   if (rank == 0)
   {
      mfem::out << " Starting uniform sampling." << endl;
   }

   // Construct the main directory name with suffix.
   std::string main_dir = "SampledData" + suffix +
                            "P" + std::to_string(ctx.order);

   // Create subdirectory for this cycle step.
   std::string cycle_dir = main_dir + "/cycle_" + std::to_string(step);

   // Construct the filename.
   std::string fname = cycle_dir + "/sampled_data_" + std::to_string(step) + ".txt";

   if (rank == 0)
   {
      mfem::out << " Creating directories: " << main_dir << " and " << cycle_dir << endl;
      if (system(("mkdir -p " + main_dir).c_str()) != 0)
         cerr << " Error creating " << main_dir << " directory!" << endl;
      if (system(("mkdir -p " + cycle_dir).c_str()) != 0)
         cerr << " Error creating " << cycle_dir << " directory!" << endl;
   }

   MPI_Barrier(comm);

   int npts = ctx.order + 1;  // sample points per coordinate direction.
   if (rank == 0)
   {
      mfem::out << " Using " << npts << " sample points per coordinate." << endl;
      mfem::out << "Number of elements: " << pmesh->GetNE() << endl;
   }

   // Local arrays to store sample data.
   vector<double> local_x, local_y, local_z;
   vector<double> local_velx, local_vely, local_velz;

   FiniteElementSpace *fes = sol->FESpace();
   int vdim = fes->GetVDim();

   // Loop over local elements and sample.
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
               double x_center = phys_coords(0);
               double y_center = phys_coords(1);
               double z_center = (phys_coords.Size() > 2) ? phys_coords(2) : 0.0;

               Vector u_val(vdim);
               sol->GetVectorValue(*Trans, ip, u_val);
               double u_x = u_val(0);
               double u_y = u_val(1);
               double u_z = (vdim > 2) ? u_val(2) : 0.0;

               local_x.push_back(x_center);
               local_y.push_back(y_center);
               local_z.push_back(z_center);
               local_velx.push_back(u_x);
               local_vely.push_back(u_y);
               local_velz.push_back(u_z);
            }
         }
      }
   }
   if (rank == 0)
   {
      mfem::out << " Finished sampling local elements." << endl;
   }

   // Gather counts from all processes.
   int local_num = local_x.size();
   vector<int> all_num_elements(size);
   vector<int> displs(size);
   MPI_Gather(&local_num, 1, MPI_INT, all_num_elements.data(), 1, MPI_INT, 0, comm);

   vector<double> all_x, all_y, all_z;
   vector<double> all_velx, all_vely, all_velz;
   if (rank == 0)
   {
      int total = 0;
      displs[0] = 0;
      for (int i = 0; i < size; i++)
      {
         total += all_num_elements[i];
         if (i > 0)
            displs[i] = displs[i - 1] + all_num_elements[i - 1];
      }
      all_x.resize(total);
      all_y.resize(total);
      all_z.resize(total);
      all_velx.resize(total);
      all_vely.resize(total);
      all_velz.resize(total);
      mfem::out << " Gathering sample data from " << size << " processes." << endl;
   }

   MPI_Gatherv(local_x.data(), local_num, MPI_DOUBLE,
               all_x.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
   MPI_Gatherv(local_y.data(), local_num, MPI_DOUBLE,
               all_y.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
   MPI_Gatherv(local_z.data(), local_num, MPI_DOUBLE,
               all_z.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
   MPI_Gatherv(local_velx.data(), local_num, MPI_DOUBLE,
               all_velx.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
   MPI_Gatherv(local_vely.data(), local_num, MPI_DOUBLE,
               all_vely.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
   MPI_Gatherv(local_velz.data(), local_num, MPI_DOUBLE,
               all_velz.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);

   if (rank == 0)
   {
      mfem::out << " Writing gathered sample data to file: " << fname << endl;
      FILE *f = fopen(fname.c_str(), "w");
      if (!f)
      {
         cerr << " Error opening file " << fname << endl;
         MPI_Abort(comm, 1);
      }
      fprintf(f, "3D Taylor Green Vortex\n");
      fprintf(f, "Order = %d\n", ctx.order);
      fprintf(f, "Step = %d\n", step);
      fprintf(f, "Time = %e\n", time);
      fprintf(f, "===================================================================");
      fprintf(f, "==========================================================================\n");
      fprintf(f, "            x                      y                      z         ");
      fprintf(f, "            vecx                   vecy                   vecz\n");
      for (size_t i = 0; i < all_x.size(); i++)
      {
         fprintf(f, "%20.16e %20.16e %20.16e %20.16e %20.16e %20.16e\n",
                 all_x[i], all_y[i], all_z[i],
                 all_velx[i], all_vely[i], all_velz[i]);
      }
      fclose(f);
      mfem::out << " Finished writing sample data." << endl;
   }
   if (rank == 0)
      mfem::out << " Exiting sampling function." << endl;
}

//------------------------------------------------------------------------------
// Function to load a checkpoint using VisItDataCollection (prints only on root).
bool LoadCheckpointVisit(const std::string &visit_dir,
                         int cycle,
                         ParMesh *& pmesh,
                         ParGridFunction *& u_gf,
                         double &t)
{
   if (Mpi::WorldRank() == 0)
      mfem::out << " Loading checkpoint from directory: " << visit_dir
           << " for cycle: " << cycle << endl;
   int precision = 16;

   VisItDataCollection *dc = new VisItDataCollection(MPI_COMM_WORLD, visit_dir, nullptr);
   dc->SetPrecision(precision);
   dc->Load(cycle);

   // Retrieve the loaded mesh
   auto *pmesh_loaded= dynamic_cast<mfem::ParMesh*>(dc->GetMesh());
   pmesh = pmesh_loaded;

   if (!pmesh)
   {
      if (Mpi::WorldRank() == 0)
         cerr << " Error: Mesh load failed." << endl;
      delete dc;
      return false;
   }

   GridFunction *temp_gf = dc->GetField("velocity");
   if (!temp_gf)
   {
      if (Mpi::WorldRank() == 0)
         cerr << " Error: velocity field not found." << endl;
      delete dc;
      return false;
   }
   ParFiniteElementSpace *vfes = dynamic_cast<ParFiniteElementSpace*>(temp_gf->FESpace());
   if (!vfes)
   {
      if (Mpi::WorldRank() == 0)
         cerr << " Error: FESpace is not a ParFiniteElementSpace." << endl;
      delete dc;
      return false;
   }
   u_gf = new ParGridFunction(vfes);
   *u_gf = *temp_gf;  // deep copy


    // Compute Linf
    mfem::real_t u_inf_loc = u_gf->Normlinf();
    mfem::real_t u_inf = mfem::GlobalLpNorm(mfem::infinity(), 
                                                  u_inf_loc, 
                                                  MPI_COMM_WORLD);
       
    if (Mpi::Root())
    {
    }

   t = dc->GetTime();
   if (Mpi::WorldRank() == 0){
      mfem::out << " Checkpoint loaded: Cycle " << cycle << ", Time " << t << endl;
      mfem::out << "After loading from checkpoint in LoadCheckpoint: u_gf Norml2 = "
                  << u_inf <<  std::endl;
      mfem::out << "Number of elements: " << pmesh->GetNE() << endl;
   }
   return true;
}

//------------------------------------------------------------------------------
// Main function: parse arguments, load checkpoint, sample data, and clean up.
int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   int myid = Mpi::WorldRank();
   if (myid == 0)
      mfem::out << " Starting main." << endl;

   if (argc < 3)
   {
      if (myid == 0)
         mfem::out << "Usage: " << argv[0] << " <visit_dir> <cycle_number>" << endl;
      Mpi::Finalize();
      return 1;
   }

   std::string visit_dir = argv[1];
   int cycle = std::atoi(argv[2]);

   if (myid == 0)
      mfem::out << " Visit directory: " << visit_dir << ", cycle: " << cycle << endl;

   ParMesh *pmesh = nullptr;
   ParGridFunction *velocity = nullptr;
   double t = 0.0;

   if (!LoadCheckpointVisit(visit_dir, cycle, pmesh, velocity, t))
   {
      if (myid == 0)
         cerr << " Failed to load checkpoint." << endl;
      Mpi::Finalize();
      return 1;
   }

   if (myid == 0){
      mfem::out << " Starting data sampling." << endl;
      mfem::out << "Number of elements: " << pmesh->GetNE() << endl;
   }
   SampledDataUniform(velocity, pmesh, cycle, t, "Velocity");
   if (myid == 0)
      mfem::out << " Data sampling complete." << endl;

   // Clean up: Uncomment these lines if you wish to free the objects.
   delete pmesh;
   delete velocity;

   if (myid == 0)
      mfem::out << " Exiting main." << endl;
   Mpi::Finalize();
   return 0;
}

