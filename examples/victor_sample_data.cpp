// sample_checkpoint_debug.cpp
//
// A simplified program that loads a checkpoint state using VisItDataCollection,
// makes a copy of the "velocity" field, and then uniformly samples the
// velocity field using the provided SampledDataUniform function.
// 
// srun -n 560 -ppdebug --exclusive ~/Documents/mfem_build/mfem/examples/victor_sample_data -o 3 -vd VisitData_Re1600NumPtsPerDir32RefLv2P4/tgv_output_visit -cyc 9000

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
   string visit_dir_inline = "Null";
   int cycle_inline = 0;
};

// Global context variable.
s_NavierContext ctx;

//------------------------------------------------------------------------------
void SampledDataUniformBothBoundaries(ParGridFunction* sol,
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
   std::string fname = cycle_dir + "/sampled_data_include_both_boundaries_" + std::to_string(step) + ".txt";

   if (rank == 0)
   {
      mfem::out << " Creating directories: " << main_dir << " and " << cycle_dir << endl;
      if (system(("mkdir -p " + main_dir).c_str()) != 0)
         cerr << " Error creating " << main_dir << " directory!" << endl;
      if (system(("mkdir -p " + cycle_dir).c_str()) != 0)
         cerr << " Error creating " << cycle_dir << " directory!" << endl;
   }

   MPI_Barrier(comm);

   // int npts = ctx.order + 1;  // sample points per coordinate direction.
   int npts = ctx.order + 2;  // sample points per coordinate direction.
   if (rank == 0)
   {
      mfem::out << " Using " << npts << " sample points per coordinate." << endl;
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
         double z_ref = static_cast<double>(iz) / (npts - 1);
         for (int iy = 0; iy < npts; iy++)
         {
            double y_ref = static_cast<double>(iy) / (npts - 1);
            for (int ix = 0; ix < npts; ix++)
            {
               double x_ref = static_cast<double>(ix) / (npts - 1);
               IntegrationPoint ip;
               ip.Set3(x_ref, y_ref, z_ref);

               Vector phys_coords(Trans->GetSpaceDim());
               Trans->Transform(ip, phys_coords);
               double x_pos = phys_coords(0);
               double y_pos = phys_coords(1);
               double z_pos = phys_coords(2);

               Vector u_val(vdim);
               sol->GetVectorValue(*Trans, ip, u_val);
               double u_x = u_val(0);
               double u_y = u_val(1);
               double u_z = u_val(2);

               local_x.push_back(x_pos);
               local_y.push_back(y_pos);
               local_z.push_back(z_pos);
               local_velx.push_back(u_x);
               local_vely.push_back(u_y);
               local_velz.push_back(u_z);
            }
         }
      }
   }

   // Prepare the data string, including the header on rank 0
   std::string data_str;
   if (rank == 0)
   {
      std::ostringstream header_stream;
      header_stream << "3D Taylor Green Vortex\n"
                    << "Order = " << ctx.order << "\n"
                    << "Step = " << step << "\n"
                    << "Time = " << std::scientific << std::setprecision(16) << time << "\n"
                    << "==================================================================="
                    << "==========================================================================\n"
                    << "            x                      y                      z                   vecx                   vecy                   vecz\n";
      data_str = header_stream.str();
   }

   // Append local data to data_str
   std::ostringstream local_data_stream;
   for (size_t i = 0; i < local_x.size(); i++)
   {
      local_data_stream << std::scientific << std::setprecision(16)
                        << std::setw(20) << local_x[i] << " "
                        << std::setw(20) << local_y[i] << " "
                        << std::setw(20) << local_z[i] << " "
                        << std::setw(20) << local_velx[i] << " "
                        << std::setw(20) << local_vely[i] << " "
                        << std::setw(20) << local_velz[i] << "\n";
   }
   data_str += local_data_stream.str();

   // Open the file collectively with MPI I/O
   MPI_File fh;
   int err = MPI_File_open(comm, fname.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
   if (err != MPI_SUCCESS)
   {
      if (rank == 0) std::cerr << "Error opening file " << fname << " with MPI I/O" << std::endl;
      MPI_Abort(comm, 1);
   }

   // All ranks write their data (including header on rank 0) in order using the shared file pointer
   MPI_File_write_ordered(fh, data_str.c_str(), data_str.size(), MPI_CHAR, MPI_STATUS_IGNORE);

   // Clear memory
   local_x.clear(); local_y.clear(); local_z.clear();
   local_velx.clear(); local_vely.clear(); local_velz.clear();
   data_str.clear();

   // Close the file
   MPI_File_close(&fh);

   // Output confirmation on rank 0
   if (rank == 0)
      mfem::out << " Finished writing sampled including both boundaries data." << endl;
}


void SampledDataUniformOneBoundary(ParGridFunction* sol,
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
   std::string fname = cycle_dir + "/sampled_data_include_one_boundary_" + std::to_string(step) + ".txt";

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
               double x_pos = phys_coords(0);
               double y_pos = phys_coords(1);
               double z_pos = phys_coords(2);

               Vector u_val(vdim);
               sol->GetVectorValue(*Trans, ip, u_val);
               double u_x = u_val(0);
               double u_y = u_val(1);
               double u_z = u_val(2);

               local_x.push_back(x_pos);
               local_y.push_back(y_pos);
               local_z.push_back(z_pos);
               local_velx.push_back(u_x);
               local_vely.push_back(u_y);
               local_velz.push_back(u_z);
            }
         }
      }
   }

   // Prepare the data string, including the header on rank 0
   std::string data_str;
   if (rank == 0)
   {
      std::ostringstream header_stream;
      header_stream << "3D Taylor Green Vortex\n"
                    << "Order = " << ctx.order << "\n"
                    << "Step = " << step << "\n"
                    << "Time = " << std::scientific << std::setprecision(16) << time << "\n"
                    << "==================================================================="
                    << "==========================================================================\n"
                    << "            x                      y                      z                   vecx                   vecy                   vecz\n";
      data_str = header_stream.str();
   }

   // Append local data to data_str
   std::ostringstream local_data_stream;
   for (size_t i = 0; i < local_x.size(); i++)
   {
      local_data_stream << std::scientific << std::setprecision(16)
                        << std::setw(20) << local_x[i] << " "
                        << std::setw(20) << local_y[i] << " "
                        << std::setw(20) << local_z[i] << " "
                        << std::setw(20) << local_velx[i] << " "
                        << std::setw(20) << local_vely[i] << " "
                        << std::setw(20) << local_velz[i] << "\n";
   }
   data_str += local_data_stream.str();

   // Open the file collectively with MPI I/O
   MPI_File fh;
   int err = MPI_File_open(comm, fname.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
   if (err != MPI_SUCCESS)
   {
      if (rank == 0) std::cerr << "Error opening file " << fname << " with MPI I/O" << std::endl;
      MPI_Abort(comm, 1);
   }

   // All ranks write their data (including header on rank 0) in order using the shared file pointer
   MPI_File_write_ordered(fh, data_str.c_str(), data_str.size(), MPI_CHAR, MPI_STATUS_IGNORE);

   // Clear memory
   local_x.clear(); local_y.clear(); local_z.clear();
   local_velx.clear(); local_vely.clear(); local_velz.clear();
   data_str.clear();

   // Close the file
   MPI_File_close(&fh);

   // Output confirmation on rank 0
   if (rank == 0)
      mfem::out << " Finished writing sampled including one boundary data." << endl;

   MPI_Barrier(comm);
}

//------------------------------------------------------------------------------
void SampledDataElementOneCenter(ParGridFunction* sol,
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
   std::string fname = cycle_dir + "/sampled_data_element_centers_" + std::to_string(step) + ".txt";

   if (rank == 0)
   {
      mfem::out << " Creating directories: " << main_dir << " and " << cycle_dir << endl;
      if (system(("mkdir -p " + main_dir).c_str()) != 0)
         cerr << " Error creating " << main_dir << " directory!" << endl;
      if (system(("mkdir -p " + cycle_dir).c_str()) != 0)
         cerr << " Error creating " << cycle_dir << " directory!" << endl;
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
      IntegrationPoint ip;
      ip.Set3(0.5, 0.5, 0.5);

      Vector phys_coords(Trans->GetSpaceDim());
      Trans->Transform(ip, phys_coords);
      double x_pos = phys_coords(0);
      double y_pos = phys_coords(1);
      double z_pos = phys_coords(2);

      Vector u_val(vdim);
      sol->GetVectorValue(*Trans, ip, u_val);
      double u_x = u_val(0);
      double u_y = u_val(1);
      double u_z = u_val(2);

      local_x.push_back(x_pos);
      local_y.push_back(y_pos);
      local_z.push_back(z_pos);
      local_velx.push_back(u_x);
      local_vely.push_back(u_y);
      local_velz.push_back(u_z);
   }

   // Prepare the data string, including the header on rank 0
   std::string data_str;
   if (rank == 0)
   {
      std::ostringstream header_stream;
      header_stream << "3D Taylor Green Vortex\n"
                    << "Order = " << ctx.order << "\n"
                    << "Step = " << step << "\n"
                    << "Time = " << std::scientific << std::setprecision(16) << time << "\n"
                    << "==================================================================="
                    << "==========================================================================\n"
                    << "            x                      y                      z                   vecx                   vecy                   vecz\n";
      data_str = header_stream.str();
   }

   // Append local data to data_str
   std::ostringstream local_data_stream;
   for (size_t i = 0; i < local_x.size(); i++)
   {
      local_data_stream << std::scientific << std::setprecision(16)
                        << std::setw(20) << local_x[i] << " "
                        << std::setw(20) << local_y[i] << " "
                        << std::setw(20) << local_z[i] << " "
                        << std::setw(20) << local_velx[i] << " "
                        << std::setw(20) << local_vely[i] << " "
                        << std::setw(20) << local_velz[i] << "\n";
   }
   data_str += local_data_stream.str();

   // Open the file collectively with MPI I/O
   MPI_File fh;
   int err = MPI_File_open(comm, fname.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
   if (err != MPI_SUCCESS)
   {
      if (rank == 0) std::cerr << "Error opening file " << fname << " with MPI I/O" << std::endl;
      MPI_Abort(comm, 1);
   }

   // All ranks write their data (including header on rank 0) in order using the shared file pointer
   MPI_File_write_ordered(fh, data_str.c_str(), data_str.size(), MPI_CHAR, MPI_STATUS_IGNORE);

   // Clear memory
   local_x.clear(); local_y.clear(); local_z.clear();
   local_velx.clear(); local_vely.clear(); local_velz.clear();
   data_str.clear();

   // Close the file
   MPI_File_close(&fh);

   // Output confirmation on rank 0
   if (rank == 0)
      mfem::out << " Finished writing cell centered data." << endl;

   MPI_Barrier(comm);
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
      return false;
   }

   GridFunction *temp_gf = dc->GetField("velocity");

   if (!temp_gf)
   {
      if (Mpi::WorldRank() == 0)
         cerr << " Error: velocity field not found." << endl;
      return false;
   }

   ParFiniteElementSpace *vfes = dynamic_cast<ParFiniteElementSpace*>(temp_gf->FESpace());
   if (!vfes)
   {
      if (Mpi::WorldRank() == 0)
         cerr << " Error: FESpace is not a ParFiniteElementSpace." << endl;
      return false;
   }

   u_gf = new ParGridFunction(vfes);
   *u_gf = *temp_gf;  // deep copy


    // Compute Linf
    mfem::real_t u_inf_loc = u_gf->Normlinf();
    mfem::real_t u_inf = mfem::GlobalLpNorm(mfem::infinity(), 
                                                  u_inf_loc, 
                                                  MPI_COMM_WORLD);
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

   OptionsParser args(argc, argv);
   args.AddOption(&ctx.order,
                  "-o",
                  "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ctx.visit_dir_inline,
                  "-vd",
                  "--vis_dir",
                  "Directory of Visit directory.");
   args.AddOption(&ctx.cycle_inline,
                  "-cyc",
                  "--cycle",
                  "Cycle number");

   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root())
      {
         args.PrintUsage(mfem::out);
      }
      return 1;
   }

   if (Mpi::Root())
   {
      args.PrintOptions(mfem::out);
   }

   if (myid == 0)
      mfem::out << " Starting main with order ." << ctx.order << endl;

   std::string visit_dir = ctx.visit_dir_inline;
   int cycle = ctx.cycle_inline;

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

   SampledDataUniformBothBoundaries(velocity, pmesh, cycle, t, "Velocity");

   MPI_Barrier(MPI_COMM_WORLD);
   SampledDataUniformOneBoundary(velocity, pmesh, cycle, t, "Velocity");

   MPI_Barrier(MPI_COMM_WORLD);
   SampledDataElementOneCenter(velocity, pmesh, cycle, t, "Velocity");

   if (myid == 0)
      mfem::out << " Data sampling complete." << endl;

   delete pmesh;
   delete velocity;

   if (myid == 0)
      mfem::out << " Exiting main." << endl;
   Mpi::Finalize();
   return 0;
}

