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

   int Wx = 0, Wy = 0; // window position
   int Ww = 350, Wh = 350; // window size
   int offx = Ww+5, offy = Wh+25; // window offsets
string space;
string direction;



real_t compute_ke(ParGridFunction *gf, string prefix);

real_t ComputeKineticEnergy(ParGridFunction &v);

void visualize(VisItDataCollection &, string, int, int, int /* visport */);

void HighToLowResolution(ParGridFunction* u, ParMesh* mesh,
                         int step, real_t time,
                         const std::string suffix, int order);

void SampleDataIntegrationPoints(mfem::ParGridFunction* sol,
                                mfem::ParMesh* pmesh,
                                int step,
                                double time,
                                const std::string &suffix);

void SampleVectorFieldNodalPoints(mfem::ParGridFunction &u, mfem::ParMesh *pmesh, 
                                  int step, double time, const std::string &suffix);
//------------------------------------------------------------------------------
// Minimal context structure to supply parameters for the sampling function.
struct s_NavierContext
{
   int order = 2;
   string visit_dir_inline = "Null";
   int cycle_inline = 0;
   string field_to_sample = "velocity";
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
                         ParGridFunction *& sol_gf,
                         double &t,
                         const std::string &field_to_sample)
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

   GridFunction *temp_gf = dc->GetField(field_to_sample);

   if (!temp_gf)
   {
      if (Mpi::WorldRank() == 0)
         cerr << " Error: " << field_to_sample << "not found." << endl;
      return false;
   }

   ParFiniteElementSpace *fes = dynamic_cast<ParFiniteElementSpace*>(temp_gf->FESpace());
   if (!fes)
   {
      if (Mpi::WorldRank() == 0)
         cerr << " Error: FESpace is not a ParFiniteElementSpace." << endl;
      return false;
   }

   sol_gf = new ParGridFunction(fes);
   *sol_gf = *temp_gf;  // deep copy


    // Compute Linf
    mfem::real_t sol_inf_loc = sol_gf->Normlinf();
    mfem::real_t sol_inf = mfem::GlobalLpNorm(mfem::infinity(), 
                                                  sol_inf_loc, 
                                                  MPI_COMM_WORLD);
   t = dc->GetTime();

   if (Mpi::WorldRank() == 0){
      mfem::out << " Checkpoint loaded: Cycle " << cycle << ", Time " << t << endl;
      mfem::out << "After loading from checkpoint in LoadCheckpoint: Field Norml2 = "
                  << sol_inf <<  std::endl;
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
   args.AddOption(&ctx.field_to_sample,
                  "-fts",
                  "--field",
                  "field to sample");

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
      mfem::out << " Starting main with order: " << ctx.order << endl;

   std::string visit_dir = ctx.visit_dir_inline;
   std::string field_to_sample = ctx.field_to_sample;
   int cycle = ctx.cycle_inline;

   if (myid == 0){
      mfem::out << " Visit directory: " << visit_dir << ", cycle: " << cycle << endl;
      mfem::out << " Field to sample: " << ctx.field_to_sample << endl;
   }

   ParMesh *pmesh = nullptr;
   ParGridFunction *sol_gf = nullptr;
   double t = 0.0;

   if (!LoadCheckpointVisit(visit_dir, cycle, pmesh, sol_gf, t,ctx.field_to_sample))
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

   SampledDataUniformBothBoundaries(sol_gf, pmesh, cycle, t, ctx.field_to_sample);

   MPI_Barrier(MPI_COMM_WORLD);
   SampledDataUniformOneBoundary(sol_gf, pmesh, cycle, t, ctx.field_to_sample);

   MPI_Barrier(MPI_COMM_WORLD);
   SampledDataElementOneCenter(sol_gf, pmesh, cycle, t, ctx.field_to_sample);

   HighToLowResolution(sol_gf, pmesh,
                         cycle, t,ctx.field_to_sample,ctx.order);

   if (myid == 0)
      mfem::out << " Data sampling complete." << endl;

   delete pmesh;
   delete sol_gf;

   if (myid == 0)
      mfem::out << " Exiting main." << endl;
   Mpi::Finalize();
   return 0;
}


void HighToLowResolution(ParGridFunction* u, ParMesh* mesh,
                         int step, real_t time,
                         const std::string suffix, int order)
{

   // Parse command-line options.
   int lref = order+1;
   int lorder = 0;
   bool vis = true;
   bool useH1 = true;
   int visport = 19916;
   bool use_pointwise_transfer = false;
   const char *device_config = "cpu";
   bool use_ea       = false;

   // Configure device
   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   int dim = mesh->Dimension();

   // Create the low-order refined mesh
   int basis_lor = BasisType::GaussLobatto; // BasisType::ClosedUniform;
   ParMesh mesh_lor = ParMesh::MakeRefined(*mesh, lref, basis_lor);

   // Create spaces
   FiniteElementCollection *fec, *fec_lor;
   if (useH1)
   {
      space = "H1";
      if (lorder == 0)
      {
         lorder = 1;
         if (Mpi::Root())
         {
            cerr << "Switching the H1 LOR space order from 0 to 1\n";
         }
      }
      fec = new H1_FECollection(order, dim);
      fec_lor = new H1_FECollection(lorder, dim);
   }
   else
   {
      space = "L2";
      fec = new L2_FECollection(order, dim);
      fec_lor = new L2_FECollection(lorder, dim);
   }
   

   ParFiniteElementSpace fespace(mesh, fec, dim);
   ParFiniteElementSpace fespace_lor(&mesh_lor, fec_lor,dim);

   ParGridFunction u_lor(&fespace_lor);

   // Data collections for vis/analysis
   VisItDataCollection HO_dc(MPI_COMM_WORLD, "HO", mesh);
   HO_dc.RegisterField("velocity", u);
   VisItDataCollection LOR_dc(MPI_COMM_WORLD, "LOR", &mesh_lor);
   LOR_dc.RegisterField("velocity", &u_lor);
      
   // HO projections
   direction = "HO -> LOR @ HO";

   real_t ho_ke = compute_ke(u, "HO         ");

   if (vis) { visualize(HO_dc, "HO", Wx, Wy, visport); Wx += offx; }

   GridTransfer *gt;
   if (use_pointwise_transfer)
   {
      gt = new InterpolationGridTransfer(fespace, fespace_lor);
   }
   else
   {
      gt = new L2ProjectionGridTransfer(fespace, fespace_lor);
   }

   // Configure element assembly for device acceleration
   gt->UseEA(use_ea);

   const Operator &R = gt->ForwardOperator();

   // HO->LOR restriction
   direction = "HO -> LOR @ LOR";
   R.Mult(*u, u_lor);
   real_t lo_ke = compute_ke(&u_lor, "R(HO)     ");

   auto global_max = [](const Vector& v)
   {
      real_t max = v.Normlinf();
      MPI_Allreduce(MPI_IN_PLACE, &max, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_MAX, MPI_COMM_WORLD);
      return max;
   };

   if (vis) { visualize(LOR_dc, "R(HO)", Wx, Wy, visport); Wx += offx; }

   if (gt->SupportsBackwardsOperator())
   {
      const Operator &P = gt->BackwardOperator();
      // LOR->HO prolongation
      direction = "HO -> LOR @ HO";
      ParGridFunction u_prev = *u;
      P.Mult(u_lor, *u);
      ho_ke = compute_ke(u, "P(R(HO)   ");

      if (vis) { visualize(HO_dc, "P(R(HO))", Wx, Wy, visport); Wx = 0; Wy += offy; }

      u_prev -= *u;
      Vector u_prev_true(fespace.GetTrueVSize());
      u_prev.GetTrueDofs(u_prev_true);
      real_t l_inf = global_max(u_prev_true);
      if (Mpi::Root())
      {
         cout.precision(12);
         cout << "|HO - P(R(HO))|_∞   = " << l_inf << endl;
      }
   }

   if (Mpi::Root()){
     mfem::out << "Saving output to visualize for later... " << "\n";
   }

   mfem::real_t u_inf_loc = u->Normlinf();
   mfem::real_t u_inf_lor_loc = u_lor.Normlinf();
   mfem::real_t u_inf_prev_loc = u->Normlinf();

   mfem::real_t u_inf = mfem::GlobalLpNorm(mfem::infinity(), 
                                                 u_inf_loc, 
                                                 MPI_COMM_WORLD);

   mfem::real_t u_inf_lor = mfem::GlobalLpNorm(mfem::infinity(), 
                                                 u_inf_lor_loc, 
                                                 MPI_COMM_WORLD);
   mfem::real_t u_inf_prev = mfem::GlobalLpNorm(mfem::infinity(), 
                                                 u_inf_prev_loc, 
                                                 MPI_COMM_WORLD);
   if (Mpi::Root()){
       mfem::out << "Linf of u: " << u_inf << "\n";
       mfem::out << "Linf of u R(HO): " << u_inf_lor << "\n";
       mfem::out << "Linf of u P(R(HO)): " << u_inf_prev << "\n";
   }


   HO_dc.SetCycle(step);
   HO_dc.SetTime(time);
   HO_dc.SetFormat(DataCollection::PARALLEL_FORMAT);
   HO_dc.Save();

   LOR_dc.SetCycle(step);
   LOR_dc.SetTime(time);
   LOR_dc.SetFormat(DataCollection::PARALLEL_FORMAT);
   LOR_dc.Save();

   if (Mpi::Root()){
     mfem::out << "Sampling data at integration points... " << "\n";
   }
   // SampleDataIntegrationPoints(&u_lor,
   //                            &mesh_lor,
   //                            step,
   //                            time,
   //                            suffix);
   SampleVectorFieldNodalPoints(u_lor, &mesh_lor, 
                              step,
                              time,
                              suffix);

   if (Mpi::Root()){
     mfem::out << "All done " << "\n";
   }
   delete fec;
   delete fec_lor;
   delete gt;

}

void SampleDataIntegrationPoints(mfem::ParGridFunction* sol,
                                mfem::ParMesh* pmesh,
                                int step,
                                double time,
                                const std::string &suffix){
   // MPI setup
   MPI_Comm comm = pmesh->GetComm();
   int rank, size;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &size);

   // Construct the main directory name with suffix.
   std::string main_dir = "SampledData" + suffix +
                            "P" + std::to_string(ctx.order);

   // Create subdirectory for this cycle step.
   std::string cycle_dir = main_dir + "/cycle_" + std::to_string(step);

   // Construct the filename.
   std::string fname = cycle_dir + "/sampled_data_element_integ_poits_" + std::to_string(step) + ".txt";

   // Create directories on rank 0
   if (rank == 0)
   {
      if (system(("mkdir -p " + main_dir).c_str()) != 0)
         std::cerr << "Error creating " << main_dir << " directory!" << std::endl;
      if (system(("mkdir -p " + cycle_dir).c_str()) != 0)
         std::cerr << "Error creating " << cycle_dir << " directory!" << std::endl;
   }

   // Synchronize all ranks before proceeding
   MPI_Barrier(MPI_COMM_WORLD);


    std::vector<double> local_x, local_y, local_z;
    std::vector<double> local_velx, local_vely, local_velz;
    std::vector<double> local_quad_weights;

    mfem::FiniteElementSpace *fes = sol->FESpace();
    int vdim = fes->GetVDim();
    const FiniteElement *fe;

   real_t integ = 0.0;
   real_t summedVolume = 0.0;
   real_t globalVolume   = 0.0;
   real_t global_integral = 0.0;

    for (int i = 0; i < fes->GetNE(); i++)
    {
      
        double volume_per_cell = 0.0;
        mfem::ElementTransformation *Trans = pmesh->GetElementTransformation(i);
        fe = fes->GetFE(i);
        int intorder = 2 * fe->GetOrder();
        mfem::out << "intorder: " << intorder << "\n";
        const IntegrationRule *ir = &IntRules.Get(fe->GetGeomType(), intorder);

        // Pre-allocate mfem::Vectors for efficiency
        mfem::Vector velx(ir->GetNPoints());
        mfem::Vector vely(ir->GetNPoints());
        mfem::Vector velz(ir->GetNPoints());

        sol->GetValues(i, *ir, velx, 1);
        sol->GetValues(i, *ir, vely, 2);
        sol->GetValues(i, *ir, velz, 3);

        // Loop over integration points
        for (int j = 0; j < ir->GetNPoints(); j++)
        {
            const IntegrationPoint &ip = ir->IntPoint(j);
            mfem::Vector phys_coords(Trans->GetSpaceDim());
            Trans->Transform(ip, phys_coords);
            double x_physical = phys_coords(0);
            double y_physical = phys_coords(1);
            double z_physical = phys_coords(2);

            // Store coordinates and velocity components
            local_x.push_back(x_physical);
            local_y.push_back(y_physical);
            local_z.push_back(z_physical);
            local_velx.push_back(velx(j));
            local_vely.push_back(vely(j));
            local_velz.push_back(velz(j));

            real_t vel2 = velx(j) * velx(j) + vely(j) * vely(j)
                          + velz(j) * velz(j);

            integ += ip.weight * Trans->Weight() * vel2;
            volume_per_cell += ip.weight*Trans->Weight();
            local_quad_weights.push_back(ip.weight*Trans->Weight());
        }
        summedVolume += volume_per_cell;
    }

   MPI_Allreduce(&summedVolume,
                 &globalVolume,
                 1,
                 MPITypeMap<real_t>::mpi_type,
                 MPI_SUM,
                 MPI_COMM_WORLD);

   MPI_Allreduce(&integ,
                 &global_integral,
                 1,
                 MPITypeMap<real_t>::mpi_type,
                 MPI_SUM,
                 MPI_COMM_WORLD);

   if (rank == 0)
     std::cout << "KE: " << 0.5*global_integral/globalVolume << std::endl;

   // Prepare the data string, including the header on rank 0
   std::string data_str;
   if (rank == 0)
   {
      std::ostringstream header_stream;
      header_stream << "PLOR Transfer\n"
                    << "Step = " << step << "\n"
                    << "Time = " << std::scientific << std::setprecision(16) << time << "\n"
                    << "==================================================================="
                    << "==========================================================================\n"
                    << "            x                      y                      z                   vecx                   vecy                   vecz         w\n";
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
                        << std::setw(20) << local_velz[i] << " "
                        << std::setw(20) << local_quad_weights[i] << "\n";
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
      std::cout << "Output element sample file saved: " << fname << std::endl;

   // Final synchronization
   MPI_Barrier(MPI_COMM_WORLD);
}

void visualize(VisItDataCollection &dc, string prefix, int x, int y,
               int visport)
{
   int w = Ww, h = Wh;

   char vishost[] = "localhost";

   socketstream sol_sockL2(vishost, visport);
   sol_sockL2 << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() <<
              "\n";
   sol_sockL2.precision(8);
   sol_sockL2 << "solution\n" << *dc.GetMesh() << *dc.GetField("velocity")
              << "window_geometry " << x << " " << y << " " << w << " " << h
              << "plot_caption '" << space << " " << prefix << " Velocity'"
              << "window_title '" << direction << "'" << flush;
}

// Compute L2 ke
real_t compute_ke(ParGridFunction *gf, string prefix)
{

  real_t ke =  ComputeKineticEnergy(*gf);

  if (Mpi::Root())
  {
     cout.precision(18);
     cout << space << " " << prefix << " Kinetic energy = " << ke<< endl;
  }

  return ke;
   
}

real_t ComputeKineticEnergy(ParGridFunction &v)
{
   Vector velx, vely, velz;
   real_t integ = 0.0;
   const FiniteElement *fe;
   ElementTransformation *T;
   FiniteElementSpace *fes = v.FESpace();
      
   real_t summedVolume = 0.0;
   real_t globalVolume   = 0.0;
   real_t global_integral = 0.0;

   for (int i = 0; i < fes->GetNE(); i++)
   {
      double volume_per_cell = 0.0;

      fe = fes->GetFE(i);
      int intorder = 2 * fe->GetOrder();
      const IntegrationRule *ir = &IntRules.Get(fe->GetGeomType(), intorder);

      v.GetValues(i, *ir, velx, 1);
      v.GetValues(i, *ir, vely, 2);
      v.GetValues(i, *ir, velz, 3);

      T = fes->GetElementTransformation(i);
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T->SetIntPoint(&ip);

         real_t vel2 = velx(j) * velx(j) + vely(j) * vely(j)
                       + velz(j) * velz(j);

         integ += ip.weight * T->Weight() * vel2;
         volume_per_cell += ip.weight*T->Weight();
      }
      summedVolume +=volume_per_cell;
   }

   MPI_Allreduce(&summedVolume,
                 &globalVolume,
                 1,
                 MPITypeMap<real_t>::mpi_type,
                 MPI_SUM,
                 MPI_COMM_WORLD);

   MPI_Allreduce(&integ,
                 &global_integral,
                 1,
                 MPITypeMap<real_t>::mpi_type,
                 MPI_SUM,
                 MPI_COMM_WORLD);

   return 0.5 * global_integral/globalVolume;
};

void SampleVectorFieldNodalPoints(mfem::ParGridFunction &u, mfem::ParMesh *pmesh, 
                                  int step, double time, const std::string &suffix)
{
   // MPI setup
   MPI_Comm comm = pmesh->GetComm();
   int rank, size;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &size);

   // Access the finite element space
   mfem::FiniteElementSpace *fes = u.FESpace();
   int vdim = fes->GetVDim(); // Number of vector components (e.g., 3 for velocity in 3D)

   // Data storage for coordinates and vector components
   std::vector<double> local_x, local_y, local_z;
   std::vector<double> local_vecx, local_vecy, local_vecz; // Assuming 3D vector field

   // Calculate the number of owned nodes
   int n_owned_nodes = fes->GetTrueVSize() / vdim;
   double *data = u.GetData(); // Local data array containing vector field values

   // Loop over owned nodal points (vertices for H1 order 1)
   for (int i = 0; i < n_owned_nodes; ++i)
   {
      // Get coordinates of the i-th owned vertex
      double *coords = pmesh->GetVertex(i);
      local_x.push_back(coords[0]);
      local_y.push_back(coords[1]);
      local_z.push_back(coords[2]);

      // Get vector field components at node i
      local_vecx.push_back(data[i * vdim + 0]); // x-component
      local_vecy.push_back(data[i * vdim + 1]); // y-component
      local_vecz.push_back(data[i * vdim + 2]); // z-component
   }

   // Prepare the output string, with header on rank 0
   std::string data_str;
   if (rank == 0)
   {
      std::ostringstream header;
      header << "Vector Field Sampling at Nodal Points\n"
             << "Step = " << step << "\n"
             << "Time = " << std::scientific << std::setprecision(16) << time << "\n"
             << "==================================================================="
             << "==========================================================================\n"
             << "            x                      y                      z                   vecx                   vecy                   vecz\n";
      data_str = header.str();
   }

   // Append local data to the string
   std::ostringstream local_data;
   for (size_t i = 0; i < local_x.size(); i++)
   {
      local_data << std::scientific << std::setprecision(16)
                 << std::setw(20) << local_x[i] << " "
                 << std::setw(20) << local_y[i] << " "
                 << std::setw(20) << local_z[i] << " "
                 << std::setw(20) << local_vecx[i] << " "
                 << std::setw(20) << local_vecy[i] << " "
                 << std::setw(20) << local_vecz[i] << "\n";
   }
   data_str += local_data.str();

   // Construct the output filename
   std::string main_dir = "SampledData" + suffix;
   std::string cycle_dir = main_dir + "/cycle_" + std::to_string(step);
   std::string fname = cycle_dir + "/sampled_vector_field_nodal_points_" + std::to_string(step) + ".txt";

   // Create directories on rank 0
   if (rank == 0)
   {
      if (system(("mkdir -p " + main_dir).c_str()) != 0)
         std::cerr << "Error creating directory: " << main_dir << std::endl;
      if (system(("mkdir -p " + cycle_dir).c_str()) != 0)
         std::cerr << "Error creating directory: " << cycle_dir << std::endl;
   }
   MPI_Barrier(comm); // Ensure directories exist before writing

   // Write data to file using MPI I/O
   MPI_File fh;
   int err = MPI_File_open(comm, fname.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
   if (err != MPI_SUCCESS)
   {
      if (rank == 0) std::cerr << "Error opening file: " << fname << std::endl;
      MPI_Abort(comm, 1);
   }

   // Write data in order across all ranks
   MPI_File_write_ordered(fh, data_str.c_str(), data_str.size(), MPI_CHAR, MPI_STATUS_IGNORE);
   MPI_File_close(&fh);

   // Confirmation output
   if (rank == 0)
      std::cout << "Vector field sampled and saved to: " << fname << std::endl;

   MPI_Barrier(comm); // Synchronize after writing
}

