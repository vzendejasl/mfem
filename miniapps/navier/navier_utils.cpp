// navier_utils.cpp
#include "navier_utils.hpp"
#include <algorithm> // for std::remove_if
#include <string>
#include <cstdio> // for popen, pclose

using namespace mfem;
using namespace navier;

// Helper function to find the last checkpoint step
int FindLastCheckpointStep(const s_NavierContext* ctx)
{
    std::string main_dir;
    std::string command;
    if (GetVisit(ctx))
    {
        main_dir = std::string("VisitData_")
                   + "Re" + std::to_string(static_cast<int>(GetReynum(ctx)))
                   + "NumPtsPerDir" + std::to_string(GetNumPts(ctx))
                   + "RefLv" + std::to_string(GetElementSubdivisions(ctx) + GetElementSubdivisionsParallel(ctx))
                   + "P" + std::to_string(GetOrder(ctx));
        command = "ls " + main_dir + " | grep mfem.root |"
                  + " sed 's/tgv_output_visit_//' | sort -n | tail -1 | sed 's/.mfem_root//'";
    }
    else if (GetConduit(ctx))
    {
        main_dir = std::string("ConduitData_")
                   + "Re" + std::to_string(static_cast<int>(GetReynum(ctx)))
                   + "NumPtsPerDir" + std::to_string(GetNumPts(ctx))
                   + "RefLv" + std::to_string(GetElementSubdivisions(ctx) + GetElementSubdivisionsParallel(ctx))
                   + "P" + std::to_string(GetOrder(ctx));
        command = "ls " + main_dir + " | grep .root |"
                  + " sed 's/tgv_output_conduit_//' | sort -n | tail -1 | sed 's/.root//'";
    }
    else
    {
        MFEM_ABORT("Can only search for visit or conduit data for restarting.");
    }

    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) return -1;
    char buffer[128];
    std::string result;
    while (fgets(buffer, 128, pipe) != NULL)
    {
        result += buffer;
    }
    pclose(pipe);

    // Trim whitespace
    result.erase(std::remove_if(result.begin(), result.end(), ::isspace), result.end());
    if (result.empty())
    {
        return -1; // No checkpoints found
    }

    return std::stoi(result);
}

bool LoadCheckpoint(ParMesh*& pmesh,
                    ParGridFunction*& u_gf,
                    ParGridFunction*& p_gf,
                    NavierSolver*& flowsolver,
                    double& t,
                    int& step,
                    int myid,
                    const s_NavierContext* ctx)
{
    int provided_step = -1;
    if (provided_step < 0)
    {
        int last_step = -1;
        if (myid == 0)
        {
            last_step = FindLastCheckpointStep(ctx);
        }

        // Broadcast to every rank
        MPI_Bcast(&last_step, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (last_step < 0) return false;
        provided_step = last_step;
    }

    GridFunction* loaded_u_gf = nullptr;
    GridFunction* loaded_p_gf = nullptr;
    int precision = 16;

    if (GetVisit(ctx))
    {
        std::string visit_dir = std::string("VisitData_")
                                + "Re" + std::to_string(static_cast<int>(GetReynum(ctx)))
                                + "NumPtsPerDir" + std::to_string(GetNumPts(ctx))
                                + "RefLv" + std::to_string(GetElementSubdivisions(ctx) + GetElementSubdivisionsParallel(ctx))
                                + "P" + std::to_string(GetOrder(ctx))
                                + "/tgv_output_visit";

        mfem::DataCollection* dc_load = new mfem::VisItDataCollection(MPI_COMM_WORLD, visit_dir, nullptr);
        dc_load->SetPrecision(precision);
        dc_load->Load(provided_step);

        auto* pmesh_loaded = dynamic_cast<mfem::ParMesh*>(dc_load->GetMesh());
        pmesh = pmesh_loaded;

        if (mfem::Mpi::Root())
        {
            if (!pmesh_loaded) mfem::out << "[ERROR] Failed to create MFEM mesh." << std::endl;
            std::cout << "Mesh data loaded from VisitDataCollection." << std::endl;
        }

        loaded_u_gf = dc_load->GetField("velocity");
        loaded_p_gf = dc_load->GetField("pressure");
        step = dc_load->GetCycle();
        t = dc_load->GetTime();
    }
    else if (GetConduit(ctx))
    {
#ifdef MFEM_USE_CONDUIT
        std::string conduit_dir = std::string("ConduitData_")
                                  + "Re" + std::to_string(static_cast<int>(GetReynum(ctx)))
                                  + "NumPtsPerDir" + std::to_string(GetNumPts(ctx))
                                  + "RefLv" + std::to_string(GetElementSubdivisions(ctx) + GetElementSubdivisionsParallel(ctx))
                                  + "P" + std::to_string(GetOrder(ctx))
                                  + "/tgv_output_conduit";

        ConduitDataCollection* cdc_load = new ConduitDataCollection(MPI_COMM_WORLD, conduit_dir, nullptr);
        cdc_load->SetPrecision(precision);
        cdc_load->SetProtocol("hdf5");
        cdc_load->Load(provided_step);

        auto* pmesh_loaded = dynamic_cast<mfem::ParMesh*>(cdc_load->GetMesh());
        pmesh = pmesh_loaded;

        if (mfem::Mpi::Root())
        {
            if (!pmesh) mfem::out << "[ERROR] Failed to create MFEM mesh." << std::endl;
            std::cout << "Mesh data loaded from ConduitDataCollection." << std::endl;
        }

        loaded_u_gf = cdc_load->GetField("velocity");
        loaded_p_gf = cdc_load->GetField("pressure");
        step = cdc_load->GetCycle();
        t = cdc_load->GetTime();
#else
        MFEM_ABORT("Must build with MFEM_USE_CONDUIT=YES for binary output.");
#endif
    }
    else
    {
        MFEM_ABORT("Can only restart with visit or conduit");
    }

    if (myid == 0)
    {
        std::cout << "Loaded time t = " << t << ", step = " << step << std::endl;
    }

    MPI_Bcast(&t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&step, 1, MPI_INT, 0, MPI_COMM_WORLD);

    flowsolver = new NavierSolver(pmesh, GetOrder(ctx), GetKinvis(ctx));
    flowsolver->EnablePA(GetPA(ctx));
    flowsolver->EnableNI(GetNI(ctx));

    u_gf = flowsolver->GetCurrentVelocity();
    p_gf = flowsolver->GetCurrentPressure();

    ParGridFunction temp_u_gf(u_gf->ParFESpace(), loaded_u_gf);
    ParGridFunction temp_p_gf(p_gf->ParFESpace(), loaded_p_gf);

    *u_gf = temp_u_gf;
    *p_gf = temp_p_gf;

    flowsolver->Setup(GetDt(ctx));

    mfem::real_t u_inf_loc = u_gf->Normlinf();
    mfem::real_t p_inf_loc = p_gf->Normlinf();
    mfem::real_t u_inf = mfem::GlobalLpNorm(mfem::infinity(), u_inf_loc, MPI_COMM_WORLD);
    mfem::real_t p_inf = mfem::GlobalLpNorm(mfem::infinity(), p_inf_loc, MPI_COMM_WORLD);

    if (Mpi::Root())
    {
        std::cout << "After loading from checkpoint in LoadCheckpoint: u_gf Norml2 = "
                  << u_inf << ", p_gf Norml2 = " << p_inf << std::endl;
    }

    return true;
}

void SamplePoints(mfem::ParGridFunction* sol,
                                mfem::ParMesh* pmesh,
                                int step,
                                double time,
                                const std::string &suffix,
                                const s_NavierContext* ctx)
{
   // MPI setup
   MPI_Comm comm = pmesh->GetComm();
   int rank, size;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &size);

   // Construct the main directory name with suffix
   std::string main_dir = "SamplePoints" + suffix +
                          "_Re" + std::to_string(static_cast<int>(GetReynum(ctx))) +
                          "NumPtsPerDir" + std::to_string(GetNumPts(ctx)) +
                   + "RefLv" + std::to_string(GetElementSubdivisions(ctx) + GetElementSubdivisionsParallel(ctx))
                   + "P" + std::to_string(GetOrder(ctx));

   // Create subdirectory for this cycle step
   std::string cycle_dir = main_dir + "/cycle_" + std::to_string(step);
   // Construct the filename inside the cycle directory
   std::string fname = cycle_dir + "/include_both_boundaries_" + std::to_string(step) + ".txt";

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

   // Sampling setup
   int npts = GetOrder(ctx) + 2;  // Number of sample points per coordinate direction

   // Local arrays to store data from the local elements
   std::vector<double> local_x, local_y, local_z;
   std::vector<double> local_velx, local_vely, local_velz;

   mfem::FiniteElementSpace *fes = sol->FESpace();
   int vdim = fes->GetVDim();

   // Loop over local elements
   for (int e = 0; e < pmesh->GetNE(); e++)
   {
      // Get element transformation for element e
      mfem::ElementTransformation *Trans = pmesh->GetElementTransformation(e);
      
      // For each element, loop over a uniform grid of points in the reference element [0,1]^d
      for (int iz = 0; iz < npts; iz++)
      {
         double z_ref = (npts == 1) ? 0.5 : static_cast<double>(iz) / (npts - 1);
         for (int iy = 0; iy < npts; iy++)
         {
            double y_ref = (npts == 1) ? 0.5 : static_cast<double>(iy) / (npts - 1);
            for (int ix = 0; ix < npts; ix++)
            {
               double x_ref = (npts == 1) ? 0.5 : static_cast<double>(ix) / (npts - 1);
               mfem::IntegrationPoint ip;
               ip.Set3(x_ref, y_ref, z_ref); // Sample point in reference element

               // Get the physical coordinates for this sample point
               mfem::Vector phys_coords(Trans->GetSpaceDim());
               Trans->Transform(ip, phys_coords);

               double x_physical = phys_coords(0);
               double y_physical = phys_coords(1);
               double z_physical = phys_coords(2);

               // Evaluate the solution at the sample point
               mfem::Vector u_val(vdim);
               sol->GetVectorValue(*Trans, ip, u_val);
               double u_x = u_val(0);
               double u_y = u_val(1);
               double u_z = u_val(2);

               // Append sample point data to local arrays
               local_x.push_back(x_physical);
               local_y.push_back(y_physical);
               local_z.push_back(z_physical);
               local_velx.push_back(u_x);
               local_vely.push_back(u_y);
               local_velz.push_back(u_z);
            } // ix
         } // iy
      } // iz
   } // for each local element

   // Prepare the data string, including the header on rank 0
   std::string data_str;
   if (rank == 0)
   {
      std::ostringstream header_stream;
      header_stream << "3D Taylor Green Vortex\n"
                    << "Order = " << GetOrder(ctx) << "\n"
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
      std::cout << "Sampled data file saved: " << fname << std::endl;

   // Final synchronization
   MPI_Barrier(MPI_COMM_WORLD);
}


