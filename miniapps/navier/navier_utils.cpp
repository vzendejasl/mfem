// navier_utils.cpp
#include "navier_utils.hpp"
#include <algorithm> // for std::remove_if
#include <string>
#include <cstdio> // for popen, pclose
// #include <adios2.h>

using namespace mfem;
using namespace navier;

// Helper function to find the last checkpoint step
int FindLastCheckpointStep(const s_NavierContext* ctx)
{
    std::string main_dir;
    std::string command;
    if (GetVisit(ctx)) {
        main_dir = std::string("VisitData_")
                   + "Re" + std::to_string(static_cast<int>(GetReynum(ctx)))
                   + "NumPtsPerDir" + std::to_string(GetNumPts(ctx))
                   + "RefLv" + std::to_string(GetElementSubdivisions(ctx) + GetElementSubdivisionsParallel(ctx))
                   + "P" + std::to_string(GetOrder(ctx));
        command = "ls " + main_dir + " | grep mfem.root |"
                  + " sed 's/output_visit_//' | sort -n | tail -1 | sed 's/.mfem_root//'";
    }
    else if (GetConduit(ctx))
    {
        main_dir = std::string("ConduitData_")
                   + "Re" + std::to_string(static_cast<int>(GetReynum(ctx)))
                   + "NumPtsPerDir" + std::to_string(GetNumPts(ctx))
                   + "RefLv" + std::to_string(GetElementSubdivisions(ctx) + GetElementSubdivisionsParallel(ctx))
                   + "P" + std::to_string(GetOrder(ctx));
        command = "ls " + main_dir + " | grep .root |"
                  + " sed 's/output_conduit_//' | sort -n | tail -1 | sed 's/.root//'";
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
                                + "/output_visit";

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
                                  + "/output_conduit";

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

     if(GetFilter(ctx)){
       flowsolver->SetFilterAlpha(1e-12); // Update later
       flowsolver->SetCutoffModes(GetOrder(ctx)-1);   // Cut off highest mode
     }

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
   std::string fname = cycle_dir + "/SampledData" + std::to_string(step) + ".txt";

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
   int npts = GetOrder(ctx);  // Number of sample points per coordinate direction
   if(GetOverSample(ctx)){
     npts = GetOrder(ctx) + 1;
   }

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
      for (int iz = 0; iz <= npts; iz++)
      {
         double z_ref = static_cast<double>(iz) / npts;
         for (int iy = 0; iy <= npts; iy++)
         {
            double y_ref = static_cast<double>(iy) / npts;
            for (int ix = 0; ix <= npts; ix++)
            {
               double x_ref = static_cast<double>(ix) / npts;
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
                    << "Order = " << GetOrder(ctx) << ", " << "Over sample = " << GetOverSample(ctx) << "\n"
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

/*
// Loop through each element and save each dof using an integration rule.
// Also, eliminate copies of DOFS
void SamplePointsAtDoFs(ParGridFunction      *sol,
                        ParMesh              *pmesh,
                        int                   step,
                        double                time,
                        const std::string    &suffix,
                        const s_NavierContext* ctx)
{
   // MPI setup
   MPI_Comm comm = pmesh->GetComm();
   int rank, size;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &size);

   // Construct the main directory name with suffix
   std::string main_dir = "SamplePointsAtDofs" + suffix +
                          "_Re" + std::to_string(static_cast<int>(GetReynum(ctx))) +
                          "NumPtsPerDir" + std::to_string(GetNumPts(ctx)) +
                   + "RefLv" + std::to_string(GetElementSubdivisions(ctx) + GetElementSubdivisionsParallel(ctx))
                   + "P" + std::to_string(GetOrder(ctx));

   // Create subdirectory for this cycle step
   std::string cycle_dir = main_dir + "/cycle_" + std::to_string(step);
   std::string fname = cycle_dir + "/SampledData" + std::to_string(step) + ".txt";

   // Create directories on rank 0
   if (rank == 0)
   {
      if (system(("mkdir -p " + main_dir).c_str()) != 0)
         std::cerr << "Error creating " << main_dir << " directory!" << std::endl;
      if (system(("mkdir -p " + cycle_dir).c_str()) != 0)
         std::cerr << "Error creating " << cycle_dir << " directory!" << std::endl;
   }

   MPI_Barrier(MPI_COMM_WORLD);

   // Get element information
   mfem::FiniteElementSpace *fes = sol->FESpace();
   int vdim = fes->GetVDim();
   const FiniteElement *fe = fes->GetFE(0);
   const IntegrationRule &fe_nodes = fe->GetNodes();
   
   // Coordinate key function
   auto coord_key = [](double x, double y, double z) -> std::string {
       std::ostringstream oss;
       oss << std::scientific << std::setprecision(17) << x << "," << y << "," << z;
       return oss.str();
   };
   
   // Phase 1: Each processor samples and deduplicates locally
   std::set<std::string> seen_coords;
   std::vector<double> local_x, local_y, local_z;
   std::vector<double> local_velx, local_vely, local_velz;
   
   int local_total_dofs = 0;
   int local_duplicates = 0;

   for (int e = 0; e < pmesh->GetNE(); e++)
   {
      mfem::ElementTransformation *Trans = pmesh->GetElementTransformation(e);

      for (int i = 0; i < fe_nodes.GetNPoints(); ++i){
        const IntegrationPoint &ip = fe_nodes.IntPoint(i);

        Trans->SetIntPoint(&ip);
        Vector phys_pt;
        Trans->Transform(ip,phys_pt);

        local_total_dofs++;
        
        // Local deduplication
        std::string coord_str = coord_key(phys_pt[0], phys_pt[1], phys_pt[2]);
        
        if (seen_coords.find(coord_str) != seen_coords.end()) {
            local_duplicates++;
            continue; // Skip local duplicate
        }
        seen_coords.insert(coord_str);

        Vector vel_val(vdim);
        sol->GetVectorValue(*Trans,ip,vel_val);

        local_x.push_back(phys_pt[0]);
        local_y.push_back(phys_pt[1]);
        local_z.push_back(phys_pt[2]);
        local_velx.push_back(vel_val[0]);
        local_vely.push_back(vel_val[1]);
        local_velz.push_back(vel_val[2]);
      }
   }

   int local_unique = local_x.size();
   
   if (rank == 0) {
       mfem::out << "Phase 1 complete: Local deduplication\n";
       mfem::out << "  Rank 0: " << local_total_dofs << " total, " 
                 << local_unique << " unique, " << local_duplicates << " local duplicates\n";
   }

   // Phase 2: Gather all locally unique data to root for global deduplication
   
   // First, gather the counts from all processors
   std::vector<int> all_counts(size);
   MPI_Gather(&local_unique, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, comm);
   
   // Calculate displacements for gathering variable-length data
   std::vector<int> displs(size);
   int total_gathered = 0;
   if (rank == 0) {
       for (int i = 0; i < size; ++i) {
           displs[i] = total_gathered;
           total_gathered += all_counts[i];
       }
       mfem::out << "Phase 2: Gathering " << total_gathered << " locally unique DOFs to root\n";
   }
   
   // Prepare arrays to receive all data on root
   std::vector<double> all_x, all_y, all_z;
   std::vector<double> all_velx, all_vely, all_velz;
   
   if (rank == 0) {
       all_x.resize(total_gathered);
       all_y.resize(total_gathered);
       all_z.resize(total_gathered);
       all_velx.resize(total_gathered);
       all_vely.resize(total_gathered);
       all_velz.resize(total_gathered);
   }
   
   // Gather all coordinate and velocity data to root
   MPI_Gatherv(local_x.data(), local_unique, MPI_DOUBLE,
               all_x.data(), all_counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
   MPI_Gatherv(local_y.data(), local_unique, MPI_DOUBLE,
               all_y.data(), all_counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
   MPI_Gatherv(local_z.data(), local_unique, MPI_DOUBLE,
               all_z.data(), all_counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
   MPI_Gatherv(local_velx.data(), local_unique, MPI_DOUBLE,
               all_velx.data(), all_counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
   MPI_Gatherv(local_vely.data(), local_unique, MPI_DOUBLE,
               all_vely.data(), all_counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
   MPI_Gatherv(local_velz.data(), local_unique, MPI_DOUBLE,
               all_velz.data(), all_counts.data(), displs.data(), MPI_DOUBLE, 0, comm);

   // Phase 3: Root processor does global deduplication
   std::vector<double> final_x, final_y, final_z;
   std::vector<double> final_velx, final_vely, final_velz;
   int global_duplicates = 0;
   int expected_total_dofs = 0;  // Will be calculated on root
   
   if (rank == 0) {
       mfem::out << "Phase 3: Global deduplication on root processor\n";
       
       // Calculate expected count dynamically
       int num_pts_per_dir = GetNumPts(ctx);  
       int order = GetOrder(ctx);              
       int expected_coords_per_dir = num_pts_per_dir * order + 1;
       expected_total_dofs = expected_coords_per_dir * expected_coords_per_dir * expected_coords_per_dir;
       
       std::set<std::string> global_seen;
       
       for (int i = 0; i < total_gathered; ++i) {
           std::string coord_str = coord_key(all_x[i], all_y[i], all_z[i]);
           
           if (global_seen.find(coord_str) != global_seen.end()) {
               global_duplicates++;
               continue; // Skip global duplicate
           }
           global_seen.insert(coord_str);
           
           // Keep this globally unique DOF
           final_x.push_back(all_x[i]);
           final_y.push_back(all_y[i]);
           final_z.push_back(all_z[i]);
           final_velx.push_back(all_velx[i]);
           final_vely.push_back(all_vely[i]);
           final_velz.push_back(all_velz[i]);
       }
       
       int final_count = final_x.size();
       
       mfem::out << "Global Deduplication Results:\n";
       mfem::out << "  Mesh: " << GetNumPts(ctx) << "^3 elements, Order " << GetOrder(ctx) << "\n";
       mfem::out << "  Expected coords per direction: " << (GetNumPts(ctx) * GetOrder(ctx) + 1) << "\n";
       mfem::out << "  Total gathered: " << total_gathered << "\n";
       mfem::out << "  Global duplicates removed: " << global_duplicates << "\n";
       mfem::out << "  Final unique DOFs: " << final_count << "\n";
       mfem::out << "  Target (calculated): " << expected_total_dofs << "\n";
       mfem::out << "  Accuracy: " << (100.0 * final_count / (double)expected_total_dofs) << "%\n";
       mfem::out << "  Perfect deduplication: " << (final_count == expected_total_dofs ? "YES" : "NO") << "\n";
   }

   // Phase 4: Root writes the final deduplicated file
   if (rank == 0) {
       std::ofstream outfile(fname);
       outfile << std::scientific << std::setprecision(16);
       
       outfile << "3D Taylor Green Vortex (Perfect MPI Deduplication)\n"
               << "Order = " << GetOrder(ctx) << "\n"
               << "Step = " << step << " "
               << "Time = " << time << "\n"
               << "Global unique DOFs = " << final_x.size() << " (target: " << expected_total_dofs << ")\n"
               << "==================================================================="
               << "==========================================================================\n"
               << "            x                      y                      z                   vecx                   vecy                   vecz\n";
       
       for (size_t i = 0; i < final_x.size(); i++) {
           outfile << std::setw(20) << final_x[i] << " "
                   << std::setw(20) << final_y[i] << " "
                   << std::setw(20) << final_z[i] << " "
                   << std::setw(20) << final_velx[i] << " "
                   << std::setw(20) << final_vely[i] << " "
                   << std::setw(20) << final_velz[i] << "\n";
       }
       
       outfile.close();
       std::cout << "Perfect deduplicated file saved: " << fname << std::endl;
   }

   MPI_Barrier(MPI_COMM_WORLD);
}*/


/*
// Loop through each element and save each dof using an integration rule
void SamplePointsAtDoFs(ParGridFunction      *sol,
                        ParMesh              *pmesh,
                        int                   step,
                        double                time,
                        const std::string    &suffix,
                        const s_NavierContext* ctx)
{
   // MPI setup
   MPI_Comm comm = pmesh->GetComm();
   int rank, size;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &size);

   // Construct the main directory name with suffix
   std::string main_dir = "SamplePointsAtDofs" + suffix +
                          "_Re" + std::to_string(static_cast<int>(GetReynum(ctx))) +
                          "NumPtsPerDir" + std::to_string(GetNumPts(ctx)) +
                   + "RefLv" + std::to_string(GetElementSubdivisions(ctx) + GetElementSubdivisionsParallel(ctx))
                   + "P" + std::to_string(GetOrder(ctx));

   // Create subdirectory for this cycle step
   std::string cycle_dir = main_dir + "/cycle_" + std::to_string(step);
   // Construct the filename inside the cycle directory
   std::string fname = cycle_dir + "/SampledData" + std::to_string(step) + ".txt";

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

   // Local arrays to store data from the local elements
   std::vector<double> local_x, local_y, local_z;
   std::vector<double> local_velx, local_vely, local_velz;

   mfem::FiniteElementSpace *fes = sol->FESpace();
   int vdim = fes->GetVDim();

   // Get element information
   const FiniteElement *fe = fes->GetFE(0);
   const IntegrationRule &fe_nodes = fe->GetNodes();
   const int NPoints = fe_nodes.GetNPoints();
   const int NElements = pmesh->GetNE();

   // Loop over local elements
   for (int e = 0; e < pmesh->GetNE(); e++)
   {
      // Get element transformation for element e
      mfem::ElementTransformation *Trans = pmesh->GetElementTransformation(e);

      for (int i = 0; i < fe_nodes.GetNPoints(); ++i){
        const IntegrationPoint &ip = fe_nodes.IntPoint(i);

        Trans->SetIntPoint(&ip);
        Vector phys_pt;
        Trans->Transform(ip,phys_pt);

        Vector vel_val(vdim);
        sol->GetVectorValue(*Trans,ip,vel_val);

        local_x.push_back(phys_pt[0]);
        local_y.push_back(phys_pt[1]);
        local_z.push_back(phys_pt[2]);
        local_velx.push_back(vel_val[0]);
        local_vely.push_back(vel_val[1]);
        local_velz.push_back(vel_val[2]);
      }
   }
      
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
*/

// Parell version of extracting unique dofs, but 
// only works in serial.
void SamplePointsAtDoFs(ParGridFunction      *sol,
                        ParMesh              *pmesh,
                        int                   step,
                        double                time,
                        const std::string    &suffix,
                        const s_NavierContext* ctx)
{
  // Get FE space
  ParFiniteElementSpace *vfes = sol->ParFESpace();
  MPI_Comm comm = vfes->GetComm();
  int rank;
  MPI_Comm_rank(comm, &rank);

  // Create position coeffients that will be used
  // to construct a new grid function
  VectorFunctionCoefficient position_coeff(3,
     [](const Vector &x, Vector &y){ y = x; });

  // Create position grid function
  ParGridFunction position_gf(vfes);
  position_gf.ProjectCoefficient(position_coeff);

  // Create hypre vectors that will store dof position
  // and velocity
  std::unique_ptr<HypreParVector> vel_tdof(sol->GetTrueDofs());
  std::unique_ptr<HypreParVector> pos_tdof(position_gf.GetTrueDofs());

  // Create pointer to velocity and position vectors
  // We are going to access these directly later
  const mfem::real_t *vel_data = vel_tdof->HostRead();
  const mfem::real_t *pos_data = pos_tdof->HostRead();

  // Compute size of velocity dof length (will be the same for position)
  // and divide by the dimension of FE space for component wise extraction
  int vel_true_size = vel_tdof->Size();
  int scalar_true_dofs = vel_true_size / vfes->GetVDim();
  
  // Each processor writes its own file 
  std::string filename = "samples_rank" + std::to_string(rank) + ".txt";
  std::ofstream out(filename);
  out << std::scientific << std::setprecision(16);
  out << "# x               y               z               "
         "u               v               w\n";
  
  // Dump the data
  for (int  i = 0; i < scalar_true_dofs; ++i)
  {
      out << std::setw(20) << pos_data[i]           << " "  
          << std::setw(20) << pos_data[i+scalar_true_dofs]         << " "  
          << std::setw(20) << pos_data[i+2*scalar_true_dofs]       << " "  
          << std::setw(20) << vel_data[i]           << " " 
          << std::setw(20) << vel_data[i+scalar_true_dofs]         << " "  
          << std::setw(20) << vel_data[i+2*scalar_true_dofs]       << "\n";
  }
  out.close();
}
 

/*
// This serial version works
void SamplePointsAtDoFs(ParGridFunction      *sol,
                        ParMesh              *pmesh,
                        int                   step,
                        double                time,
                        const std::string    &suffix,
                        const s_NavierContext* ctx)
{
   ParFiniteElementSpace *vfes = sol->ParFESpace();
   MPI_Comm comm = vfes->GetComm();
   int rank; MPI_Comm_rank(comm, &rank);

   const int scalar_dofs = vfes->GetNDofs();

   VectorFunctionCoefficient position_coeff(3,
      [](const Vector &x, Vector &y){ y = x; });

   ParGridFunction position_gf(vfes);
   position_gf.ProjectCoefficient(position_coeff);

   std::unique_ptr<HypreParVector> vel_tdof (sol->GetTrueDofs());
   std::unique_ptr<HypreParVector> pos_tdof (position_gf.GetTrueDofs());

   const double *vel_data = vel_tdof->Read();
   const double *pos_data = pos_tdof->Read();
   
   // vel_tdof, xyz_tdof are HypreParVector* (by-VDIM ordering assumed)
   auto vel = std::unique_ptr<Vector>(vel_tdof->GlobalVector());
   auto xyz = std::unique_ptr<Vector>(pos_tdof->GlobalVector());

   const double *V = vel->Read();              // [   u …   v …   w … ]
   const double *X = xyz->Read();              // [   x …   y …   z … ]
   const HYPRE_BigInt n = vel->Size() / 3;     // DOFs per component
   
   std::ofstream out("samples.txt");
   out << std::scientific << std::setprecision(16);
   out << "# x               y               z               "
          "u               v               w\n";
   
   for (int i = 0; i < n; ++i)
   {
       out << std::setw(20) << X[i]           << " "
           << std::setw(20) << X[i+n]         << " "
           << std::setw(20) << X[i+2*n]       << " "
           << std::setw(20) << V[i]           << " "
           << std::setw(20) << V[i+n]         << " "
           << std::setw(20) << V[i+2*n]       << "\n";
   }


   // std::vector<double> all_x(scalar_dofs), all_y(scalar_dofs), all_z(scalar_dofs);
   // std::vector<double> all_u(scalar_dofs), all_v(scalar_dofs), all_w(scalar_dofs);

   // for (int i = 0; i < scalar_dofs; ++i)
   // {
   //    all_x[i] = pos_data[i];
   //    all_y[i] = pos_data[i + scalar_dofs];
   //    all_z[i] = pos_data[i + 2*scalar_dofs];

   //    all_u[i] = vel_data[i];
   //    all_v[i] = vel_data[i + scalar_dofs];
   //    all_w[i] = vel_data[i + 2*scalar_dofs];
   // }

   // std::string main_dir = "SamplePointsAtDofs" + suffix +
   //     "_Re" + std::to_string(static_cast<int>(GetReynum(ctx))) +
   //     "NumPtsPerDir" + std::to_string(GetNumPts(ctx)) +
   //     "RefLv" + std::to_string(GetElementSubdivisions(ctx) +
   //                              GetElementSubdivisionsParallel(ctx)) +
   //     "P" + std::to_string(GetOrder(ctx));

   // std::string cycle_dir = main_dir + "/cycle_" + std::to_string(step);
   // std::string fname     = cycle_dir + "/SampledData" + std::to_string(step) + ".txt";

   // if (rank == 0)
   // {
   //    if (system(("mkdir -p " + cycle_dir).c_str()) != 0)
   //       std::cerr << "Error creating " << cycle_dir << std::endl;
   // }
   // MPI_Barrier(comm);   // ensure directory exists before any rank opens file

   // if (rank == 0)
   // {
   //    std::ofstream outfile(fname);
   //    if (!outfile.is_open())
   //    {
   //       std::cerr << "Error opening file " << fname << std::endl;
   //       return;
   //    }

   //    outfile << std::scientific << std::setprecision(16);
   //    outfile << "3D Taylor Green Vortex (PARALLEL true-DOFs)\n"
   //            << "Order = " << GetOrder(ctx) << " (sampling at DoFs)\n"
   //            << "Step = "  << step << ", Time = " << time << ", "
   //            << "Total DOFs per component = " << scalar_dofs << "\n"
   //            << "====================================================================================\n"
   //            << "            x                      y                      z"
   //            << "                   valx                   valy                   valz\n";

   //    for (int i = 0; i < scalar_dofs; ++i)
   //    {
   //       outfile << std::setw(20) << all_x[i] << " "
   //               << std::setw(20) << all_y[i] << " "
   //               << std::setw(20) << all_z[i] << " "
   //               << std::setw(20) << all_u[i] << " "
   //               << std::setw(20) << all_v[i] << " "
   //               << std::setw(20) << all_w[i] << "\n";
   //    }
   //    std::cout << "Sampled data file saved: " << fname << std::endl;
   // }
}
*/




/*
// -------------------------------------------------------------
// Writes (x,y,z,u_x,u_y,u_z) for each sample in parallel via ADIOS2
// -------------------------------------------------------------
void SamplePointsAdios(mfem::ParGridFunction* sol,
                       mfem::ParMesh* pmesh,
                       int step,
                       double time,
                       const std::string &suffix,
                       bool oversample,
                       const s_NavierContext* ctx)
{
    MPI_Comm comm = pmesh->GetComm();
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Build output directories
    std::string main_dir = "SamplePoints" + suffix
        + "_Re"  + std::to_string(static_cast<int>(GetReynum(ctx)))
        + "NumPtsPerDir" + std::to_string(GetNumPts(ctx))
        + "RefLv" + std::to_string(GetElementSubdivisions(ctx)
                                 + GetElementSubdivisionsParallel(ctx))
        + "P"    + std::to_string(GetOrder(ctx));
    std::string cycle_dir = main_dir + "/cycle_" + std::to_string(step);
    std::string mode = oversample ? "WithOverSample" : "WithoutOverSample";
    std::string fname = cycle_dir + "/SampledData" + mode
                      + std::to_string(step) + ".bp";

    if (rank == 0)
    {
        system(("mkdir -p " + main_dir).c_str());
        system(("mkdir -p " + cycle_dir).c_str());
    }
    MPI_Barrier(comm);

    // Number of samples per direction
    int npts = GetOrder(ctx) + (oversample ? 1 : 0);
    if (rank == 0)
    {
        std::cout << "Sampling " << npts << "^3 per element\n";
    }

    // Gather sampled data
    std::vector<double> local_x, local_y, local_z;
    std::vector<double> local_velx, local_vely, local_velz;
    auto fes  = sol->FESpace();
    int  vdim = fes->GetVDim();
    int  localNE = pmesh->GetNE();

    for (int e = 0; e < localNE; ++e)
    {
        auto Trans = pmesh->GetElementTransformation(e);
        for (int iz = 0; iz <= npts; ++iz)
        for (int iy = 0; iy <= npts; ++iy)
        for (int ix = 0; ix <= npts; ++ix)
        {
            IntegrationPoint ip;
            ip.Set3(double(ix)/npts,
                    double(iy)/npts,
                    double(iz)/npts);

            Vector phys(Trans->GetSpaceDim());
            Trans->Transform(ip, phys);

            Vector u_val(vdim);
            sol->GetVectorValue(*Trans, ip, u_val);

            local_x .push_back(phys(0));
            local_y .push_back(phys(1));
            local_z .push_back(phys(2));
            local_velx.push_back(u_val(0));
            local_vely.push_back(u_val(1));
            local_velz.push_back(u_val(2));
        }
    }

    // Compute global count & per-rank offset
    size_t localCount = local_x.size();
    size_t globalCount = 0;
    MPI_Allreduce(&localCount, &globalCount, 1,
                  MPI_UNSIGNED_LONG, MPI_SUM, comm);

    size_t offset = 0;
    if (rank > 0)
    {
        MPI_Exscan(&localCount, &offset, 1,
                   MPI_UNSIGNED_LONG, MPI_SUM, comm);
    }

    // Pack into [N_local × 6] buffer: (x,y,z, u_x,u_y,u_z)
    std::vector<double> buf(6*localCount);
    for (size_t i = 0; i < localCount; ++i)
    {
        buf[6*i + 0] = local_x [i];
        buf[6*i + 1] = local_y [i];
        buf[6*i + 2] = local_z [i];
        buf[6*i + 3] = local_velx[i];
        buf[6*i + 4] = local_vely[i];
        buf[6*i + 5] = local_velz[i];
    }

    // ADIOS2
    adios2::ADIOS   adios(comm);
    auto io = adios.DeclareIO("SampleIO");
    io.SetEngine("BP4");
    io.SetParameters({{"NumAggregator","1"}});

    auto var = io.DefineVariable<double>(
        "samples",
        { globalCount, 6ULL },
        { offset,      0ULL },
        { localCount,  6ULL },
        adios2::DataType::double
    );

    auto eng = io.Open(fname, adios2::Mode::Write);
    eng.Put(var, buf.data());
    eng.Close();

    if (rank == 0)
    {
        std::cout << "ADIOS2 wrote " << globalCount
                  << " samples to " << fname << "\n";
    }
    MPI_Barrier(comm);
}*/

