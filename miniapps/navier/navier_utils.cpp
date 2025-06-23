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
// Working serial version
void SamplePointsAtDoFs(ParGridFunction      *sol,     // velocity (u,v,w)
                        ParMesh              *pmesh,   // mesh (serial assumed)
                        int                   step,    // output index / cycle
                        double                time,    // physical time
                        const std::string    &suffix,  // optional tag
                        const s_NavierContext* ctx)
{
   ParFiniteElementSpace *vfes = sol->ParFESpace();
   MPI_Comm comm = sol->ParFESpace()->GetComm();
   int rank; MPI_Comm_rank(comm, &rank);

   const int vdim = vfes->GetVDim();
   MFEM_VERIFY(vdim == 3,
               "SamplePointsAtDoFs expects a 3-component velocity field.");

   // Build output directory and filename
   std::string main_dir = "SamplePointsAtDofs" + suffix +
      "_Re" + std::to_string(static_cast<int>(GetReynum(ctx))) +
      "RefLv" + std::to_string(GetElementSubdivisions(ctx) + GetElementSubdivisionsParallel(ctx)) +
      "P" + std::to_string(GetOrder(ctx));

   std::string cycle_dir = main_dir + "/cycle_" + std::to_string(step);
   std::string fname = cycle_dir + "/SampledData" + std::to_string(step) + ".txt";

   if (rank == 0)
   {
      if (system(("mkdir -p " + main_dir).c_str()) != 0)
         std::cerr << "Error creating " << main_dir << " directory!" << std::endl;
      if (system(("mkdir -p " + cycle_dir).c_str()) != 0)
         std::cerr << "Error creating " << cycle_dir << " directory!" << std::endl;
   }
   MPI_Barrier(comm);

   // 1) Build coordinate grid-function (x,y,z) on the same FE space
   VectorFunctionCoefficient xyz_coeff(3,
      [](const Vector &x, Vector &y) { y = x; });
   ParGridFunction xyz(vfes);
   xyz.ProjectCoefficient(xyz_coeff);

   const int ND = vfes->GetNDofs();        // true dofs per component

   // 2) Assemble dump Vector: [x | y | z | u | v | w]
   Vector dump(6*ND);
   const double *cdata = xyz.HostRead();   // coordinates
   const double *vdata = sol->HostRead();  // velocity

   double ke_sum = 0.0;                    // accumulate ½|u|² over nodes
   for (int i = 0; i < ND; ++i)
   {
      // coords
      dump[i           ] = cdata[i];
      dump[i +   ND    ] = cdata[i + ND];
      dump[i + 2*ND    ] = cdata[i + 2*ND];

      // velocity
      double u = vdata[i];
      double v = vdata[i + ND];
      double w = vdata[i + 2*ND];

      dump[i + 3*ND    ] = u;
      dump[i + 4*ND    ] = v;
      dump[i + 5*ND    ] = w;

      ke_sum += 0.5 * (u*u + v*v + w*w);
   }
   double ke_avg = ke_sum / ND;            

   // 3) Open file, write header, then delegate to Vector::Print
   std::ofstream ofs(fname);
   ofs << std::setprecision(15) << std::scientific;

   ofs << "# Sampled nodal values\n"
       << "# Step   " << step  << "\n"
       << "# Time   " << time  << "\n"
       << "# AvgKE  " << ke_avg << "   (0.5*|u|² averaged over " << ND << " nodes)\n"
       << "# Layout: block-wise [x y z u v w], " << ND << " entries per block\n";

   dump.Print(ofs);        // one number per line / MFEM default formatting
   ofs.close();
}
*/


// Parallel version -----------------------------------------------------------
void SamplePointsAtDoFs(ParGridFunction      *sol,     // velocity (u,v,w)
                        ParMesh              *pmesh,   // mesh (parallel)
                        int                   step,    // output index / cycle
                        double                time,    // physical time
                        const std::string    &suffix,  // optional tag
                        const s_NavierContext* ctx)
{
   ParFiniteElementSpace *vfes = sol->ParFESpace();
   MPI_Comm              comm  = vfes->GetComm();
   int rank, nprocs;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &nprocs);

   const int vdim = vfes->GetVDim();
   MFEM_VERIFY(vdim == 3,
               "SamplePointsAtDoFs expects a 3-component velocity field.");

   // ------------------------------------------------------------------ 0) I/O paths
   std::string main_dir  = "SamplePointsAtDofs" + suffix +
                           "_Re"     + std::to_string(static_cast<int>(GetReynum(ctx))) +
                           "RefLv"   + std::to_string(GetElementSubdivisions(ctx) +
                                                    GetElementSubdivisionsParallel(ctx)) +
                           "P"       + std::to_string(GetOrder(ctx));
   std::string cycle_dir = main_dir + "/cycle_" + std::to_string(step);
   std::string fname     = cycle_dir + "/SampledData" +
                           std::to_string(step) + ".txt";

   if (rank == 0)
   {
      if (system(("mkdir -p " + main_dir ).c_str()) != 0)
         mfem::err << "Error creating " << main_dir  << " directory!\n";
      if (system(("mkdir -p " + cycle_dir).c_str()) != 0)
         mfem::err << "Error creating " << cycle_dir << " directory!\n";
   }
   MPI_Barrier(comm);

   // ------------------------------------------------------------------ 1) build (x,y,z)   on the same FE space
   VectorFunctionCoefficient xyz_coeff(3,
      [](const Vector &x, Vector &y) { y = x; });
   ParGridFunction xyz(vfes);
   xyz.ProjectCoefficient(xyz_coeff);

   // ------------------------------------------------------------------ 2) unique true-dof vectors – OWNED rows only
   auto xyz_hpv = std::unique_ptr<HypreParVector>(xyz.ParallelAssemble());
   auto vel_hpv = std::unique_ptr<HypreParVector>(sol->ParallelAssemble());
   
   const HYPRE_Int *part   = xyz_hpv->GetPartitioning();      // size nprocs+1
   HYPRE_Int        loc_owned_gl = part[rank+1] - part[rank]; // rows truly owned
   MFEM_VERIFY(loc_owned_gl % vdim == 0,
               "vdim does not divide owned rows!");
   const HYPRE_Int loc_nd = loc_owned_gl / vdim;           // owned DOFs/comp
   
   const double *cdata = xyz_hpv->GetData();               // data[0:loc_owned_gl)
   const double *vdata = vel_hpv->GetData();
   
   std::vector<double> X(loc_nd), Y(loc_nd), Z(loc_nd),
                       U(loc_nd), V(loc_nd), W(loc_nd);
   
   double ke_sum_local = 0.0;
   for (HYPRE_Int i = 0; i < loc_nd; ++i)
   {
      // coords (first 3 * loc_nd entries are owned x,y,z)
      X[i] = cdata[i];
      Y[i] = cdata[i +      loc_nd];
      Z[i] = cdata[i + 2 *  loc_nd];
   
      // velocity
      U[i] = vdata[i];
      V[i] = vdata[i +      loc_nd];
      W[i] = vdata[i + 2 *  loc_nd];
   
      ke_sum_local += 0.5*(U[i]*U[i] + V[i]*V[i] + W[i]*W[i]);
   }
   


   // ------------------------------------------------------------------ 3) global tallies
   long long ND_global_ll = 0;
   long long ND_local_ll  = static_cast<long long>(loc_nd);
   MPI_Allreduce(&ND_local_ll, &ND_global_ll, 1, MPI_LONG_LONG, MPI_SUM, comm);
   const long long ND_global = ND_global_ll;

   double ke_sum_global = 0.0;
   MPI_Allreduce(&ke_sum_local, &ke_sum_global, 1, MPI_DOUBLE, MPI_SUM, comm);
   const double ke_avg = ke_sum_global / static_cast<double>(ND_global);

   // ------------------------------------------------------------------ 4) gather counts/displs for Gatherv
   std::vector<int> counts(nprocs), displs(nprocs);
   int loc_nd_int = static_cast<int>(loc_nd);
   MPI_Gather(&loc_nd_int, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm);
   if (rank == 0)
   {
      displs[0] = 0;
      for (int p = 1; p < nprocs; ++p)
         displs[p] = displs[p-1] + counts[p-1];
   }

   // Root allocates global buffers
   std::vector<double> gX, gY, gZ, gU, gV, gW;
   if (rank == 0)
   {
      gX.resize(ND_global);
      gY.resize(ND_global);
      gZ.resize(ND_global);
      gU.resize(ND_global);
      gV.resize(ND_global);
      gW.resize(ND_global);
   }

   // ------------------------------------------------------------------ 5) gather component-wise
   auto gather = [&](const std::vector<double>& local,
                     std::vector<double>&       global)
   {
      MPI_Gatherv(const_cast<double*>(local.data()),   // sendbuf
                  loc_nd_int, MPI_DOUBLE,
                  global.data(), counts.data(), displs.data(),
                  MPI_DOUBLE, 0, comm);
   };

   gather(X, gX);  gather(Y, gY);  gather(Z, gZ);
   gather(U, gU);  gather(V, gV);  gather(W, gW);

   // ------------------------------------------------------------------ 6) write on rank 0
   if (rank == 0)
   {
      // pack into MFEM::Vector so we can keep the original "dump.Print(ofs)" line
      mfem::Vector dump(6 * ND_global);
      for (long long i = 0; i < ND_global; ++i)
      {
         dump[i]                   = gX[i];
         dump[i +   ND_global   ]  = gY[i];
         dump[i + 2*ND_global  ]  = gZ[i];
         dump[i + 3*ND_global  ]  = gU[i];
         dump[i + 4*ND_global  ]  = gV[i];
         dump[i + 5*ND_global  ]  = gW[i];
      }

      std::ofstream ofs(fname);
      ofs << std::setprecision(15) << std::scientific;

      ofs << "# Sampled nodal values\n"
          << "# Step   "  << step      << "\n"
          << "# Time   "  << time      << "\n"
          << "# AvgKE  "  << ke_avg    << "   (0.5*|u|² averaged over "
          << ND_global    << " nodes)\n"
          << "# Layout: block-wise [x y z u v w], "
          << ND_global    << " entries per block\n";

      dump.Print(ofs);   // one number per line (MFEM default)
   }
}




/*
void SamplePointsAtDoFs(mfem::ParGridFunction* sol,
                        mfem::ParMesh* pmesh,
                        int step,
                        double time,
                        const std::string &suffix,
                        const s_NavierContext* ctx)
{
   MPI_Comm comm = pmesh->GetComm();
   int rank, size;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &size);

   // Build output directory and filename
   std::string main_dir = "SamplePointsAtDofs" + suffix +
      "_Re" + std::to_string(static_cast<int>(GetReynum(ctx))) +
      "RefLv" + std::to_string(GetElementSubdivisions(ctx) + GetElementSubdivisionsParallel(ctx)) +
      "P" + std::to_string(GetOrder(ctx));

   std::string cycle_dir = main_dir + "/cycle_" + std::to_string(step);
   std::string fname = cycle_dir + "/SampledData" + std::to_string(step) + ".txt";

   if (rank == 0)
   {
      if (system(("mkdir -p " + main_dir).c_str()) != 0)
         std::cerr << "Error creating " << main_dir << " directory!" << std::endl;
      if (system(("mkdir -p " + cycle_dir).c_str()) != 0)
         std::cerr << "Error creating " << cycle_dir << " directory!" << std::endl;
   }
   MPI_Barrier(comm);

   mfem::FiniteElementSpace *fes = sol->FESpace();
   const int vdim = fes->GetVDim();
   const int ndofs = fes->GetVSize(); // total dofs (including all vector components)

   // Prepare arrays to store DoF coordinates and values
   std::vector<double> local_x, local_y, local_z;
   std::vector<double> local_valx, local_valy, local_valz;

   mfem::Array<int> vdofs;
   mfem::Vector dof_val(vdim);

   for (int e = 0; e < fes->GetNE(); e++)
   {
      mfem::ElementTransformation *Trans = fes->GetMesh()->GetElementTransformation(e);
      const mfem::FiniteElement *fe = fes->GetFE(e);
      mfem::Array<int> vdofs;
      fes->GetElementVDofs(e, vdofs);

      const mfem::IntegrationRule &nodes = fe->GetNodes();
      MFEM_VERIFY(nodes != nullptr, "FiniteElement does not have nodes (not a nodal basis)!");

      for (int i = 0; i < fe->GetDof(); i++)
      {
         const mfem::IntegrationPoint &ip = nodes.IntPoint(i);
         mfem::Vector phys_coord(3); phys_coord = 0.0;
         Trans->Transform(ip, phys_coord);

         for (int d = 0; d < vdim; d++)
         {
            int vdof = vdofs[i + d * fe->GetDof()];
            double value = (*sol)[vdof];
            local_x.push_back(phys_coord(0));
            local_y.push_back(phys_coord.Size() > 1 ? phys_coord(1) : 0.0);
            local_z.push_back(phys_coord.Size() > 2 ? phys_coord(2) : 0.0);
            local_valx.push_back(d == 0 ? value : 0.0);
            local_valy.push_back(d == 1 ? value : 0.0);
            local_valz.push_back(d == 2 ? value : 0.0);
         }
      }
   }




   // Prepare the data string, including the header on rank 0
   std::string data_str;
   if (rank == 0)
   {
      std::ostringstream header_stream;
      header_stream << "3D Taylor Green Vortex\n"
                    << "Order = " << GetOrder(ctx) << " (sampling at DoFs)\n"
                    << "Step = " << step << "\n"
                    << "Time = " << std::scientific << std::setprecision(16) << time << "\n"
                    << "====================================================================================\n"
                    << "            x                      y                      z                   valx                   valy                   valz\n";
      data_str = header_stream.str();
   }

   // Append local data
   std::ostringstream local_data_stream;
   for (size_t i = 0; i < local_x.size(); i++)
   {
      local_data_stream << std::scientific << std::setprecision(16)
                        << std::setw(20) << local_x[i] << " "
                        << std::setw(20) << local_y[i] << " "
                        << std::setw(20) << local_z[i] << " "
                        << std::setw(20) << local_valx[i] << " "
                        << std::setw(20) << local_valy[i] << " "
                        << std::setw(20) << local_valz[i] << "\n";
   }
   data_str += local_data_stream.str();

   // Write out using MPI I/O
   MPI_File fh;
   int err = MPI_File_open(comm, fname.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
   if (err != MPI_SUCCESS)
   {
      if (rank == 0) std::cerr << "Error opening file " << fname << " with MPI I/O" << std::endl;
      MPI_Abort(comm, 1);
   }

   MPI_File_write_ordered(fh, data_str.c_str(), data_str.size(), MPI_CHAR, MPI_STATUS_IGNORE);

   MPI_File_close(&fh);

   if (rank == 0)
      std::cout << "Sampled data file saved: " << fname << std::endl;

   MPI_Barrier(comm);
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

