// navier_utils.cpp
#include "navier_utils.hpp"
#include <algorithm> // for std::remove_if
#include <string>
#include <cstdio> // for popen, pclose
#include "hdf5.h"
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
/*
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
}*/

// Parallel-only HDF5 writer (collective MPI-IO).
// On disk:
//   row_major = true  -> dataset shape [N_total_rows, ncols]
//   row_major = false -> dataset shape [ncols, N_total_rows]
// Your local buffer is always row-major: n_local_rows contiguous rows of length ncols.


#ifndef H5_HAVE_PARALLEL
#error "Parallel HDF5 not found. Rebuild/point HDF5 to a build configured with MPI (H5_HAVE_PARALLEL)."
#endif

static herr_t WriteH5Parallel2D(MPI_Comm           comm,
                                const std::string &path,
                                const std::string &dset_name,
                                const double      *local_rows,   // row-major buffer
                                hsize_t            n_local_rows,
                                int                ncols,
                                bool               row_major)
{
    int rank; MPI_Comm_rank(comm, &rank);

    // Global rows and my starting row (exclusive scan)
    hsize_t N_total = 0, my_start = 0;
    MPI_Allreduce(&n_local_rows, &N_total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
    MPI_Exscan(&n_local_rows, &my_start, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
    if (rank == 0) my_start = 0;

    // Parallel file driver
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL);

    // Create/truncate file
    hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    if (file < 0) return file;

    herr_t st = 0;
    hid_t fspace = -1, mspace = -1, dset = -1;

    if (row_major) {
        // Dataset [N_total, ncols]; write my row slab
        const hsize_t dims[2]   = { N_total, (hsize_t)ncols };
        const hsize_t offset[2] = { my_start, 0 };
        const hsize_t count[2]  = { n_local_rows, (hsize_t)ncols };

        fspace = H5Screate_simple(2, dims, nullptr);
        dset   = H5Dcreate2(file, dset_name.c_str(), H5T_NATIVE_DOUBLE,
                            fspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        if (n_local_rows > 0) H5Sselect_hyperslab(fspace, H5S_SELECT_SET, offset, nullptr, count, nullptr);
        else                  H5Sselect_none(fspace);

        if (n_local_rows > 0) {
            const hsize_t mdims[2] = { n_local_rows, (hsize_t)ncols };
            mspace = H5Screate_simple(2, mdims, nullptr);
            H5Sselect_all(mspace);
        } else {
            const hsize_t mdims[2] = { 1, (hsize_t)ncols };
            mspace = H5Screate_simple(2, mdims, nullptr);
            H5Sselect_none(mspace);
        }

        // Collective write
        hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE);
        const void *buf = (n_local_rows > 0) ? (const void*)local_rows : nullptr;
        st = H5Dwrite(dset, H5T_NATIVE_DOUBLE, mspace, fspace, dxpl, buf);
        H5Pclose(dxpl);
    } else {
        // Dataset [ncols, N_total]; write column slab by locally transposing
        std::vector<double> tmp((size_t)ncols * (size_t)std::max<hsize_t>(n_local_rows, 1));
        for (hsize_t r = 0; r < n_local_rows; ++r) {
            const double *src = local_rows + (size_t)r * (size_t)ncols;
            for (int c = 0; c < ncols; ++c)
                tmp[(size_t)c * (size_t)std::max<hsize_t>(n_local_rows,1) + (size_t)r] = src[c];
        }

        const hsize_t dims[2]   = { (hsize_t)ncols, N_total };
        const hsize_t offset[2] = { 0, my_start };
        const hsize_t count[2]  = { (hsize_t)ncols, n_local_rows };

        fspace = H5Screate_simple(2, dims, nullptr);
        dset   = H5Dcreate2(file, dset_name.c_str(), H5T_NATIVE_DOUBLE,
                            fspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        if (n_local_rows > 0) H5Sselect_hyperslab(fspace, H5S_SELECT_SET, offset, nullptr, count, nullptr);
        else                  H5Sselect_none(fspace);

        if (n_local_rows > 0) {
            const hsize_t mdims[2] = { (hsize_t)ncols, n_local_rows };
            mspace = H5Screate_simple(2, mdims, nullptr);
            H5Sselect_all(mspace);
        } else {
            const hsize_t mdims[2] = { (hsize_t)ncols, 1 };
            mspace = H5Screate_simple(2, mdims, nullptr);
            H5Sselect_none(mspace);
        }

        hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE);
        const void *buf = (n_local_rows > 0) ? (const void*)tmp.data() : nullptr;
        st = H5Dwrite(dset, H5T_NATIVE_DOUBLE, mspace, fspace, dxpl, buf);
        H5Pclose(dxpl);
    }

    // Cleanup
    if (dset   >= 0) H5Dclose(dset);
    if (mspace >= 0) H5Sclose(mspace);
    if (fspace >= 0) H5Sclose(fspace);
    H5Fclose(file);
    return st;  // >=0 on success, <0 on error
}



// -------------------------------
// Minimal collective text writer
// -------------------------------
static int WriteTextCollective(MPI_Comm comm,
                               const std::string &path,
                               const std::string &header_rank0,
                               const std::string &local_text)
{
    int rank; MPI_Comm_rank(comm, &rank);

    MPI_File fh;
    int err = MPI_File_open(comm, path.c_str(),
                            MPI_MODE_CREATE | MPI_MODE_WRONLY,
                            MPI_INFO_NULL, &fh);
    if (err != MPI_SUCCESS) return err;

    // (1) rank 0 writes header at offset 0
    MPI_Offset header_bytes = 0;
    if (rank == 0) {
        const MPI_Offset hsz = (MPI_Offset)header_rank0.size();
        if (hsz > 0) {
            MPI_Status st0;
            err = MPI_File_write_at(fh, 0,
                                    header_rank0.data(), (int)hsz,
                                    MPI_BYTE, &st0);
            if (err != MPI_SUCCESS) { MPI_File_close(&fh); return err; }
        }
        header_bytes = hsz;
    }
    MPI_Bcast(&header_bytes, 1, MPI_OFFSET, 0, comm);

    // (2) exclusive scan of local byte sizes
    const MPI_Offset my_bytes  = (MPI_Offset)local_text.size();
    MPI_Offset my_prefix = 0;
    MPI_Exscan(&my_bytes, &my_prefix, 1, MPI_OFFSET, MPI_SUM, comm);
    if (rank == 0) my_prefix = 0;

    // (3) byte-wise view + collective write
    MPI_File_set_view(fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);

    const MPI_Offset my_file_offset = header_bytes + my_prefix;
    MPI_Status st_all;
    const void *buf = local_text.size() ? (const void*)local_text.data() : (const void*)"";
    err = MPI_File_write_at_all(fh, my_file_offset,
                                buf, (int)my_bytes, MPI_BYTE, &st_all);

    MPI_File_close(&fh);
    return err; // MPI_SUCCESS on success
}

// -------------------------------
// Helpers: endpoint snapping & global-max-plane ownership
// -------------------------------
static inline void snap_endpoints(double &x, double a, double b, double eps)
{
    if (std::abs(x - a) <= eps) x = a;
    else if (std::abs(x - b) <= eps) x = b;
}

// Decide if this element "owns" the global max plane along a given axis,
// by checking if any of its 8 vertices touches the global max within eps.
static bool OwnsGlobalMaxPlane(mfem::ElementTransformation *T,
                               int axis,
                               double b, double eps)
{
    static const double c[2] = {0.0, 1.0};
    mfem::IntegrationPoint ip;
    mfem::Vector X(T->GetSpaceDim());
    double vmax = -std::numeric_limits<double>::infinity();

    for (int iz = 0; iz < 2; ++iz)
    for (int iy = 0; iy < 2; ++iy)
    for (int ix = 0; ix < 2; ++ix) {
        ip.Set3(c[ix], c[iy], c[iz]);
        T->Transform(ip, X);
        vmax = std::max(vmax, X(axis));
    }
    return (b - vmax) <= eps;
}

// -------------------------------
// Closed-grid sampler (hex meshes), moving-mesh safe
// -------------------------------
void SamplePoints(mfem::ParGridFunction* sol,
                  mfem::ParMesh*         pmesh,
                  int                    step,
                  double                 time,
                  const std::string&     suffix,
                  const s_NavierContext* ctx)
{
    MPI_Comm comm = pmesh->GetComm();
    int rank; MPI_Comm_rank(comm, &rank);

    // ---- 0) Output paths
    std::string main_dir = std::string("SamplePoints") + suffix +
                           "_Re" + std::to_string((int)GetReynum(ctx)) +
                           "NumPtsPerDir" + std::to_string(GetNumPts(ctx)) +
                           "RefLv" + std::to_string(GetElementSubdivisions(ctx) +
                                                    GetElementSubdivisionsParallel(ctx)) +
                           "P" + std::to_string(GetOrder(ctx));
    std::string cycle_dir = main_dir + "/cycle_" + std::to_string(step);
    std::string fname     = cycle_dir + "/SampledData" + std::to_string(step) + ".txt";

    if (rank == 0) {
        (void)system(("mkdir -p " + main_dir ).c_str());
        (void)system(("mkdir -p " + cycle_dir).c_str());
    }
    MPI_Barrier(comm);

    // ---- 1) Sampling resolution
    int npts = GetOrder(ctx);
    if (GetOverSample(ctx)) npts = GetOrder(ctx) + 1;
    MFEM_VERIFY(npts > 0, "npts must be positive.");
    const double inv_n = 1.0 / double(npts);

    // ---- 2) Global bounds + eps (recomputed each call → moving mesh OK)
    mfem::Vector bbmin(pmesh->SpaceDimension()), bbmax(pmesh->SpaceDimension());
    pmesh->GetBoundingBox(bbmin, bbmax);
    const double ax = bbmin(0), bx = bbmax(0);
    const double ay = bbmin(1), by = bbmax(1);
    const double az = bbmin(2), bz = bbmax(2);

    // eps scaled to domain size; tolerant to motion/curvature
    const double ex = 1e-12 * std::max(1.0, bx - ax);
    const double ey = 1e-12 * std::max(1.0, by - ay);
    const double ez = 1e-12 * std::max(1.0, bz - az);

    // ---- 3) Local payload
    std::ostringstream local_ss;
    local_ss.setf(std::ios::scientific);
    local_ss << std::setprecision(16);

    mfem::FiniteElementSpace *fes = sol->FESpace();
    const int vdim = fes->GetVDim();
    MFEM_VERIFY(vdim == 3, "Adjust printing if not 3D.");

    // Hexes assumed (TGV). Extend if you use mixed meshes.
    MFEM_VERIFY(pmesh->GetElementBaseGeometry(0) == mfem::Geometry::Type::CUBE,
                "This implementation assumes hexahedra.");

    const int NE = pmesh->GetNE();
    for (int e = 0; e < NE; ++e)
    {
        mfem::ElementTransformation *T = pmesh->GetElementTransformation(e);

        // Only decide ownership for the *global* max planes:
        const bool own_max_x = OwnsGlobalMaxPlane(T, 0, bx, ex);
        const bool own_max_y = OwnsGlobalMaxPlane(T, 1, by, ey);
        const bool own_max_z = OwnsGlobalMaxPlane(T, 2, bz, ez);

        for (int iz = 0; iz <= npts; ++iz)
        {
            for (int iy = 0; iy <= npts; ++iy)
            {
                for (int ix = 0; ix <= npts; ++ix)
                {
                    // Closed-grid rule:
                    // - Always keep the min planes (ix==0,iy==0,iz==0)
                    // - Keep the global max plane only if we "own" it on that axis.
                    if ((ix == npts && !own_max_x) ||
                        (iy == npts && !own_max_y) ||
                        (iz == npts && !own_max_z)) {
                        continue;
                    }

                    mfem::IntegrationPoint ip;
                    ip.Set3(ix*inv_n, iy*inv_n, iz*inv_n);

                    // Physical coordinate at current ALE configuration
                    mfem::Vector Xphys(T->GetSpaceDim());
                    T->Transform(ip, Xphys);
                    double Xx = Xphys(0), Xy = Xphys(1), Xz = Xphys(2);

                    // Snap to exact endpoints for clean boundaries
                    snap_endpoints(Xx, ax, bx, ex);
                    snap_endpoints(Xy, ay, by, ey);
                    snap_endpoints(Xz, az, bz, ez);

                    // Field value at current configuration
                    mfem::Vector u_val(vdim);
                    sol->GetVectorValue(*T, ip, u_val);

                    // Emit row
                    local_ss << std::setw(20) << Xx << " "
                             << std::setw(20) << Xy << " "
                             << std::setw(20) << Xz << " "
                             << std::setw(20) << u_val(0) << " "
                             << std::setw(20) << u_val(1) << " "
                             << std::setw(20) << u_val(2) << "\n";
                }
            }
        }
    }

    // ---- 4) Header
    std::string header;
    if (rank == 0) {
        std::ostringstream h;
        h.setf(std::ios::scientific); h << std::setprecision(16);
        h << "3D Taylor Green Vortex\n"
          << "Order = " << GetOrder(ctx) << ", Over sample = " << GetOverSample(ctx) << "\n"
          << "Step = " << step << "\n"
          << "Time = " << std::scientific << std::setprecision(16) << time << "\n"
          << "==================================================================="
          << "==========================================================================\n"
          << "            x                      y                      z                   vecx                   vecy                   vecz\n";
        header = h.str();
    }

    // ---- 5) Collective text write
    const int werr = WriteTextCollective(comm, fname, header, local_ss.str());
    if (werr != MPI_SUCCESS) {
        if (rank == 0) std::cerr << "WriteTextCollective failed (MPI err=" << werr << ")\n";
        MPI_Abort(comm, 1);
    }
    if (rank == 0) {
        std::cout << "Sampled closed-grid data (unique outer boundary, endpoints kept) saved: "
                  << fname << std::endl;
    }
}




// Writes unique H1 nodal true-DOFs (coords + vector values) to one file via MPI-IO.
// Closed-domain emit: includes BOTH endpoints by mirroring boundary true-DOFs.
void SamplePointsAtDoFs(mfem::ParGridFunction      *u,
                        mfem::ParMesh              *pmesh,
                        int                         step,
                        double                      time,
                        const std::string          &suffix,
                        const s_NavierContext      *ctx)
{
    MPI_Comm comm = pmesh->GetComm();
    int rank; MPI_Comm_rank(comm, &rank);

    // ---- 0) Output paths
    std::string main_dir = std::string("SamplePoints") + suffix +
                           "_Re" + std::to_string((int)GetReynum(ctx)) +
                           "NumPtsPerDir" + std::to_string(GetNumPts(ctx)) +
                           "RefLv" + std::to_string(GetElementSubdivisions(ctx) +
                                                    GetElementSubdivisionsParallel(ctx)) +
                           "P" + std::to_string(GetOrder(ctx));
    std::string cycle_dir = main_dir + "/cycle_" + std::to_string(step);
    std::string fname     = cycle_dir + "/SampledDataAtDofs" + std::to_string(step) + ".txt";

    if (rank == 0) {
        (void)system(("mkdir -p " + main_dir).c_str());
        (void)system(("mkdir -p " + cycle_dir).c_str());
    }
    MPI_Barrier(comm);

    // ---- 1) Preconditions
    mfem::ParFiniteElementSpace *fes = u->ParFESpace();
    MFEM_VERIFY(fes->GetFE(0)->GetNodes() != nullptr,
                "This method requires a nodal (H1) velocity space.");
    const int vdim = fes->GetVDim();
    const int sdim = pmesh->SpaceDimension();
    MFEM_VERIFY(vdim == 3 && sdim == 3, "Adjust printing if not 3D.");

    // ---- 2) Coordinates on SAME FES as 'u'
    mfem::ParGridFunction X_on_u(fes);
    mfem::PositionVectorCoefficient pos(sdim);
    X_on_u.ProjectCoefficient(pos);     // (x,y,z) at velocity nodal points
    X_on_u.ParallelAverage();

    // Average velocity on a copy (consistent shared-DoFs)
    mfem::ParGridFunction u_avg(*u);
    u_avg.ParallelAverage();

    // ---- 3) True DOFs (unique across ranks)
    mfem::HypreParVector U_true, X_true;
    u_avg.GetTrueDofs(U_true);
    X_on_u.GetTrueDofs(X_true);

    const double *U = U_true.Read();   // local partition
    const double *X = X_true.Read();

    const int tloc = fes->GetTrueVSize() / vdim;  // local scalar true size
    const auto ord = fes->GetOrdering();

    auto get_comp = [&](const double *D, int i, int c)->double {
        // i in [0, tloc), c in [0, vdim)
        return (ord == mfem::Ordering::byNODES) ? D[i + c * tloc]
                                                : D[c + i * vdim];
    };

    // ---- 4) Bounds, eps, and a tiny helper to build mirrored coordinate lists
    mfem::Vector bbmin(sdim), bbmax(sdim);
    pmesh->GetBoundingBox(bbmin, bbmax);
    const double ax = bbmin(0), bx = bbmax(0);
    const double ay = bbmin(1), by = bbmax(1);
    const double az = bbmin(2), bz = bbmax(2);

    const double ex = 1e-12 * std::max(1.0, bx - ax);
    const double ey = 1e-12 * std::max(1.0, by - ay);
    const double ez = 1e-12 * std::max(1.0, bz - az);

    auto near_min = [](double x, double a, double eps){ return (x - a) <= eps; };
    auto near_max = [](double x, double b, double eps){ return (b - x) <= eps; };

    // Given a value and [a,b], return {val} plus its mirror if on a face
    auto variants_1d = [&](double val, double a, double b, double eps)
    {
        std::vector<double> v; v.reserve(2);
        // Snap exact endpoints (stable printing)
        if (std::abs(val - a) <= eps) val = a;
        else if (std::abs(val - b) <= eps) val = b;
        v.push_back(val);
        if (near_min(val, a, eps)) v.push_back(b);
        else if (near_max(val, b, eps)) v.push_back(a);
        return v; // size 1 (interior) or 2 (on a face)
    };

    // ---- 5) Build payload (rank-local).
    std::ostringstream data_ss;
    data_ss.setf(std::ios::scientific);
    data_ss << std::setprecision(16);

    auto emit = [&](double Xx, double Xy, double Xz,
                    double Ux, double Uy, double Uz)
    {
        data_ss << std::setw(20) << Xx << " "
                << std::setw(20) << Xy << " "
                << std::setw(20) << Xz << " "
                << std::setw(20) << Ux << " "
                << std::setw(20) << Uy << " "
                << std::setw(20) << Uz << "\n";
    };

    for (int i = 0; i < tloc; ++i)
    {
        const double Xx0 = get_comp(X, i, 0);
        const double Xy0 = get_comp(X, i, 1);
        const double Xz0 = get_comp(X, i, 2);

        const double Ux  = get_comp(U, i, 0);
        const double Uy  = get_comp(U, i, 1);
        const double Uz  = get_comp(U, i, 2);

        const auto vx = variants_1d(Xx0, ax, bx, ex);
        const auto vy = variants_1d(Xy0, ay, by, ey);
        const auto vz = variants_1d(Xz0, az, bz, ez);

        for (double xx : vx)
        for (double yy : vy)
        for (double zz : vz)
            emit(xx, yy, zz, Ux, Uy, Uz);
    }

    const std::string data = data_ss.str();

    // ---- 6) Header (rank 0 only)
    std::string header;
    if (rank == 0) {
        std::ostringstream h;
        h.setf(std::ios::scientific); h << std::setprecision(16);
        h << "3D Taylor Green Vortex\n"
          << "Order = " << GetOrder(ctx) << ", Over sample = " << GetOverSample(ctx) << "\n"
          << "Step = " << step << "\n"
          << "Time = " << std::scientific << std::setprecision(16) << time << "\n"
          << "==================================================================="
          << "==========================================================================\n"
          << "            x                      y                      z                   vecx                   vecy                   vecz\n";
        header = h.str();
    }

    // ---- 7) Simple write helper (same as your other function)
    const int werr = WriteTextCollective(comm, fname, header, data);
    if (werr != MPI_SUCCESS) {
        if (rank == 0) std::cerr << "WriteTextCollective failed (MPI err=" << werr << ")\n";
        MPI_Abort(comm, 1);
    }

    if (rank == 0) {
        std::cout << "Sampled closed-domain DOF data saved: " << fname << std::endl;
    }
}


/*
// Writes unique H1 nodal true-DOFs (coords + vector values) to one file via MPI-IO.
// Closed-domain emit: includes both endpoints by mirroring points at each end to the opposite end.
// Coordinates are NOT ParallelAveraged (to avoid smearing periodic endpoints).
void SamplePointsAtDoFs(mfem::ParGridFunction      *u,
                        mfem::ParMesh              *pmesh,
                        int                         step,
                        double                      time,
                        const std::string          &suffix,
                        const s_NavierContext      *ctx)
{
    MPI_Comm comm = pmesh->GetComm();
    int rank; MPI_Comm_rank(comm, &rank);

    // ---- 0) Output paths
    std::string main_dir = std::string("SamplePoints") + suffix +
                           "_Re" + std::to_string((int)GetReynum(ctx)) +
                           "NumPtsPerDir" + std::to_string(GetNumPts(ctx)) +
                           "RefLv" + std::to_string(GetElementSubdivisions(ctx) +
                                                    GetElementSubdivisionsParallel(ctx)) +
                           "P" + std::to_string(GetOrder(ctx));
    std::string cycle_dir = main_dir + "/cycle_" + std::to_string(step);
    std::string fname     = cycle_dir + "/SampledDataAtDofs" + std::to_string(step) + ".txt";

    if (rank == 0) {
        (void)system(("mkdir -p " + main_dir).c_str());
        (void)system(("mkdir -p " + cycle_dir).c_str());
    }
    MPI_Barrier(comm);

    // ---- 1) Preconditions
    mfem::ParFiniteElementSpace *fes = u->ParFESpace();
    MFEM_VERIFY(fes->GetFE(0)->GetNodes() != nullptr,
                "This method requires a nodal (H1) velocity space.");
    const int vdim = fes->GetVDim();
    const int sdim = pmesh->SpaceDimension();
    MFEM_VERIFY(vdim == 3 && sdim == 3, "Adjust printing if not 3D.");

    // ---- 2) Coordinates on SAME FES as 'u' (NO ParallelAverage on coords)
    mfem::ParGridFunction X_on_u(fes);
    mfem::PositionVectorCoefficient pos(sdim);
    X_on_u.ProjectCoefficient(pos);   // (x,y,z) at velocity nodal points
    // DO NOT: X_on_u.ParallelAverage();  // <-- removing this avoids endpoint smearing

    // Average velocity on a copy (safe & keeps values consistent across shared DOFs)
    mfem::ParGridFunction u_avg(*u);
    u_avg.ParallelAverage();

    // ---- 3) True DOFs (unique & partitioned)
    mfem::HypreParVector U_true, X_true;
    u_avg.GetTrueDofs(U_true);
    X_on_u.GetTrueDofs(X_true);

    const double *U = U_true.Read();   // local partition
    const double *X = X_true.Read();

    const int tloc = fes->GetTrueVSize() / vdim;   // local scalar true size
    const auto ord = fes->GetOrdering();

    auto get_comp = [&](const double *D, int i, int c)->double {
        // i in [0, tloc), c in [0, vdim)
        if (ord == mfem::Ordering::byNODES) {      // XXX..., YYY..., ZZZ...
            return D[i + c * tloc];
        } else {                                    // byVDIM: XYZ, XYZ, ...
            return D[c + i * vdim];
        }
    };

    // ---- 4) Endpoint snapping (both ends) + mirroring both directions
    mfem::Vector bbmin(sdim), bbmax(sdim);
    pmesh->GetBoundingBox(bbmin, bbmax);
    const double ax = bbmin(0), bx = bbmax(0);
    const double ay = bbmin(1), by = bbmax(1);
    const double az = bbmin(2), bz = bbmax(2);
    const double ex = 1e-12 * std::max(1.0, bx - ax);
    const double ey = 1e-12 * std::max(1.0, by - ay);
    const double ez = 1e-12 * std::max(1.0, bz - az);

    auto snap = [](double &x, double a, double b, double eps){
        if (std::abs(x - a) <= eps) x = a;
        else if (std::abs(x - b) <= eps) x = b;
    };

    // Build per-rank data payload (no header here)
    std::ostringstream data_ss;
    data_ss.setf(std::ios::scientific);
    data_ss << std::setprecision(16);

    auto emit_line = [&](double Xx, double Xy, double Xz,
                         double Ux, double Uy, double Uz)
    {
        data_ss << std::setw(20) << Xx << " "
                << std::setw(20) << Xy << " "
                << std::setw(20) << Xz << " "
                << std::setw(20) << Ux << " "
                << std::setw(20) << Uy << " "
                << std::setw(20) << Uz << "\n";
    };

    for (int i = 0; i < tloc; ++i) {
        // Base (true DOF) values
        double Xx = get_comp(X, i, 0);
        double Xy = get_comp(X, i, 1);
        double Xz = get_comp(X, i, 2);
        const double Ux = get_comp(U, i, 0);
        const double Uy = get_comp(U, i, 1);
        const double Uz = get_comp(U, i, 2);

        // Snap to exact endpoints (but keep whichever side they already are on)
        snap(Xx, ax, bx, ex);
        snap(Xy, ay, by, ey);
        snap(Xz, az, bz, ez);

        // Determine if each coord is at an endpoint
        const bool at_ax = std::abs(Xx - ax) <= ex;
        const bool at_bx = std::abs(Xx - bx) <= ex;
        const bool at_ay = std::abs(Xy - ay) <= ey;
        const bool at_by = std::abs(Xy - by) <= ey;
        const bool at_az = std::abs(Xz - az) <= ez;
        const bool at_bz = std::abs(Xz - bz) <= ez;

        // Emit the base point
        emit_line(Xx, Xy, Xz, Ux, Uy, Uz);

        // Mirror rules (both directions):
        // If at 'a', also emit at 'b' (same y,z); if at 'b', also emit at 'a'.
        auto mirror_if = [&](bool cond, double a, double b, double &coord,
                             double cx, double cy, double cz)
        {
            if (cond) {
                double mcoord = (std::abs(coord - a) <= 1e-300 ? b : a);
                emit_line( ( &coord==&Xx ? mcoord : cx ),
                           ( &coord==&Xy ? mcoord : cy ),
                           ( &coord==&Xz ? mcoord : cz ),
                           Ux, Uy, Uz);
            }
        };

        // Mirror along each axis independently (faces)
        if (at_ax || at_bx) { double cx=Xx, cy=Xy, cz=Xz; double &ref=Xx;
            if (at_ax) { double tmp = bx; emit_line(tmp, cy, cz, Ux, Uy, Uz); }
            if (at_bx) { double tmp = ax; emit_line(tmp, cy, cz, Ux, Uy, Uz); }
        }
        if (at_ay || at_by) { double cx=Xx, cy=Xy, cz=Xz;
            if (at_ay) { double tmp = by; emit_line(cx, tmp, cz, Ux, Uy, Uz); }
            if (at_by) { double tmp = ay; emit_line(cx, tmp, cz, Ux, Uy, Uz); }
        }
        if (at_az || at_bz) { double cx=Xx, cy=Xy, cz=Xz;
            if (at_az) { double tmp = bz; emit_line(cx, cy, tmp, Ux, Uy, Uz); }
            if (at_bz) { double tmp = az; emit_line(cx, cy, tmp, Ux, Uy, Uz); }
        }

        // Edges/corners: if multiple coords are at endpoints, emit the cross-combinations.
        // This ensures full population of the closed tensor grid.
        auto vec = [&](bool a0,bool b0,double aV,double bV,double base)->std::vector<double>{
            std::vector<double> out{base};
            if (a0) out.push_back(bV);
            if (b0) out.push_back(aV);
            return out;
        };
        const std::vector<double> Xxv = vec(at_ax, at_bx, ax, bx, Xx);
        const std::vector<double> Xyv = vec(at_ay, at_by, ay, by, Xy);
        const std::vector<double> Xzv = vec(at_az, at_bz, az, bz, Xz);

        for (double xx : Xxv)
        for (double yy : Xyv)
        for (double zz : Xzv)
        {
            // Skip the base point; it was already emitted.
            if (xx==Xx && yy==Xy && zz==Xz) continue;
            emit_line(xx, yy, zz, Ux, Uy, Uz);
        }
    }

    const std::string data = data_ss.str();

    // ---- 5) Header (rank 0 only)
    std::string header;
    if (rank == 0) {
        std::ostringstream h;
        h.setf(std::ios::scientific); h << std::setprecision(16);
        h << "3D Taylor Green Vortex\n"
          << "Order = " << GetOrder(ctx) << ", Over sample = " << GetOverSample(ctx) << "\n"
          << "Step = " << step << "\n"
          << "Time = " << std::scientific << std::setprecision(16) << time << "\n"
          << "==================================================================="
          << "==========================================================================\n"
          << "            x                      y                      z                   vecx                   vecy                   vecz\n";
        header = h.str();
    }

    // ---- 6) Two-phase parallel write (header-at-0 + explicit offsets)
    MPI_File fh;
    int err = MPI_File_open(comm, fname.c_str(),
                            MPI_MODE_CREATE | MPI_MODE_WRONLY,
                            MPI_INFO_NULL, &fh);
    if (err != MPI_SUCCESS) {
        if (rank==0) std::cerr << "Error opening " << fname << " with MPI I/O\n";
        MPI_Abort(comm, 1);
    }

    MPI_Offset header_bytes = 0;
    if (rank == 0) {
        MPI_Status st;
        MPI_File_write_at(fh, 0, header.data(), (int)header.size(), MPI_CHAR, &st);
        header_bytes = (MPI_Offset)header.size();
    }
    MPI_Bcast(&header_bytes, 1, MPI_OFFSET, 0, comm);

    const MPI_Offset my_bytes = (MPI_Offset)data.size();
    MPI_Offset my_offset = 0;
    MPI_Exscan(&my_bytes, &my_offset, 1, MPI_OFFSET, MPI_SUM, comm);
    if (rank == 0) my_offset = 0;

    MPI_Status st;
    MPI_File_write_at_all(fh, header_bytes + my_offset,
                          data.data(), (int)my_bytes, MPI_CHAR, &st);

    MPI_File_close(&fh);

    if (rank == 0) {
        std::cout << "Sampled closed-domain DOF data saved: " << fname << std::endl;
    }
}*/

/*
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
}*/
 

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

