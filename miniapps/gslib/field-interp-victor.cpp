// Copyright (c) 2010-2025,
// Lawrence Livermore National Security, LLC. LLNL-CODE-806117.
// See LICENSE and NOTICE for details.

#include "mfem.hpp"
#include <fstream>
#include <vector>
#include <array>
#include <iomanip>

// =============================================================
// Problem setup
// =============================================================

static double g_amp = 0.05; // perturbation amplitude

void PerturbMeshTransform(const mfem::Vector &x_in, mfem::Vector &x_out)
{
   const double freq = 2.0 * M_PI;
   x_out = x_in;

   x_out[0] += g_amp * std::sin(freq * x_in[1]) * std::cos(freq * x_in[2]);
   x_out[1] += g_amp * std::cos(freq * x_in[0]) * std::sin(freq * x_in[2]) * 0.8;
   x_out[2] += g_amp * std::sin(freq * x_in[0]) * std::cos(freq * x_in[1]) * 0.6;
}

void vector_func(const mfem::Vector &p, mfem::Vector &F)
{
   const double xi = 2.0 * M_PI * p(0);
   const double yi = 2.0 * M_PI * p(1);
   const double zi = 2.0 * M_PI * p(2);

   F(0) = std::sin(xi) * std::cos(yi) * std::cos(zi);
   F(1) = -std::cos(xi) * std::sin(yi) * std::cos(zi);
   F(2) = 0.0;
}

// =============================================================
// Helper to extract subset of coordinates for missing points
// =============================================================

void ExtractMissingCoords(const mfem::Vector &all_coords,
                          int dim,
                          int nodes_cnt,
                          const std::vector<int> &missing_indices,
                          mfem::Vector &subset_coords)
{
   const int N = missing_indices.size();
   subset_coords.SetSize(dim * N);

   for (int d = 0; d < dim; d++)
   {
      for (int i = 0; i < N; i++)
      {
         subset_coords[d * N + i] = all_coords[d * nodes_cnt + missing_indices[i]];
      }
   }
}

// =============================================================
// Helper to detect field type from FiniteElementCollection
// =============================================================

int GetFieldType(const mfem::FiniteElementCollection *fec)
{
   // dynamic_cast safely checks if fec is actually one of these derived types
   // Returns nullptr if the cast fails, non-null if successful
   
   if (dynamic_cast<const mfem::H1_FECollection *>(fec)) 
   {
      return 0;  // H1
   }
   if (dynamic_cast<const mfem::L2_FECollection *>(fec)) 
   {
      return 1;  // L2
   }
   if (dynamic_cast<const mfem::RT_FECollection *>(fec)) 
   {
      return 2;  // RT (H(div))
   }
   if (dynamic_cast<const mfem::ND_FECollection *>(fec)) 
   {
      return 3;  // ND (H(curl))
   }
   
   MFEM_ABORT("Unsupported source GF type.");
   return -1;
}

// =============================================================
// Periodic Field Interpolation Function
// =============================================================

mfem::ParGridFunction* InterpolateFieldPeriodic(
   mfem::ParMesh &src_mesh,
   mfem::ParGridFunction &src_gf,
   mfem::ParMesh &tar_mesh,
   int fieldtype,
   int order,
   double Lx, double Ly, double Lz)
{
   const int myid = mfem::Mpi::WorldRank();
   const int nprocs = mfem::Mpi::WorldSize();
   
   const int dim = src_mesh.Dimension();
   
   // Detect what type of field the source is
   const mfem::FiniteElementCollection *src_fec = src_gf.FESpace()->FEColl();
   const int src_fieldtype = GetFieldType(src_fec);
   
   const int src_vdim = src_gf.VectorDim();
   const int mesh_poly_deg = tar_mesh.GetNodes()->FESpace()->GetElementOrder(0);
   
   // Auto-detect target field type if not specified
   if (fieldtype < 0) { fieldtype = src_fieldtype; }
   
   // ---------------- Target field setup ----------------
   mfem::FiniteElementCollection *tar_fec = nullptr;
   int tar_vdim = src_vdim;
   
   if (fieldtype == 0)
   {
      tar_fec = new mfem::H1_FECollection(order, dim);
      if (src_fieldtype > 1) { tar_vdim = dim; }
   }
   else if (fieldtype == 1)
   {
      tar_fec = new mfem::L2_FECollection(order, dim);
      if (src_fieldtype > 1) { tar_vdim = dim; }
   }
   else if (fieldtype == 2)
   {
      tar_fec = new mfem::RT_FECollection(order, dim);
      tar_vdim = 1;
      MFEM_VERIFY(src_fieldtype > 1, "Cannot interpolate scalar to H(div).");
   }
   else if (fieldtype == 3)
   {
      tar_fec = new mfem::ND_FECollection(order, dim);
      tar_vdim = 1;
      MFEM_VERIFY(src_fieldtype > 1, "Cannot interpolate scalar to H(curl).");
   }
   else
   {
      MFEM_ABORT("Invalid target fieldtype.");
   }
   
   mfem::ParFiniteElementSpace *tar_fes =
      new mfem::ParFiniteElementSpace(&tar_mesh, tar_fec, tar_vdim, 
                                      src_gf.FESpace()->GetOrdering());
   mfem::ParGridFunction *func_target = new mfem::ParGridFunction(tar_fes);
   func_target->MakeOwner(tar_fec);  // Grid function will own and delete the FEC
   
   // ---------------- Build query points ----------------
   const int NE = tar_mesh.GetNE();
   const int nsp = tar_fes->GetTypicalFE()->GetNodes().GetNPoints();
   const int tar_ncomp = func_target->VectorDim();
   
   mfem::Vector vxyz;
   int point_ordering;
   
   if (fieldtype == 0 && order == mesh_poly_deg)
   {
      vxyz = *tar_mesh.GetNodes();
      point_ordering = tar_mesh.GetNodes()->FESpace()->GetOrdering();
   }
   else
   {
      vxyz.SetSize(nsp * NE * dim);
      for (int i = 0; i < NE; i++)
      {
         const mfem::FiniteElement *fe = tar_fes->GetFE(i);
         const mfem::IntegrationRule ir = fe->GetNodes();
         mfem::ElementTransformation *et = tar_fes->GetElementTransformation(i);
   
         mfem::DenseMatrix pos;
         et->Transform(ir, pos);
   
         mfem::Vector rowx(vxyz.GetData() + i * nsp, nsp);
         mfem::Vector rowy(vxyz.GetData() + i * nsp + NE * nsp, nsp);
         mfem::Vector rowz;
         if (dim == 3)
         {
            rowz.SetDataAndSize(vxyz.GetData() + i * nsp + 2 * NE * nsp, nsp);
         }
   
         pos.GetRow(0, rowx);
         pos.GetRow(1, rowy);
         if (dim == 3) { pos.GetRow(2, rowz); }
      }
      point_ordering = mfem::Ordering::byNODES;
   }
   
   const int nodes_cnt = vxyz.Size() / dim;
   
   // ---------------- FindPoints interpolation ----------------
   mfem::Vector interp_vals(nodes_cnt * tar_ncomp);
   interp_vals = 0.0;
   
   mfem::FindPointsGSLIB finder(MPI_COMM_WORLD);
   finder.Setup(src_mesh);
   if (fieldtype == 1)
   {
      finder.SetL2AvgType(mfem::FindPointsGSLIB::ARITHMETIC);
   }
   
   // First pass: standard interpolation
   finder.Interpolate(vxyz, src_gf, interp_vals, point_ordering);
   
   // Track which points were not found
   const mfem::Array<unsigned int> &codes = finder.GetCode();
   std::vector<int> missing;
   for (int i = 0; i < nodes_cnt; i++)
   {
      if (codes[i] == 2) { missing.push_back(i); }
   }
   
   int local_missing = missing.size();
   int global_missing = 0;
   MPI_Allreduce(&local_missing, &global_missing, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   
   if (myid == 0)
   {
      std::cout << "First pass: " << (1LL * nodes_cnt * nprocs - global_missing)
                << " / " << (1LL * nodes_cnt * nprocs) << " points found\n";
   }
   
   // ---------------- Periodic retry for missing points ----------------
   if (global_missing > 0)
   {
      if (myid == 0)
      {
         std::cout << "Retrying " << global_missing << " missing points with periodic offsets...\n";
      }
   
      // Define all periodic offsets to try
      std::vector<std::array<double, 3>> offsets;
      
      // Faces (6)
      offsets.push_back({ Lx, 0.0, 0.0});
      offsets.push_back({-Lx, 0.0, 0.0});
      offsets.push_back({0.0,  Ly, 0.0});
      offsets.push_back({0.0, -Ly, 0.0});
      offsets.push_back({0.0, 0.0,  Lz});
      offsets.push_back({0.0, 0.0, -Lz});
   
      // Edges (12)
      offsets.push_back({ Lx,  Ly, 0.0});
      offsets.push_back({ Lx, -Ly, 0.0});
      offsets.push_back({-Lx,  Ly, 0.0});
      offsets.push_back({-Lx, -Ly, 0.0});
      offsets.push_back({ Lx, 0.0,  Lz});
      offsets.push_back({ Lx, 0.0, -Lz});
      offsets.push_back({-Lx, 0.0,  Lz});
      offsets.push_back({-Lx, 0.0, -Lz});
      offsets.push_back({0.0,  Ly,  Lz});
      offsets.push_back({0.0,  Ly, -Lz});
      offsets.push_back({0.0, -Ly,  Lz});
      offsets.push_back({0.0, -Ly, -Lz});
   
      // Corners (8)
      for (double ox : {Lx, -Lx})
      {
         for (double oy : {Ly, -Ly})
         {
            for (double oz : {Lz, -Lz})
            {
               offsets.push_back({ox, oy, oz});
            }
         }
      }
   
      // Try each offset
      for (const auto &offset : offsets)
      {
         // Check if ANY rank still has missing points (collective check)
         int local_has_missing = missing.empty() ? 0 : 1;
         int global_has_missing = 0;
         MPI_Allreduce(&local_has_missing, &global_has_missing, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
         
         if (global_has_missing == 0) { break; }  // All ranks done
      
         const int N = missing.size();
         
         // Extract coordinates for missing points (or empty if none)
         mfem::Vector offset_coords;
         if (N > 0)
         {
            ExtractMissingCoords(vxyz, dim, nodes_cnt, missing, offset_coords);
         
            // Apply periodic offset
            for (int i = 0; i < N; i++)
            {
               offset_coords[i] += offset[0];
               offset_coords[N + i] += offset[1];
               if (dim == 3) { offset_coords[2*N + i] += offset[2]; }
            }
         }
         else
         {
            offset_coords.SetSize(0);
         }
      
         // Interpolate with offset coordinates (COLLECTIVE)
         mfem::Vector subset_vals(N * tar_ncomp);
         subset_vals = 0.0;
         finder.Interpolate(offset_coords, src_gf, subset_vals, mfem::Ordering::byNODES);
      
         const mfem::Array<unsigned int> &subset_codes = finder.GetCode();
      
         // Copy successful results back to main array
         for (int d = 0; d < tar_ncomp; d++)
         {
            for (int i = 0; i < N; i++)
            {
               if (subset_codes[i] != 2)
               {
                  interp_vals[d * nodes_cnt + missing[i]] = subset_vals[d * N + i];
               }
            }
         }
      
         // Update missing list
         std::vector<int> still_missing;
         for (int i = 0; i < N; i++)
         {
            if (subset_codes[i] == 2)
            {
               still_missing.push_back(missing[i]);
            }
         }
         missing = still_missing;
      }
   }
   
   // Final statistics
   int final_missing = missing.size();
   int final_global_missing = 0;
   MPI_Allreduce(&final_missing, &final_global_missing, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   
   if (myid == 0)
   {
      const long long total = 1LL * nodes_cnt * nprocs;
      const long long found = total - final_global_missing;
      std::cout << "Final: " << found << " / " << total << " points found ("
                << (100.0 * found / total) << "%)\n";
   }
   
   // ---------------- Project to target space ----------------
   if (fieldtype <= 1) // H1 or L2
   {
      if ((fieldtype == 0 && order == mesh_poly_deg) || fieldtype == 1)
      {
         *func_target = interp_vals;
      }
      else
      {
         mfem::Array<int> vdofs;
         mfem::Vector elem_dof_vals(nsp * tar_ncomp);
   
         for (int i = 0; i < NE; i++)
         {
            tar_fes->GetElementVDofs(i, vdofs);
            for (int j = 0; j < nsp; j++)
            {
               for (int d = 0; d < tar_ncomp; d++)
               {
                  const int idx = d * (nsp * NE) + i * nsp + j;
                  elem_dof_vals(j + d * nsp) = interp_vals(idx);
               }
            }
            func_target->SetSubVector(vdofs, elem_dof_vals);
         }
      }
   }
   else // H(div) or H(curl)
   {
      mfem::Array<int> vdofs;
      mfem::Vector vals;
      mfem::Vector elem_dof_vals(nsp * tar_ncomp);
   
      for (int i = 0; i < NE; i++)
      {
         tar_fes->GetElementVDofs(i, vdofs);
         vals.SetSize(vdofs.Size());
   
         for (int j = 0; j < nsp; j++)
         {
            for (int d = 0; d < tar_ncomp; d++)
            {
               const int idx = d * (nsp * NE) + i * nsp + j;
               elem_dof_vals(j * tar_ncomp + d) = interp_vals(idx);
            }
         }
   
         tar_fes->GetFE(i)->ProjectFromNodes(elem_dof_vals,
                                             *tar_fes->GetElementTransformation(i),
                                             vals);
         func_target->SetSubVector(vdofs, vals);
      }
   }
   
   func_target->SetTrueVector();
   func_target->SetFromTrueVector();
   
   return func_target;
}

// =============================================================
// Main
// =============================================================

int main(int argc, char *argv[])
{
   mfem::Mpi::Init(argc, argv);
   const int myid = mfem::Mpi::WorldRank();
   const int nprocs = mfem::Mpi::WorldSize();
   mfem::Hypre::Init();

#ifndef MFEM_USE_GSLIB
   if (myid == 0)
   {
      std::cerr << "ERROR: Requires MFEM built with GSLIB (MFEM_USE_GSLIB=YES).\n";
   }
   return 2;
#endif

   // ---------------- Parameters ----------------
   int nx = 16;
   int order = 2;
   int ref_levels = 0;
   int src_fieldtype = 0;   // 0-H1, 1-L2, 2-RT, 3-ND
   int src_ncomp = 1;
   int src_gf_ordering = 0; // 0-byNodes, 1-byVDim
   int fieldtype = -1;      // -1 => match source
   bool visualization = true;
   bool visit_output = false;
   int visport = 19916;

   mfem::OptionsParser args(argc, argv);
   args.AddOption(&nx, "-n", "--num_el", "Elements per direction.");
   args.AddOption(&order, "-o", "--order", "Polynomial order.");
   args.AddOption(&ref_levels, "-r", "--refine", "Target mesh refinements.");
   args.AddOption(&g_amp, "-amp", "--amplitude", "Mesh perturbation amplitude.");
   args.AddOption(&src_fieldtype, "-fts", "--field-type-src", "0-H1, 1-L2, 2-RT, 3-ND.");
   args.AddOption(&src_ncomp, "-nc", "--ncomp", "Components for H1/L2.");
   args.AddOption(&src_gf_ordering, "-gfo", "--gfo", "GridFunction ordering: 0(byNodes), 1(byVDim).");
   args.AddOption(&fieldtype, "-ft", "--field-type", "Target: -1(same), 0-H1, 1-L2, 2-H(div), 3-H(curl).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis", "--no-visualization", "GLVis on/off.");
   args.AddOption(&visit_output, "-visit", "--visit-output", "-no-visit", "--no-visit-output", "VisIt on/off.");
   args.Parse();

   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(std::cout); }
      return 1;
   }

   if (myid == 0)
   {
      args.PrintOptions(std::cout);
      std::cout << "\n=== Periodic Field Interpolation with FindPoints ===\n";
      std::cout << "MPI ranks: " << nprocs << "\n";
      std::cout << "Grid: " << nx << "^3, Order: " << order << "\n";
   }

   // ---------------- Create SOURCE mesh (perturbed) ----------------
   const double Lx = 1.0, Ly = 1.0, Lz = 1.0;
   
   mfem::Mesh mesh_init = mfem::Mesh::MakeCartesian3D(nx, nx, nx, mfem::Element::HEXAHEDRON,
                                                       Lx, Ly, Lz, false);
   
   mfem::Vector xT(3); xT = 0.0; xT[0] = Lx;
   mfem::Vector yT(3); yT = 0.0; yT[1] = Ly;
   mfem::Vector zT(3); zT = 0.0; zT[2] = Lz;
   std::vector<mfem::Vector> translations = {xT, yT, zT};
   std::vector<int> v2v = mesh_init.CreatePeriodicVertexMapping(translations);
   
   mfem::Mesh mesh_1_serial = mfem::Mesh::MakePeriodic(mesh_init, v2v);
   mesh_1_serial.Transform(PerturbMeshTransform);
   
   const int dim = mesh_1_serial.Dimension();
   
   // ---------------- Create TARGET mesh (clean) ----------------
   mfem::Mesh mesh_2_init = mfem::Mesh::MakeCartesian3D(nx, nx, nx, mfem::Element::HEXAHEDRON,
                                                         Lx, Ly, Lz, false);
   std::vector<int> v2v2 = mesh_2_init.CreatePeriodicVertexMapping(translations);
   mfem::Mesh mesh_2_serial = mfem::Mesh::MakePeriodic(mesh_2_init, v2v2);
   
   for (int l = 0; l < ref_levels; l++)
   {
      mesh_2_serial.UniformRefinement();
   }
   
   if (!mesh_1_serial.GetNodes()) { mesh_1_serial.SetCurvature(1); }
   if (!mesh_2_serial.GetNodes()) { mesh_2_serial.SetCurvature(1); }
   
   // Create parallel meshes
   mfem::ParMesh mesh_1(MPI_COMM_WORLD, mesh_1_serial);
   mfem::ParMesh mesh_2(MPI_COMM_WORLD, mesh_2_serial);
   mesh_1_serial.Clear();
   mesh_2_serial.Clear();

   {
      const std::string mesh_dir = "saved_mesh";
      const std::string fname_base = "uniform-parallel";

      if (myid == 0)
      {
         std::string cmd = "mkdir -p " + mesh_dir;
         int ret = system(cmd.c_str());
         MFEM_VERIFY(ret == 0, "Failed to create directory: " + mesh_dir);
      }

      MPI_Barrier(MPI_COMM_WORLD);  // Wait for directory creation

      // --- Save parallel mesh: each rank writes its partition ---
      std::ostringstream mesh_name;
      mesh_name << mesh_dir << "/" << fname_base << "." 
                << std::setfill('0') << std::setw(6) << myid;

      std::ofstream mesh_ofs(mesh_name.str());
      MFEM_VERIFY(mesh_ofs.good(), "Failed to open mesh file for writing: " + mesh_name.str());
      mesh_ofs.precision(17);
      mesh_2.ParPrint(mesh_ofs);
      mesh_ofs.close();

      MPI_Barrier(MPI_COMM_WORLD);  // Ensure all ranks finish writing

      if (myid == 0)
      {
         std::cout << "Mesh saved to directory: " << mesh_dir << "/" << std::endl;
      }

      // --- Load parallel mesh: each rank reads its partition ---
      std::ostringstream mesh_load_name;
      mesh_load_name << mesh_dir << "/" << fname_base << "." 
                     << std::setfill('0') << std::setw(6) << myid;

      std::ifstream mesh_ifs(mesh_load_name.str());
      MFEM_VERIFY(mesh_ifs.good(), "Failed to open mesh file for reading: " + mesh_load_name.str());

      mfem::ParMesh reloaded(MPI_COMM_WORLD, mesh_ifs);
      mesh_ifs.close();

      if (myid == 0)
      {
         std::cout << "Mesh loaded from directory: " << mesh_dir << "/" << std::endl;
      }

      // Replace old mesh
      mesh_2 = std::move(reloaded);
   }


   // ---------------- Source field setup ----------------
   int src_vdim = src_ncomp;
   mfem::FiniteElementCollection *src_fec = nullptr;

   if (src_fieldtype == 0)
   {
      src_fec = new mfem::H1_FECollection(order, dim);
   }
   else if (src_fieldtype == 1)
   {
      src_fec = new mfem::L2_FECollection(order, dim);
   }
   else if (src_fieldtype == 2)
   {
      src_fec = new mfem::RT_FECollection(order, dim);
      src_ncomp = 1;
      src_vdim = dim;
   }
   else if (src_fieldtype == 3)
   {
      src_fec = new mfem::ND_FECollection(order, dim);
      src_ncomp = 1;
      src_vdim = dim;
   }
   else
   {
      MFEM_ABORT("Invalid src_fieldtype.");
   }

   mfem::ParFiniteElementSpace src_fes(&mesh_1, src_fec, src_ncomp, src_gf_ordering);
   mfem::ParGridFunction func_source(&src_fes);
   mfem::VectorFunctionCoefficient F(src_vdim, vector_func);
   func_source.ProjectCoefficient(F);

   double source_norm = 0.0;
   double source_mass = 0.0;
   double source_ke   = 0.0;
   {
      if (func_source.VectorDim() == 1)
      {
         mfem::ConstantCoefficient zero(0.0);
         source_norm = func_source.ComputeL2Error(zero);
      }
      else
      {
         mfem::Vector zero_v(func_source.VectorDim()); zero_v = 0.0;
         mfem::VectorConstantCoefficient zero_c(zero_v);
         source_norm = func_source.ComputeL2Error(zero_c);
      }
      source_ke = 0.5 * source_norm * source_norm;

      // Calculate Mass: Integral of the field components
      mfem::ConstantCoefficient one(1.0);
      mfem::ParLinearForm lf(&src_fes);
      lf.AddDomainIntegrator(new mfem::DomainLFIntegrator(one));
      lf.Assemble();
      source_mass = lf(func_source);
      double global_mass = 0.0;
      MPI_Allreduce(&source_mass, &global_mass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      source_mass = global_mass;

      if (myid == 0)
      {
         std::cout << "Source L2 norm: " << source_norm << std::endl;
         std::cout << "Source Mass:    " << source_mass << std::endl;
         std::cout << "Source KE:      " << source_ke << std::endl;
      }
   }

   // Create desired solution for comparison
   mfem::ParFiniteElementSpace des_fes(&mesh_2, src_fec, src_ncomp, src_gf_ordering);
   mfem::ParGridFunction func_desired(&des_fes);
   func_desired.ProjectCoefficient(F);

   // ---------------- Interpolate using modular function ----------------
   mfem::ParGridFunction *func_target = InterpolateFieldPeriodic(
      mesh_1, func_source, mesh_2, fieldtype, order, Lx, Ly, Lz);

   double target_norm = 0.0;
   double target_mass = 0.0;
   double target_ke   = 0.0;
   {
      if (func_target->VectorDim() == 1)
      {
         mfem::ConstantCoefficient zero(0.0);
         target_norm = func_target->ComputeL2Error(zero);
      }
      else
      {
         mfem::Vector zero_v(func_target->VectorDim()); zero_v = 0.0;
         mfem::VectorConstantCoefficient zero_c(zero_v);
         target_norm = func_target->ComputeL2Error(zero_c);
      }
      target_ke = 0.5 * target_norm * target_norm;

      // Calculate Mass
      mfem::ConstantCoefficient one(1.0);
      mfem::ParLinearForm lf(func_target->ParFESpace());
      lf.AddDomainIntegrator(new mfem::DomainLFIntegrator(one));
      lf.Assemble();
      target_mass = lf(*func_target);
      double global_mass = 0.0;
      MPI_Allreduce(&target_mass, &global_mass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      target_mass = global_mass;

      if (myid == 0)
      {
         std::cout << "Target L2 norm: " << target_norm << std::endl;
         std::cout << "Target Mass:    " << target_mass << std::endl;
         std::cout << "Target KE:      " << target_ke << std::endl;
      }
   }

   // Calculate L2 error vs desired
   double l2_error = func_target->ComputeL2Error(F);
   if (myid == 0)
   {
      std::cout << "L2 Error vs Exact: " << l2_error << std::endl;
   }

   // ---------------- Visualization ----------------
   if (visualization)
   {
      char vishost[] = "localhost";
      
      // Source mesh visualization
      mfem::socketstream sout1;
      sout1.open(vishost, visport);
      if (!sout1)
      {
         mfem::out << "Unable to connect to GLVis server at " 
                   << vishost << ':' << visport << std::endl;
      }
      else
      {
         sout1.precision(8);
         sout1 << "parallel " << nprocs << " " << myid << "\n";
         sout1 << "solution\n" << mesh_1 << func_source
               << "window_title 'Source mesh + field'\n"
               << "window_geometry 0 0 600 600\n";
         if (dim == 3) { sout1 << "keys mA\n"; }
         sout1 << std::flush;
      }
      
      // Target mesh visualization
      mfem::socketstream sout2;
      sout2.open(vishost, visport);
      if (!sout2)
      {
         mfem::out << "Unable to connect to GLVis server at " 
                   << vishost << ':' << visport << std::endl;
      }
      else
      {
         sout2.precision(8);
         sout2 << "parallel " << nprocs << " " << myid << "\n";
         sout2 << "solution\n" << mesh_2 << *func_target
               << "window_title 'Target mesh + interpolated field'\n"
               << "window_geometry 620 0 600 600\n";
         if (dim == 3) { sout2 << "keys mA\n"; }
         sout2 << std::flush;
      }
   }

   // ---------------- Output ----------------
   if (visit_output)
   {
      mfem::VisItDataCollection dc_src("SourceMesh", &mesh_1);
      dc_src.SetPrecision(8);
      dc_src.RegisterField("u_source", &func_source);
      dc_src.SetCycle(0);
      dc_src.SetTime(0.0);
      dc_src.Save();

      mfem::VisItDataCollection dc("TargetMesh", &mesh_2);
      dc.SetPrecision(8);
      dc.RegisterField("u_interp", func_target);
      dc.RegisterField("u_exact", &func_desired);
      dc.SetCycle(0);
      dc.SetTime(0.0);
      dc.Save();
   }

   if (myid == 0)
   {
      std::ofstream ofs("interpolated.gf");
      ofs.precision(8);
      func_target->Save(ofs);

      // Save results to CSV
      const char *csv_name = "simulation_results.csv";
      bool exists = std::ifstream(csv_name).good();
      std::ofstream csv(csv_name, std::ios::app);
      if (!exists)
      {
         csv << std::setw(24) << "nx" << ","
             << std::setw(24) << "order" << ","
             << std::setw(24) << "ref_levels" << ","
             << std::setw(24) << "src_fieldtype" << ","
             << std::setw(24) << "ncomp" << ","
             << std::setw(24) << "amplitude" << ","
             << std::setw(24) << "source_norm" << ","
             << std::setw(24) << "target_norm" << ","
             << std::setw(24) << "source_mass" << ","
             << std::setw(24) << "target_mass" << ","
             << std::setw(24) << "source_ke" << ","
             << std::setw(24) << "target_ke" << ","
             << std::setw(24) << "l2_error" << "\n";
      }
      csv << std::scientific << std::setprecision(16)
          << std::setw(24) << (double)nx << ","
          << std::setw(24) << (double)order << ","
          << std::setw(24) << (double)ref_levels << ","
          << std::setw(24) << (double)src_fieldtype << ","
          << std::setw(24) << (double)src_ncomp << ","
          << std::setw(24) << g_amp << ","
          << std::setw(24) << source_norm << ","
          << std::setw(24) << target_norm << ","
          << std::setw(24) << source_mass << ","
          << std::setw(24) << target_mass << ","
          << std::setw(24) << source_ke << ","
          << std::setw(24) << target_ke << ","
          << std::setw(24) << l2_error << "\n";
   }

   // ---------------- Cleanup ----------------
   delete func_target;  // This also deletes the FEC since we called MakeOwner
   delete src_fec;

   return 0;
}
