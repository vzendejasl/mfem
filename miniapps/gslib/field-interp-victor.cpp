// Copyright (c) 2010-2025,
// Lawrence Livermore National Security, LLC. LLNL-CODE-806117.
// See LICENSE and NOTICE for details.

#include "mfem.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <numeric>
#include <vector>

using namespace mfem;
using namespace std;

// =============================================================
// Problem setup
// =============================================================

static double g_amp = 0.15; // perturbation amplitude

void PerturbMeshTransform(const Vector &x_in, Vector &x_out)
{
   const double freq = 2.0 * M_PI;
   x_out = x_in;

   x_out[0] += g_amp * std::sin(freq * x_in[1]) * std::cos(freq * x_in[2]);
   x_out[1] += g_amp * std::cos(freq * x_in[0]) * std::sin(freq * x_in[2]) * 0.8;
   x_out[2] += g_amp * std::sin(freq * x_in[0]) * std::cos(freq * x_in[1]) * 0.6;
}

void vector_func(const Vector &p, Vector &F)
{
   const double xi = 2.0 * M_PI * p(0);
   const double yi = 2.0 * M_PI * p(1);
   const double zi = 2.0 * M_PI * p(2);

   F(0) = std::sin(xi) * std::cos(yi) * std::cos(zi);
   F(1) = -std::cos(xi) * std::sin(yi) * std::cos(zi);
   F(2) = 0.0;
}

// =============================================================
// Helpers
// =============================================================

static void ReorderCoordsToByNodes(const Vector &in,
                                   int ordering,
                                   int nodes_cnt,
                                   int dim,
                                   Vector &out)
{
   out.SetSize(in.Size());

   if (ordering == Ordering::byNODES)
   {
      out = in;
      return;
   }

   // ordering == byVDIM: (x0,y0,z0, x1,y1,z1, ...)
   for (int i = 0; i < nodes_cnt; ++i)
   {
      for (int d = 0; d < dim; ++d)
      {
         out[d * nodes_cnt + i] = in[i * dim + d];
      }
   }
}

static void BuildSubsetCoordsByNodes(const Vector &all_bn,
                                     int dim,
                                     const std::vector<int> &indices,
                                     Vector &subset_bn)
{
   const int N = static_cast<int>(indices.size());
   subset_bn.SetSize(dim * N);

   const int fullN = static_cast<int>(all_bn.Size() / dim);

   for (int k = 0; k < N; ++k)
   {
      const int i = indices[k];

      for (int d = 0; d < dim; ++d)
      {
         subset_bn[d * N + k] = all_bn[d * fullN + i];
      }
   }
}

static void ShowInGLVis(const ParMesh &pmesh,
                        const ParGridFunction &pgf,
                        const char *title,
                        int visport,
                        int x = 0,
                        int y = 0)
{
   char vishost[] = "localhost";
   socketstream sout;

   sout.open(vishost, visport);
   if (!sout)
   {
      mfem::out << "GLVis: could not connect to " << vishost << ":" << visport << "\n";
      return;
   }

   sout.precision(8);
   sout << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() << "\n";
   sout << "solution\n" << pmesh << pgf
        << "window_title '" << title << "'\n"
        << "window_geometry " << x << " " << y << " 600 600\n";

   if (pmesh.Dimension() == 3)
   {
      sout << "keys mA\n";
   }

   sout << std::flush;
}

// =============================================================
// Main
// =============================================================

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);

   const int myid   = Mpi::WorldRank();
   const int nprocs = Mpi::WorldSize();

   Hypre::Init();

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
   int src_ncomp = 1;       // for H1/L2
   int src_gf_ordering = 0; // 0-byNodes, 1-byVDim
   int fieldtype = -1;      // -1 => match source

   bool visualization = true;
   bool visit_output = false;
   int visport = 19916;

   double bb_rel     = 0.20; // per-element AABB padding
   double bdr_frac   = 1; // boundary distance tol as fraction of h
   double plane_mult = 1;  // plane window = plane_mult * bdr_tol

   OptionsParser args(argc, argv);
   args.AddOption(&nx, "-n", "--num_el", "Elements per direction.");
   args.AddOption(&order, "-o", "--order", "Polynomial order.");
   args.AddOption(&ref_levels, "-r", "--refine", "Target mesh refinements.");
   args.AddOption(&src_fieldtype, "-fts", "--field-type-src", "0-H1, 1-L2, 2-RT, 3-ND.");
   args.AddOption(&src_ncomp, "-nc", "--ncomp", "Components for H1/L2.");
   args.AddOption(&src_gf_ordering, "-gfo", "--gfo", "GridFunction ordering: 0(byNodes), 1(byVDim).");
   args.AddOption(&fieldtype, "-ft", "--field-type", "Target GF: -1(same), 0-H1, 1-L2, 2-H(div), 3-H(curl).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis", "--no-visualization", "GLVis on/off.");
   args.AddOption(&visit_output, "-visit", "--visit-output", "-no-visit", "--no-visit-output", "VisIt on/off.");
   args.AddOption(&bb_rel, "--bb", "--bb", "FindPoints per-element AABB padding.");
   args.AddOption(&bdr_frac, "--bdrfrac", "--bdrfrac", "Boundary tolerance as fraction of h.");
   args.AddOption(&plane_mult, "--plane-mult", "--plane-mult", "Plane window multiplier (×bdr_tol).");
   args.Parse();

   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }

   if (myid == 0)
   {
      args.PrintOptions(cout);
      cout << "\n=== Parallel FindPoints Interpolation (collective retries + final brute-force) ===\n";
      cout << "MPI ranks: " << nprocs << "\n";
      cout << "Grid: " << nx << " x " << nx << " x " << nx << "\n";
      cout << "Order: " << order << "\n";
      cout << "Amplitude: " << g_amp << "\n";
   }

   // ---------------- Periodic unit box ----------------
   const double Lx = 1.0, Ly = 1.0, Lz = 1.0;

   Mesh mesh_1_init = Mesh::MakeCartesian3D(nx, nx, nx,
                                            Element::HEXAHEDRON,
                                            Lx, Ly, Lz,
                                            /*generate_edges=*/false);

   // Periodic translations
   Vector xT(3); xT = 0.0; xT[0] = Lx;
   Vector yT(3); yT = 0.0; yT[1] = Ly;
   Vector zT(3); zT = 0.0; zT[2] = Lz;

   std::vector<Vector> translations = {xT, yT, zT};
   std::vector<int> v2v = mesh_1_init.CreatePeriodicVertexMapping(translations);

   Mesh mesh_1_serial = Mesh::MakePeriodic(mesh_1_init, v2v);
   Mesh mesh_2_serial = Mesh::MakePeriodic(mesh_1_init, v2v);

   // Perturb source mesh
   mesh_1_serial.Transform(PerturbMeshTransform);

   const int dim = mesh_1_serial.Dimension();
   MFEM_ASSERT(dim == mesh_2_serial.Dimension(), "Source/target dim mismatch.");
   MFEM_VERIFY(dim > 1, "Requires 2D or 3D.");

   for (int l = 0; l < ref_levels; l++)
   {
      mesh_2_serial.UniformRefinement();
   }

   if (!mesh_1_serial.GetNodes())
   {
      mesh_1_serial.SetCurvature(1);
   }
   if (!mesh_2_serial.GetNodes())
   {
      mesh_2_serial.SetCurvature(1);
   }

   const int mesh_poly_deg =
      mesh_2_serial.GetNodes()->FESpace()->GetElementOrder(0);

   // ---------------- Parallel meshes ----------------
   ParMesh mesh_1(MPI_COMM_WORLD, mesh_1_serial);
   ParMesh mesh_2(MPI_COMM_WORLD, mesh_2_serial);
   mesh_1_serial.Clear();
   mesh_2_serial.Clear();

   // ---------------- Source FE space & field ----------------
   int src_vdim = src_ncomp;
   FiniteElementCollection *src_fec = nullptr;

   if (src_fieldtype == 0)
   {
      src_fec = new H1_FECollection(order, dim);
   }
   else if (src_fieldtype == 1)
   {
      src_fec = new L2_FECollection(order, dim);
   }
   else if (src_fieldtype == 2)
   {
      src_fec = new RT_FECollection(order, dim);
      src_ncomp = 1;
      src_vdim = dim;
   }
   else if (src_fieldtype == 3)
   {
      src_fec = new ND_FECollection(order, dim);
      src_ncomp = 1;
      src_vdim = dim;
   }
   else
   {
      MFEM_ABORT("Invalid src_fieldtype.");
   }

   MFEM_VERIFY(src_gf_ordering == 0 || src_gf_ordering == 1,
               "Source ordering must be 0(byNodes) or 1(byVDim).");

   ParFiniteElementSpace *src_fes =
      new ParFiniteElementSpace(&mesh_1, src_fec, src_ncomp, src_gf_ordering);

   ParGridFunction *func_source = new ParGridFunction(src_fes);
   {
      VectorFunctionCoefficient F(src_vdim, vector_func);
      func_source->ProjectCoefficient(F);
   }

   ParFiniteElementSpace *des_fes =
      new ParFiniteElementSpace(&mesh_2, src_fec, src_ncomp, src_gf_ordering);

   ParGridFunction *func_desired = new ParGridFunction(des_fes);
   {
      VectorFunctionCoefficient F(src_vdim, vector_func);
      func_desired->ProjectCoefficient(F);
   }

   // Global TRUE DoFs (portable reduction)
   {
      HYPRE_Int local_tv = src_fes->GetTrueVSize();
      long long local_ll = static_cast<long long>(local_tv);
      long long global_ll = 0;
      MPI_Allreduce(&local_ll, &global_ll, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

      if (myid == 0)
      {
         cout << "Global TRUE DoFs: " << global_ll << "\n";
      }
   }

   // ---------------- Target FE space ----------------
   if (fieldtype < 0)
   {
      const FiniteElementCollection *fec_in = func_source->FESpace()->FEColl();

      if (dynamic_cast<const H1_FECollection *>(fec_in))
      {
         fieldtype = 0;
      }
      else if (dynamic_cast<const L2_FECollection *>(fec_in))
      {
         fieldtype = 1;
      }
      else if (dynamic_cast<const RT_FECollection *>(fec_in))
      {
         fieldtype = 2;
      }
      else if (dynamic_cast<const ND_FECollection *>(fec_in))
      {
         fieldtype = 3;
      }
      else
      {
         MFEM_ABORT("Unsupported source GF type.");
      }
   }

   FiniteElementCollection *tar_fec = nullptr;
   int tar_vdim = src_vdim;

   if (fieldtype == 0)
   {
      tar_fec = new H1_FECollection(order, dim);
      if (src_fieldtype > 1) { tar_vdim = dim; }
   }
   else if (fieldtype == 1)
   {
      tar_fec = new L2_FECollection(order, dim);
      if (src_fieldtype > 1) { tar_vdim = dim; }
   }
   else if (fieldtype == 2)
   {
      tar_fec = new RT_FECollection(order, dim);
      tar_vdim = 1;
      MFEM_VERIFY(src_fieldtype > 1, "Cannot interpolate scalar to H(div).");
   }
   else if (fieldtype == 3)
   {
      tar_fec = new ND_FECollection(order, dim);
      tar_vdim = 1;
      MFEM_VERIFY(src_fieldtype > 1, "Cannot interpolate scalar to H(curl).");
   }
   else
   {
      MFEM_ABORT("Invalid target fieldtype.");
   }

   ParFiniteElementSpace *tar_fes =
      new ParFiniteElementSpace(&mesh_2, tar_fec, tar_vdim, src_fes->GetOrdering());

   ParGridFunction func_target(tar_fes);

   // ---------------- Build query points ----------------
   const int NE = mesh_2.GetNE();
   const int nsp = tar_fes->GetTypicalFE()->GetNodes().GetNPoints();
   const int tar_ncomp = func_target.VectorDim();

   Vector vxyz_raw;
   int point_ordering;

   const bool using_same_nodes = (fieldtype == 0 && order == mesh_poly_deg);

   if (using_same_nodes)
   {
      vxyz_raw = *mesh_2.GetNodes();
      point_ordering = mesh_2.GetNodes()->FESpace()->GetOrdering();
   }
   else
   {
      vxyz_raw.SetSize(nsp * NE * dim);

      for (int i = 0; i < NE; i++)
      {
         const FiniteElement *fe = tar_fes->GetFE(i);
         const IntegrationRule ir = fe->GetNodes();
         ElementTransformation *et = tar_fes->GetElementTransformation(i);

         DenseMatrix pos;
         et->Transform(ir, pos);

         Vector rowx(vxyz_raw.GetData() + i * nsp, nsp);
         Vector rowy(vxyz_raw.GetData() + i * nsp + NE * nsp, nsp);
         Vector rowz;

         pos.GetRow(0, rowx);

         if (dim >= 2)
         {
            pos.GetRow(1, rowy);
         }

         if (dim == 3)
         {
            rowz.SetDataAndSize(vxyz_raw.GetData() + i * nsp + 2 * NE * nsp, nsp);
            pos.GetRow(2, rowz);
         }
      }

      point_ordering = Ordering::byNODES;
   }

   const int nodes_cnt = static_cast<int>(vxyz_raw.Size() / dim);

   Vector vxyz_bn; // byNODES from here
   ReorderCoordsToByNodes(vxyz_raw, point_ordering, nodes_cnt, dim, vxyz_bn);
   point_ordering = Ordering::byNODES;

   // Characteristic size and tolerances
   const double h       = Lx / nx;
   const double bdr_tol = std::max(bdr_frac * h, 1e-6);
   const double eps_plane = std::max(plane_mult * bdr_tol, 2e-6);

   // ---------------- FindPoints: first pass ----------------
   Vector interp_vals(nodes_cnt * tar_ncomp);
   interp_vals = 0.0;

   FindPointsGSLIB finder(MPI_COMM_WORLD);
   finder.Setup(mesh_1, bb_rel);
   finder.SetDistanceToleranceForPointsFoundOnBoundary(bdr_tol);
   if (fieldtype == 1)
   {
      finder.SetL2AvgType(mfem::FindPointsGSLIB::ARITHMETIC);
   }

   finder.Interpolate(vxyz_bn, *func_source, interp_vals, point_ordering);

   const Array<unsigned int> &codes = finder.GetCode(); // 0=in, 1=boundary, 2=not found

   std::vector<int> missing;
   missing.reserve(nodes_cnt);

   int c_in = 0, c_bdr = 0, c_miss = 0;
   for (int i = 0; i < nodes_cnt; ++i)
   {
      if (codes[i] == 0) { c_in++; }
      else if (codes[i] == 1) { c_bdr++; }
      else { c_miss++; missing.push_back(i); }
   }

   int g_in=0, g_bdr=0, g_miss=0;
   MPI_Allreduce(&c_in,   &g_in,   1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(&c_bdr,  &g_bdr,  1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(&c_miss, &g_miss, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

   if (myid == 0)
   {
      const long long total = 1LL * nodes_cnt * nprocs;
      std::cout << "First pass codes: inside=" << g_in
                << " boundary=" << g_bdr
                << " notfound=" << g_miss
                << "  (total=" << total << ")\n";
   }

   // // ---------------- Retry keys (deterministic order on all ranks) ----------------
   // std::vector<std::array<int,3>> retry_keys;

   // // Faces
   // retry_keys.push_back({+1,  0,  0});
   // retry_keys.push_back({-1,  0,  0});
   // if (dim > 1)
   // {
   //    retry_keys.push_back({0, +1,  0});
   //    retry_keys.push_back({0, -1,  0});
   // }
   // if (dim > 2)
   // {
   //    retry_keys.push_back({0,  0, +1});
   //    retry_keys.push_back({0,  0, -1});
   // }

   // // Edges
   // if (dim > 1)
   // {
   //    for (int sx : {-1, +1})
   //    {
   //       for (int sy : {-1, +1})
   //       {
   //          retry_keys.push_back({sx, sy, 0});
   //       }
   //    }
   // }
   // if (dim > 2)
   // {
   //    for (int sx : {-1, +1})
   //    {
   //       for (int sz : {-1, +1})
   //       {
   //          retry_keys.push_back({sx, 0, sz});
   //       }
   //    }
   //    for (int sy : {-1, +1})
   //    {
   //       for (int sz : {-1, +1})
   //       {
   //          retry_keys.push_back({0, sy, sz});
   //       }
   //    }
   // }

   // // Corners
   // if (dim > 2)
   // {
   //    for (int sx : {-1, +1})
   //    {
   //       for (int sy : {-1, +1})
   //       {
   //          for (int sz : {-1, +1})
   //          {
   //             retry_keys.push_back({sx, sy, sz});
   //          }
   //       }
   //    }
   // }

   // auto select_candidates_for_key =
   //    [&](const std::array<int,3>& key,
   //        const std::vector<int>& pool,
   //        std::vector<int>& out_ids)
   // {
   //    out_ids.clear();

   //    const int sx = key[0], sy = key[1], sz = key[2];

   //    for (int idx : pool)
   //    {
   //       const double x = vxyz_bn[idx];
   //       const double y = (dim > 1) ? vxyz_bn[nodes_cnt + idx]       : 0.0;
   //       const double z = (dim > 2) ? vxyz_bn[2 * nodes_cnt + idx]   : 0.0;

   //       bool pass = true;

   //       if (sx == +1) { pass = pass && (x < eps_plane); }
   //       if (sx == -1) { pass = pass && (x > Lx - eps_plane); }

   //       if (dim > 1)
   //       {
   //          if (sy == +1) { pass = pass && (y < eps_plane); }
   //          if (sy == -1) { pass = pass && (y > Ly - eps_plane); }
   //       }

   //       if (dim > 2)
   //       {
   //          if (sz == +1) { pass = pass && (z < eps_plane); }
   //          if (sz == -1) { pass = pass && (z > Lz - eps_plane); }
   //       }

   //       if (pass)
   //       {
   //          out_ids.push_back(idx);
   //       }
   //    }
   // };

   // // ---------------- Collective retry loop (plane-filtered) ----------------
   // int found_faces = 0;
   // int found_edges_corners = 0;

   // std::vector<int> local_group;
   // local_group.reserve(missing.size());

   // for (size_t k = 0; k < retry_keys.size(); ++k)
   // {
   //    const std::array<int,3> key = retry_keys[k];

   //    // Face if exactly one non-zero in 2D, or exactly one zero in 3D? Simpler:
   //    const bool is_face =
   //       ( (key[0] == 0) + (key[1] == 0) + (key[2] == 0) ==
   //         (dim == 3 ? 2 : 1) );

   //    select_candidates_for_key(key, missing, local_group);

   //    int local_has = local_group.empty() ? 0 : 1;
   //    int global_has = 0;
   //    MPI_Allreduce(&local_has, &global_has, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

   //    if (global_has == 0)
   //    {
   //       continue; // nothing to do for this key on any rank
   //    }

   //    Vector sub_coords;
   //    BuildSubsetCoordsByNodes(vxyz_bn, dim, local_group, sub_coords);
   //    const int subN = static_cast<int>(local_group.size());

   //    Vector sub_vals(subN * tar_ncomp);
   //    sub_vals = 0.0;

   //    finder.Interpolate(sub_coords, *func_source, sub_vals, Ordering::byNODES);
   //    const Array<unsigned int> &sub_codes = finder.GetCode();

   //    int local_new_found = 0;

   //    for (int d = 0; d < tar_ncomp; ++d)
   //    {
   //       double *dst = interp_vals.GetData() + d * nodes_cnt;
   //       const double *src = sub_vals.GetData() + d * subN;

   //       for (int i = 0; i < subN; ++i)
   //       {
   //          if (sub_codes[i] != 2)
   //          {
   //             dst[ local_group[i] ] = src[i];
   //             local_new_found++;
   //          }
   //       }
   //    }

   //    int global_new_found = 0;
   //    MPI_Allreduce(&local_new_found, &global_new_found, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

   //    if (is_face) { found_faces += global_new_found; }
   //    else         { found_edges_corners += global_new_found; }

   //    // Rebuild local missing pool
   //    if (subN > 0)
   //    {
   //       std::vector<char> is_found(nodes_cnt, 0);
   //       for (int i = 0; i < subN; ++i)
   //       {
   //          if (sub_codes[i] != 2)
   //          {
   //             is_found[ local_group[i] ] = 1;
   //          }
   //       }

   //       std::vector<int> next_missing;
   //       next_missing.reserve(missing.size());
   //       for (int idx : missing)
   //       {
   //          if (!is_found[idx])
   //          {
   //             next_missing.push_back(idx);
   //          }
   //       }
   //       missing.swap(next_missing);
   //    }
   // }

   // if (myid == 0)
   // {
   //    std::cout << "Plane-filtered retries found: faces=" << found_faces
   //              << ", edges/corners=" << found_edges_corners << "\n";
   // }

   // ---------------- Final brute-force collective pass (no plane filter) ----------------
   // If anything remains, try all offsets again on the entire remaining set.
   int remaining_local = static_cast<int>(missing.size());
   int remaining_global = 0;
   MPI_Allreduce(&remaining_local, &remaining_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

   if (remaining_global > 0)
   {
      if (myid == 0)
      {
         std::cout << "Final brute-force retries on remaining " << remaining_global << " points.\n";
      }

      auto collective_retry_whole = [&](double ox, double oy, double oz)
      {
         // Everyone participates once per offset
         Vector sub_coords;
         BuildSubsetCoordsByNodes(vxyz_bn, dim, missing, sub_coords);
         const int subN = static_cast<int>(missing.size());

         // Apply offsets to local chunk
         for (int k = 0; k < subN; ++k) { sub_coords[k] += ox; }
         if (dim > 1) { for (int k = 0; k < subN; ++k) { sub_coords[subN + k] += oy; } }
         if (dim > 2) { for (int k = 0; k < subN; ++k) { sub_coords[2 * subN + k] += oz; } }

         Vector sub_vals(subN * tar_ncomp);
         sub_vals = 0.0;

         finder.Interpolate(sub_coords, *func_source, sub_vals, Ordering::byNODES);
         const Array<unsigned int> &sub_codes = finder.GetCode();

         // Scatter successes
         for (int d = 0; d < tar_ncomp; ++d)
         {
            double *dst = interp_vals.GetData() + d * nodes_cnt;
            const double *src = sub_vals.GetData() + d * subN;

            for (int i = 0; i < subN; ++i)
            {
               if (sub_codes[i] != 2)
               {
                  dst[ missing[i] ] = src[i];
               }
            }
         }

         // Shrink local missing
         std::vector<int> still;
         still.reserve(subN);
         for (int i = 0; i < subN; ++i)
         {
            if (sub_codes[i] == 2)
            {
               still.push_back(missing[i]);
            }
         }
         missing.swap(still);
      };

      // Faces
      collective_retry_whole( Lx, 0.0, 0.0);
      collective_retry_whole(-Lx, 0.0, 0.0);
      if (dim > 1)
      {
         collective_retry_whole(0.0,  Ly, 0.0);
         collective_retry_whole(0.0, -Ly, 0.0);
      }
      if (dim > 2)
      {
         collective_retry_whole(0.0, 0.0,  Lz);
         collective_retry_whole(0.0, 0.0, -Lz);
      }

      // Edges
      if (dim > 1)
      {
         collective_retry_whole( Lx,  Ly, 0.0);
         collective_retry_whole( Lx, -Ly, 0.0);
         collective_retry_whole(-Lx,  Ly, 0.0);
         collective_retry_whole(-Lx, -Ly, 0.0);
      }
      if (dim > 2)
      {
         collective_retry_whole( Lx, 0.0,  Lz);
         collective_retry_whole( Lx, 0.0, -Lz);
         collective_retry_whole(-Lx, 0.0,  Lz);
         collective_retry_whole(-Lx, 0.0, -Lz);

         collective_retry_whole(0.0,  Ly,  Lz);
         collective_retry_whole(0.0,  Ly, -Lz);
         collective_retry_whole(0.0, -Ly,  Lz);
         collective_retry_whole(0.0, -Ly, -Lz);
      }

      // Corners
      if (dim > 2)
      {
         for (double ox : { Lx, -Lx })
         {
            for (double oy : { Ly, -Ly })
            {
               for (double oz : { Lz, -Lz })
               {
                  collective_retry_whole(ox, oy, oz);
               }
            }
         }
      }
   }

   int final_missing_local = static_cast<int>(missing.size());
   int final_missing_global = 0;
   MPI_Allreduce(&final_missing_local, &final_missing_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

   if (myid == 0)
   {
      const long long total_pts = 1LL * nodes_cnt * nprocs;
      const long long found = total_pts - final_missing_global;
      std::cout << "Final: found " << found << " / " << total_pts
                << " (" << 100.0 * double(found) / double(total_pts) << "%)\n";

      if (final_missing_global > 0)
      {
         std::cout << "NOTE: remaining misses = " << final_missing_global
                   << ". Try --bb 0.35..0.45 or --bdrfrac 0.14..0.18 or --plane-mult 4.0.\n";
      }
   }

   // ---------------- Project to target space ----------------
   if (fieldtype <= 1) // H1 or L2
   {
      const bool direct_assign =
         (fieldtype == 1) || (fieldtype == 0 && order == mesh_poly_deg);

      if (direct_assign)
      {
         func_target = interp_vals;
      }
      else
      {
         Array<int> vdofs;
         Vector vals;
         Vector elem_dof_vals(nsp * tar_ncomp);

         for (int i = 0; i < mesh_2.GetNE(); i++)
         {
            tar_fes->GetElementVDofs(i, vdofs);
            vals.SetSize(vdofs.Size());

            for (int j = 0; j < nsp; j++)
            {
               for (int d = 0; d < tar_ncomp; d++)
               {
                  const int idx = d * (nsp * NE) + i * nsp + j;
                  elem_dof_vals(j + d * nsp) = interp_vals(idx);
               }
            }
            func_target.SetSubVector(vdofs, elem_dof_vals);
         }
      }
   }
   else // H(div) or H(curl)
   {
      Array<int> vdofs;
      Vector vals;
      Vector elem_dof_vals(nsp * tar_ncomp);

      for (int i = 0; i < mesh_2.GetNE(); i++)
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

         func_target.SetSubVector(vdofs, vals);
      }
   }

   func_target.SetTrueVector();
   func_target.SetFromTrueVector();

   // Global TRUE DoFs (portable reduction)
   {
      HYPRE_Int local_tv = tar_fes->GetTrueVSize();
      long long local_ll = static_cast<long long>(local_tv);
      long long global_ll = 0;
      MPI_Allreduce(&local_ll, &global_ll, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

      if (myid == 0)
      {
         cout << "Target Global TRUE DoFs: " << global_ll << "\n";
      }
   }

   // ---------------- Visualization ----------------
   if (visualization)
   {
      ShowInGLVis(mesh_1, *func_source, "Source mesh + source field", visport, 0, 0);
      ShowInGLVis(mesh_2,  func_target, "Target mesh + interpolated field", visport, 620, 0);
   }

   // ---------------- Output ----------------
   if (visit_output && myid == 0)
   {
      VisItDataCollection dc("TargetMesh", &mesh_2);
      dc.SetPrecision(8);
      dc.RegisterField("u_interp", &func_target);
      dc.RegisterField("u_exact",  func_desired);
      dc.SetCycle(0);
      dc.SetTime(0.0);
      dc.Save();
   }

   if (myid == 0)
   {
      ofstream ofs("interpolated.gf");
      ofs.precision(8);
      func_target.Save(ofs);
   }

   // ---------------- Cleanup ----------------
   // (finder.FreeData() is optional; destructor will handle it)
   // finder.FreeData();

   delete func_source;
   delete func_desired;
   delete src_fes;
   delete des_fes;
   delete src_fec;
   delete tar_fes;
   delete tar_fec;

   return 0;
}

/*
Build:
mpicxx -std=c++17 -O3 -I$MFEM_DIR -L$MFEM_DIR -o field-interp-victor \
       field-interp-victor.cpp -lmfem -lHYPRE -lmetis

Examples:
mpirun -np 1 ./field-interp-victor -n 16 -o 2 -fts 0 -nc 3 -ft 0 --bb 0.3 --bdrfrac 0.12
mpirun -np 4 ./field-interp-victor -n 16 -o 2 -fts 0 -nc 3 -ft 0 --bb 0.35 --bdrfrac 0.14 --plane-mult 4.0

Notes:
- If you still see tiny face speckles, try:
    --plane-mult 4.0   (thicker plane window)
    --bb 0.35..0.45    (bigger per-element AABB)
    --bdrfrac 0.14..0.18
- You can also test with --no-wrap to compare behavior without the initial wrap.
*/
