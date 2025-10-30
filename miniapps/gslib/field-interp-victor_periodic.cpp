// Copyright (c) 2010-2025,
// Lawrence Livermore National Security, LLC. LLNL-CODE-806117.
// See LICENSE and NOTICE for details.

#include "mfem.hpp"
#include <fstream>
#include <vector>
#include <array>

using namespace mfem;
using namespace std;

// =============================================================
// Problem setup - UNCHANGED FROM WORKING VERSION
// =============================================================

static double g_amp = 0.05;

void PerturbMeshTransform(const Vector &x_in, Vector &x_out)
{
   const double freq = 2.0 * M_PI;
   x_out = x_in;
   x_out[0] += g_amp * sin(freq * x_in[1]) * cos(freq * x_in[2]);
   x_out[1] += g_amp * cos(freq * x_in[0]) * sin(freq * x_in[2]) * 0.8;
   x_out[2] += g_amp * sin(freq * x_in[0]) * cos(freq * x_in[1]) * 0.6;
}

void vector_func(const Vector &p, Vector &F)
{
   const double xi = 2.0 * M_PI * p(0);
   const double yi = 2.0 * M_PI * p(1);
   const double zi = 2.0 * M_PI * p(2);

   F(0) = sin(xi) * cos(yi) * cos(zi);
   F(1) = -cos(xi) * sin(yi) * cos(zi);
   F(2) = 0.0;
}

// =============================================================
// Helper Functions - UNCHANGED FROM WORKING VERSION
// =============================================================

static void ReorderCoordsToByNodes(const Vector &in, int ordering, int nodes_cnt, int dim, Vector &out)
{
   out.SetSize(in.Size());
   if (ordering == Ordering::byNODES) {
      out = in;
      return;
   }
   for (int i = 0; i < nodes_cnt; ++i) {
      for (int d = 0; d < dim; ++d) {
         out[d * nodes_cnt + i] = in[i * dim + d];
      }
   }
}

static void BuildSubsetCoordsByNodes(const Vector &all_bn, int dim, const vector<int> &indices, Vector &subset_bn)
{
   const int N = indices.size();
   subset_bn.SetSize(dim * N);
   const int fullN = all_bn.Size() / dim;
   for (int k = 0; k < N; ++k) {
      const int i = indices[k];
      for (int d = 0; d < dim; ++d) {
         subset_bn[d * N + k] = all_bn[d * fullN + i];
      }
   }
}

static void ShowInGLVis(const ParMesh &pmesh, const ParGridFunction &pgf, const char *title, int visport, int x = 0, int y = 0)
{
   char vishost[] = "localhost";
   socketstream sout;
   sout.open(vishost, visport);
   if (!sout) return;
   sout.precision(8);
   sout << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() << "\n";
   sout << "solution\n" << pmesh << pgf << "window_title '" << title << "'\n"
        << "window_geometry " << x << " " << y << " 600 600\n";
   if (pmesh.Dimension() == 3) sout << "keys mA\n";
   sout << flush;
}

// =============================================================
// Main Function - MINIMAL CLEANUP, SAME LOGIC
// =============================================================

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   const int myid = Mpi::WorldRank();
   const int nprocs = Mpi::WorldSize();
   Hypre::Init();

#ifndef MFEM_USE_GSLIB
   if (myid == 0) cerr << "ERROR: Requires MFEM built with GSLIB (MFEM_USE_GSLIB=YES).\n";
   return 2;
#endif

   // Parameters - SAME AS ORIGINAL
   int nx = 16, order = 2, ref_levels = 0;
   int src_fieldtype = 0, src_ncomp = 1, src_gf_ordering = 0, fieldtype = -1;
   bool visualization = true, visit_output = false;
   double bb_rel = 0.20, bdr_frac = 1.0, plane_mult = 1.0;
   int visport = 19916;

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

   if (!args.Good()) {
      if (myid == 0) args.PrintUsage(cout);
      return 1;
   }

   if (myid == 0) {
      args.PrintOptions(cout);
      cout << "\n=== Parallel FindPoints Interpolation ===\n";
   }

   // Create periodic meshes - SAME AS ORIGINAL
   const double Lx = 1.0, Ly = 1.0, Lz = 1.0;
   Mesh mesh_1_init = Mesh::MakeCartesian3D(nx, nx, nx, Element::HEXAHEDRON, Lx, Ly, Lz, false);

   Vector xT(3); xT = 0.0; xT[0] = Lx;
   Vector yT(3); yT = 0.0; yT[1] = Ly;
   Vector zT(3); zT = 0.0; zT[2] = Lz;
   vector<Vector> translations = {xT, yT, zT};
   vector<int> v2v = mesh_1_init.CreatePeriodicVertexMapping(translations);

   Mesh mesh_1_serial = Mesh::MakePeriodic(mesh_1_init, v2v);
   Mesh mesh_2_serial = Mesh::MakePeriodic(mesh_1_init, v2v);
   mesh_1_serial.Transform(PerturbMeshTransform);

   const int dim = mesh_1_serial.Dimension();
   for (int l = 0; l < ref_levels; l++) {
      mesh_2_serial.UniformRefinement();
   }

   if (!mesh_1_serial.GetNodes()) mesh_1_serial.SetCurvature(1);
   if (!mesh_2_serial.GetNodes()) mesh_2_serial.SetCurvature(1);

   const int mesh_poly_deg = mesh_2_serial.GetNodes()->FESpace()->GetElementOrder(0);

   // Create parallel meshes
   ParMesh mesh_1(MPI_COMM_WORLD, mesh_1_serial);
   ParMesh mesh_2(MPI_COMM_WORLD, mesh_2_serial);
   mesh_1_serial.Clear();
   mesh_2_serial.Clear();

   // Create source FE space & field - SAME AS ORIGINAL
   int src_vdim = src_ncomp;
   FiniteElementCollection *src_fec = nullptr;
   switch(src_fieldtype) {
      case 0: src_fec = new H1_FECollection(order, dim); break;
      case 1: src_fec = new L2_FECollection(order, dim); break;
      case 2: src_fec = new RT_FECollection(order, dim); src_vdim = dim; break;
      case 3: src_fec = new ND_FECollection(order, dim); src_vdim = dim; break;
      default: MFEM_ABORT("Invalid src_fieldtype.");
   }

   ParFiniteElementSpace *src_fes = new ParFiniteElementSpace(&mesh_1, src_fec, src_ncomp, src_gf_ordering);
   ParGridFunction *func_source = new ParGridFunction(src_fes);
   {
      VectorFunctionCoefficient F(src_vdim, vector_func);
      func_source->ProjectCoefficient(F);
   }

   ParFiniteElementSpace *des_fes = new ParFiniteElementSpace(&mesh_2, src_fec, src_ncomp, src_gf_ordering);
   ParGridFunction *func_desired = new ParGridFunction(des_fes);
   {
      VectorFunctionCoefficient F(src_vdim, vector_func);
      func_desired->ProjectCoefficient(F);
   }

   // Determine target field type - SAME AS ORIGINAL
   if (fieldtype < 0) {
      const FiniteElementCollection *fec_in = func_source->FESpace()->FEColl();
      if (dynamic_cast<const H1_FECollection*>(fec_in)) fieldtype = 0;
      else if (dynamic_cast<const L2_FECollection*>(fec_in)) fieldtype = 1;
      else if (dynamic_cast<const RT_FECollection*>(fec_in)) fieldtype = 2;
      else if (dynamic_cast<const ND_FECollection*>(fec_in)) fieldtype = 3;
      else MFEM_ABORT("Unsupported source GF type.");
   }

   // Create target FE space - SAME AS ORIGINAL
   FiniteElementCollection *tar_fec = nullptr;
   int tar_vdim = src_vdim;
   switch(fieldtype) {
      case 0: 
         tar_fec = new H1_FECollection(order, dim);
         if (src_fieldtype > 1) tar_vdim = dim;
         break;
      case 1:
         tar_fec = new L2_FECollection(order, dim);
         if (src_fieldtype > 1) tar_vdim = dim;
         break;
      case 2:
         tar_fec = new RT_FECollection(order, dim);
         tar_vdim = 1;
         MFEM_VERIFY(src_fieldtype > 1, "Cannot interpolate scalar to H(div).");
         break;
      case 3:
         tar_fec = new ND_FECollection(order, dim);
         tar_vdim = 1;
         MFEM_VERIFY(src_fieldtype > 1, "Cannot interpolate scalar to H(curl).");
         break;
      default: MFEM_ABORT("Invalid target fieldtype.");
   }

   ParFiniteElementSpace *tar_fes = new ParFiniteElementSpace(&mesh_2, tar_fec, tar_vdim, src_fes->GetOrdering());
   ParGridFunction func_target(tar_fes);

   // Build query points - SAME AS ORIGINAL
   const int NE = mesh_2.GetNE();
   const int nsp = tar_fes->GetFE(0)->GetNodes().GetNPoints();
   const int tar_ncomp = func_target.VectorDim();

   Vector vxyz_raw;
   int point_ordering;
   const bool using_same_nodes = (fieldtype == 0 && order == mesh_poly_deg);

   if (using_same_nodes) {
      vxyz_raw = *mesh_2.GetNodes();
      point_ordering = mesh_2.GetNodes()->FESpace()->GetOrdering();
   } else {
      vxyz_raw.SetSize(nsp * NE * dim);
      for (int i = 0; i < NE; i++) {
         const FiniteElement *fe = tar_fes->GetFE(i);
         const IntegrationRule &ir = fe->GetNodes();
         ElementTransformation *et = tar_fes->GetElementTransformation(i);
         DenseMatrix pos;
         et->Transform(ir, pos);

         Vector rowx(vxyz_raw.GetData() + i * nsp, nsp);
         Vector rowy(vxyz_raw.GetData() + i * nsp + NE * nsp, nsp);
         pos.GetRow(0, rowx);
         if (dim >= 2) pos.GetRow(1, rowy);
         if (dim == 3) {
            Vector rowz(vxyz_raw.GetData() + i * nsp + 2 * NE * nsp, nsp);
            pos.GetRow(2, rowz);
         }
      }
      point_ordering = Ordering::byNODES;
   }

   const int nodes_cnt = vxyz_raw.Size() / dim;
   Vector vxyz_bn;
   ReorderCoordsToByNodes(vxyz_raw, point_ordering, nodes_cnt, dim, vxyz_bn);

   // Setup FindPoints and interpolate - SAME AS ORIGINAL
   const double h = Lx / nx;
   const double bdr_tol = max(bdr_frac * h, 1e-6);
   const double eps_plane = max(plane_mult * bdr_tol, 2e-6);

   Vector interp_vals(nodes_cnt * tar_ncomp);
   FindPointsGSLIB finder(MPI_COMM_WORLD);
   finder.Setup(mesh_1, bb_rel);
   finder.SetDistanceToleranceForPointsFoundOnBoundary(bdr_tol);
   if (fieldtype == 1) finder.SetL2AvgType(FindPointsGSLIB::ARITHMETIC);

   finder.Interpolate(vxyz_bn, *func_source, interp_vals, Ordering::byNODES);
   const Array<unsigned int> &codes = finder.GetCode();

   // Find missing points - SAME AS ORIGINAL
   vector<int> missing;
   for (int i = 0; i < nodes_cnt; ++i) {
      if (codes[i] == 2) missing.push_back(i);
   }

   // Retry keys - SAME AS ORIGINAL
   vector<array<int,3>> retry_keys;
   // Faces
   retry_keys.push_back({+1,0,0}); retry_keys.push_back({-1,0,0});
   if (dim > 1) { retry_keys.push_back({0,+1,0}); retry_keys.push_back({0,-1,0}); }
   if (dim > 2) { retry_keys.push_back({0,0,+1}); retry_keys.push_back({0,0,-1}); }
   // Edges
   if (dim > 1) for (int sx : {-1,+1}) for (int sy : {-1,+1}) retry_keys.push_back({sx,sy,0});
   if (dim > 2) {
      for (int sx : {-1,+1}) for (int sz : {-1,+1}) retry_keys.push_back({sx,0,sz});
      for (int sy : {-1,+1}) for (int sz : {-1,+1}) retry_keys.push_back({0,sy,sz});
   }
   // Corners
   if (dim > 2) for (int sx : {-1,+1}) for (int sy : {-1,+1}) for (int sz : {-1,+1}) retry_keys.push_back({sx,sy,sz});

// Collective retry loop - FIXED FOR PARALLEL
for (const auto &key : retry_keys) {
    if (missing.empty()) break;
    vector<int> local_group;
    const int sx = key[0], sy = key[1], sz = key[2];
    
    for (int idx : missing) {
        const double x = vxyz_bn[idx];
        const double y = (dim > 1) ? vxyz_bn[nodes_cnt + idx] : 0.0;
        const double z = (dim > 2) ? vxyz_bn[2 * nodes_cnt + idx] : 0.0;
        bool pass = true;
        if (sx == +1) pass = pass && (x < eps_plane);
        if (sx == -1) pass = pass && (x > Lx - eps_plane);
        if (dim > 1) {
            if (sy == +1) pass = pass && (y < eps_plane);
            if (sy == -1) pass = pass && (y > Ly - eps_plane);
        }
        if (dim > 2) {
            if (sz == +1) pass = pass && (z < eps_plane);
            if (sz == -1) pass = pass && (z > Lz - eps_plane);
        }
        if (pass) local_group.push_back(idx);
    }

    // FIX: All ranks must participate in Interpolate calls
    int local_has = local_group.empty() ? 0 : 1;
    int global_has = 0;
    MPI_Allreduce(&local_has, &global_has, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (global_has == 0) continue;

    Vector sub_coords;
    BuildSubsetCoordsByNodes(vxyz_bn, dim, local_group, sub_coords);
    const int subN = local_group.size();

    Vector sub_vals(subN * tar_ncomp);
    // All ranks call Interpolate, even if they have no points
    finder.Interpolate(sub_coords, *func_source, sub_vals, Ordering::byNODES);
    const Array<unsigned int> &sub_codes = finder.GetCode();

    for (int d = 0; d < tar_ncomp; ++d) {
        double *dst = interp_vals.GetData() + d * nodes_cnt;
        const double *src = sub_vals.GetData() + d * subN;
        for (int i = 0; i < subN; ++i) {
            if (sub_codes[i] != 2) dst[local_group[i]] = src[i];
        }
    }

    vector<int> next_missing;
    for (int idx : missing) {
        bool found = false;
        for (size_t i = 0; i < local_group.size(); i++) {
            if (local_group[i] == idx && sub_codes[i] != 2) {
                found = true;
                break;
            }
        }
        if (!found) next_missing.push_back(idx);
    }
    missing = next_missing;
}

// Final brute-force - FIXED FOR PARALLEL
if (!missing.empty()) {
    vector<array<double,3>> offsets = {
        {Lx,0,0}, {-Lx,0,0}, {0,Ly,0}, {0,-Ly,0}, {0,0,Lz}, {0,0,-Lz},
        {Lx,Ly,0}, {Lx,-Ly,0}, {-Lx,Ly,0}, {-Lx,-Ly,0},
        {Lx,0,Lz}, {Lx,0,-Lz}, {-Lx,0,Lz}, {-Lx,0,-Lz},
        {0,Ly,Lz}, {0,Ly,-Lz}, {0,-Ly,Lz}, {0,-Ly,-Lz},
        {Lx,Ly,Lz}, {Lx,Ly,-Lz}, {Lx,-Ly,Lz}, {Lx,-Ly,-Lz},
        {-Lx,Ly,Lz}, {-Lx,Ly,-Lz}, {-Lx,-Ly,Lz}, {-Lx,-Ly,-Lz}
    };

    for (const auto &offset : offsets) {
        // FIX: Check globally if any points remain
        int remaining_local = missing.size();
        int remaining_global = 0;
        MPI_Allreduce(&remaining_local, &remaining_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (remaining_global == 0) break;

        Vector sub_coords;
        BuildSubsetCoordsByNodes(vxyz_bn, dim, missing, sub_coords);
        const int subN = missing.size();

        for (int k = 0; k < subN; ++k) sub_coords[k] += offset[0];
        if (dim > 1) for (int k = 0; k < subN; ++k) sub_coords[subN + k] += offset[1];
        if (dim > 2) for (int k = 0; k < subN; ++k) sub_coords[2 * subN + k] += offset[2];

        Vector sub_vals(subN * tar_ncomp);
        // All ranks call Interpolate
        finder.Interpolate(sub_coords, *func_source, sub_vals, Ordering::byNODES);
        const Array<unsigned int> &sub_codes = finder.GetCode();

        vector<int> still_missing;
        for (int i = 0; i < subN; ++i) {
            if (sub_codes[i] != 2) {
                for (int d = 0; d < tar_ncomp; ++d) {
                    interp_vals[d * nodes_cnt + missing[i]] = sub_vals[d * subN + i];
                }
            } else {
                still_missing.push_back(missing[i]);
            }
        }
        missing = still_missing;
    }
}


   // Final brute-force - SAME AS ORIGINAL
   if (!missing.empty()) {
      vector<array<double,3>> offsets = {
         {Lx,0,0}, {-Lx,0,0}, {0,Ly,0}, {0,-Ly,0}, {0,0,Lz}, {0,0,-Lz},
         {Lx,Ly,0}, {Lx,-Ly,0}, {-Lx,Ly,0}, {-Lx,-Ly,0},
         {Lx,0,Lz}, {Lx,0,-Lz}, {-Lx,0,Lz}, {-Lx,0,-Lz},
         {0,Ly,Lz}, {0,Ly,-Lz}, {0,-Ly,Lz}, {0,-Ly,-Lz},
         {Lx,Ly,Lz}, {Lx,Ly,-Lz}, {Lx,-Ly,Lz}, {Lx,-Ly,-Lz},
         {-Lx,Ly,Lz}, {-Lx,Ly,-Lz}, {-Lx,-Ly,Lz}, {-Lx,-Ly,-Lz}
      };

      for (const auto &offset : offsets) {
         if (missing.empty()) break;
         Vector sub_coords;
         BuildSubsetCoordsByNodes(vxyz_bn, dim, missing, sub_coords);
         const int subN = missing.size();

         for (int k = 0; k < subN; ++k) sub_coords[k] += offset[0];
         if (dim > 1) for (int k = 0; k < subN; ++k) sub_coords[subN + k] += offset[1];
         if (dim > 2) for (int k = 0; k < subN; ++k) sub_coords[2 * subN + k] += offset[2];

         Vector sub_vals(subN * tar_ncomp);
         finder.Interpolate(sub_coords, *func_source, sub_vals, Ordering::byNODES);
         const Array<unsigned int> &sub_codes = finder.GetCode();

         vector<int> still_missing;
         for (int i = 0; i < subN; ++i) {
            if (sub_codes[i] != 2) {
               for (int d = 0; d < tar_ncomp; ++d) {
                  interp_vals[d * nodes_cnt + missing[i]] = sub_vals[d * subN + i];
               }
            } else {
               still_missing.push_back(missing[i]);
            }
         }
         missing = still_missing;
      }
   }

   // Project to target space - SAME AS ORIGINAL
   if (fieldtype <= 1) {
      const bool direct_assign = (fieldtype == 1) || (fieldtype == 0 && order == mesh_poly_deg);
      if (direct_assign) {
         func_target = interp_vals;
      } else {
         Array<int> vdofs;
         Vector elem_dof_vals(nsp * tar_ncomp);
         for (int i = 0; i < mesh_2.GetNE(); i++) {
            tar_fes->GetElementVDofs(i, vdofs);
            for (int j = 0; j < nsp; j++) {
               for (int d = 0; d < tar_ncomp; d++) {
                  const int idx = d * (nsp * NE) + i * nsp + j;
                  elem_dof_vals(j + d * nsp) = interp_vals(idx);
               }
            }
            func_target.SetSubVector(vdofs, elem_dof_vals);
         }
      }
   } else {
      Array<int> vdofs;
      Vector vals, elem_dof_vals(nsp * tar_ncomp);
      for (int i = 0; i < mesh_2.GetNE(); i++) {
         tar_fes->GetElementVDofs(i, vdofs);
         vals.SetSize(vdofs.Size());
         for (int j = 0; j < nsp; j++) {
            for (int d = 0; d < tar_ncomp; d++) {
               const int idx = d * (nsp * NE) + i * nsp + j;
               elem_dof_vals(j * tar_ncomp + d) = interp_vals(idx);
            }
         }
         tar_fes->GetFE(i)->ProjectFromNodes(elem_dof_vals, *tar_fes->GetElementTransformation(i), vals);
         func_target.SetSubVector(vdofs, vals);
      }
   }

   func_target.SetTrueVector();
   func_target.SetFromTrueVector();

   // Visualization and output - SAME AS ORIGINAL
   if (visualization) {
      ShowInGLVis(mesh_1, *func_source, "Source Field", visport, 0, 0);
      ShowInGLVis(mesh_2, func_target, "Interpolated Field", visport, 620, 0);
   }

   if (visit_output && myid == 0) {
      VisItDataCollection dc("TargetMesh", &mesh_2);
      dc.SetPrecision(8);
      dc.RegisterField("u_interp", &func_target);
      dc.RegisterField("u_exact", func_desired);
      dc.Save();
   }

   if (myid == 0) {
      ofstream ofs("interpolated.gf");
      ofs.precision(8);
      func_target.Save(ofs);
   }

   // Cleanup
   delete func_source;
   delete func_desired;
   delete src_fes;
   delete des_fes;
   delete src_fec;
   delete tar_fes;
   delete tar_fec;

   return 0;
}