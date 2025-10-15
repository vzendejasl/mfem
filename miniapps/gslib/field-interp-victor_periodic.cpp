// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mfem.hpp"
#include <fstream>
#include <cmath>
#include <iomanip>

using namespace mfem;
using namespace std;

// Mesh transformation for perturbation
static double g_amp = 0.05;
void PerturbMeshTransform(const Vector &x_in, Vector &x_out)
{
   const double freq = 2.0*M_PI;
   x_out = x_in;
   x_out[0] += g_amp * std::sin(freq*x_in[1]) * std::cos(freq*x_in[2]);
   x_out[1] += g_amp * std::cos(freq*x_in[0]) * std::sin(freq*x_in[2]) * 0.8;
   x_out[2] += g_amp * std::sin(freq*x_in[0]) * std::cos(freq*x_in[1]) * 0.6;
}

// Create periodic mesh
Mesh MakePeriodicMesh(int nx, int ny, int nz, double L)
{
   Mesh base_mesh = Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON, 
                                          L, L, L, false);
   
   Vector x_trans({L, 0.0, 0.0});
   Vector y_trans({0.0, L, 0.0});
   Vector z_trans({0.0, 0.0, L});
   std::vector<Vector> translations = {x_trans, y_trans, z_trans};
   
   std::vector<int> v2v = base_mesh.CreatePeriodicVertexMapping(translations);
   return Mesh::MakePeriodic(base_mesh, v2v);
}

// Scalar function to project
double scalar_func(const Vector &x)
{
   return std::sin(M_PI*x[0]) * std::cos(M_PI*x[1]) * std::sin(0.5*M_PI*x[2]);
}

// Vector field function
void vector_func(const Vector &p, Vector &F)
{
   double xi = 2*M_PI*p(0);
   double yi = 2*M_PI*p(1);
   double zi = 2*M_PI*p(2);

   F(0) = sin(xi) * cos(yi) * cos(zi);
   F(1) = -cos(xi) * sin(yi) * cos(zi);
   F(2) = 0.0;
}

// Helper function to ensure mesh has nodes
void EnsureNodes(Mesh &mesh, int order)
{
   if (!mesh.GetNodes())
   {
      mesh.SetCurvature(order, false, mesh.Dimension(), Ordering::byNODES);
   }
}

int main (int argc, char *argv[])
{
   // Initialize MPI
   Mpi::Init(argc, argv);
   int myid = Mpi::WorldRank();
   int nprocs = Mpi::WorldSize();
   Hypre::Init();

   // Set the method's default parameters.
   int nx = 8;
   int order = 3;
   double L = 1.0;
   bool visualization = false;
   bool visit_output = true;
   int visport = 19916;
   bool vector_field = false;

   // Parse command-line options.
   OptionsParser args(argc, argv);
   args.AddOption(&nx, "-nx", "--nx", "Number of elements per dimension.");
   args.AddOption(&order, "-o", "--order", "Finite element order.");
   args.AddOption(&g_amp, "-amp", "--amplitude", "Perturbation amplitude.");
   args.AddOption(&L, "-L", "--domain-size", "Domain size.");
   args.AddOption(&vector_field, "-vec", "--vector-field", "-no-vec", 
                  "--no-vector-field", "Use vector field instead of scalar field.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable or disable GLVis visualization.");
   args.AddOption(&visit_output, "-visit", "--visit-output", "-no-visit", 
                  "--no-visit-output", "Enable or disable VisIt output.");
   args.AddOption(&visport, "-p", "--send-port", "Socket for GLVis.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

#ifndef MFEM_USE_GSLIB
   if (myid == 0)
   {
      std::cerr << "ERROR: Requires MFEM built with GSLIB (MFEM_USE_GSLIB=YES).\n";
   }
   return 2;
#endif

   if (myid == 0)
   {
      cout << "\n=== Parallel FindPoints Interpolation ===\n";
      cout << "MPI ranks: " << nprocs << "\n";
      cout << "Grid: " << nx << " x " << nx << " x " << nx << "\n";
      cout << "Order: " << order << "\n";
      cout << "Field type: " << (vector_field ? "Vector (3D)" : "Scalar") << "\n";
      cout << "Amplitude: " << g_amp << "\n";
   }

   // Create serial meshes on all ranks (identical)
   Mesh mesh_1_serial = MakePeriodicMesh(nx, nx, nx, L);  // Source: periodic
   mesh_1_serial.Transform(PerturbMeshTransform);         // Apply perturbation
   
   Mesh mesh_2_serial = Mesh::MakeCartesian3D(nx, nx, nx, Element::HEXAHEDRON, 
                                              L, L, L, false);  // Target: non-periodic

   const int dim = mesh_1_serial.Dimension();
   MFEM_VERIFY(dim == 3, "This code is for 3D meshes");

   EnsureNodes(mesh_1_serial, order);
   EnsureNodes(mesh_2_serial, order);

   if (myid == 0)
   {
      cout << "Source mesh curvature: " << mesh_1_serial.GetNodes()->OwnFEC()->Name() << endl;
      cout << "Target mesh curvature: " << mesh_2_serial.GetNodes()->OwnFEC()->Name() << endl;
   }

   // Create parallel meshes
   if (myid == 0) { cout << "Creating parallel meshes...\n"; }
   
   ParMesh mesh_1(MPI_COMM_WORLD, mesh_1_serial);  // Source
   ParMesh mesh_2(MPI_COMM_WORLD, mesh_2_serial);  // Target
   
   // Clear serial meshes to save memory
   mesh_1_serial.Clear();
   mesh_2_serial.Clear();

   // Setup finite element spaces
   const int vdim = vector_field ? 3 : 1;
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fes_src(&mesh_1, &fec, vdim);  // Source space
   ParFiniteElementSpace fes_tar(&mesh_2, &fec, vdim);  // Target space

   HYPRE_BigInt glob_dofs_src = fes_src.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Global TRUE DoFs: " << glob_dofs_src 
           << " (" << (vector_field ? "3 components" : "scalar") << ")\n";
   }

   // Create source grid function
   ParGridFunction func_source(&fes_src);
   if (vector_field)
   {
      VectorFunctionCoefficient vec_coeff(vdim, vector_func);
      func_source.ProjectCoefficient(vec_coeff);
   }
   else
   {
      FunctionCoefficient scalar_coeff(scalar_func);
      func_source.ProjectCoefficient(scalar_coeff);
   }

   // Get target mesh node coordinates
   ParGridFunction *tar_nodes = dynamic_cast<ParGridFunction*>(mesh_2.GetNodes());
   MFEM_VERIFY(tar_nodes && tar_nodes->VectorDim() == dim, "Expected mesh nodes.");
   
   const int local_ndofs = tar_nodes->FESpace()->GetNDofs();
   
   if (myid == 0)
   {
      cout << "Local DOFs per rank (avg): " << local_ndofs << "\n";
   }

   // Pack coordinates in byNODES format
   Vector vxyz(dim * local_ndofs);
   for (int d = 0; d < dim; ++d)
   {
      const double *comp = tar_nodes->GetData() + d*local_ndofs;
      for (int i = 0; i < local_ndofs; ++i) 
      { 
         vxyz[d*local_ndofs + i] = comp[i]; 
      }
   }

   // Setup FindPointsGSLIB
   if (myid == 0) { cout << "\n=== Using Parallel FindPoints ===\n"; }
   
   FindPointsGSLIB finder(MPI_COMM_WORLD);
   finder.Setup(mesh_1);
   finder.SetDistanceToleranceForPointsFoundOnBoundary(std::max(1e-12 * L, 2.0 * g_amp));
   
   // Find points
   finder.FindPoints(vxyz, Ordering::byNODES);

   // Get status information
   Array<unsigned int> code_out = finder.GetCode();
   Array<unsigned int> task_id_out = finder.GetProc();

   // Count results locally
   int local_inside = 0, local_border = 0, local_miss = 0;
   for (int i = 0; i < local_ndofs; ++i)
   {
      if (code_out[i] == 0u) ++local_inside;
      else if (code_out[i] == 1u) ++local_border;
      else ++local_miss;
   }

   // Global reduction
   int global_inside, global_border, global_miss, global_ndofs;
   MPI_Allreduce(&local_inside, &global_inside, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(&local_border, &global_border, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(&local_miss, &global_miss, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(&local_ndofs, &global_ndofs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

   if (myid == 0)
   {
      cout << "FindPoints results:\n";
      cout << "  Inside elements: " << global_inside << "\n";
      cout << "  On boundaries:   " << global_border << "\n";  
      cout << "  Not found:       " << global_miss << "\n";
      cout << "  Total DOFs:      " << global_ndofs << "\n";
      double success_rate = (global_ndofs > 0) ? 100.0 * (global_inside + global_border) / global_ndofs : 0.0;
      cout << "  Success rate:    " << fixed << setprecision(2) << success_rate << "%\n";
   }

   // Create target grid function
   ParGridFunction func_target(&fes_tar);

   // Interpolate - handle scalar vs vector
   if (vector_field)
   {
      Vector interp_vals(local_ndofs * vdim);
      finder.Interpolate(func_source, interp_vals);
      
      // Handle NaN values
      int local_fallback = 0;
      for (int i = 0; i < local_ndofs; ++i)
      {
         bool has_nan = false;
         for (int c = 0; c < vdim; ++c)
         {
            int idx = c * local_ndofs + i;
            if (std::isnan(interp_vals[idx]) || code_out[i] >= 2)
            {
               has_nan = true;
               break;
            }
         }
         
         if (has_nan)
         {
            Vector pt(dim);
            for (int d = 0; d < dim; ++d) 
            {
               pt[d] = vxyz[d*local_ndofs + i];
            }
            Vector val(vdim);
            vector_func(pt, val);
            for (int c = 0; c < vdim; ++c)
            {
               interp_vals[c * local_ndofs + i] = val[c];
            }
            ++local_fallback;
         }
      }
      
      // Direct assignment
      double *func_target_data = func_target.GetData();
      for (int i = 0; i < local_ndofs * vdim; ++i)
      {
         func_target_data[i] = interp_vals[i];
      }
      
      int global_fallback;
      MPI_Allreduce(&local_fallback, &global_fallback, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      if (myid == 0 && global_fallback > 0)
      {
         cout << "Fallback to exact function: " << global_fallback << " points\n";
      }
   }
   else  // Scalar field
   {
      Vector interp_vals(local_ndofs);
      finder.Interpolate(func_source, interp_vals);
      
      // Handle NaN values
      int local_fallback = 0;
      for (int i = 0; i < local_ndofs; ++i)
      {
         if (std::isnan(interp_vals[i]) || code_out[i] >= 2)
         {
            Vector pt(dim);
            for (int d = 0; d < dim; ++d) 
            {
               pt[d] = vxyz[d*local_ndofs + i];
            }
            interp_vals[i] = scalar_func(pt);
            ++local_fallback;
         }
      }
      
      // Direct assignment
      double *func_target_data = func_target.GetData();
      for (int i = 0; i < local_ndofs; ++i) 
      { 
         func_target_data[i] = interp_vals[i]; 
      }
      
      int global_fallback;
      MPI_Allreduce(&local_fallback, &global_fallback, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      if (myid == 0 && global_fallback > 0)
      {
         cout << "Fallback to exact function: " << global_fallback << " points\n";
      }
   }

   // Compute errors
   if (myid == 0) { cout << "\n=== Computing Errors ===\n"; }
   
   // Create exact solution on target mesh
   ParGridFunction func_exact(&fes_tar);
   
   if (vector_field)
   {
      VectorFunctionCoefficient exact_coeff(vdim, vector_func);
      func_exact.ProjectCoefficient(exact_coeff);
      
      double l2_err_interp = func_target.ComputeL2Error(exact_coeff);
      double l2_err_exact = func_exact.ComputeL2Error(exact_coeff);
      double l2_err_src = func_source.ComputeL2Error(exact_coeff);
      
      // Compute norm
      Vector zero_vec(vdim);
      zero_vec = 0.0;
      VectorConstantCoefficient zero_coeff(zero_vec);
      double l2_norm = func_exact.ComputeL2Error(zero_coeff);
      double rel_err = (l2_norm > 0.0) ? (l2_err_interp / l2_norm) : l2_err_interp;
      
      if (myid == 0)
      {
         cout << scientific << setprecision(12);
         cout << "L2 error of u_interp:  " << l2_err_interp << "\n";
         cout << "L2 error of u_exact:   " << l2_err_exact << " (sanity check)\n";
         cout << "L2 error of u_src:     " << l2_err_src << " (on perturbed mesh)\n";
         cout << "L2 norm of u_exact:    " << l2_norm << "\n";
         cout << "Relative L2 error:     " << rel_err*100.0 << " %\n";
         
         if (l2_err_exact > 1e-10)
         {
            cout << "WARNING: u_exact has non-trivial error vs analytical solution!\n";
         }
      }
   }
   else
   {
      FunctionCoefficient exact_coeff(scalar_func);
      func_exact.ProjectCoefficient(exact_coeff);
      
      double l2_err_interp = func_target.ComputeL2Error(exact_coeff);
      double l2_err_exact = func_exact.ComputeL2Error(exact_coeff);
      double l2_err_src = func_source.ComputeL2Error(exact_coeff);
      
      ConstantCoefficient zero_coeff(0.0);
      double l2_norm = func_exact.ComputeL2Error(zero_coeff);
      double rel_err = (l2_norm > 0.0) ? (l2_err_interp / l2_norm) : l2_err_interp;
      
      if (myid == 0)
      {
         cout << scientific << setprecision(12);
         cout << "L2 error of u_interp:  " << l2_err_interp << "\n";
         cout << "L2 error of u_exact:   " << l2_err_exact << " (sanity check)\n";
         cout << "L2 error of u_src:     " << l2_err_src << " (on perturbed mesh)\n";
         cout << "L2 norm of u_exact:    " << l2_norm << "\n";
         cout << "Relative L2 error:     " << rel_err*100.0 << " %\n";
         
         if (l2_err_exact > 1e-10)
         {
            cout << "WARNING: u_exact has non-trivial error vs analytical solution!\n";
         }
      }
   }

   // Compute pointwise error
   ParGridFunction func_error(&fes_tar);
   func_error = func_exact;
   func_error -= func_target;

   // VisIt output
   if (visit_output)
   {
      if (myid == 0) { cout << "\n=== Creating VisIt Output ===\n"; }
      
      VisItDataCollection src_dc("SourceMesh", &mesh_1);
      src_dc.SetPrecision(8);
      src_dc.RegisterField("u_src", &func_source);
      src_dc.SetCycle(0);
      src_dc.SetTime(0.0);
      src_dc.Save();
      
      VisItDataCollection tar_dc("TargetMesh", &mesh_2);
      tar_dc.SetPrecision(8);
      tar_dc.RegisterField("u_interp", &func_target);
      tar_dc.RegisterField("u_exact", &func_exact);
      tar_dc.RegisterField("u_error", &func_error);
      tar_dc.SetCycle(0);
      tar_dc.SetTime(0.0);
      tar_dc.Save();
      
      if (myid == 0)
      {
         cout << "VisIt output saved to SourceMesh/ and TargetMesh/\n";
      }
   }

   // GLVis visualization
   if (visualization)
   {
      char vishost[] = "localhost";
      socketstream sout;
      sout.open(vishost, visport);
      if (!sout)
      {
         if (myid == 0)
         {
            cout << "Unable to connect to GLVis server at "
                 << vishost << ':' << visport << endl;
         }
      }
      else
      {
         sout << "parallel " << nprocs << " " << myid << "\n";
         sout.precision(8);
         sout << "solution\n" << mesh_2 << func_target
              << "window_title 'Interpolated Solution'"
              << "keys mA\n" << flush;
      }
   }

   if (myid == 0)
   {
      cout << "\nDone!\n";
   }

   // Free the internal gslib data
   finder.FreeData();

   return 0;
}

/*
COMPILATION:
mpicxx -std=c++11 -I$MFEM_DIR -L$MFEM_DIR -o field_interp_parallel \
       field_interp_parallel.cpp -lmfem -lHYPRE -lmetis

USAGE:
# Scalar field (default)
mpirun -np 1 ./field_interp_parallel -amp 0.02
mpirun -np 2 ./field_interp_parallel -amp 0.02  
mpirun -np 4 ./field_interp_parallel -amp 0.02

# Vector field
mpirun -np 1 ./field_interp_parallel -amp 0.02 -vec
mpirun -np 2 ./field_interp_parallel -amp 0.02 -vec
mpirun -np 4 ./field_interp_parallel -amp 0.02 -vec

# Higher resolution
mpirun -np 4 ./field_interp_parallel -nx 16 -o 3 -amp 0.05 -vec
*/