#include "mfem.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <limits>

using namespace mfem;
using namespace std;

// ---------------- test fields ----------------
static double scalar_func(const Vector &x)
{
   return std::sin(M_PI*x[0]) * std::cos(M_PI*x[1]) * std::sin(0.5*M_PI*x[2])
        + x[0]*x[1]*x[2];
}

// Vector field function
static void vector_func(const Vector &x, Vector &v)
{
   v.SetSize(3);
   v[0] = std::sin(M_PI*x[1]) * std::cos(M_PI*x[2]) + x[0]*x[1];
   v[1] = std::cos(M_PI*x[0]) * std::sin(M_PI*x[2]) + x[1]*x[2];
   v[2] = std::sin(M_PI*x[0]) * std::cos(M_PI*x[1]) + x[2]*x[0];
}

// Sine-wave mesh perturbation (vector coefficient in physical coords)
class MeshPerturbationCoefficient : public VectorCoefficient
{
   double amp, freq;
public:
   MeshPerturbationCoefficient(double amplitude, double frequency = 2.0*M_PI)
   : VectorCoefficient(3), amp(amplitude), freq(frequency) {}

   virtual void Eval(Vector &u, ElementTransformation &T, const IntegrationPoint &ip)
   {
      Vector x(3);
      T.Transform(ip, x);
      u.SetSize(3);
      u[0] = amp * std::sin(freq*x[1]) * std::cos(freq*x[2]);
      u[1] = amp * std::cos(freq*x[0]) * std::sin(freq*x[2]) * 0.8;
      u[2] = amp * std::sin(freq*x[0]) * std::cos(freq*x[1]) * 0.6;
   }
};

// Vector function coefficient wrapper
class ExactVectorCoefficient : public VectorCoefficient
{
public:
   ExactVectorCoefficient() : VectorCoefficient(3) {}
   
   virtual void Eval(Vector &v, ElementTransformation &T, const IntegrationPoint &ip)
   {
      Vector x(3);
      T.Transform(ip, x);
      vector_func(x, v);
   }
};

// ---------------- helpers ----------------
static void EnsureNodes(Mesh &mesh, int order)
{
   if (!mesh.GetNodes())
   {
      mesh.SetCurvature(order, false, mesh.Dimension(), Ordering::byNODES);
   }
}

static double MaxVectorMagnitudeAtNodes(const GridFunction &v)
{
   const int vdim = v.VectorDim();
   MFEM_VERIFY(vdim == 3, "Expect 3D displacement.");
   const int ndofs = v.FESpace()->GetNDofs();
   const double *data = v.Read();
   double max_mag = 0.0;
   for (int i = 0; i < ndofs; ++i)
   {
      const double x = data[0*ndofs + i];
      const double y = data[1*ndofs + i];
      const double z = data[2*ndofs + i];
      const double m = std::sqrt(x*x + y*y + z*z);
      if (m > max_mag) max_mag = m;
   }
   return max_mag;
}

template <typename GF>
static void ComputeError(const GF &a, const GF &b, GF &err)
{
   MFEM_ASSERT(a.Size() == b.Size() && a.Size() == err.Size(),
               "GridFunction size mismatch.");
   err = a; err -= b;
}

// ---------------- main ----------------
int main(int argc, char *argv[])
{
   // Initialize MPI
   Mpi::Init(argc, argv);
   int myid = Mpi::WorldRank();
   int nprocs = Mpi::WorldSize();
   Hypre::Init();

   int nx = 8, ny = 8, nz = 8;
   int order = 2;
   double amp = 0.05;
   double L = 1.0;
   bool visualization = false, visit_output = true;
   int visport = 19916;
   bool vector_field = false;  // New option for vector field

   OptionsParser args(argc, argv);
   args.AddOption(&nx, "-nx", "--nx", "Elements in x.");
   args.AddOption(&ny, "-ny", "--ny", "Elements in y.");
   args.AddOption(&nz, "-nz", "--nz", "Elements in z.");
   args.AddOption(&order, "-o", "--order", "H1 order (scalar/vector).");
   args.AddOption(&amp, "-amp", "--amplitude", "Perturbation amplitude (absolute).");
   args.AddOption(&L, "-L", "--domain-size", "Cube side length.");
   args.AddOption(&vector_field, "-vec", "--vector-field", "-no-vec", "--no-vector-field",
                  "Use vector field instead of scalar field.");
   args.AddOption(&visualization, "-vis", "--visualization",
                               "-no-vis", "--no-visualization", "GLVis on/off.");
   args.AddOption(&visit_output, "-visit", "--visit-output",
                               "-no-visit", "--no-visit-output", "VisIt dump on/off.");
   args.AddOption(&visport, "-p", "--send-port", "GLVis port.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(std::cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(std::cout); }

#ifndef MFEM_USE_GSLIB
   if (myid == 0)
   {
      std::cerr << "ERROR: Requires MFEM built with GSLIB (MFEM_USE_GSLIB=YES).\n";
   }
   return 2;
#endif

   if (myid == 0)
   {
      std::cout << "\n=== Parallel FindPoints Interpolation ===\n";
      std::cout << "MPI ranks: " << nprocs << "\n";
      std::cout << "Grid: " << nx << " x " << ny << " x " << nz << "\n";
      std::cout << "Order: " << order << "\n";
      std::cout << "Field type: " << (vector_field ? "Vector (3D)" : "Scalar") << "\n";
      std::cout << "Amplitude: " << amp << "\n";
   }

   // Create serial meshes on all ranks (identical)
   Mesh clean_smesh = Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON, L, L, L);
   Mesh pert_smesh(clean_smesh);  // Copy for perturbation
   
   EnsureNodes(clean_smesh, order);
   EnsureNodes(pert_smesh, order);

   // Apply perturbation to SERIAL perturbed mesh
   if (myid == 0) { std::cout << "\n=== Applying Perturbation to Serial Mesh ===\n"; }
   
   H1_FECollection serial_disp_fec(order, 3);
   FiniteElementSpace serial_disp_fes(&pert_smesh, &serial_disp_fec, /*vdim=*/3);
   GridFunction serial_displacement(&serial_disp_fes);
   
   MeshPerturbationCoefficient disp_coeff(amp);
   serial_displacement.ProjectCoefficient(disp_coeff);
   
   // Apply displacement to serial mesh nodes
   GridFunction *serial_nodes = pert_smesh.GetNodes();
   MFEM_VERIFY(serial_nodes && serial_nodes->VectorDim() == 3, "Expected 3D serial nodes.");
   *serial_nodes += serial_displacement;
   
   // Compute max displacement on serial mesh
   double max_disp_serial = MaxVectorMagnitudeAtNodes(serial_displacement);
   if (myid == 0)
   {
      std::cout << "Max node displacement (serial): " << std::setprecision(12)
                << max_disp_serial << "\n";
   }

   // Partition both meshes
   if (myid == 0) { std::cout << "Creating parallel meshes...\n"; }
   
   ParMesh clean_mesh(MPI_COMM_WORLD, clean_smesh);
   ParMesh perturbed_mesh(MPI_COMM_WORLD, pert_smesh);
   
   // Clear serial meshes to save memory
   clean_smesh.Clear();
   pert_smesh.Clear();

   // Create finite element spaces on parallel meshes
   const int vdim = vector_field ? 3 : 1;
   H1_FECollection fec(order, 3);
   ParFiniteElementSpace fes_src(&perturbed_mesh, &fec, vdim); // source field space
   ParFiniteElementSpace fes_dst(&clean_mesh, &fec, vdim);     // destination (clean)

   HYPRE_BigInt glob_dofs_src = fes_src.GlobalTrueVSize();
   if (myid == 0)
   {
      std::cout << "Global TRUE DoFs: " << glob_dofs_src 
                << " (" << (vector_field ? "3 components" : "scalar") << ")\n";
   }

   // Define field on perturbed mesh, and exact on clean mesh
   ParGridFunction u_src(&fes_src);
   ParGridFunction u_exact(&fes_dst);
   
   if (vector_field)
   {
      ExactVectorCoefficient vec_coeff;
      u_src.ProjectCoefficient(vec_coeff);
      u_exact.ProjectCoefficient(vec_coeff);
   }
   else
   {
      FunctionCoefficient f(scalar_func);
      u_src.ProjectCoefficient(f);
      u_exact.ProjectCoefficient(f);
   }

   // Use the pfindpts approach
   if (myid == 0)
   {
      std::cout << "\n=== Using Parallel FindPoints ===\n";
   }

   ParGridFunction u_interp(&fes_dst);

   // Get ALL the clean mesh node coordinates on each rank
   ParGridFunction *clean_nodes = dynamic_cast<ParGridFunction*>(clean_mesh.GetNodes());
   MFEM_VERIFY(clean_nodes, "Clean par-mesh must have nodes.");
   
   const int dim = 3;
   const int local_ndofs = clean_nodes->FESpace()->GetNDofs(); // Local DOFs (including shared)
   
   if (myid == 0)
   {
      std::cout << "Local DOFs per rank (avg): " << local_ndofs << "\n";
   }

   // Pack coordinates in byNODES format
   Vector vxyz(dim * local_ndofs);
   for (int d = 0; d < dim; ++d)
   {
      const double *comp = clean_nodes->GetData() + d*local_ndofs;
      for (int i = 0; i < local_ndofs; ++i) 
      { 
         vxyz[d*local_ndofs + i] = comp[i]; 
      }
   }

   // Create FindPointsGSLIB
   FindPointsGSLIB finder(MPI_COMM_WORLD);
   finder.Setup(perturbed_mesh);
   finder.SetDistanceToleranceForPointsFoundOnBoundary(std::max(1e-12 * L, 2.0 * amp));
   
   // Use pfindpts-style FindPoints call
   finder.FindPoints(vxyz, Ordering::byNODES);

   // Get status information
   Array<unsigned int> code_out = finder.GetCode();
   Array<unsigned int> task_id_out = finder.GetProc();
   Vector dist_p_out = finder.GetDist();

   // Count results
   int local_inside = 0, local_border = 0, local_miss = 0;
   for (int i = 0; i < local_ndofs; ++i)
   {
      if (code_out[i] == 0u) ++local_inside;
      else if (code_out[i] == 1u) ++local_border;
      else ++local_miss;
   }

   int global_inside, global_border, global_miss, global_ndofs;
   MPI_Allreduce(&local_inside, &global_inside, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(&local_border, &global_border, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(&local_miss, &global_miss, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(&local_ndofs, &global_ndofs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

   if (myid == 0)
   {
      std::cout << "FindPoints results:\n";
      std::cout << "  Inside elements: " << global_inside << "\n";
      std::cout << "  On boundaries:   " << global_border << "\n";  
      std::cout << "  Not found:       " << global_miss << "\n";
      std::cout << "  Total DOFs:      " << global_ndofs << "\n";
      double success_rate = (global_ndofs > 0) ? 100.0 * (global_inside + global_border) / global_ndofs : 0.0;
      std::cout << "  Success rate:    " << std::fixed << std::setprecision(2) << success_rate << "%\n";
   }

   // Interpolate - handle scalar vs vector
   if (vector_field)
   {
   // For vector fields, interpolate all components at once
   // The finder expects points, we give it the mesh nodes
   // But we need to interpolate the field which might have different DOFs
   
   // We need to interpolate at the actual field DOFs, not mesh nodes
   // For H1, the DOFs are at the same locations as mesh nodes if orders match
   // But if field order != mesh order, we have a problem
   
   // Simple approach: interpolate all components together
   Vector interp_vals(local_ndofs * vdim);
   finder.Interpolate(u_src, interp_vals);
   
   // Handle NaN values
   int local_fallback = 0;
   for (int i = 0; i < local_ndofs; ++i)
   {
      bool has_nan = false;
      for (int c = 0; c < vdim; ++c)
      {
         // Check byNODES ordering: [x0,x1,...,xN,y0,y1,...,yN,z0,z1,...,zN]
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
   
   // Copy to u_interp - need to be careful about ordering
   double *u_interp_data = u_interp.GetData();
   for (int i = 0; i < local_ndofs * vdim; ++i)
   {
      u_interp_data[i] = interp_vals[i];
   }
   
   int global_fallback;
   MPI_Allreduce(&local_fallback, &global_fallback, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   if (myid == 0 && global_fallback > 0)
   {
      std::cout << "Fallback to exact function: " << global_fallback << " points\n";
   }


   }
   else
   {
      // Scalar field - original code
      Vector interp_vals(local_ndofs);
      finder.Interpolate(u_src, interp_vals);
      
      // Handle any NaN values with exact function evaluation
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
      
      // Set the interpolated values in the parallel GridFunction
      double *u_interp_data = u_interp.GetData();
      for (int i = 0; i < local_ndofs; ++i) 
      { 
         u_interp_data[i] = interp_vals[i]; 
      }
      
      int global_fallback;
      MPI_Allreduce(&local_fallback, &global_fallback, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      if (myid == 0 && global_fallback > 0)
      {
         std::cout << "Fallback to exact function: " << global_fallback << " points\n";
      }
   }

   // You actually don't need this
   // // Critical for rank-invariant results
   // u_interp.ExchangeFaceNbrData();
   // u_interp.ParallelAverage();

   // Compute errors
   if (myid == 0) { std::cout << "Computing errors using ComputeL2Error...\n"; }

   // For visualization, compute pointwise error (used by both scalar and vector)
   ParGridFunction u_err(&fes_dst);
   ComputeError(u_exact, u_interp, u_err);

   if (vector_field)
   {
      // Vector field error computation
      ExactVectorCoefficient exact_vec_coeff;
      
      // Create zero vector coefficient properly
      Vector zero_vec(vdim);
      zero_vec = 0.0;
      VectorConstantCoefficient zero_vec_coeff(zero_vec);
      
      double l2_err_interp = u_interp.ComputeL2Error(exact_vec_coeff);
      double l2_err_exact = u_exact.ComputeL2Error(exact_vec_coeff);
      double l2_err_src = u_src.ComputeL2Error(exact_vec_coeff);
      
      // Compute norm using ComputeL2Error with zero coefficient
      double l2_norm_exact = u_exact.ComputeL2Error(zero_vec_coeff);
      double rel_err = (l2_norm_exact > 0.0) ? (l2_err_interp / l2_norm_exact) : l2_err_interp;
      
      if (myid == 0)
      {
         std::cout << "\n=== L2 Error Analysis (Vector Field) ===\n";
         std::cout << std::setprecision(12) << std::scientific;
         std::cout << "L2 error of u_interp  : " << l2_err_interp << "\n";
         std::cout << "L2 error of u_exact   : " << l2_err_exact << " (sanity check)\n";
         std::cout << "L2 error of u_src     : " << l2_err_src << " (on perturbed mesh)\n";
         std::cout << "L2 norm of u_exact    : " << l2_norm_exact << " (using ||u-0||)\n";
         std::cout << "Relative L2 error     : " << rel_err*100.0 << " %\n";
         
         if (l2_err_exact > 1e-10)
         {
            std::cout << "WARNING: u_exact has non-trivial error vs analytical solution!\n";
         }
      }
   }
   else
   {
      // Scalar field error computation
      FunctionCoefficient exact_coeff(scalar_func);
      ConstantCoefficient zero_coeff(0.0);
      
      double l2_err_interp = u_interp.ComputeL2Error(exact_coeff);
      double l2_err_exact = u_exact.ComputeL2Error(exact_coeff);
      double l2_err_src = u_src.ComputeL2Error(exact_coeff);
      
      double l2_norm_exact = u_exact.ComputeL2Error(zero_coeff);
      double rel_err = (l2_norm_exact > 0.0) ? (l2_err_interp / l2_norm_exact) : l2_err_interp;
      
      if (myid == 0)
      {
         std::cout << "\n=== L2 Error Analysis (Scalar Field) ===\n";
         std::cout << std::setprecision(12) << std::scientific;
         std::cout << "L2 error of u_interp  : " << l2_err_interp << "\n";
         std::cout << "L2 error of u_exact   : " << l2_err_exact << " (sanity check)\n";
         std::cout << "L2 error of u_src     : " << l2_err_src << " (on perturbed mesh)\n";
         std::cout << "L2 norm of u_exact    : " << l2_norm_exact << " (using ||u-0||)\n";
         std::cout << "Relative L2 error     : " << rel_err*100.0 << " %\n";
         
         if (l2_err_exact > 1e-10)
         {
            std::cout << "WARNING: u_exact has non-trivial error vs analytical solution!\n";
         }
      }
   }
   
   // VisIt output - same directory names for both scalar and vector
   if (visit_output)
   {
      if (myid == 0) { std::cout << "\n=== Creating VisIt Output ===\n"; }
      
      VisItDataCollection pert_dc("PerturbedMesh_pfindpts", &perturbed_mesh);
      pert_dc.SetPrecision(8);
      pert_dc.RegisterField("u_src", &u_src);
      pert_dc.SetCycle(0); pert_dc.SetTime(0.0);
      pert_dc.Save();
      
      VisItDataCollection clean_dc("CleanMesh_pfindpts", &clean_mesh);
      clean_dc.SetPrecision(8);
      clean_dc.RegisterField("u_interp", &u_interp);
      clean_dc.RegisterField("u_exact",  &u_exact);
      clean_dc.RegisterField("u_error",  &u_err);
      clean_dc.SetCycle(0); clean_dc.SetTime(0.0);
      clean_dc.Save();
   }
   
   return 0;
}

/*
COMPILATION:
mpicxx -std=c++17 -I$MFEM_DIR -L$MFEM_DIR -o field_interp \
       field_interp.cpp -lmfem -lHYPRE -lmetis

USAGE:
# Scalar field (default)
mpirun -np 1 ./field_interp -amp 0.02
mpirun -np 2 ./field_interp -amp 0.02  
mpirun -np 4 ./field_interp -amp 0.02

# Vector field
mpirun -np 1 ./field_interp -amp 0.02 -vec
mpirun -np 2 ./field_interp -amp 0.02 -vec
mpirun -np 4 ./field_interp -amp 0.02 -vec
*/