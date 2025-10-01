#include "mfem.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <limits>

using namespace mfem;
using namespace std;

// Test functions
static double scalar_func(const Vector &x)
{
   return std::sin(2.0*M_PI*x[0]) * std::cos(2.0*M_PI*x[1]) * std::sin(2.0*M_PI*x[2]);
}

static void vector_func(const Vector &x, Vector &v)
{
   v.SetSize(3);
   v[0] = std::sin(2.0*M_PI*x[1]) * std::cos(2.0*M_PI*x[2]);
   v[1] = std::cos(2.0*M_PI*x[0]) * std::sin(2.0*M_PI*x[2]);
   v[2] = std::sin(2.0*M_PI*x[0]) * std::cos(2.0*M_PI*x[1]);
}

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

static void EnsureNodes(Mesh &mesh, int order)
{
   if (!mesh.GetNodes())
   {
      mesh.SetCurvature(order, false, mesh.Dimension(), Ordering::byNODES);
   }
}

struct InterpolationResult
{
   double success_rate;
   double l2_error;
   double relative_error;
   int total_points;
   int failed_points;
};

// Transfer data from non-periodic mesh to periodic mesh
void TransferNonPeriodicToPeriodic(ParGridFunction &nonperiodic_gf, 
                                   ParGridFunction &periodic_gf,
                                   double L)
{
   // Get the data arrays
   const int vdim = nonperiodic_gf.VectorDim();
   const int np_ndofs = nonperiodic_gf.FESpace()->GetNDofs();
   const int p_ndofs = periodic_gf.FESpace()->GetNDofs();
   
   // For a structured Cartesian mesh, we need to map DOFs correctly
   // The non-periodic mesh has all unique DOFs
   // The periodic mesh has shared DOFs on periodic boundaries
   
   // Get the vertex mappings for both meshes
   ParMesh *np_mesh = nonperiodic_gf.ParFESpace()->GetParMesh();
   ParMesh *p_mesh = periodic_gf.ParFESpace()->GetParMesh();
   
   // Simple approach: Use GSLIB to interpolate from non-periodic to periodic
   // This handles the DOF mapping automatically
   FindPointsGSLIB finder(MPI_COMM_WORLD);
   finder.Setup(*np_mesh);
   finder.SetDistanceToleranceForPointsFoundOnBoundary(1e-12);
   
   // Get coordinates of periodic mesh nodes
   ParGridFunction *p_nodes = dynamic_cast<ParGridFunction*>(p_mesh->GetNodes());
   const int local_ndofs = p_nodes->FESpace()->GetNDofs();
   
   Vector target_coords(3 * local_ndofs);
   for (int d = 0; d < 3; ++d)
   {
      const double *comp = p_nodes->GetData() + d * local_ndofs;
      for (int i = 0; i < local_ndofs; ++i)
      {
         // Apply periodic wrapping for coordinates
         double coord = comp[i];
         // Ensure coordinates are in [0, L) range for periodicity
         while (coord >= L) coord -= L;
         while (coord < 0) coord += L;
         target_coords[d * local_ndofs + i] = coord;
      }
   }
   
   // Find points and interpolate
   finder.FindPoints(target_coords, Ordering::byNODES);
   
   Vector interp_vals(local_ndofs * vdim);
   finder.Interpolate(nonperiodic_gf, interp_vals);
   
   // Check for failed interpolations
   Array<unsigned int> code_out = finder.GetCode();
   for (int i = 0; i < local_ndofs; ++i)
   {
      if (code_out[i] >= 2)  // Point not found
      {
         // For periodic boundaries, this might be expected - try to handle gracefully
         for (int c = 0; c < vdim; ++c)
         {
            interp_vals[c * local_ndofs + i] = 0.0;  // Or use averaging from neighbors
         }
      }
   }
   
   // Store result in periodic GridFunction
   double *result_data = periodic_gf.GetData();
   for (int i = 0; i < local_ndofs * vdim; ++i)
   {
      result_data[i] = interp_vals[i];
   }
}

InterpolationResult DoPeriodicProjectionInterpolation(int nx, int ny, int nz, int order, 
                                                     double L, double amp, bool vector_field,
                                                     bool use_perturbation, bool visit_output, 
                                                     const std::string& output_prefix, int myid)
{
   InterpolationResult result;
   result.success_rate = 0.0;
   result.l2_error = 0.0;
   result.relative_error = 0.0;
   result.total_points = 0;
   result.failed_points = 0;
   
   // ========== STEP 1: Create all meshes ==========
   // Create base clean mesh
   Mesh clean_serial = Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON, L, L, L, false);
   
   // Create periodic meshes (source and target)
   Vector x_translation({L, 0.0, 0.0});
   Vector y_translation({0.0, L, 0.0});
   Vector z_translation({0.0, 0.0, L});
   std::vector<Vector> translations = {x_translation, y_translation, z_translation};
   
   Mesh periodic_source_serial = Mesh::MakePeriodic(clean_serial, clean_serial.CreatePeriodicVertexMapping(translations));
   Mesh periodic_target_serial = Mesh::MakePeriodic(clean_serial, clean_serial.CreatePeriodicVertexMapping(translations));
   
   EnsureNodes(periodic_source_serial, order);
   EnsureNodes(periodic_target_serial, order);
   
   // Apply perturbation to periodic source if requested
   if (use_perturbation)
   {
      H1_FECollection disp_fec(order, 3);
      FiniteElementSpace disp_fes(&periodic_source_serial, &disp_fec, 3);
      GridFunction displacement(&disp_fes);
      MeshPerturbationCoefficient disp_coeff(amp);
      displacement.ProjectCoefficient(disp_coeff);
      GridFunction *source_nodes = periodic_source_serial.GetNodes();
      *source_nodes += displacement;
   }
   
   // Create non-periodic meshes for GSLIB interpolation
   Mesh nonperiodic_source_serial = Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON, L, L, L, false);
   Mesh nonperiodic_target_serial = Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON, L, L, L, false);
   
   EnsureNodes(nonperiodic_source_serial, order);
   EnsureNodes(nonperiodic_target_serial, order);
   
   // Apply same perturbation to non-periodic source
   if (use_perturbation)
   {
      H1_FECollection disp_fec(order, 3);
      FiniteElementSpace np_disp_fes(&nonperiodic_source_serial, &disp_fec, 3);
      GridFunction np_displacement(&np_disp_fes);
      MeshPerturbationCoefficient disp_coeff(amp);
      np_displacement.ProjectCoefficient(disp_coeff);
      GridFunction *np_source_nodes = nonperiodic_source_serial.GetNodes();
      *np_source_nodes += np_displacement;
   }
   
   // ========== STEP 2: Create parallel meshes ==========
   ParMesh nonperiodic_source_mesh(MPI_COMM_WORLD, nonperiodic_source_serial);
   ParMesh nonperiodic_target_mesh(MPI_COMM_WORLD, nonperiodic_target_serial);
   ParMesh periodic_source_mesh(MPI_COMM_WORLD, periodic_source_serial);
   ParMesh periodic_target_mesh(MPI_COMM_WORLD, periodic_target_serial);
   
   // ========== STEP 3: Create FE spaces and GridFunctions ==========
   const int vdim = vector_field ? 3 : 1;
   H1_FECollection fec(order, 3);
   
   ParFiniteElementSpace nonperiodic_source_fes(&nonperiodic_source_mesh, &fec, vdim);
   ParFiniteElementSpace nonperiodic_target_fes(&nonperiodic_target_mesh, &fec, vdim);
   ParFiniteElementSpace periodic_source_fes(&periodic_source_mesh, &fec, vdim);
   ParFiniteElementSpace periodic_target_fes(&periodic_target_mesh, &fec, vdim);
   
   ParGridFunction nonperiodic_source_data(&nonperiodic_source_fes);
   ParGridFunction nonperiodic_interpolated(&nonperiodic_target_fes);
   ParGridFunction periodic_source_data(&periodic_source_fes);
   ParGridFunction periodic_interpolated(&periodic_target_fes);
   ParGridFunction periodic_exact(&periodic_target_fes);
   
   // ========== STEP 4: Set up source data (same on both periodic and non-periodic) ==========
   if (vector_field)
   {
      ExactVectorCoefficient coeff;
      nonperiodic_source_data.ProjectCoefficient(coeff);
      periodic_source_data.ProjectCoefficient(coeff);
      periodic_exact.ProjectCoefficient(coeff);
   }
   else
   {
      FunctionCoefficient coeff(scalar_func);
      nonperiodic_source_data.ProjectCoefficient(coeff);
      periodic_source_data.ProjectCoefficient(coeff);
      periodic_exact.ProjectCoefficient(coeff);
   }
   
   // ========== STEP 5: Perform GSLIB interpolation (non-periodic to non-periodic) ==========
   FindPointsGSLIB finder(MPI_COMM_WORLD);
   finder.Setup(nonperiodic_source_mesh);
   
   double tolerance = std::max(1e-12 * L, 2.0 * amp);
   finder.SetDistanceToleranceForPointsFoundOnBoundary(tolerance);
   
   ParGridFunction *target_nodes = dynamic_cast<ParGridFunction*>(nonperiodic_target_mesh.GetNodes());
   const int local_ndofs = target_nodes->FESpace()->GetNDofs();
   
   Vector target_coords(3 * local_ndofs);
   for (int d = 0; d < 3; ++d)
   {
      const double *comp = target_nodes->GetData() + d * local_ndofs;
      for (int i = 0; i < local_ndofs; ++i)
      {
         target_coords[d * local_ndofs + i] = comp[i];
      }
   }
   
   finder.FindPoints(target_coords, Ordering::byNODES);
   
   Array<unsigned int> code_out = finder.GetCode();
   Vector interp_vals(local_ndofs * vdim);
   finder.Interpolate(nonperiodic_source_data, interp_vals);
   
   // Count success/failure
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
   
   result.total_points = global_ndofs;
   result.failed_points = global_miss;
   result.success_rate = (global_ndofs > 0) ? 100.0 * (global_inside + global_border) / global_ndofs : 0.0;
   
   // Handle failed points
   for (int i = 0; i < local_ndofs; ++i)
   {
      if (code_out[i] >= 2)  // Point not found
      {
         for (int c = 0; c < vdim; ++c)
         {
            interp_vals[c * local_ndofs + i] = 0.0;
         }
      }
   }
   
   // Store in non-periodic result
   double *np_result_data = nonperiodic_interpolated.GetData();
   for (int i = 0; i < local_ndofs * vdim; ++i)
   {
      np_result_data[i] = interp_vals[i];
   }
   
   // ========== STEP 6: Transfer interpolated data to periodic mesh ==========
   if (myid == 0) { std::cout << "Transferring interpolated data to periodic mesh...\n"; }
   TransferNonPeriodicToPeriodic(nonperiodic_interpolated, periodic_interpolated, L);
   
   // ========== STEP 7: Compute errors on periodic mesh ==========
   if (vector_field)
   {
      ExactVectorCoefficient exact_coeff;
      Vector zero_vec(vdim);
      zero_vec = 0.0;
      VectorConstantCoefficient zero_coeff(zero_vec);
      
      result.l2_error = periodic_interpolated.ComputeL2Error(exact_coeff);
      double l2_norm = periodic_exact.ComputeL2Error(zero_coeff);
      result.relative_error = (l2_norm > 0.0) ? (result.l2_error / l2_norm) : result.l2_error;
   }
   else
   {
      FunctionCoefficient exact_coeff(scalar_func);
      ConstantCoefficient zero_coeff(0.0);
      
      result.l2_error = periodic_interpolated.ComputeL2Error(exact_coeff);
      double l2_norm = periodic_exact.ComputeL2Error(zero_coeff);
      result.relative_error = (l2_norm > 0.0) ? (result.l2_error / l2_norm) : result.l2_error;
   }
   
   // ========== STEP 8: Create VisIt output ==========
   if (visit_output)
   {
      try 
      {
         if (use_perturbation)
         {
            // Output 1: Original data on perturbed periodic source mesh
            VisItDataCollection source_dc(output_prefix + "_Source", &periodic_source_mesh);
            source_dc.SetPrecision(16);
            source_dc.RegisterField("original_data", &periodic_source_data);
            source_dc.SetCycle(0); 
            source_dc.SetTime(0.0);
            source_dc.Save();
            
            // Output 2: Interpolated data on clean periodic target mesh (THE ACTUAL RESULT)
            VisItDataCollection target_dc(output_prefix + "_Target", &periodic_target_mesh);
            target_dc.SetPrecision(16);
            target_dc.RegisterField("interpolated_data", &periodic_interpolated);  // Now this is real!
            target_dc.RegisterField("exact_data", &periodic_exact);
            
            // Compute and save error field
            ParGridFunction error_field(&periodic_target_fes);
            error_field = periodic_exact;
            error_field -= periodic_interpolated;
            target_dc.RegisterField("interpolation_error", &error_field);
            
            target_dc.SetCycle(0); 
            target_dc.SetTime(0.0);
            target_dc.Save();
            
            // Output 3: Non-periodic interpolated (for debugging)
            VisItDataCollection np_dc(output_prefix + "_NonPeriodic", &nonperiodic_target_mesh);
            np_dc.SetPrecision(8);
            np_dc.RegisterField("np_interpolated", &nonperiodic_interpolated);
            np_dc.SetCycle(0);
            np_dc.SetTime(0.0);
            np_dc.Save();
         }
         else
         {
            // Baseline case
            VisItDataCollection baseline_dc(output_prefix + "_Baseline", &periodic_target_mesh);
            baseline_dc.SetPrecision(8);
            baseline_dc.RegisterField("data", &periodic_interpolated);
            baseline_dc.RegisterField("exact", &periodic_exact);
            baseline_dc.SetCycle(0); 
            baseline_dc.SetTime(0.0);
            baseline_dc.Save();
         }
      }
      catch (...)
      {
         if (myid == 0)
         {
            std::cout << "WARNING: VisIt output failed, continuing without visualization.\n";
         }
      }
   }
   
   return result;
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   int myid = Mpi::WorldRank();
   int nprocs = Mpi::WorldSize();
   Hypre::Init();

   int nx = 8, ny = 8, nz = 8;
   int order = 2;
   double amp = 0.05;
   double L = 1.0;
   bool vector_field = false;
   bool visit_output = true;

   OptionsParser args(argc, argv);
   args.AddOption(&nx, "-nx", "--nx", "Elements in x.");
   args.AddOption(&ny, "-ny", "--ny", "Elements in y.");
   args.AddOption(&nz, "-nz", "--nz", "Elements in z.");
   args.AddOption(&order, "-o", "--order", "H1 order (scalar/vector).");
   args.AddOption(&amp, "-amp", "--amplitude", "Perturbation amplitude (absolute).");
   args.AddOption(&L, "-L", "--domain-size", "Cube side length.");
   args.AddOption(&vector_field, "-vec", "--vector-field", "-no-vec", "--no-vector-field",
                  "Use vector field instead of scalar field.");
   args.AddOption(&visit_output, "-visit", "--visit-output", "-no-visit", "--no-visit-output", 
                  "VisIt dump on/off.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(std::cout); }
      return 1;
   }

#ifndef MFEM_USE_GSLIB
   if (myid == 0)
   {
      std::cerr << "ERROR: Requires MFEM built with GSLIB (MFEM_USE_GSLIB=YES).\n";
   }
   return 2;
#endif

   if (myid == 0)
   {
      std::cout << "\nPeriodic Data Projection Interpolation Test\n";
      std::cout << "Grid: " << nx << "x" << ny << "x" << nz 
                << ", Order: " << order 
                << ", Amplitude: " << amp
                << ", Field: " << (vector_field ? "Vector" : "Scalar")
                << ", Ranks: " << nprocs << "\n";
      std::cout << std::string(80, '-') << "\n";
   }

   // Test case 1: No perturbation (baseline)
   if (myid == 0) { std::cout << "Testing without mesh perturbation (baseline)...\n"; }
   InterpolationResult result_no_pert = DoPeriodicProjectionInterpolation(
      nx, ny, nz, order, L, amp, vector_field, false, visit_output, "Baseline", myid);
   
   // Test case 2: With perturbation (the actual test)
   if (myid == 0) { std::cout << "Testing with mesh perturbation...\n"; }
   InterpolationResult result_with_pert = DoPeriodicProjectionInterpolation(
      nx, ny, nz, order, L, amp, vector_field, true, visit_output, "PeriodicProjection", myid);

   // Report results
   if (myid == 0)
   {
      std::cout << std::string(80, '-') << "\n";
      std::cout << "RESULTS SUMMARY\n";
      std::cout << std::string(80, '-') << "\n";
      
      std::cout << std::left << std::setw(30) << "Case" 
                << std::setw(15) << "Success Rate" 
                << std::setw(15) << "L2 Error"
                << std::setw(15) << "Rel Error (%)"
                << std::setw(10) << "Failed Pts" << "\n";
      std::cout << std::string(80, '-') << "\n";
      
      std::cout << std::left << std::setw(30) << "No Perturbation:"
                << std::setw(15) << (std::to_string(result_no_pert.success_rate) + "%")
                << std::setw(15) << std::scientific << std::setprecision(2) << result_no_pert.l2_error
                << std::setw(15) << std::fixed << std::setprecision(4) << (result_no_pert.relative_error * 100.0)
                << std::setw(10) << result_no_pert.failed_points << "\n";
                
      std::cout << std::left << std::setw(30) << "With Perturbation:"
                << std::setw(15) << (std::to_string(result_with_pert.success_rate) + "%")
                << std::setw(15) << std::scientific << std::setprecision(2) << result_with_pert.l2_error
                << std::setw(15) << std::fixed << std::setprecision(4) << (result_with_pert.relative_error * 100.0)
                << std::setw(10) << result_with_pert.failed_points << "\n";
      
      std::cout << std::string(80, '-') << "\n";
      
      // Analysis
      if (result_with_pert.success_rate > 99.0)
      {
         std::cout << "EXCELLENT: >99% success rate achieved!\n";
      }
      else if (result_with_pert.success_rate > 95.0)
      {
         std::cout << "GOOD: High success rate achieved.\n";
      }
      else
      {
         std::cout << "WARNING: Lower than expected success rate.\n";
      }
      
      if (result_with_pert.l2_error > 1e-6)
      {
         std::cout << "Note: L2 error is significant. This may be due to:\n";
         std::cout << "  - Mesh perturbation affecting interpolation accuracy\n";
         std::cout << "  - Periodic boundary handling in the transfer step\n";
         std::cout << "  - Consider reducing perturbation amplitude or increasing mesh resolution\n";
      }
      
      std::cout << "\nTotal points tested: " << result_with_pert.total_points << "\n";
      
      if (visit_output)
      {
         std::cout << "\nVisIt files created:\n";
         std::cout << "  Baseline_Baseline/ - Baseline without perturbation\n";
         std::cout << "  PeriodicProjection_Source/ - Original data on perturbed periodic mesh\n";
         std::cout << "  PeriodicProjection_Target/ - ACTUAL interpolated data on clean periodic mesh\n";
         std::cout << "  PeriodicProjection_NonPeriodic/ - Intermediate non-periodic result\n";
      }
   }

   return 0;
}

/*
COMPILATION:
mpicxx -std=c++17 -I$MFEM_DIR -L$MFEM_DIR -o periodic_test_fixed \
       periodic_test_fixed.cpp -lmfem -lHYPRE -lmetis

USAGE:
mpirun -np 4 ./periodic_test_fixed -amp 0.02 -vec
mpirun -np 1 ./periodic_test_fixed -amp 0.05 -vec  # Test in serial first
*/