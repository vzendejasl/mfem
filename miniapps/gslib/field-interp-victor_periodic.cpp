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
   
   // ========== STEP 1: Create periodic meshes ==========
   Mesh clean_serial = Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON, L, L, L, false);
   
   Vector x_translation({L, 0.0, 0.0});
   Vector y_translation({0.0, L, 0.0});
   Vector z_translation({0.0, 0.0, L});
   std::vector<Vector> translations = {x_translation, y_translation, z_translation};
   
   Mesh periodic_source_serial = Mesh::MakePeriodic(clean_serial, clean_serial.CreatePeriodicVertexMapping(translations));
   Mesh periodic_target_serial = Mesh::MakePeriodic(clean_serial, clean_serial.CreatePeriodicVertexMapping(translations));
   
   EnsureNodes(periodic_source_serial, order);
   EnsureNodes(periodic_target_serial, order);
   
   // ========== STEP 2: Apply perturbation to source mesh only ==========
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
   
   // ========== STEP 3: Create parallel meshes ==========
   ParMesh periodic_source_mesh(MPI_COMM_WORLD, periodic_source_serial);
   ParMesh periodic_target_mesh(MPI_COMM_WORLD, periodic_target_serial);
   
   // ========== STEP 4: Create FE spaces and GridFunctions ==========
   const int vdim = vector_field ? 3 : 1;
   H1_FECollection fec(order, 3);
   
   ParFiniteElementSpace periodic_source_fes(&periodic_source_mesh, &fec, vdim);
   ParFiniteElementSpace periodic_target_fes(&periodic_target_mesh, &fec, vdim);
   
   ParGridFunction periodic_source_data(&periodic_source_fes);
   ParGridFunction periodic_interpolated(&periodic_target_fes);
   ParGridFunction periodic_exact(&periodic_target_fes);
   
   // ========== STEP 5: Set up source data ==========
   if (vector_field)
   {
      ExactVectorCoefficient coeff;
      periodic_source_data.ProjectCoefficient(coeff);
      periodic_exact.ProjectCoefficient(coeff);
   }
   else
   {
      FunctionCoefficient coeff(scalar_func);
      periodic_source_data.ProjectCoefficient(coeff);
      periodic_exact.ProjectCoefficient(coeff);
   }
   
   // CRITICAL: Synchronize shared DOFs on periodic boundaries
   periodic_source_data.SetFromTrueVector();
   periodic_exact.SetFromTrueVector();
   
   // ========== STEP 6: Set up GSLIB interpolation ==========
   FindPointsGSLIB finder(MPI_COMM_WORLD);
   finder.Setup(periodic_source_mesh);
   
   // Set tolerance based on perturbation amplitude
   double tolerance = std::max(1e-12 * L, 2.0 * amp);
   finder.SetDistanceToleranceForPointsFoundOnBoundary(tolerance);
   
   // ========== STEP 7: Get target coordinates (using true DOFs) ==========
   ParGridFunction *target_nodes = dynamic_cast<ParGridFunction*>(periodic_target_mesh.GetNodes());
   
   // Get the true DOF coordinates (handles periodic boundaries correctly)
   Vector target_coords_true;
   target_nodes->GetTrueDofs(target_coords_true);
   
   const int true_vsize = periodic_target_fes.GetTrueVSize();
   const int local_true_ndofs = true_vsize / vdim;
   
   // ========== STEP 8: Find points and interpolate ==========
   finder.FindPoints(target_coords_true, Ordering::byNODES);
   
   Array<unsigned int> code_out = finder.GetCode();
   Vector interp_vals_true(true_vsize);
   finder.Interpolate(periodic_source_data, interp_vals_true);
   
   // ========== STEP 9: Count success/failure ==========
   int local_inside = 0, local_border = 0, local_miss = 0;
   for (int i = 0; i < local_true_ndofs; ++i)
   {
      if (code_out[i] == 0u) ++local_inside;
      else if (code_out[i] == 1u) ++local_border;
      else ++local_miss;
   }
   
   int global_inside, global_border, global_miss, global_true_ndofs;
   MPI_Allreduce(&local_inside, &global_inside, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(&local_border, &global_border, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(&local_miss, &global_miss, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(&local_true_ndofs, &global_true_ndofs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   
   result.total_points = global_true_ndofs;
   result.failed_points = global_miss;
   result.success_rate = (global_true_ndofs > 0) ? 100.0 * (global_inside + global_border) / global_true_ndofs : 0.0;
   
   // ========== STEP 10: Handle failed points ==========
   for (int i = 0; i < local_true_ndofs; ++i)
   {
      if (code_out[i] >= 2)  // Point not found
      {
         for (int c = 0; c < vdim; ++c)
         {
            interp_vals_true[c * local_true_ndofs + i] = 0.0;
         }
      }
   }
   
   // CRITICAL: Set from true DOFs (handles periodic boundaries)
   periodic_interpolated.SetFromTrueDofs(interp_vals_true);
   
   // ========== STEP 11: Compute errors ==========
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
   
   // ========== STEP 12: Create VisIt output ==========
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
            
            // Output 2: Interpolated data on clean periodic target mesh
            VisItDataCollection target_dc(output_prefix + "_Target", &periodic_target_mesh);
            target_dc.SetPrecision(16);
            target_dc.RegisterField("interpolated_data", &periodic_interpolated);
            target_dc.RegisterField("exact_data", &periodic_exact);
            
            // Compute and save error field
            ParGridFunction error_field(&periodic_target_fes);
            error_field = periodic_exact;
            error_field -= periodic_interpolated;
            target_dc.RegisterField("interpolation_error", &error_field);
            
            target_dc.SetCycle(0); 
            target_dc.SetTime(0.0);
            target_dc.Save();
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
      std::cout << "\nPeriodic Data Projection Interpolation Test (CORRECTED)\n";
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
         std::cout << "✓ EXCELLENT: >99% success rate achieved!\n";
      }
      else if (result_with_pert.success_rate > 95.0)
      {
         std::cout << "✓ GOOD: High success rate achieved.\n";
      }
      else
      {
         std::cout << "⚠ WARNING: Lower than expected success rate.\n";
      }
      
      if (result_with_pert.relative_error < 1e-6)
      {
         std::cout << "✓ EXCELLENT: Very low relative error!\n";
      }
      else if (result_with_pert.relative_error < 1e-3)
      {
         std::cout << "✓ GOOD: Acceptable relative error.\n";
      }
      else
      {
         std::cout << "⚠ Note: Significant L2 error. This may be due to:\n";
         std::cout << "  - Mesh perturbation affecting interpolation accuracy\n";
         std::cout << "  - Consider reducing perturbation amplitude or increasing mesh resolution\n";
      }
      
      std::cout << "\nTotal points tested: " << result_with_pert.total_points << "\n";
      
      if (visit_output)
      {
         std::cout << "\nVisIt files created:\n";
         std::cout << "  Baseline_Baseline/ - Baseline without perturbation\n";
         std::cout << "  PeriodicProjection_Source/ - Original data on perturbed periodic mesh\n";
         std::cout << "  PeriodicProjection_Target/ - Interpolated data on clean periodic mesh\n";
         std::cout << "     - Fields: interpolated_data, exact_data, interpolation_error\n";
      }
      
      std::cout << "\n" << std::string(80, '=') << "\n";
      std::cout << "KEY IMPROVEMENTS IN THIS VERSION:\n";
      std::cout << "  • Uses SetFromTrueVector() after projection (syncs periodic boundaries)\n";
      std::cout << "  • Uses GetTrueDofs() for target coordinates (respects periodicity)\n";
      std::cout << "  • Uses SetFromTrueDofs() for interpolated result (correct DOF mapping)\n";
      std::cout << "  • Eliminated unnecessary non-periodic mesh complexity\n";
      std::cout << "  • Direct periodic-to-periodic interpolation via GSLIB\n";
      std::cout << std::string(80, '=') << "\n";
   }

   return 0;
}

/*
COMPILATION:
============
Make sure MFEM is built with GSLIB support (MFEM_USE_GSLIB=YES)

mpicxx -std=c++17 -I$MFEM_DIR -L$MFEM_DIR -o periodic_test_corrected \
       periodic_test_corrected.cpp -lmfem -lHYPRE -lmetis

USAGE EXAMPLES:
===============
# Serial test with scalar field
mpirun -np 1 ./periodic_test_corrected -amp 0.05

# Parallel test with vector field
mpirun -np 4 ./periodic_test_corrected -amp 0.02 -vec

# High-resolution test
mpirun -np 8 ./periodic_test_corrected -nx 16 -ny 16 -nz 16 -o 3 -vec

# Test without visualization output
mpirun -np 4 ./periodic_test_corrected -amp 0.05 -no-visit

WHAT TO EXPECT:
===============
- Baseline test should have ~0% error (identity mapping)
- Perturbed test success rate should be >99% (maybe 100%)
- L2 error should be small (dependent on amplitude and resolution)
- Lower perturbation amplitude = lower error
- Higher order and finer mesh = lower error
*/