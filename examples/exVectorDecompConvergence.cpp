#include "mfem.hpp"
#include <fstream>
#include <algorithm>
#include <iostream>
#include <string>

using namespace mfem;

void vel_tgv(const Vector &x, real_t t, Vector &u)
{
   real_t xi = x(0);
   real_t yi = x(1);
   real_t zi = x(2);

   u(0) = sin(xi) * cos(yi) * cos(zi);
   u(1) = -cos(xi) * sin(yi) * cos(zi);
   u(2) = 0.0;
}

void ComputeCurl3D(ParGridFunction &u, ParGridFunction &cu)
{
   FiniteElementSpace *fes = u.FESpace();

   // AccumulateAndCountZones.
   Array<int> zones_per_vdof;
   zones_per_vdof.SetSize(fes->GetVSize());
   zones_per_vdof = 0;

   cu = 0.0;

   // Local interpolation.
   int elndofs;
   Array<int> vdofs;
   Vector vals;
   Vector loc_data;
   int vdim = fes->GetVDim();
   DenseMatrix grad_hat;
   DenseMatrix dshape;
   DenseMatrix grad;
   Vector curl;

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      fes->GetElementVDofs(e, vdofs);
      u.GetSubVector(vdofs, loc_data);
      vals.SetSize(vdofs.Size());
      ElementTransformation *tr = fes->GetElementTransformation(e);
      const FiniteElement *el = fes->GetFE(e);
      elndofs = el->GetDof();
      int dim = el->GetDim();
      dshape.SetSize(elndofs, dim);

      for (int dof = 0; dof < elndofs; ++dof)
      {
         // Project.
         const IntegrationPoint &ip = el->GetNodes().IntPoint(dof);
         tr->SetIntPoint(&ip);

         // Eval and GetVectorGradientHat.
         el->CalcDShape(tr->GetIntPoint(), dshape);
         grad_hat.SetSize(vdim, dim);
         DenseMatrix loc_data_mat(loc_data.GetData(), elndofs, vdim);
         MultAtB(loc_data_mat, dshape, grad_hat);

         const DenseMatrix &Jinv = tr->InverseJacobian();
         grad.SetSize(grad_hat.Height(), Jinv.Width());
         Mult(grad_hat, Jinv, grad);

         curl.SetSize(3);
         curl(0) = grad(2, 1) - grad(1, 2);
         curl(1) = grad(0, 2) - grad(2, 0);
         curl(2) = grad(1, 0) - grad(0, 1);

         for (int j = 0; j < curl.Size(); ++j)
         {
            vals(elndofs * j + dof) = curl(j);
         }
      }

      // Accumulate values in all dofs, count the zones.
      for (int j = 0; j < vdofs.Size(); j++)
      {
         int ldof = vdofs[j];
         cu(ldof) += vals[j];
         zones_per_vdof[ldof]++;
      }
   }

   // Communication

   // Count the zones globally.
   GroupCommunicator &gcomm = u.ParFESpace()->GroupComm();
   gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
   gcomm.Bcast(zones_per_vdof);

   // Accumulate for all vdofs.
   gcomm.Reduce<real_t>(cu.GetData(), GroupCommunicator::Sum);
   gcomm.Bcast<real_t>(cu.GetData());

   // Compute means.
   for (int i = 0; i < cu.Size(); i++)
   {
      const int nz = zones_per_vdof[i];
      if (nz)
      {
         cu(i) /= nz;
      }
   }
}

void ComputeVorticalPart(ParGridFunction &u,
                         ParGridFunction &w_gf,
                         ParGridFunction &u_vort)
{

  // This routine solves for the vortical part of the
  // velocity field Ax=b where A is the vector diffusion
  // integrator and b is - vorticity and x are the vector
  // coefficients that represent the vortical solution.
  // Note that this routine assumes triple preiodicity
  // and we are not setting the mean of b to zero. 
   ParFiniteElementSpace *vfes = u.ParFESpace();

   Array<int> ess_tdof_list;
   // if (pmesh.bdr_attributes.Size())
   // {
   //    Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   //    ess_bdr = 0;
   //    // Apply boundary conditions on all external boundaries:
   //    pmesh.MarkExternalBoundaries(ess_bdr);
   //    // Boundary conditions can also be applied based on named attributes:
   //    // pmesh.MarkNamedBoundaries(set_name, ess_bdr)

   //    vfes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   // }

   // Extract ceofficients of gridfunction to form rhs
   VectorGridFunctionCoefficient w_coeff(&w_gf);
   ParLinearForm b(vfes);
   b.AddDomainIntegrator(new VectorDomainLFIntegrator(w_coeff));
   b.Assemble();

   ParBilinearForm vLap(vfes);
   vLap.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   ConstantCoefficient one(1.0); 
   vLap.AddDomainIntegrator(new VectorDiffusionIntegrator(one));

   vLap.Assemble();

   OperatorPtr A;
   Vector B,X;

   ParGridFunction x(vfes);
   x = 0.0;

   vLap.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   Solver *prec = new OperatorJacobiSmoother(vLap, ess_tdof_list);
   
   CGSolver cg(MPI_COMM_WORLD);
   if (prec) { cg.SetPreconditioner(*prec); }
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(0);
   cg.SetOperator(*A);
   cg.Mult(B, X);

   vLap.RecoverFEMSolution(X, b, x);

   ComputeCurl3D(x,u_vort);

}


void ComputeError(int num_pts,int  order,
    int element_subdivisions_parallel, int &dofs, double &h_min, double &error);

// Check to make sure mesh is periodic
template<typename T>
bool InArray(const T* begin, size_t sz, T i)
{
   const T *end = begin + sz;
   return std::find(begin, end, i) != end;
}

bool IndicesAreConnected(const Table &t, int i, int j)
{
   return InArray(t.GetRow(i), t.RowSize(i), j)
          && InArray(t.GetRow(j), t.RowSize(j), i);
}

void VerifyPeriodicMesh(Mesh *mesh, const int num_pts);

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   int myid = Mpi::WorldRank();
   Hypre::Init();

   int element_subdivisions = 0;
   int element_subdivisions_parallel = 0;
   int order = 2;
   bool visualization = false;
   int num_pts = 64;
   bool visit = true;
   int max_lref = 5;

   OptionsParser args(argc, argv);
   args.AddOption(&max_lref, "-lref", "--max-lor-refinement", "Max LOR refinement level.");
   args.AddOption(&order,
                  "-o",
                  "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&num_pts,
                  "-num_pts_per_dir",
                  "--grid-points-xyz",
                  "Number of grid points in xyz.");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root())
      {
         args.PrintUsage(mfem::out);
      }
      return 1;
   }
   if (Mpi::Root())
   {
      args.PrintOptions(mfem::out);
   }

   if (Mpi::Root()) {
     std::cout << "# Vector potential convergence_rate study\n";
       args.PrintOptions(std::cout);
       std::cout << std::setw(8)  << "ref"
            << std::setw(14) << "DOF"
            << std::setw(14) << "h_min"
            << std::setw(18) << "L2 Error"
            << std::setw(12) << "Rate"
            << std::endl << std::string(80, '-') << std::endl;
   }

   std::vector<double> ref_list, dofs_list, h_list, error_list, rate_list;
   double prev_error = 0.0, prev_h = 0.0;

   for (int lref = 1; lref <= max_lref; ++lref)
   {
       double  h_min= 0.0, error;
       int dofs;
       ComputeError(num_pts, order, lref, dofs, h_min, error);

       double rate = 0.0;
       if (lref > 1 && error > 1e-16 && prev_error > 1e-16 && h_min < prev_h && prev_h > 0.0)
           rate = log(error/prev_error) / log(h_min/prev_h);

       if (Mpi::Root()) {
         std::cout << std::setw(8)  << lref
                << std::setw(14) << (long long)dofs
                << std::setw(14) << h_min
                << std::setw(18) << error
                << std::setw(12) << rate << std::endl;
       }

      ref_list.push_back(lref);
      dofs_list.push_back(dofs);
      h_list.push_back(h_min);
      error_list.push_back(error);
      rate_list.push_back(rate);

       prev_error = error;
       prev_h = h_min;
   }

   if (Mpi::Root())
   {
       std::ofstream ofs("vector_potential_convergence_rate.txt");
       ofs << "# ref DOFs h_min L2Error Rate" << std::endl;
       for (size_t i = 0; i < ref_list.size(); ++i)
       {
           ofs << ref_list[i] << " "
               << dofs_list[i] << " "
               << dofs_list[i] << " "
               << h_list[i] << " "
               << error_list[i] << " "
               << rate_list[i] << std::endl;
       }
       ofs.close();
   }

   Mpi::Finalize();
   return 0;
}



void ComputeError(int num_pts, int order,
   int element_subdivisions_parallel, int &dofs, double &h_min, double &error)
{
   ParMesh *pmesh = nullptr;
   Mesh *mesh = nullptr;

   double t = 0.0;
   int step = 0;
   int global_cycle = 0;

      // Initialize as mesh
      Mesh *init_mesh;

      real_t length = 2*M_PI;
      init_mesh = new Mesh(Mesh::MakeCartesian3D(num_pts,
                                                 num_pts,
                                                 num_pts,
                                                 Element::HEXAHEDRON,
                                                 length,
                                                 length,
                                                 length, false));

      Vector x_translation({length, 0.0, 0.0});
      Vector y_translation({0.0, length, 0.0});
      Vector z_translation({0.0, 0.0, length});

      std::vector<Vector> translations = {x_translation, y_translation, z_translation};

      mesh = new Mesh(Mesh::MakePeriodic(*init_mesh, init_mesh->CreatePeriodicVertexMapping(translations)));

      if (Mpi::Root())
      {
         VerifyPeriodicMesh(mesh,num_pts);
      }
      

      // Create the parallel mesh
      pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
      pmesh->Finalize(true);

      // Parallel Mesh refinement
      for (int lev = 0; lev < element_subdivisions_parallel; lev++)
      {
         pmesh->UniformRefinement();
      }

      delete init_mesh;

      auto *vfec = new H1_FECollection(order, pmesh->Dimension());
      auto *vfes = new ParFiniteElementSpace(pmesh, vfec, pmesh->Dimension());
      ParGridFunction u_gf(vfes);
      ParGridFunction w_gf(vfes);
      ParGridFunction u_vort(vfes);

      // Set the initial condition
      VectorFunctionCoefficient u_excoeff(pmesh->Dimension(), vel_tgv);

      u_gf.ProjectCoefficient(u_excoeff);

      ComputeCurl3D(u_gf,w_gf);
      
      ComputeVorticalPart(u_gf, w_gf, u_vort);

      VectorGridFunctionCoefficient u_vort_coef(&u_vort);

      error = u_gf.ComputeL2Error(u_vort_coef);
      dofs  = vfes->GlobalTrueVSize();

      double h_max, kappa_min, kappa_max;
      pmesh->GetCharacteristics(h_min, h_max, kappa_min, kappa_max);
  
      int nel = pmesh->GetGlobalNE();

      delete mesh;
}

void VerifyPeriodicMesh(mfem::Mesh *mesh, const int num_pts)
{
    int n = num_pts;
    const mfem::Table &e2e = mesh->ElementToElementTable();
    int n2 = n * n;

    if (!mesh->GetNV() == pow(n - 1, 3) + 3 * pow(n - 1, 2) + 3 * (n - 1) + 1) {
      MFEM_ABORT("Mesh does not have the correct number of vertices for a periodic mesh.");
    }

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            // Check periodicity in z direction
            if (!IndicesAreConnected(e2e, i + j * n, i + j * n + n2 * (n - 1))) {
                MFEM_ABORT("Mesh is not periodic in the z direction.");
            }

            // Check periodicity in y direction
            if (!IndicesAreConnected(e2e, i + j * n2, i + j * n2 + n * (n - 1))) {
                MFEM_ABORT("Mesh is not periodic in the y direction.");
            }

            // Check periodicity in x direction
            if (!IndicesAreConnected(e2e, i * n + j * n2, i * n + j * n2 + n - 1)) {
                MFEM_ABORT("Mesh is not periodic in the x direction.");
            }
        }
    }
            
}


