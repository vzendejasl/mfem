#include "mfem.hpp"
#include <fstream>
#include <algorithm>
#include <iostream>
#include <string>

using namespace mfem;

struct s_NavierContext
{
   int element_subdivisions = 0;
   int element_subdivisions_parallel = 0;
   int order = 2;
   bool visualization = false;
   int num_pts = 8;
   bool visit = true;

} ctx;



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
   // Get the vector finite element space for u.
   ParFiniteElementSpace *vfes = u.ParFESpace();
   Array<int> ess_tdof_list;  // No essential degrees-of-freedom assumed.
     
   // This works with LORSolver
   vfes->GetBoundaryTrueDofs(ess_tdof_list);

   // Assemble the right-hand side b using the (negative) vorticity w_gf.
   VectorGridFunctionCoefficient w_coeff(&w_gf);
   ParLinearForm b(vfes);
   b.AddDomainIntegrator(new VectorDomainLFIntegrator(w_coeff));
   b.Assemble();

   // Assemble the vector diffusion operator.
   ParBilinearForm vLap(vfes);
   vLap.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   ConstantCoefficient one(1.0);
   vLap.AddDomainIntegrator(new VectorDiffusionIntegrator(one));
   vLap.Assemble();

   // Set the initial guess for the solution.
   ParGridFunction x(vfes);
   x = 0.0;
   // This works with the LOR Solver
   x = u;

   // Form the linear system A * X = B.
   OperatorHandle A;
   Vector X, B;
   vLap.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   // Set up the LOR solver using HypreBoomerAMG as the preconditioner.
   std::unique_ptr<Solver> lor_solver;
   // lor_solver.reset(new LORSolver<HypreBoomerAMG>(vLap, ess_tdof_list));
   lor_solver.reset(new OperatorJacobiSmoother(vLap, ess_tdof_list));

   // Set up the Conjugate Gradient (CG) solver.
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetAbsTol(0.0);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(500);
   cg.SetPrintLevel(3);
   cg.SetOperator(*A);
   cg.SetPreconditioner(*lor_solver);
   cg.Mult(B, X);

   // Recover the finite element solution from the linear system solution.
   vLap.RecoverFEMSolution(X, b, x);
   // u_vort = x;

   // Compute the curl of the computed vector field x to obtain the vortical part.
   ComputeCurl3D(x, u_vort);
}







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

void VerifyPeriodicMesh(Mesh *mesh);

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   int myid = Mpi::WorldRank();
   Hypre::Init();

   OptionsParser args(argc, argv);
   args.AddOption(&ctx.element_subdivisions,
                  "-es",
                  "--element-subdivisions",
                  "Number of 1d uniform subdivisions for each element.");
   args.AddOption(&ctx.element_subdivisions_parallel,
                  "-esp",
                  "--element-subdivisions-parallel",
                  "Number of 1d uniform subdivisions for each element.");
   args.AddOption(&ctx.order,
                  "-o",
                  "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ctx.visualization,
                  "-vis",
                  "--visualization",
                  "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&ctx.visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&ctx.num_pts,
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
   int order = ctx.order;

   ParMesh *pmesh = nullptr;
   Mesh *mesh = nullptr;

   double t = 0.0;
   int step = 0;
   int global_cycle = 0;

     if (Mpi::Root())
     {
        std::cout << "Creating the mesh..." << std::endl;
     }

      // Initialize as mesh
      Mesh *init_mesh;

      real_t length = 2.0*M_PI;
      init_mesh = new Mesh(Mesh::MakeCartesian3D(ctx.num_pts,
                                                 ctx.num_pts,
                                                 ctx.num_pts,
                                                 Element::HEXAHEDRON,
                                                 length,
                                                 length,
                                                 length, false));

      Vector x_translation({length, 0.0, 0.0});
      Vector y_translation({0.0, length, 0.0});
      Vector z_translation({0.0, 0.0, length});

      std::vector<Vector> translations = {x_translation, y_translation, z_translation};
      mesh = new Mesh(Mesh::MakePeriodic(*init_mesh, init_mesh->CreatePeriodicVertexMapping(translations)));


      // const char *mesh_file = "../data/periodic-cube.mesh";
      // mesh = new Mesh(mesh_file,1,1);


      if (Mpi::Root())
      {
         VerifyPeriodicMesh(mesh);
      }
      
      if (Mpi::Root() && (ctx.element_subdivisions >= 1))
      {
         mfem::out << "Serial refining the mesh... " << std::endl;
      }

      // Serial Mesh refinement
      for (int lev = 0; lev < ctx.element_subdivisions; lev++)
      {
         mesh->UniformRefinement();
      }

      // Create the parallel mesh
      pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
      pmesh->Finalize(true);

      if (Mpi::Root() && (ctx.element_subdivisions_parallel >= 1))
      {
         mfem::out << "Parallel refining the mesh... " << std::endl;
      }

      // Parallel Mesh refinement
      for (int lev = 0; lev < ctx.element_subdivisions_parallel; lev++)
      {
         pmesh->UniformRefinement();
      }

      delete init_mesh;

      if (Mpi::Root())
      {
         mfem::out << "Done creating the mesh. Creating the flowsolver. " << std::endl;
      }

   
      auto *vfec = new H1_FECollection(order, pmesh->Dimension());
      auto *vfes = new ParFiniteElementSpace(pmesh, vfec, pmesh->Dimension());
      ParGridFunction u_gf(vfes);
      ParGridFunction w_gf(vfes);
      ParGridFunction u_vort(vfes);

      // Set the initial condition
      VectorFunctionCoefficient u_excoeff(pmesh->Dimension(), vel_tgv);
      u_gf.ProjectCoefficient(u_excoeff);

      ComputeCurl3D(u_gf, w_gf);
      
      ComputeVorticalPart(u_gf, w_gf, u_vort);

      int nel = pmesh->GetGlobalNE();
      if (Mpi::Root())
      {
         mfem::out << "Number of elements: " << nel << std::endl;
      }

      DataCollection *dc = NULL;
      if (ctx.visit)
      {
            std::string visit_dir = std::string("VisitData_") 
                                                     + "NumPtsPerDir" +std::to_string(ctx.num_pts) 
                                                     + "RefLv" + std::to_string(
                                                         ctx.element_subdivisions 
                                                       + ctx.element_subdivisions_parallel) 
                                                     + "P" + std::to_string(ctx.order)
                                                     + "/tgv_output_visit";

            dc = new VisItDataCollection(MPI_COMM_WORLD,visit_dir, pmesh);
         int precision = 16;
         dc->SetPrecision(precision);
         dc->SetCycle(global_cycle + step);
         dc->SetTime(t);
         dc->SetFormat(DataCollection::PARALLEL_FORMAT);
         dc->RegisterField("velocity", &u_gf);
         dc->RegisterField("vorticity", &w_gf);
         dc->RegisterField("u_vort", &u_vort); 
         dc->Save();
         }

       {
          // Save the solution and mesh to disk. The output can be viewed using
          // GLVis as follows: "glvis -np <np> -m mesh -g sol"
          u_vort.Save("u_vort");
          pmesh->Save("mesh");
       }


        delete pmesh;
        delete mesh;

        return 0;
}

void VerifyPeriodicMesh(mfem::Mesh *mesh)
{
    int n = ctx.num_pts;
    const mfem::Table &e2e = mesh->ElementToElementTable();
    int n2 = n * n;

    std::cout << "Checking to see if mesh is periodic.." << std::endl;

    if (mesh->GetNV() == pow(n - 1, 3) + 3 * pow(n - 1, 2) + 3 * (n - 1) + 1) {
        std::cout << "Total number of vertices match a periodic mesh." << std::endl;
    } else {
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
            
    std::cout << "Done checking... Periodic in all directions." << std::endl;
}


