// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// 3D Taylor-Green vortex benchmark example at Re=1600
// Unsteady flow of a decaying vortex is computed and compared against a known,
// analytical solution.
//
// TODO: Incorporate restart. We can read a mesh from a restart file, but we
// need to save all the field variales as well and load those too. 

#include "navier_solver.hpp"
#include <fstream>
#include <algorithm>
#include <iostream>

using namespace mfem;
using namespace navier;

struct s_NavierContext
{
   int element_subdivisions = 0;
   int element_subdivisions_parallel = 0;
   int order = 2;
   real_t kinvis = 1.0 / 1600.0;
   real_t t_final = 10 * 1e-3;
   real_t dt = 1e-3;
   bool pa = true;
   bool ni = false;
   bool visualization = false;
   bool checkres = false;
   int num_pts = 64;
   bool visit = true;
   bool paraview = false;
   bool binary = false;
   bool restart = false;
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

class QuantitiesOfInterest
{
public:
   QuantitiesOfInterest(ParMesh *pmesh)
   {
      H1_FECollection h1fec(1);
      ParFiniteElementSpace h1fes(pmesh, &h1fec);

      onecoeff.constant = 1.0;
      mass_lf = new ParLinearForm(&h1fes);
      mass_lf->AddDomainIntegrator(new DomainLFIntegrator(onecoeff));
      mass_lf->Assemble();

      ParGridFunction one_gf(&h1fes);
      one_gf.ProjectCoefficient(onecoeff);

      volume = mass_lf->operator()(one_gf);
   };

   real_t ComputeKineticEnergy(ParGridFunction &v)
   {
      Vector velx, vely, velz;
      real_t integ = 0.0;
      const FiniteElement *fe;
      ElementTransformation *T;
      FiniteElementSpace *fes = v.FESpace();

      for (int i = 0; i < fes->GetNE(); i++)
      {
         fe = fes->GetFE(i);
         int intorder = 2 * fe->GetOrder();
         const IntegrationRule *ir = &IntRules.Get(fe->GetGeomType(), intorder);

         v.GetValues(i, *ir, velx, 1);
         v.GetValues(i, *ir, vely, 2);
         v.GetValues(i, *ir, velz, 3);

         T = fes->GetElementTransformation(i);
         for (int j = 0; j < ir->GetNPoints(); j++)
         {
            const IntegrationPoint &ip = ir->IntPoint(j);
            T->SetIntPoint(&ip);

            real_t vel2 = velx(j) * velx(j) + vely(j) * vely(j)
                          + velz(j) * velz(j);

            integ += ip.weight * T->Weight() * vel2;
         }
      }

      real_t global_integral = 0.0;
      MPI_Allreduce(&integ,
                    &global_integral,
                    1,
                    MPITypeMap<real_t>::mpi_type,
                    MPI_SUM,
                    MPI_COMM_WORLD);

      return 0.5 * global_integral / volume;
   };

  real_t ComputeEnstrophy(ParGridFunction &w)
  {
      Vector wx, wy, wz;
      real_t integ = 0.0;
      const FiniteElement *fe;
      ElementTransformation *T;
      FiniteElementSpace *fes = w.FESpace();
  
      for (int i = 0; i < fes->GetNE(); i++)
      {
          fe = fes->GetFE(i);
          int intorder = 2 * fe->GetOrder();
          const IntegrationRule *ir = &IntRules.Get(fe->GetGeomType(), intorder);
  
          w.GetValues(i, *ir, wx, 1);
          w.GetValues(i, *ir, wy, 2);
          w.GetValues(i, *ir, wz, 3);
  
          T = fes->GetElementTransformation(i);
          for (int j = 0; j < ir->GetNPoints(); j++)
          {
              const IntegrationPoint &ip = ir->IntPoint(j);
              T->SetIntPoint(&ip);
  
              real_t w2 = wx(j) * wx(j) + wy(j) * wy(j) + wz(j) * wz(j);
  
              integ += ip.weight * T->Weight() * w2;
          }
      }
  
      real_t global_integral = 0.0;
      MPI_Allreduce(&integ,
                    &global_integral,
                    1,
                    MPITypeMap<real_t>::mpi_type,
                    MPI_SUM,
                    MPI_COMM_WORLD);
  
      return 0.5 * global_integral / volume;
  }
   

   ~QuantitiesOfInterest() { delete mass_lf; };

private:
   ConstantCoefficient onecoeff;
   ParLinearForm *mass_lf;
   real_t volume;
};

template<typename T>
T sq(T x)
{
   return x * x;
}

// Computes Q = 0.5*(tr(\nabla u)^2 - tr(\nabla u \cdot \nabla u))
void ComputeQCriterion(ParGridFunction &u, ParGridFunction &q)
{
   FiniteElementSpace *v_fes = u.FESpace();
   FiniteElementSpace *fes = q.FESpace();

   // AccumulateAndCountZones
   Array<int> zones_per_vdof;
   zones_per_vdof.SetSize(fes->GetVSize());
   zones_per_vdof = 0;

   q = 0.0;

   // Local interpolation
   int elndofs;
   Array<int> v_dofs, dofs;
   Vector vals;
   Vector loc_data;
   int vdim = v_fes->GetVDim();
   DenseMatrix grad_hat;
   DenseMatrix dshape;
   DenseMatrix grad;

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      fes->GetElementVDofs(e, dofs);
      v_fes->GetElementVDofs(e, v_dofs);
      u.GetSubVector(v_dofs, loc_data);
      vals.SetSize(dofs.Size());
      ElementTransformation *tr = fes->GetElementTransformation(e);
      const FiniteElement *el = fes->GetFE(e);
      elndofs = el->GetDof();
      int dim = el->GetDim();
      dshape.SetSize(elndofs, dim);

      for (int dof = 0; dof < elndofs; ++dof)
      {
         // Project
         const IntegrationPoint &ip = el->GetNodes().IntPoint(dof);
         tr->SetIntPoint(&ip);

         // Eval
         // GetVectorGradientHat
         el->CalcDShape(tr->GetIntPoint(), dshape);
         grad_hat.SetSize(vdim, dim);
         DenseMatrix loc_data_mat(loc_data.GetData(), elndofs, vdim);
         MultAtB(loc_data_mat, dshape, grad_hat);

         const DenseMatrix &Jinv = tr->InverseJacobian();
         grad.SetSize(grad_hat.Height(), Jinv.Width());
         Mult(grad_hat, Jinv, grad);

         real_t q_val = 0.5 * (sq(grad(0, 0)) + sq(grad(1, 1)) + sq(grad(2, 2)))
                        + grad(0, 1) * grad(1, 0) + grad(0, 2) * grad(2, 0)
                        + grad(1, 2) * grad(2, 1);

         vals(dof) = q_val;
      }

      // Accumulate values in all dofs, count the zones.
      for (int j = 0; j < dofs.Size(); j++)
      {
         int ldof = dofs[j];
         q(ldof) += vals[j];
         zones_per_vdof[ldof]++;
      }
   }

   // Communication

   // Count the zones globally.
   GroupCommunicator &gcomm = q.ParFESpace()->GroupComm();
   gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
   gcomm.Bcast(zones_per_vdof);

   // Accumulate for all vdofs.
   gcomm.Reduce<real_t>(q.GetData(), GroupCommunicator::Sum);
   gcomm.Bcast<real_t>(q.GetData());

   // Compute means
   for (int i = 0; i < q.Size(); i++)
   {
      const int nz = zones_per_vdof[i];
      if (nz)
      {
         q(i) /= nz;
      }
   }
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

void ComputeElementCenterValues(ParGridFunction *sol, ParMesh *pmesh, int step);
void ComputeElementCenterValuesScalar(ParGridFunction *sol, ParMesh *pmesh);

void SaveCheckpoint(ParMesh *pmesh, ParGridFunction *u_gf, ParGridFunction *p_gf,
                    double t, int step, int myid);

bool LoadCheckpoint(ParMesh *&pmesh, ParGridFunction *&u_gf, ParGridFunction *&p_gf,
                    NavierSolver *&flowsolver, double &t, int &step, int myid, const s_NavierContext &ctx);

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
   args.AddOption(&ctx.dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&ctx.t_final, "-tf", "--final-time", "Final time.");
   args.AddOption(&ctx.pa,
                  "-pa",
                  "--enable-pa",
                  "-no-pa",
                  "--disable-pa",
                  "Enable partial assembly.");
   args.AddOption(&ctx.ni,
                  "-ni",
                  "--enable-ni",
                  "-no-ni",
                  "--disable-ni",
                  "Enable numerical integration rules.");
   args.AddOption(&ctx.visualization,
                  "-vis",
                  "--visualization",
                  "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&ctx.visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&ctx.paraview, "-paraview", "--paraview-datafiles", "-no-paraview",
                  "--no-paraview-datafiles",
                  "Save data files for ParaView (paraview.org) visualization.");
   args.AddOption(&ctx.binary, "-binary", "--binary-datafiles", "-ascii",
                  "--ascii-datafiles",
                  "Use binary (Sidre) or ascii format for VisIt data files.");
   args.AddOption(&ctx.num_pts,
                  "-num_pts_per_dir",
                  "--grid-points-xyz",
                  "Number of grid points in xyz.");
   args.AddOption(&ctx.restart, "-res", "--restart", "-no-res", "--no-restart",
                  "Restart computation from the last checkpoint.");
   args.AddOption(
      &ctx.checkres,
      "-cr",
      "--checkresult",
      "-no-cr",
      "--no-checkresult",
      "Enable or disable checking of the result. Returns -1 on failure.");
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

   ParMesh *pmesh = nullptr;
   ParGridFunction *u_gf = nullptr;
   ParGridFunction *p_gf = nullptr;
   double t = 0.0;
   int step = 0;
   int initial_step = 0;
   NavierSolver *flowsolver = nullptr;

   bool restart_files_found = false;

   if (ctx.restart)
   {
      // Try to load the checkpoint files
      restart_files_found = LoadCheckpoint(pmesh, u_gf, p_gf, flowsolver, t, step, myid, ctx);
      if (restart_files_found)
      {
         if (Mpi::Root())
         {
            std::cout << "Restart files found. Continuing from checkpoint at time t = " << t << std::endl;
         }
         // Store the initial step number at restart
         initial_step = step + 1;
      }
      else
      {
         if (Mpi::Root())
         {
            std::cout << "Restart files not found. Starting from initial conditions." << std::endl;
         }
      }


   }

   if (!ctx.restart || !restart_files_found)
   {
      // Initialize as usual
      Mesh *init_mesh;
      Mesh *mesh;

      init_mesh = new Mesh(Mesh::MakeCartesian3D(ctx.num_pts,
                                                 ctx.num_pts,
                                                 ctx.num_pts,
                                                 Element::HEXAHEDRON,
                                                 2.0 * M_PI,
                                                 2.0 * M_PI,
                                                 2.0 * M_PI, false));

      Vector x_translation({2.0 * M_PI, 0.0, 0.0});
      Vector y_translation({0.0, 2.0 * M_PI, 0.0});
      Vector z_translation({0.0, 0.0, 2.0 * M_PI});

      std::vector<Vector> translations = {x_translation, y_translation, z_translation};

      mesh = new Mesh(Mesh::MakePeriodic(*init_mesh, init_mesh->CreatePeriodicVertexMapping(translations)));

      VerifyPeriodicMesh(mesh);

      // Define a translation function for the mesh nodes
      VectorFunctionCoefficient translate(mesh->Dimension(), [&](const Vector &x_in, Vector &x_out)
                                          {
         double shift = -M_PI;

         x_out[0] = x_in[0] + shift; // Translate x-coordinate
         x_out[1] = x_in[1] + shift; // Translate y-coordinate
         if (mesh->Dimension() == 3){
           x_out[2] = x_in[2] + shift; // Translate z-coordinate
         } });

      mesh->Transform(translate);

      // Serial Mesh refinement
      for (int lev = 0; lev < ctx.element_subdivisions; lev++)
      {
         mesh->UniformRefinement();
      }

      // Create the parallel mesh
      pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);

      // Parallel Mesh refinement
      for (int lev = 0; lev < ctx.element_subdivisions_parallel; lev++)
      {
         pmesh->UniformRefinement();
      }

      delete mesh;
      delete init_mesh;

      // Create the flow solver
      flowsolver = new NavierSolver(pmesh, ctx.order, ctx.kinvis);
      flowsolver->EnablePA(ctx.pa);
      flowsolver->EnableNI(ctx.ni);

      // Set the initial condition
      u_gf = flowsolver->GetCurrentVelocity();
      VectorFunctionCoefficient u_excoeff(pmesh->Dimension(), vel_tgv);
      u_gf->ProjectCoefficient(u_excoeff);

      p_gf = flowsolver->GetCurrentPressure();

      // Set up the flow solver
      flowsolver->Setup(ctx.dt);

      ComputeElementCenterValues(u_gf, pmesh, step);
      // ComputeElementCenterValuesScalar(u_gf, pmesh);
   }

   int nel = pmesh->GetNE();
   if (Mpi::Root())
   {
      mfem::out << "Number of elements: " << nel << std::endl;
   }

   // Initialize w_gf and q_gf using the finite element spaces
   ParFiniteElementSpace *velocity_fespace = u_gf->ParFESpace();
   ParFiniteElementSpace *pressure_fespace = p_gf->ParFESpace();
   
   ParGridFunction w_gf(velocity_fespace);
   ParGridFunction q_gf(pressure_fespace);

   flowsolver->ComputeCurl3D(*u_gf, w_gf);
   ComputeQCriterion(*u_gf, q_gf);
   QuantitiesOfInterest kin_energy(pmesh);

   ParaViewDataCollection *pvdc = NULL;
   if (ctx.paraview)
   {
      pvdc = new ParaViewDataCollection("DatOutputParaivew/tgv_output", pmesh);
      pvdc->SetDataFormat(VTKFormat::BINARY32);
      pvdc->SetHighOrderOutput(true);
      pvdc->SetLevelsOfDetail(ctx.order);
      pvdc->SetCycle(step + initial_step);
      pvdc->SetTime(t);
      pvdc->RegisterField("velocity", u_gf);
      pvdc->RegisterField("pressure", p_gf);
      pvdc->RegisterField("vorticity", &w_gf);
      pvdc->RegisterField("qcriterion", &q_gf);
      pvdc->Save();
   }

   DataCollection *dc = NULL;
   if (ctx.visit)
   {
      if (ctx.binary)
      {
#ifdef MFEM_USE_SIDRE
         dc = new SidreDataCollection("tgv_output_sidre", pmesh);
#else
         MFEM_ABORT("Must build with MFEM_USE_SIDRE=YES for binary output.");
#endif
      }
      else
      {
         int precision = 8;
         dc = new VisItDataCollection("DataOutputVisit/tgv_output_visit", pmesh);
         dc->SetPrecision(precision);
      }
      dc->SetCycle(step + initial_step);
      dc->SetTime(t);
      dc->RegisterField("velocity", u_gf);
      dc->RegisterField("pressure", p_gf);
      dc->RegisterField("vorticity", &w_gf);
      dc->RegisterField("qcriterion", &q_gf);
      dc->Save();
   }

   real_t u_inf_loc = u_gf->Normlinf();
   real_t p_inf_loc = p_gf->Normlinf();
   real_t u_inf = GlobalLpNorm(infinity(), u_inf_loc, MPI_COMM_WORLD);
   real_t p_inf = GlobalLpNorm(infinity(), p_inf_loc, MPI_COMM_WORLD);
   real_t ke = kin_energy.ComputeKineticEnergy(*u_gf);
   real_t enstrophy = kin_energy.ComputeEnstrophy(w_gf);

   std::string fname = "tgv_out_p_" + std::to_string(ctx.order) + ".txt";
   FILE *f = NULL;

   if (Mpi::Root())
   {
      int nel1d = static_cast<int>(std::round(pow(nel, 1.0 / 3.0)));
      int ngridpts = p_gf->ParFESpace()->GlobalVSize();
      printf("%11s %11s %11s %11s %11s %11s\n", "Time", "dt", "u_inf", "p_inf", "ke", "enstrophy");
      printf("%.5E %.5E %.5E %.5E %.5E %.5E\n", t, ctx.dt, u_inf, p_inf, ke, enstrophy);


      // Determine the file mode based on whether we're restarting

      const char *file_mode = "w"; // Default write mode

      if (ctx.restart && restart_files_found)
      {
        file_mode = "a"; // Switch to append mode if restarting
      }

      f = fopen(fname.c_str(), file_mode);

      if (!f)
      {
        std::cerr << "Error opening file " << fname << std::endl;
        MPI_Abort(MPI_COMM_WORLD,1);
      }

      if (!(ctx.restart && restart_files_found))
      {
          // Write header only if not restarting
          fprintf(f, "3D Taylor Green Vortex\n");
          fprintf(f, "order = %d\n", ctx.order);
          fprintf(f, "grid = %d x %d x %d\n", nel1d, nel1d, nel1d);
          fprintf(f, "dofs per component = %d\n", ngridpts);
          fprintf(f, "===============================================================================\n");
          fprintf(f, "        time                   kinetic energy                   enstrophy\n");

          // Write the initial data point
          fprintf(f, "%20.16e     %20.16e     %20.16e\n", t, ke, enstrophy);
      } 

      fflush(f);
      fflush(stdout);
   }


   real_t dt = ctx.dt;
   real_t t_final = ctx.t_final;
   bool last_step = false;

   // reset step from restart for flow solver
   step = 0;
   for (; !last_step; ++step)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      flowsolver->Step(t, dt, step);

      if ((step - initial_step) % 100 == 0 || last_step)
      {
         // If restarting, skip the first saved checkpoint
         if (!(ctx.restart && step == 0 && restart_files_found))
         {
            flowsolver->ComputeCurl3D(*u_gf, w_gf);
            ComputeQCriterion(*u_gf, q_gf);

            if (ctx.paraview)
            {
               pvdc->SetCycle(step + initial_step);
               pvdc->SetTime(t);
               pvdc->Save();
               if (Mpi::Root())
               {
                  std::cout << "\nParaview file saved." << std::endl;
               }
            }

            if (ctx.visit)
            {
               dc->SetCycle(step + initial_step);
               dc->SetTime(t);
               dc->Save();
               if (Mpi::Root())
               {
                  std::cout << "\nVisit file saved at cycle " << step + initial_step << "." << std::endl;
               }
            }

            ComputeElementCenterValues(u_gf, pmesh, step + initial_step);
            if (Mpi::Root())
            {
               std::cout << "\nOutput element center file saved at cycle " << step + initial_step << "." << std::endl;
            }

            // Save the checkpoint files
            SaveCheckpoint(pmesh, u_gf, p_gf, t, step, myid);
            if (Mpi::Root())
            {
               std::cout << "\nCheckpoint saved." << std::endl;
            }
         }
      }

      u_inf_loc = u_gf->Normlinf();
      p_inf_loc = p_gf->Normlinf();

      u_inf = GlobalLpNorm(infinity(), u_inf_loc, MPI_COMM_WORLD);
      p_inf = GlobalLpNorm(infinity(), p_inf_loc, MPI_COMM_WORLD);

      ke = kin_energy.ComputeKineticEnergy(*u_gf);
      flowsolver->ComputeCurl3D(*u_gf, w_gf);
      enstrophy = kin_energy.ComputeEnstrophy(w_gf);

      if (Mpi::Root())
      {
         // If restarting, skip the first saved checkpoint
         if (!(ctx.restart && step == 0 && restart_files_found))
         {
           printf("%.5E %.5E %.5E %.5E %.5E %.5E\n", t, dt, u_inf, p_inf, ke, enstrophy);
           fprintf(f, "%20.16e     %20.16e     %20.16e\n", t, ke, enstrophy);
           fflush(f);
           fflush(stdout);
         }
      }
   }

   flowsolver->PrintTimingData();

   // Test if the result for the test run is as expected.
   if (ctx.checkres)
   {
      real_t tol = 2e-5;
      real_t ke_expected = 1.25e-1;
      if (fabs(ke - ke_expected) > tol)
      {
         if (Mpi::Root())
         {
            mfem::out << "Result has a larger error than expected."
                      << std::endl;
         }
         return -1;
      }
   }

   delete flowsolver;
   delete pmesh;

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

/*
// For debuging: Does not work with serial or parallel refinement!
void ComputeElementCenterValues(ParGridFunction* sol, ParMesh* pmesh, int step)
{
  
  
    // Local arrays to store the data
    std::vector<double> local_x, local_y, local_z;
    std::vector<double> local_velx, local_vely, local_velz;

    FiniteElementSpace *fes = sol->FESpace();

    // Set the integration point to the center of the reference element
    IntegrationPoint ip;
    ip.Set3(0.5, 0.5, 0.5);  // Center of the reference element

    // Loop over local elements
    int num_elems = pmesh->GetNE();
    for (int i = 0; i < num_elems; i++)
    {
        // Get the element transformation
        ElementTransformation *Trans = pmesh->GetElementTransformation(i);

        // Evaluate the solution at the element center
        Trans->SetIntPoint(&ip);

        // Create a vector to hold the velocity components
        Vector u_val(fes->GetVDim()); // Typically 3 for velocity

        // Get the vector value at the integration point
        sol->GetVectorValue(*Trans, ip, u_val);

        // Extract the components
        double u_x = u_val(0);
        double u_y = u_val(1);
        double u_z = u_val(2);

        // Transform the reference point to physical coordinates
        Vector phys_coords(3);
        Trans->Transform(ip, phys_coords);

        double x_center = phys_coords[0];
        double y_center = phys_coords[1];
        double z_center = phys_coords[2];

        // Store the data
        local_x.push_back(x_center);
        local_y.push_back(y_center);
        local_z.push_back(z_center);

        local_velx.push_back(u_x);
        local_vely.push_back(u_y);
        local_velz.push_back(u_z);
    }

    // In serial, we have all data locally, so we can directly write to file.
    // Create a filename that includes the step number
    std::ostringstream fname_stream;
    // fname_stream << "element_centers_step_" << step << ".txt";
    fname_stream << "element_centers_vec" << ".txt";
    std::string fname = fname_stream.str();

    FILE *f = fopen(fname.c_str(), "w");
    if (!f)
    {
        std::cerr << "Error opening file " << fname << std::endl;
        abort();
    }

    // Write header
    fprintf(f, "3D Taylor Green Vortex\n");
    fprintf(f, "order = %d\n", ctx.order);
    fprintf(f, "step = %d\n", step);
    fprintf(f, "===================================================================");
    fprintf(f, "========================================================================\n");
    fprintf(f, "            x                      y                      z         ");
    fprintf(f, "            velx                   vely                   velz\n");

    // Write data with aligned columns
    for (size_t i = 0; i < local_x.size(); ++i)
    {
        fprintf(f, "%20.16e %20.16e %20.16e %20.16e %20.16e %20.16e\n",
                local_x[i], local_y[i], local_z[i],
                local_velx[i], local_vely[i], local_velz[i]);
    }

    fclose(f);


    std::ofstream ofs("element_centers_vec.txt");
    if (!ofs.is_open())
    {
        std::cerr << "Error: Unable to open element_centers.txt for writing." << std::endl;
        return;
    }

    // Write descriptive header
    ofs << "3D Element Centers\n";
    ofs << "===============================================================================\n";

    // Write column headers with variable name
    ofs << std::left << std::setw(20) << " "
        << std::left << std::setw(20) << "x"
        << std::left << std::setw(20) << "y"
        << std::left << std::setw(20) << "z"
        << std::left << std::setw(20) << ("u_x")
        << std::left << std::setw(20) << ("u_y")
        << std::left << std::setw(20) << ("u_z")
        << "\n";

    // Set formatting options
    ofs << std::scientific << std::setprecision(16);

    // Write data with aligned columns
    for (size_t i = 0; i < local_x.size(); ++i)
    {
        ofs << std::setw(20) << local_x[i] << " "
            << std::setw(20) << local_y[i] << " "
            << std::setw(20) << local_z[i] << " "
            << std::setw(20) << local_velx[i] << " "
            << std::setw(20) << local_vely[i] << " "
            << std::setw(20) << local_velz[i] << "\n";
    }

    ofs.close();
    std::cout << "Element center values written to element_centers.txt" << std::endl;
    
}

*/

void ComputeElementCenterValues(ParGridFunction* sol, ParMesh* pmesh, int step)
{
    // Local arrays to store the data
    std::vector<double> local_x, local_y, local_z, local_value;
    std::vector<double> local_velx, local_vely, local_velz;

    FiniteElementSpace *fes = sol->FESpace();

    // Set the integration point to the center of the reference element
    IntegrationPoint ip;
    ip.Set3(0.5, 0.5, 0.5);  // Center of the reference element

    // Loop over local elements
    for (int i = 0; i < pmesh->GetNE(); i++)
    {
        // Get the element transformation
        ElementTransformation *Trans = pmesh->GetElementTransformation(i);

        // Evaluate the solution at the element center
        Trans->SetIntPoint(&ip);

        // Create a vector to hold the velocity components
        Vector u_val(sol->FESpace()->GetVDim()); // GetVDim() returns the vector dimension (should be 3)
        
        // Get the vector value at the integration point
        sol->GetVectorValue(*Trans, ip, u_val);
        
        // Extract the components
        double u_x = u_val(0);
        double u_y = u_val(1);
        double u_z = u_val(2);
        
        // Transform the reference point to physical coordinates
        Vector phys_coords(3);
        Trans->Transform(ip, phys_coords);

        double x_center = phys_coords[0];
        double y_center = phys_coords[1];
        double z_center = phys_coords[2];

        // Store the data
        local_x.push_back(x_center);
        local_y.push_back(y_center);
        local_z.push_back(z_center);

        local_velx.push_back(u_x);
        local_vely.push_back(u_y);
        local_velz.push_back(u_z);
    }

    // MPI setup
    MPI_Comm comm = pmesh->GetComm();
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Gather all data on rank 0
    std::vector<double> all_x, all_y, all_z;
    std::vector<double> all_velx, all_vely, all_velz;
    int local_num_elements = local_x.size();
    std::vector<int> all_num_elements(size);
    std::vector<int> displs(size);

    MPI_Gather(&local_num_elements, 1, MPI_INT, 
        all_num_elements.data(), 1, MPI_INT, 0, comm);

    if (rank == 0)
    {
        int total_elements = 0;
        displs[0] = 0;
        for (int i = 0; i < size; ++i)
        {
            total_elements += all_num_elements[i];
            if (i > 0)
            {
                displs[i] = displs[i - 1] + all_num_elements[i - 1];
            }
        }

        all_x.resize(total_elements);
        all_y.resize(total_elements);
        all_z.resize(total_elements);
        all_velx.resize(total_elements);
        all_vely.resize(total_elements);
        all_velz.resize(total_elements);
    }

    MPI_Gatherv(local_x.data(), local_num_elements, MPI_DOUBLE, 
        all_x.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(local_y.data(), local_num_elements, MPI_DOUBLE, 
        all_y.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(local_z.data(), local_num_elements, MPI_DOUBLE, 
        all_z.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);

    MPI_Gatherv(local_velx.data(), local_num_elements, MPI_DOUBLE, 
        all_velx.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(local_vely.data(), local_num_elements, MPI_DOUBLE, 
        all_vely.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(local_velz.data(), local_num_elements, MPI_DOUBLE, 
        all_velz.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);

    // Write the data to a file in a human-readable format on rank 0
    if (rank == 0)
    {
   
      std::string fname = "element_centers_" + std::to_string(step) + ".txt";
      FILE *f = NULL;
      f = fopen(fname.c_str(), "w");
      if (!f)
      {
        std::cerr << "Error opening file " << fname << std::endl;
        MPI_Abort(MPI_COMM_WORLD,1);
      }

      // Write header only if not restarting
      fprintf(f, "3D Taylor Green Vortex\n");
      fprintf(f, "order = %d\n", ctx.order);
      fprintf(f, "step = %d\n", step);
      fprintf(f, "===================================================================");
      fprintf(f, "========================================================================\n");
      fprintf(f, "            x                      y                      z         ");
      fprintf(f, "            velx                   vely                   velz\n");

      // Write data with aligned columns
      for (size_t i = 0; i < all_x.size(); ++i)
      {
        // Write the initial data point
        fprintf(f, "%20.16e %20.16e %20.16e %20.16e %20.16e %20.16e\n", all_x[i], all_y[i],all_z[i],
                                                                        all_velx[i], all_vely[i], all_velz[i]);
      }

      fflush(f);
      fflush(stdout);
    }
    
}

void ComputeElementCenterValuesScalar(ParGridFunction* sol, ParMesh* pmesh)
{
    // Local arrays to store the data
    std::vector<double> local_x, local_y, local_z, local_value;
    Vector velx, vely, velz;

    FiniteElementSpace *fes = sol->FESpace();

    // Set the integration point to the center of the reference element
    IntegrationPoint ip;
    ip.Set3(0.5, 0.5, 0.5);  // Center of the reference element

    // Loop over local elements
    for (int i = 0; i < pmesh->GetNE(); i++)
    {
        // Get the element transformation
        ElementTransformation *Trans = pmesh->GetElementTransformation(i);

        // Evaluate the solution at the element center
        Trans->SetIntPoint(&ip);
        // double value = sol->GetValue(*Trans, ip);
        double value = sol->GetValue(*Trans, ip, 1);

        // Transform the reference point to physical coordinates
        Vector phys_coords(3);
        Trans->Transform(ip, phys_coords);

        double x_center = phys_coords[0];
        double y_center = phys_coords[1];
        double z_center = phys_coords[2];

        // Store the data
        local_x.push_back(x_center);
        local_y.push_back(y_center);
        local_z.push_back(z_center);
        local_value.push_back(value);
    }

    // MPI setup
    MPI_Comm comm = pmesh->GetComm();
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Gather all data on rank 0
    std::vector<double> all_x, all_y, all_z, all_value;
    int local_num_elements = local_x.size();
    std::vector<int> all_num_elements(size);
    std::vector<int> displs(size);

    MPI_Gather(&local_num_elements, 1, MPI_INT, 
        all_num_elements.data(), 1, MPI_INT, 0, comm);

    if (rank == 0)
    {
        int total_elements = 0;
        displs[0] = 0;
        for (int i = 0; i < size; ++i)
        {
            total_elements += all_num_elements[i];
            if (i > 0)
            {
                displs[i] = displs[i - 1] + all_num_elements[i - 1];
            }
        }

        all_x.resize(total_elements);
        all_y.resize(total_elements);
        all_z.resize(total_elements);
        all_value.resize(total_elements);
    }

    MPI_Gatherv(local_x.data(), local_num_elements, MPI_DOUBLE, 
        all_x.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(local_y.data(), local_num_elements, MPI_DOUBLE, 
        all_y.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(local_z.data(), local_num_elements, MPI_DOUBLE, 
        all_z.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(local_value.data(), local_num_elements, MPI_DOUBLE, 
        all_value.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);

    // Write the data to a file in a human-readable format on rank 0
    if (rank == 0)
    {
        std::ofstream ofs("element_centers_scalar.txt");
        for (size_t i = 0; i < all_x.size(); ++i)
        {
            ofs << all_x[i] << " " << all_y[i] << " " << all_z[i] << " " << all_value[i] << std::endl;
        }
        ofs.close();
    }

    // // Write the data to a file in a human-readable format on rank 0
    // if (rank == 0)
    // {
   
    //   // std::string fname = "element_centers_" + std::to_string(step) + ".txt";
    //   std::string fname = "element_centers_scalar.txt";
    //   FILE *f = NULL;
    //   f = fopen(fname.c_str(), "w");
    //   if (!f)
    //   {
    //     std::cerr << "Error opening file " << fname << std::endl;
    //     MPI_Abort(MPI_COMM_WORLD,1);
    //   }

    //   // Write header only if not restarting
    //   fprintf(f, "3D Taylor Green Vortex\n");
    //   fprintf(f, "order = %d\n", ctx.order);
    //   fprintf(f, "===================================================================");
    //   fprintf(f, "========================================================================\n");
    //   fprintf(f, "            x                      y                      z         ");
    //   fprintf(f, "            p     \n");

    //   // Write data with aligned columns
    //   for (size_t i = 0; i < all_x.size(); ++i)
    //   {
    //     // Write the initial data point
    //     fprintf(f, "%20.16e %20.16e %20.16e %20.16e \n", all_x[i], all_y[i],all_z[i],
    //                                                                     all_value[i]);
    //   }

    //   fflush(f);
    //   fflush(stdout);
    // 
    // }
}

/*
void ComputeElementCenterValues(ParGridFunction* sol, ParMesh* pmesh)
{
    // Local arrays to store the data
    std::vector<double> local_x, local_y, local_z, local_value;
    Vector velx, vely, velz;

    FiniteElementSpace *fes = sol->FESpace();

    // Set the integration point to the center of the reference element
    IntegrationPoint ip;
    ip.Set3(0.5, 0.5, 0.5);  // Center of the reference element

    // Loop over local elements
    for (int i = 0; i < pmesh->GetNE(); i++)
    {
        // Get the element transformation
        ElementTransformation *Trans = pmesh->GetElementTransformation(i);

        // Evaluate the solution at the element center
        Trans->SetIntPoint(&ip);
        // double value = sol->GetValue(*Trans, ip);
        double value = sol->GetValue(*Trans, ip, 1);

        // Transform the reference point to physical coordinates
        Vector phys_coords(3);
        Trans->Transform(ip, phys_coords);

        double x_center = phys_coords[0];
        double y_center = phys_coords[1];
        double z_center = phys_coords[2];

        // Store the data
        local_x.push_back(x_center);
        local_y.push_back(y_center);
        local_z.push_back(z_center);
        local_value.push_back(value);
    }

    // MPI setup
    MPI_Comm comm = pmesh->GetComm();
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Gather all data on rank 0
    std::vector<double> all_x, all_y, all_z, all_value;
    int local_num_elements = local_x.size();
    std::vector<int> all_num_elements(size);
    std::vector<int> displs(size);

    MPI_Gather(&local_num_elements, 1, MPI_INT, 
        all_num_elements.data(), 1, MPI_INT, 0, comm);

    if (rank == 0)
    {
        int total_elements = 0;
        displs[0] = 0;
        for (int i = 0; i < size; ++i)
        {
            total_elements += all_num_elements[i];
            if (i > 0)
            {
                displs[i] = displs[i - 1] + all_num_elements[i - 1];
            }
        }

        all_x.resize(total_elements);
        all_y.resize(total_elements);
        all_z.resize(total_elements);
        all_value.resize(total_elements);
    }

    MPI_Gatherv(local_x.data(), local_num_elements, MPI_DOUBLE, 
        all_x.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(local_y.data(), local_num_elements, MPI_DOUBLE, 
        all_y.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(local_z.data(), local_num_elements, MPI_DOUBLE, 
        all_z.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(local_value.data(), local_num_elements, MPI_DOUBLE, 
        all_value.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);

    // Write the data to a file in a human-readable format on rank 0
    if (rank == 0)
    {
        std::ofstream ofs("element_centers.txt");
        for (size_t i = 0; i < all_x.size(); ++i)
        {
            ofs << all_x[i] << " " << all_y[i] << " " << all_z[i] << " " << all_value[i] << std::endl;
        }
        ofs.close();
    }
    
}
*/

void SaveCheckpoint(ParMesh *pmesh, ParGridFunction *u_gf, ParGridFunction *p_gf,
                    double t, int step, int myid)
{
    // Helper function to print ParGridFunction properties
    auto PrintGridFunctionProperties = [](const ParGridFunction &gf, const std::string &name, int myid)
    {
        const ParFiniteElementSpace *fes = gf.ParFESpace();
        std::cout << "Processor " << myid << " - Grid Function: " << name << std::endl;
        std::cout << "  VDim: " << fes->GetVDim() << std::endl;
        std::cout << "  NDofs: " << fes->GetNDofs() << std::endl;
        std::cout << "  Global VSize: " << fes->GlobalVSize() << std::endl;
        std::cout << "  Local VSize: " << gf.Size() << std::endl;
        std::cout << "  Norm L2: " << gf.Norml2() << std::endl;
        std::cout << "  Mesh Elements: " << fes->GetMesh()->GetNE() << std::endl;
        std::cout << "  ParMesh Group Size: " << fes->GetParMesh()->GetNGroups() << std::endl;
    };

    // Initialize error flag
    bool error_flag = false;

    // Save the mesh
    std::string mesh_fname = MakeParFilename("tgv-checkpoint.mesh.", myid);

    // Open an output file stream for the mesh
    std::ofstream mesh_ofs(mesh_fname.c_str());
    mesh_ofs.precision(16);

    if (!mesh_ofs.good())
    {
        error_flag = true;
    }
    else
    {
        // Use ParPrint to write the mesh to the file
        pmesh->ParPrint(mesh_ofs);
        mesh_ofs.close();
    }

    // Save the time and step number (only on root processor)
    if (myid == 0)
    {
        std::ofstream t_ofs("tgv-checkpoint.time");
        t_ofs.precision(16);
        if (!t_ofs.good())
        {
            error_flag = true;
        }
        else
        {
            t_ofs << t << std::endl;
            t_ofs << step << std::endl;
            t_ofs.close();
        }
    }

    // Print properties of u_gf and p_gf before saving
    PrintGridFunctionProperties(*u_gf, "u_gf", myid);
    PrintGridFunctionProperties(*p_gf, "p_gf", myid);

    // Save the velocity field
    std::string u_fname = MakeParFilename("tgv-checkpoint.u.", myid);
    std::ofstream u_ofs(u_fname.c_str());
    u_ofs.precision(16);

    if (!u_ofs.good())
    {
        error_flag = true;
    }
    else
    {
        u_gf->Save(u_ofs);
        u_ofs.close();
    }

    // Save the pressure field
    std::string p_fname = MakeParFilename("tgv-checkpoint.p.", myid);
    std::ofstream p_ofs(p_fname.c_str());
    p_ofs.precision(16);

    if (!p_ofs.good())
    {
        error_flag = true;
    }
    else
    {
        p_gf->Save(p_ofs);
        p_ofs.close();
    }

    // Use MPI to check if any process has encountered an error
    int global_error_flag = 0;
    int local_error_flag = error_flag ? 1 : 0;
    MPI_Allreduce(&local_error_flag, &global_error_flag, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);

    if (global_error_flag)
    {
        if (myid == 0)
        {
            std::cerr << "Error occurred during checkpoint saving. Aborting." << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Compute norms before reloading
    double u_norm = u_gf->Norml2();
    double p_norm = p_gf->Norml2();

    if (myid == 0)
    {
        std::cout << "Saving checkpoint: t = " << t << ", step = " << step << std::endl;
        std::cout << "u_gf Norml2 = " << u_norm << ", p_gf Norml2 = " << p_norm << std::endl;
    }

    // Reload the data and compute norms
    // Open input file streams for the velocity and pressure fields
    std::ifstream u_ifs(u_fname.c_str());
    std::ifstream p_ifs(p_fname.c_str());

    // Check for errors during file opening
    error_flag = false;
    if (!u_ifs.good())
    {
        error_flag = true;
    }
    if (!p_ifs.good())
    {
        error_flag = true;
    }

    // Use MPI to check if any process has encountered an error
    local_error_flag = error_flag ? 1 : 0;
    MPI_Allreduce(&local_error_flag, &global_error_flag, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);

    if (global_error_flag)
    {
        if (myid == 0)
        {
            std::cerr << "Error opening checkpoint files for reloading. Aborting." << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Create temporary grid functions by reading from the files
    ParGridFunction temp_u_gf(pmesh, u_ifs);
    ParGridFunction temp_p_gf(pmesh, p_ifs);

    u_ifs.close();
    p_ifs.close();

    // Print properties of temp_u_gf and temp_p_gf after loading
    PrintGridFunctionProperties(temp_u_gf, "temp_u_gf", myid);
    PrintGridFunctionProperties(temp_p_gf, "temp_p_gf", myid);

    // Compute norms after reloading
    double temp_u_norm = temp_u_gf.Norml2();
    double temp_p_norm = temp_p_gf.Norml2();

    // Compare norms to check if they match
    double u_diff = std::abs(u_norm - temp_u_norm);
    double p_diff = std::abs(p_norm - temp_p_norm);
    const double tol = 1e-8; // Adjusted tolerance for floating-point comparison

    if (myid == 0)
    {
        std::cout << "After reloading: u_gf Norml2 = " << temp_u_norm
                  << ", p_gf Norml2 = " << temp_p_norm << std::endl;

        if (u_diff < tol && p_diff < tol)
        {
            std::cout << "Checkpoint verification successful: norms match." << std::endl;
        }
        else
        {
            std::cerr << "Checkpoint verification failed: norms do not match!" << std::endl;
            std::cerr << "Difference in u_gf Norml2: " << u_diff << std::endl;
            std::cerr << "Difference in p_gf Norml2: " << p_diff << std::endl;
        }
    }

    // Ensure all processes reach this point before proceeding
    MPI_Barrier(MPI_COMM_WORLD);
}

bool LoadCheckpoint(ParMesh *&pmesh, ParGridFunction *&u_gf, ParGridFunction *&p_gf,
                    NavierSolver *&flowsolver, double &t, int &step, int myid, const s_NavierContext &ctx)
{
    // Helper function to print ParGridFunction properties
    auto PrintGridFunctionProperties = [](const ParGridFunction &gf, const std::string &name, int myid)
    {
        const ParFiniteElementSpace *fes = gf.ParFESpace();
        std::cout << "Processor " << myid << " - Grid Function: " << name << std::endl;
        std::cout << "  VDim: " << fes->GetVDim() << std::endl;
        std::cout << "  NDofs: " << fes->GetNDofs() << std::endl;
        std::cout << "  Global VSize: " << fes->GlobalVSize() << std::endl;
        std::cout << "  Local VSize: " << gf.Size() << std::endl;
        std::cout << "  Norm L2: " << gf.Norml2() << std::endl;
        std::cout << "  Mesh Elements: " << fes->GetMesh()->GetNE() << std::endl;
        std::cout << "  ParMesh Group Size: " << fes->GetParMesh()->GetNGroups() << std::endl;
    };

    // Check if all checkpoint files exist
    bool all_files_exist = true;

    // Declare filenames at the beginning
    std::string mesh_fname = MakeParFilename("tgv-checkpoint.mesh.", myid);
    std::string u_fname = MakeParFilename("tgv-checkpoint.u.", myid);
    std::string p_fname = MakeParFilename("tgv-checkpoint.p.", myid);

    // Check mesh file
    std::ifstream mesh_ifs(mesh_fname);
    if (!mesh_ifs.good())
    {
        all_files_exist = false;
    }
    mesh_ifs.close();

    // Check time file (only on root processor)
    if (myid == 0)
    {
        std::ifstream t_ifs("tgv-checkpoint.time");
        if (!t_ifs.good())
        {
            all_files_exist = false;
        }
        t_ifs.close();
    }

    // Check velocity field file
    std::ifstream u_ifs(u_fname);
    if (!u_ifs.good())
    {
        all_files_exist = false;
    }
    u_ifs.close();

    // Check pressure field file
    std::ifstream p_ifs(p_fname);
    if (!p_ifs.good())
    {
        all_files_exist = false;
    }
    p_ifs.close();

    // Use MPI to ensure all processors agree on the existence of files
    int all_files_exist_int = all_files_exist ? 1 : 0;
    int global_all_files_exist_int;
    MPI_Allreduce(&all_files_exist_int, &global_all_files_exist_int, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

    if (global_all_files_exist_int == 0)
    {
        // Files not found
        return false;
    }

    // Now proceed to load the files
    // Read the mesh
    std::ifstream mesh_ifs2(mesh_fname);
    pmesh = new ParMesh(MPI_COMM_WORLD, mesh_ifs2);
    mesh_ifs2.close();

    // Read the time and step number (only on root processor)
    if (myid == 0)
    {
        std::ifstream t_ifs2("tgv-checkpoint.time");
        t_ifs2 >> t;
        t_ifs2 >> step;
        t_ifs2.close();
        std::cout << "Loaded time t = " << t << ", step = " << step << std::endl;
    }

    // Broadcast the time and step number to all processors
    MPI_Bcast(&t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&step, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Create the flow solver with the loaded mesh
    flowsolver = new NavierSolver(pmesh, ctx.order, ctx.kinvis);
    flowsolver->EnablePA(ctx.pa);
    flowsolver->EnableNI(ctx.ni);

    // Get the existing grid functions from the flowsolver
    u_gf = flowsolver->GetCurrentVelocity();
    p_gf = flowsolver->GetCurrentPressure();

    // Load the velocity field into a temporary grid function
    std::ifstream u_ifs2(u_fname);
    if (!u_ifs2.good())
    {
        if (myid == 0)
        {
            std::cerr << "Error opening velocity checkpoint file " << u_fname << std::endl;
        }
        return false;
    }
    ParGridFunction temp_u_gf(pmesh, u_ifs2);
    u_ifs2.close();

    // Load the pressure field into a temporary grid function
    std::ifstream p_ifs2(p_fname);
    if (!p_ifs2.good())
    {
        if (myid == 0)
        {
            std::cerr << "Error opening pressure checkpoint file " << p_fname << std::endl;
        }
        return false;
    }
    ParGridFunction temp_p_gf(pmesh, p_ifs2);
    p_ifs2.close();

    // Copy the data from the temporary grid functions to the existing ones
    *u_gf = temp_u_gf;
    *p_gf = temp_p_gf;

    // Print norms for debugging
    double u_norm = u_gf->Norml2();
    double p_norm = p_gf->Norml2();

    if (myid == 0)
    {
        std::cout << "Loaded checkpoint: u_gf Norml2 = " << u_norm
                  << ", p_gf Norml2 = " << p_norm << std::endl;
    }

    // Set up the flow solver to initialize internal structures
    flowsolver->Setup(ctx.dt);

    // Print properties of u_gf and p_gf before saving
    PrintGridFunctionProperties(*u_gf, "u_gf", myid);
    PrintGridFunctionProperties(*p_gf, "p_gf", myid);

    return true;
}
