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

// TODO: 
// 1. Add option to set cycle for saving checkpoint files and post process
// files seperatelty
// 2. Store Element data at center in binary for effiecnecy?
// 3. Compute fft of data directly?

#include "navier_solver.hpp"
#include <fstream>
#include <algorithm>
#include <iostream>
#include <string>

using namespace mfem;
using namespace navier;

struct s_NavierContext
{
   int element_subdivisions = 0;
   int element_subdivisions_parallel = 0;
   int order = 2;
   real_t reynum = 1600;
   real_t kinvis = 1.0 / reynum;
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
   int checkpoint_cycle = 100;
   int element_center_cycle = 100;
   int data_dump_cycle = 100;

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

void ComputeElementCenterValues(ParGridFunction *sol, ParMesh *pmesh, int step, double time, const std::string &suffix);
void ComputeElementCenterValuesScalar(ParGridFunction *sol, ParMesh *pmesh,int step, double time);

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
   args.AddOption(&ctx.reynum, "-Re", "--Renolds-number", "Reynolds Number.");
   args.AddOption(&ctx.checkpoint_cycle, "-cpc", "--Checkpoint-Cycle", "Checkpoint Cycle.");
   args.AddOption(&ctx.element_center_cycle, "-ecc", "--Element-Center-Cycle", "Element Center Cycle.");
   args.AddOption(&ctx.data_dump_cycle, "-ddc", "--Data-Dump-Cycle", "Data Dump Cycle.");
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

      // Initialize as mesh
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

      if (Mpi::Root())
      {
         VerifyPeriodicMesh(mesh);
      }

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

      if (Mpi::Root() && (ctx.element_subdivisions > 1))
      {
         mfem::out << "Refining the mesh... " << std::endl;
      }
      // Serial Mesh refinement
      for (int lev = 0; lev < ctx.element_subdivisions; lev++)
      {
         mesh->UniformRefinement();
      }

      // Create the parallel mesh
      pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
      pmesh->Finalize(true);
      // Parallel Mesh refinement
      for (int lev = 0; lev < ctx.element_subdivisions_parallel; lev++)
      {
         pmesh->UniformRefinement();
      }

      delete mesh;
      delete init_mesh;

      if (Mpi::Root())
      {
         mfem::out << "Creating the flowsolver. " << std::endl;
      }

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

      if (Mpi::Root())
      {
         mfem::out << "Done setting up the flowsolver. " << std::endl;
      }

      ComputeElementCenterValues(u_gf, pmesh, step, t,"Velocity");
      // ComputeElementCenterValuesScalar(u_gf, pmesh,step, t);
   }

   int nel = pmesh->GetGlobalNE();
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
      std::string paraview_dir = std::string("ParaviewData_") 
                                               + "Re" + std::to_string(static_cast<int>(ctx.reynum)) 
                                               + "NumPtsPerDir" +std::to_string(ctx.num_pts) 
                                               + "Order" + std::to_string(ctx.order)
                                               + "/tgv_output_paraview";

      pvdc = new ParaViewDataCollection(paraview_dir, pmesh);
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
         std::string visit_dir = std::string("VisitData_") 
                                                  + "Re" + std::to_string(static_cast<int>(ctx.reynum)) 
                                                  + "NumPtsPerDir" +std::to_string(ctx.num_pts) 
                                                  + "P" + std::to_string(ctx.order)
                                                  + "/tgv_output_visit";
         dc = new VisItDataCollection(visit_dir,pmesh);
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

   // Compute the cfl
   real_t cfl;
   cfl = flowsolver->ComputeCFL(*u_gf, ctx.dt);

   std::string fname = std::string("tgv_out_") 
                                            + "Re" + std::to_string(static_cast<int>(ctx.reynum)) 
                                            + "NumPtsPerDir" +std::to_string(ctx.num_pts) 
                                            + "P" + std::to_string(ctx.order)
                                            + ".txt";
   FILE *f = NULL;

   if (Mpi::Root())
   {
      int nel1d = static_cast<int>(std::round(pow(nel, 1.0 / 3.0)));
      int ngridpts = p_gf->ParFESpace()->GlobalVSize();
      printf("%11s %11s %11s %11s %11s %11s %11s\n", "Time", "dt", "u_inf", "p_inf", "ke", "enstrophy", "CFL");
      printf("%.5E %.5E %.5E %.5E %.5E %.5E %.5E\n", t, ctx.dt, u_inf, p_inf, ke, enstrophy, cfl);

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
          fprintf(f, "Reynolds Number = %d\n", static_cast<int>(ctx.reynum));
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
      cfl = flowsolver->ComputeCFL(*u_gf, ctx.dt);

      if ((step - initial_step) % ctx.data_dump_cycle == 0 || last_step)
      {
         // If restarting, skip the first saved checkpoint
         if (!(ctx.restart && step == 0 && restart_files_found))
         {
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

         }
      }

      if ((step - initial_step) % ctx.element_center_cycle == 0 || last_step)
      {
         // If restarting, skip the first saved checkpoint
         if (!(ctx.restart && step == 0 && restart_files_found))
         {
            flowsolver->ComputeCurl3D(*u_gf, w_gf);

            ComputeElementCenterValues( u_gf, pmesh, step + initial_step, t, "Velocity");
            ComputeElementCenterValues(&w_gf, pmesh, step + initial_step, t, "Vorticity");

            if (Mpi::Root())
            {
               std::cout << "\nOutput element center file saved at cycle " << step + initial_step << "." << std::endl;
            }

         }
      }

      if ((step - initial_step) % ctx.checkpoint_cycle == 0 || last_step)
      {
         // If restarting, skip the first saved checkpoint
         if (!(ctx.restart && step == 0 && restart_files_found))
         {
            // Save the checkpoint files
            SaveCheckpoint(pmesh, u_gf, p_gf, t, step + initial_step, myid);
            if (Mpi::Root())
            {
               std::cout << "\nCheckpoint file saved at cycle " << step + initial_step << "." << std::endl;
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
           printf("%.5E %.5E %.5E %.5E %.5E %.5E %.5E\n", t, ctx.dt, u_inf, p_inf, ke, enstrophy, cfl);
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

void ComputeElementCenterValues(ParGridFunction* sol,
                                ParMesh* pmesh,
                                int step,
                                double time,
                                const std::string &suffix)
{
    // MPI setup
    MPI_Comm comm = pmesh->GetComm();
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Construct the main directory name with suffix
    std::string main_dir = std::string("ElementCenters") + suffix
                                             + "_Re" + std::to_string(static_cast<int>(ctx.reynum)) 
                                             + "NumPtsPerDir" +std::to_string(ctx.num_pts) 
                                             + "P" + std::to_string(ctx.order);

    // Create subdirectory for this cycle step
    std::string cycle_dir = main_dir + "/cycle_" + std::to_string(step);

    // Construct the filename inside the cycle directory
    std::string fname = cycle_dir + "/element_centers_" + std::to_string(step) + ".txt";

    if (rank == 0){
      {
        // Create main directory for element centers with suffix
        std::string command = "mkdir -p " + main_dir;
        int ret = system(command.c_str());
        if (ret != 0 && rank == 0)
        {
            std::cerr << "Error creating " << main_dir << " directory!" << std::endl;
        }
      }

      {
          std::string command = "mkdir -p " + cycle_dir;
          int ret = system(command.c_str());
          if (ret != 0 && rank == 0)
          {
              std::cerr << "Error creating " << cycle_dir << " directory!" << std::endl;
          }
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Local arrays to store data
    std::vector<double> local_x, local_y, local_z;
    std::vector<double> local_velx, local_vely, local_velz;

    FiniteElementSpace *fes = sol->FESpace();

    // Set the integration point to the center of the reference element
    IntegrationPoint ip;
    ip.Set3(0.5, 0.5, 0.5); // Center of element

    // Loop over local elements
    for (int i = 0; i < pmesh->GetNE(); i++)
    {
        // Get element transformation
        ElementTransformation *Trans = pmesh->GetElementTransformation(i);

        // Evaluate at center
        Trans->SetIntPoint(&ip);

        // Get vector dimension (should be 3 for velocity)
        int vdim = fes->GetVDim();
        Vector u_val(vdim);
        
        // Get vector value at ip
        sol->GetVectorValue(*Trans, ip, u_val);

        double u_x = u_val(0);
        double u_y = u_val(1);
        double u_z = u_val(2);
        
        // Physical coordinates of element center
        Vector phys_coords(3);
        Trans->Transform(ip, phys_coords);

        double x_center = phys_coords[0];
        double y_center = phys_coords[1];
        double z_center = phys_coords[2];

        // Store data locally
        local_x.push_back(x_center);
        local_y.push_back(y_center);
        local_z.push_back(z_center);
        local_velx.push_back(u_x);
        local_vely.push_back(u_y);
        local_velz.push_back(u_z);
    }

    // Gather all data on rank 0
    int local_num_elements = (int)local_x.size();
    std::vector<int> all_num_elements(size);
    std::vector<int> displs(size);

    MPI_Gather(&local_num_elements, 1, MPI_INT, 
               all_num_elements.data(), 1, MPI_INT, 0, comm);

    std::vector<double> all_x, all_y, all_z;
    std::vector<double> all_velx, all_vely, all_velz;

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

    if (rank == 0)
    {
        FILE *f = fopen(fname.c_str(), "w");
        if (!f)
        {
            std::cerr << "Error opening file " << fname << std::endl;
            MPI_Abort(MPI_COMM_WORLD,1);
        }

        // Write header
        fprintf(f, "3D Taylor Green Vortex\n");
        fprintf(f, "Order = %d\n", ctx.order);
        fprintf(f, "Step = %d\n", step);
        fprintf(f, "Time = %e\n", time);
        fprintf(f, "===================================================================");
        fprintf(f, "==========================================================================\n");
        fprintf(f, "            x                      y                      z         ");
        fprintf(f, "            vecx                   vecy                   velc\n");

        // Write data
        for (size_t i = 0; i < all_x.size(); ++i)
        {
            fprintf(f, "%20.16e %20.16e %20.16e %20.16e %20.16e %20.16e\n",
                    all_x[i], all_y[i], all_z[i],
                    all_velx[i], all_vely[i], all_velz[i]);
        }

        fflush(f);
        fclose(f);
    }
}


void ComputeElementCenterValuesScalar(ParGridFunction* sol, ParMesh* pmesh, int step, double time)
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
      std::string fname = "element_centers_scalar_" + std::to_string(step) + ".txt";
      FILE *f = NULL;
      f = fopen(fname.c_str(), "w");
      if (!f)
      {
        std::cerr << "Error opening file " << fname << std::endl;
        MPI_Abort(MPI_COMM_WORLD,1);
      }

      // Write header only if not restarting
      fprintf(f, "3D Taylor Green Vortex\n");
      fprintf(f, "Order = %d\n", ctx.order);
      fprintf(f, "Step = %d\n", step);
      fprintf(f, "Time = %d\n", time);
      fprintf(f, "===================================================================");
      fprintf(f, "========================================================================\n");
      fprintf(f, "            x                      y                      z         ");
      fprintf(f, "            p     \n");

      // Write data with aligned columns
      for (size_t i = 0; i < all_x.size(); ++i)
      {
        // Write the initial data point
        fprintf(f, "%20.16e %20.16e %20.16e %20.16e \n", all_x[i], all_y[i],all_z[i],
                                                                        all_value[i]);
      }

      fflush(f);
      fflush(stdout);
    
    }
}


void SaveCheckpoint(ParMesh *pmesh, ParGridFunction *u_gf, ParGridFunction *p_gf,
                    double t, int step, int myid)
{
    // Construct the main directory name with suffix
    std::string main_dir = std::string("CheckPoint_")
                                             + "Re" + std::to_string(static_cast<int>(ctx.reynum)) 
                                             + "NumPtsPerDir" +std::to_string(ctx.num_pts) 
                                             + "P" + std::to_string(ctx.order);

    // Create subdirectory for this cycle step
    std::string cycle_dir = main_dir + "/cycle_" + std::to_string(step);

    if (myid == 0){
        // Create main checkpoint directory
        {
            std::string command = "mkdir -p " + main_dir;
            int ret = system(command.c_str());
            if (ret != 0 && myid == 0)
            {
                std::cerr << "Error creating tgv_check_point directory!" << std::endl;
            }
        }

        {
            std::string command = "mkdir -p " + cycle_dir;
            int ret = system(command.c_str());
            if (ret != 0 && myid == 0)
            {
                std::cerr << "Error creating " << cycle_dir << " directory!" << std::endl;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Adjust filenames to be in the cycle directory
    std::string mesh_fname = cycle_dir + "/tgv-checkpoint.mesh." + std::to_string(myid);
    std::string u_fname    = cycle_dir + "/tgv-checkpoint.u." + std::to_string(myid);
    std::string p_fname    = cycle_dir + "/tgv-checkpoint.p." + std::to_string(myid);

    bool error_flag = false;

    // Save the mesh
    {
        std::ofstream mesh_ofs(mesh_fname.c_str());
        mesh_ofs.precision(16);

        if (!mesh_ofs.good())
        {
            error_flag = true;
        }
        else
        {
            pmesh->ParPrint(mesh_ofs);
            mesh_ofs.close();
        }
    }

    // Save the time and step number (only on root processor)
    if (myid == 0)
    {
        std::ofstream t_ofs((cycle_dir + "/tgv-checkpoint.time").c_str());
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

    // Save the velocity field
    {
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
    }

    // Save the pressure field
    {
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

    // Compute norms before reloading (original fields)
    double u_norm = u_gf->Norml2();
    double p_norm = p_gf->Norml2();

    if (myid == 0)
    {
        std::cout << "Checkpoint saved at step " << step << ", time " << t << std::endl;
        std::cout << "Original fields: u_gf Norml2 = " << u_norm << ", p_gf Norml2 = " << p_norm << std::endl;
    }

    // Reload the data and compute norms
    {
        std::ifstream u_ifs(u_fname.c_str());
        std::ifstream p_ifs(p_fname.c_str());

        // Check for errors during file opening
        bool reload_error = false;
        if (!u_ifs.good()) { reload_error = true; }
        if (!p_ifs.good()) { reload_error = true; }

        int reload_error_flag = reload_error ? 1 : 0;
        int global_reload_error_flag;
        MPI_Allreduce(&reload_error_flag, &global_reload_error_flag, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);

        if (global_reload_error_flag)
        {
            if (myid == 0)
            {
                std::cerr << "Error opening checkpoint files for reloading. Aborting." << std::endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        ParGridFunction temp_u_gf(pmesh, u_ifs);
        ParGridFunction temp_p_gf(pmesh, p_ifs);

        u_ifs.close();
        p_ifs.close();

        double temp_u_norm = temp_u_gf.Norml2();
        double temp_p_norm = temp_p_gf.Norml2();

        if (myid == 0)
        {
            std::cout << "After reloading in SaveCheckpoint: u_gf Norml2 = " << temp_u_norm
                      << ", p_gf Norml2 = " << temp_p_norm << std::endl;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

// Helper function to find the last checkpoint step if none is provided
// This scans "tgv_check_point/cycle_XXXX" directories and finds the maximum step number.
int FindLastCheckpointStep()
{
    // We'll rely on a shell command to list directories; adapt as needed.
    // "ls tgv_check_point | grep cycle_" will list cycle directories.
    // We'll parse the largest number from "cycle_<step>".

    // Construct the main directory name with suffix
    std::string main_dir = std::string("CheckPoint_")
                                             + "Re" + std::to_string(static_cast<int>(ctx.reynum)) 
                                             + "NumPtsPerDir" +std::to_string(ctx.num_pts) 
                                             + "P" + std::to_string(ctx.order);

    // Using popen to run shell command and read output
    std::string command = "ls " + main_dir + " | grep cycle_ | sed 's/cycle_//' | sort -n | tail -1";
    FILE *pipe = popen(command.c_str(),"r");
    if (!pipe) return -1;
    char buffer[128];
    std::string result = "";
    while (fgets(buffer, 128, pipe) != NULL)
    {
        result += buffer;
    }
    pclose(pipe);

    // Trim whitespace
    result.erase(std::remove_if(result.begin(), result.end(), ::isspace), result.end());
    if (result.empty())
    {
        return -1; // no checkpoints
    }

    return std::stoi(result);
}

bool LoadCheckpoint(ParMesh *&pmesh, ParGridFunction *&u_gf, ParGridFunction *&p_gf,
                    NavierSolver *&flowsolver, double &t, int &step, int myid, const s_NavierContext &ctx)
{
    // If no step given, find last checkpoint step
    int provided_step = -1; // Assume no step provided
    if (provided_step < 0)
    {
        int last_step = -1;
        if (myid == 0)
        {
          last_step = FindLastCheckpointStep();
        }

        // now broadcast to every rank
        MPI_Bcast(&last_step, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (last_step < 0)
        {
            // No checkpoints found
            return false;
        }
        provided_step = last_step;
    }

    // Construct the main directory name with suffix
    std::string main_dir = std::string("CheckPoint_")
                                             + "Re" + std::to_string(static_cast<int>(ctx.reynum)) 
                                             + "NumPtsPerDir" +std::to_string(ctx.num_pts) 
                                             + "P" + std::to_string(ctx.order);

    // Construct directory for given step
    std::string cycle_dir = main_dir + "/cycle_" + std::to_string(provided_step);

    std::string mesh_fname = cycle_dir + "/tgv-checkpoint.mesh." + std::to_string(myid);
    std::string u_fname = cycle_dir + "/tgv-checkpoint.u." + std::to_string(myid);
    std::string p_fname = cycle_dir + "/tgv-checkpoint.p." + std::to_string(myid);

    // Check files
    {
        std::ifstream mesh_ifs(mesh_fname); 
        if (!mesh_ifs.good()) return false; 
        mesh_ifs.close();
    }

    if (myid == 0)
    {
        std::ifstream t_ifs((cycle_dir + "/tgv-checkpoint.time").c_str());
        if (!t_ifs.good()) return false;
        t_ifs.close();
    }

    {
        std::ifstream u_ifs(u_fname);
        if (!u_ifs.good()) return false;
        u_ifs.close();
    }

    {
        std::ifstream p_ifs(p_fname);
        if (!p_ifs.good()) return false;
        p_ifs.close();
    }

    int all_files_exist_int = 1;
    int global_all_files_exist_int;
    MPI_Allreduce(&all_files_exist_int, &global_all_files_exist_int, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    if (global_all_files_exist_int == 0)
    {
        return false;
    }

    // Load the mesh
    {
        std::ifstream mesh_ifs2(mesh_fname);
        pmesh = new ParMesh(MPI_COMM_WORLD, mesh_ifs2);
        mesh_ifs2.close();
    }

    // Read the time and step number (only on root processor)
    if (myid == 0)
    {
        std::ifstream t_ifs2((cycle_dir + "/tgv-checkpoint.time").c_str());
        t_ifs2 >> t;
        t_ifs2 >> step;
        t_ifs2.close();
        std::cout << "Loaded time t = " << t << ", step = " << step << std::endl;
    }

    MPI_Bcast(&t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&step, 1, MPI_INT, 0, MPI_COMM_WORLD);

    flowsolver = new NavierSolver(pmesh, ctx.order, ctx.kinvis);
    flowsolver->EnablePA(ctx.pa);
    flowsolver->EnableNI(ctx.ni);

    u_gf = flowsolver->GetCurrentVelocity();
    p_gf = flowsolver->GetCurrentPressure();

    {
        std::ifstream u_ifs2(u_fname);
        ParGridFunction temp_u_gf(pmesh, u_ifs2);
        u_ifs2.close();

        std::ifstream p_ifs2(p_fname);
        ParGridFunction temp_p_gf(pmesh, p_ifs2);
        p_ifs2.close();

        *u_gf = temp_u_gf;
        *p_gf = temp_p_gf;
    }

    flowsolver->Setup(ctx.dt);

    // Compute norms after loading
    double u_norm = u_gf->Norml2();
    double p_norm = p_gf->Norml2();

    if (myid == 0)
    {
        std::cout << "After loading from checkpoint in LoadCheckpoint: u_gf Norml2 = "
                  << u_norm << ", p_gf Norml2 = " << p_norm << std::endl;
    }

    return true;
}


