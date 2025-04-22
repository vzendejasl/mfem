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
//
// Navier double shear layer example
//
// Solve the double shear problem in the following configuration.
//
//       +-------------------+
//       |                   |
//       |      u0 = ua      |
//       |                   |
//  -------------------------------- y = 0.5
//       |                   |
//       |      u0 = ub      |
//       |                   |
//       +-------------------+
//
// The initial condition u0 is chosen to be a varying velocity in the y
// direction. It includes a perturbation at x = 0.5 which leads to an
// instability and the dynamics of the flow. The boundary conditions are fully
// periodic.

#include "navier_solver.hpp"
#include <fstream>

using namespace mfem;
using namespace navier;

struct s_NavierContext
{
   int order = 7;
   // real_t kinvis = 1.0 / 100000.0;
   real_t kinvis = 1.0 / 500.0;
   real_t t_final = 10 * 1e1;
   // real_t t_final = 10 * 1e-3;
   // real_t dt = 1e-3;
   real_t dt = 5e-4;
} ctx;

void vel_shear_ic(const Vector &x, real_t t, Vector &u)
{
   real_t xi = x(0);
   real_t yi = x(1);
   real_t zi = x(2);

   real_t rho = 30.0;
   real_t delta = 0.05;

   // if (yi <= 0.5)
   // {
   //    u(0) = tanh(rho * (yi - 0.25));
   // }
   // else
   // {
   //    u(0) = tanh(rho * (0.75 - yi));
   // }
   
   u(0) = tanh(rho * (yi - 1.0));

   u(1) = delta * sin(4.0 * M_PI * x(0)) * sin(M_PI * x(1))*cos(M_PI*x(2));
   u(2) = 0.0; 
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   int serial_refinements = 0;
   // int serial_refinements = 2;

   // Mesh *mesh = new Mesh("../../data/periodic-square.mesh");

   // Initialize as mesh
   Mesh *init_mesh;

   real_t length = 2.0;
   real_t lengthz = 1.0;
   int num_pts = 12;
   int num_ptsz = 6;
   // init_mesh = new Mesh(Mesh::MakeCartesian2D(num_pts,
   //                                            num_pts,
   //                                            Element::QUADRILATERAL,
   //                                            true,
   //                                            length,
   //                                            length, false));

   init_mesh = new Mesh(Mesh::MakeCartesian3D(num_pts,
                                              num_pts,
                                              num_ptsz,
                                              Element::HEXAHEDRON,
                                              length,
                                              length,
                                              lengthz, false));

   Vector x_translation({length, 0.0, 0.0});
   Vector y_translation({0.0, length, 0.0});
   Vector z_translation({0.0, 0.0, length});

   // std::vector<Vector> translations = {x_translation, y_translation, z_translation};
   std::vector<Vector> translations = {x_translation, z_translation};

   Mesh *mesh = new Mesh(Mesh::MakePeriodic(*init_mesh, init_mesh->CreatePeriodicVertexMapping(translations)));


   mesh->EnsureNodes();
   GridFunction *nodes = mesh->GetNodes();
   // *nodes -= -1.0;
   // *nodes /= 2.0;

   for (int i = 0; i < serial_refinements; ++i)
   {
      mesh->UniformRefinement();
   }

   if (Mpi::Root())
   {
      std::cout << "Number of elements: " << mesh->GetNE() << std::endl;
   }

   auto *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // Create the flow solver.
   NavierSolver flowsolver(pmesh, ctx.order, ctx.kinvis);
   flowsolver.EnablePA(true);

   // Set the initial condition.
   ParGridFunction *u_ic = flowsolver.GetCurrentVelocity();
   VectorFunctionCoefficient u_excoeff(pmesh->Dimension(), vel_shear_ic);
   u_ic->ProjectCoefficient(u_excoeff);


   real_t t = 0.0;
   real_t dt = ctx.dt;
   real_t t_final = ctx.t_final;
   bool last_step = false;

   flowsolver.Setup(dt);

   ParGridFunction *u_gf = flowsolver.GetCurrentVelocity();
   ParGridFunction *p_gf = flowsolver.GetCurrentPressure();

   ParGridFunction w_gf(*u_gf);
   // flowsolver.ComputeCurl2D(*u_gf, w_gf);
   flowsolver.ComputeCurl3D(*u_gf, w_gf);

   // ParaViewDataCollection pvdc("shear_output", pmesh);
   // pvdc.SetDataFormat(VTKFormat::BINARY32);
   // pvdc.SetHighOrderOutput(true);
   // pvdc.SetLevelsOfDetail(ctx.order);
   // pvdc.SetCycle(0);
   // pvdc.SetTime(t);
   // pvdc.RegisterField("velocity", u_gf);
   // pvdc.RegisterField("pressure", p_gf);
   // pvdc.RegisterField("vorticity", &w_gf);
   // pvdc.Save();

   VisItDataCollection pvdc("shear_output", pmesh);
   pvdc.SetFormat(DataCollection::PARALLEL_FORMAT);
   pvdc.SetLevelsOfDetail(ctx.order);
   pvdc.SetCycle(0);
   pvdc.SetTime(t);
   pvdc.RegisterField("velocity", u_gf);
   pvdc.RegisterField("pressure", p_gf);
   pvdc.RegisterField("vorticity", &w_gf);
   pvdc.Save();

   for (int step = 0; !last_step; ++step)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      flowsolver.Step(t, dt, step);

      if (step % 500 == 0)
      {
         // flowsolver.ComputeCurl2D(*u_gf, w_gf);
         flowsolver.ComputeCurl3D(*u_gf, w_gf);
         pvdc.SetCycle(step);
         pvdc.SetTime(t);
         pvdc.Save();
      }

      if (Mpi::Root())
      {
         printf("%11s %11s\n", "Time", "dt");
         printf("%.5E %.5E\n", t, dt);
         fflush(stdout);
      }
   }

   flowsolver.PrintTimingData();

   delete pmesh;

   return 0;
}

