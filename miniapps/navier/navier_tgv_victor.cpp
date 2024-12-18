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

void ComputeElementCenterValues(ParGridFunction* sol, ParMesh* pmesh);

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

   

   ParMesh *pmesh;

   if (!ctx.restart)
   {
      Mesh *init_mesh;
      Mesh *mesh;

      init_mesh = new Mesh(Mesh::MakeCartesian3D(ctx.num_pts,
                                                 ctx.num_pts,
                                                 ctx.num_pts,
                                                 Element::HEXAHEDRON,
                                                 2.0*M_PI,
                                                 2.0*M_PI, 
                                                 2.0*M_PI,false));

      Vector x_translation({2.0*M_PI, 0.0, 0.0});
      Vector y_translation({0.0,2.0*M_PI, 0.0});
      Vector z_translation({0.0, 0.0,2.0*M_PI});

      std::vector<Vector> translations = {x_translation, y_translation,z_translation};

      mesh = new Mesh(Mesh::MakePeriodic(*init_mesh, init_mesh->CreatePeriodicVertexMapping(translations)));

      VerifyPeriodicMesh(mesh);

      // Define a translation function for the mesh nodes
      VectorFunctionCoefficient translate(mesh->Dimension(), [&](const Vector &x_in, Vector &x_out){
         double shift = -M_PI; 

         x_out[0] = x_in[0] + shift; // Translate x-coordinate 
         x_out[1] = x_in[1] + shift; // Translate y-coordinate 
         if (mesh->Dimension() == 3){
           x_out[2] = x_in[2] + shift; // Translate y-coordinate 
         }
      });

      mesh->Transform(translate);


      // Serial Mesh refinement
      for (int lev = 0; lev < ctx.element_subdivisions; lev++)
      {
         mesh->UniformRefinement();
      }

      // auto *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
      pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);

      // Parallel Mesh refinement
      for (int lev = 0; lev < ctx.element_subdivisions_parallel; lev++)
      {
         pmesh->UniformRefinement();
      }

      delete mesh;
      delete init_mesh;
   }
   else
   {
      std::string fname(MakeParFilename("tgv-checkpoint.", myid));
      std::ifstream ifs(fname);
      MFEM_VERIFY(ifs.good(), "Checkpoint file " << fname << " not found.");
      pmesh = new ParMesh(MPI_COMM_WORLD, ifs);
   }

   int nel = pmesh->GetNE();
   if (Mpi::Root())
   {
      mfem::out << "Number of elements: " << nel << std::endl;
   }

   // Create the flow solver.
   NavierSolver flowsolver(pmesh, ctx.order, ctx.kinvis);
   flowsolver.EnablePA(ctx.pa);
   flowsolver.EnableNI(ctx.ni);

   // Set the initial condition.
   ParGridFunction *u_ic = flowsolver.GetCurrentVelocity();
   VectorFunctionCoefficient u_excoeff(pmesh->Dimension(), vel_tgv);
   u_ic->ProjectCoefficient(u_excoeff);

   real_t t = 0.0;
   real_t dt = ctx.dt;
   real_t t_final = ctx.t_final;
   bool last_step = false;

   flowsolver.Setup(dt);

   ParGridFunction *u_gf = flowsolver.GetCurrentVelocity();
   ParGridFunction *p_gf = flowsolver.GetCurrentPressure();

   ParGridFunction w_gf(*u_gf);
   ParGridFunction q_gf(*p_gf);
   flowsolver.ComputeCurl3D(*u_gf, w_gf);
   ComputeQCriterion(*u_gf, q_gf);

   QuantitiesOfInterest kin_energy(pmesh);

   ParaViewDataCollection *pvdc = NULL;
   if (ctx.paraview)
   {
     pvdc = new ParaViewDataCollection("DatOutputParaivew/tgv_output", pmesh);
     pvdc->SetDataFormat(VTKFormat::BINARY32);
     pvdc->SetHighOrderOutput(true);
     pvdc->SetLevelsOfDetail(ctx.order);
     pvdc->SetCycle(0);
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
      dc->SetCycle(0);
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
      printf("%.5E %.5E %.5E %.5E %.5E %.5E\n", t, dt, u_inf, p_inf, ke, enstrophy);

      f = fopen(fname.c_str(), "w");
      fprintf(f, "3D Taylor Green Vortex\n");
      fprintf(f, "order = %d\n", ctx.order);
      fprintf(f, "grid = %d x %d x %d\n", nel1d, nel1d, nel1d);
      fprintf(f, "dofs per component = %d\n", ngridpts);
      fprintf(f, "===============================================================================\n");
      fprintf(f, "        time                   kinetic energy                   enstrophy\n");
      fprintf(f, "%20.16e     %20.16e     %20.16e\n", t, ke, enstrophy);
      fflush(f);
      fflush(stdout);
   }

   ComputeElementCenterValues(u_gf, pmesh);

   for (int step = 0; !last_step; ++step)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      flowsolver.Step(t, dt, step);

      if ((step + 1) % 100 == 0 || last_step)
      {
         flowsolver.ComputeCurl3D(*u_gf, w_gf);
         ComputeQCriterion(*u_gf, q_gf);

         if (ctx.paraview)
         {
           pvdc->SetCycle(step);
           pvdc->SetTime(t);
           pvdc->Save();
           if (Mpi::Root())
           {
             std::cout << "\nParaview file saved." << std::endl;
           }
         }

         if (ctx.visit)
         {
           dc->SetCycle(step);
           dc->SetTime(t);
           dc->Save();
           if (Mpi::Root())
           {
             std::cout << "\nVisit file saved." << std::endl;
           }
         }

         if (ctx.restart)
         {
           std::ofstream ofs(MakeParFilename("tgv-checkpoint.", myid));
            ofs.precision(8);
            pmesh->ParPrint(ofs);

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
      flowsolver.ComputeCurl3D(*u_gf, w_gf);
      enstrophy = kin_energy.ComputeEnstrophy(w_gf);

      if (Mpi::Root())
      {
         printf("%.5E %.5E %.5E %.5E %.5E %.5E\n", t, dt, u_inf, p_inf, ke, enstrophy);
         fprintf(f, "%20.16e     %20.16e     %20.16e\n", t, ke, enstrophy);
         fflush(f);
         fflush(stdout);
      }
   }

   flowsolver.PrintTimingData();

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
