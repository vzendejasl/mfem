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
//   -----------------------------------------------------------------------
//   Parallel LOR Transfer Miniapp:  Map functions between HO and LOR spaces
//   -----------------------------------------------------------------------
//
// This miniapp visualizes the maps between a high-order (HO) finite element
// space, typically using high-order functions on a high-order mesh, and a
// low-order refined (LOR) finite element space, typically defined by 0th or 1st
// order functions on a low-order refinement of the HO mesh.
//
// The grid transfer operators are represented using either
// InterpolationGridTransfer or L2ProjectionGridTransfer (depending on the
// options requested by the user). The two transfer operators are then:
//
//  1. R: HO -> LOR, defined by GridTransfer::ForwardOperator
//  2. P: LOR -> HO, defined by GridTransfer::BackwardOperator
//
// While defined generally, these operators have some nice properties for
// particular finite element spaces. For example they satisfy PR=I, plus mass
// conservation in both directions for L2 fields.
//
// Compile with: make plor-transfer
//
// Sample runs:  plor-transfer
//               plor-transfer -h1
//               plor-transfer -t
//               plor-transfer -lref 4 -o 4 -lo 0 
//               plor-transfer -lref 5 -o 4 -lo 0 
//               plor-transfer -lref 5 -o 4 -lo 3 
//               plor-transfer -lref 5 -o 4 -lo 0 

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int problem = 1; // problem type

int Wx = 0, Wy = 0; // window position
int Ww = 350, Wh = 350; // window size
int offx = Ww+5, offy = Wh+25; // window offsets

string space;
string direction;

// Exact functions to project
real_t RHO_exact(const Vector &x);

// Exact solution function
double u_exact(const Vector &x, Vector &u);

real_t compute_ke(ParGridFunction *gf, string prefix);

real_t ComputeKineticEnergy(ParGridFunction &v);

void ComputeKeGridFunction(ParGridFunction &u, ParGridFunction &ke);

// Helper functioTs
void visualize(VisItDataCollection &, string, string, int, int, int /* visport */);

void ComputeElementCenterValues(mfem::ParGridFunction* sol,
                                mfem::ParMesh* pmesh,
                                int step,
                                double time,
                                const std::string &suffix);

real_t compute_mass(ParFiniteElementSpace *, real_t, VisItDataCollection &,
                    string);
int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   Hypre::Init();

   // Parse command-line options.
   int order = 2;
   int lref = order+1;
   int lorder = 0;
   bool vis = true;
   bool useH1 = false;
   int visport = 19916;
   bool use_pointwise_transfer = false;
   const char *device_config = "cpu";
   bool use_ea       = false;

   OptionsParser args(argc, argv);
   args.AddOption(&problem, "-p", "--problem",
                  "Problem type (see the RHO_exact function).");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&lref, "-lref", "--lor-ref-level", "LOR refinement level.");
   args.AddOption(&lorder, "-lo", "--lor-order",
                  "LOR space order (polynomial degree, zero by default).");
   args.AddOption(&vis, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&useH1, "-h1", "--use-h1", "-l2", "--use-l2",
                  "Use H1 spaces instead of L2.");
   args.AddOption(&use_pointwise_transfer, "-t", "--use-pointwise-transfer",
                  "-no-t", "--dont-use-pointwise-transfer",
                  "Use pointwise transfer operators instead of L2 projection.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&use_ea, "-ea", "--ea-version", "-no-ea",
                  "--no-ea-version", "Use element assembly version.");
   args.ParseCheck();

   // Configure device
   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   Mesh periodic_mesh;
   Mesh *init_mesh;

   int nx, ny, nz;
   nx = 5;
   ny = nx;
   nz = nx;

   // Mesh boundaries (unit cube)
   double x1 = 0.0, x2 = 1.0;
   double y1 = 0.0, y2 = 1.0;
   double z1 = 0.0, z2 = 1.0;


   init_mesh = new Mesh(Mesh::MakeCartesian3D(nx,
                                              ny,
                                              nz,
                                              Element::HEXAHEDRON,
                                              x2 - x1,
                                              y2 - y1,
                                              z2 - z1));

   Vector x_translation({x2 - x1, 0.0, 0.0});
   Vector y_translation({0.0, y2 - y1, 0.0});
   Vector z_translation({0.0, 0.0, z2 - z1});

   std::vector<Vector> translations = {x_translation, y_translation, z_translation};

   periodic_mesh = Mesh(Mesh::MakePeriodic(*init_mesh,
                                      init_mesh->CreatePeriodicVertexMapping(translations)));


   ParMesh mesh(MPI_COMM_WORLD, periodic_mesh);
   // ParMesh mesh(MPI_COMM_WORLD, *init_mesh);
   delete init_mesh;

   int dim = mesh.Dimension();

   // Make initial refinement on serial mesh.
   for (int l = 0; l < 1; l++)
   {
      mesh.UniformRefinement();
   }

   // Create the low-order refined mesh
   int basis_lor = BasisType::GaussLobatto; // BasisType::ClosedUniform;
   ParMesh mesh_lor = ParMesh::MakeRefined(mesh, lref, basis_lor);

         
   if (Mpi::Root()){
     mfem::out << "\n";
     mfem::out << "Peforming high to low order refined operation." << "\n";
   }
   // Create spaces
   FiniteElementCollection *fec, *fec_lor;
   if (useH1)
   {
      space = "H1";
      if (lorder == 0)
      {
         lorder = 1;
         if (Mpi::Root())
         {
            cerr << "Switching the H1 LOR space order from 0 to 1\n";
         }
      }
      fec = new H1_FECollection(order, dim);
      fec_lor = new H1_FECollection(lorder, dim);
   }
   else
   {
      space = "L2";
      fec = new L2_FECollection(order, dim);
      fec_lor = new L2_FECollection(lorder, dim);
   }

   ParFiniteElementSpace fespace(&mesh, fec, dim);
   ParFiniteElementSpace fespace_lor(&mesh_lor, fec_lor,dim);

   ParGridFunction u(&fespace);
   ParGridFunction u_lor(&fespace_lor);

   // Data collections for vis/analysis
   VisItDataCollection HO_dc(MPI_COMM_WORLD, "HO", &mesh);
   HO_dc.RegisterField("velocity", &u);
   VisItDataCollection LOR_dc(MPI_COMM_WORLD, "LOR", &mesh_lor);
   LOR_dc.RegisterField("velocity", &u_lor);
      
   // HO projections
   direction = "HO -> LOR @ HO";
   VectorFunctionCoefficient u_ex_coeff(dim,u_exact);
   u.ProjectCoefficient(u_ex_coeff);

   // Make sure AMR constraints are satisfied
   u.SetTrueVector();
   u.SetFromTrueVector();

   real_t ho_ke = compute_ke(&u, "HO        ");

   if (vis) { visualize(HO_dc,"velocity", "HO", Wx, Wy, visport); Wx += offx; }

   GridTransfer *gt;
   if (use_pointwise_transfer)
   {
      gt = new InterpolationGridTransfer(fespace, fespace_lor);
   }
   else
   {
      gt = new L2ProjectionGridTransfer(fespace, fespace_lor);
   }

   // Configure element assembly for device acceleration
   gt->UseEA(use_ea);

   const Operator &R = gt->ForwardOperator();

   // HO->LOR restriction
   direction = "HO -> LOR @ LOR";
   R.Mult(u, u_lor);
   real_t lo_ke = compute_ke(&u_lor, "R(HO)     ");
   if (vis) { visualize(LOR_dc,"velocity", "R(HO)", Wx, Wy, visport); Wx += offx; }
   auto global_max = [](const Vector& v)
   {
      real_t max = v.Normlinf();
      MPI_Allreduce(MPI_IN_PLACE, &max, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_MAX, MPI_COMM_WORLD);
      return max;
   };

   if (gt->SupportsBackwardsOperator())
   {
      const Operator &P = gt->BackwardOperator();
      // LOR->HO prolongation
      direction = "HO -> LOR @ HO";
      ParGridFunction u_prev = u;
      P.Mult(u_lor, u);
      // compute_mass(&fespace, ho_mass, HO_dc, "P(R(HO)) ");
      ho_ke = compute_ke(&u, "P(R(HO)   ");
      if (vis) { visualize(HO_dc,"velocity", "P(R(HO))", Wx, Wy, visport); Wx = 0; Wy += offy; }

      u_prev -= u;
      Vector u_prev_true(fespace.GetTrueVSize());
      u_prev.GetTrueDofs(u_prev_true);
      real_t l_inf = global_max(u_prev_true);
      if (Mpi::Root())
      {
         cout.precision(12);
         cout << "|HO - P(R(HO))|_∞   = " << l_inf << endl;
      }
   }

   HO_dc.SetCycle(0);
   HO_dc.SetTime(0.0);
   HO_dc.SetFormat(DataCollection::PARALLEL_FORMAT);
   HO_dc.Save();

   LOR_dc.SetCycle(0);
   LOR_dc.SetTime(0.0);
   LOR_dc.SetFormat(DataCollection::PARALLEL_FORMAT);
   LOR_dc.Save();


   // ComputeElementCenterValues(&u_lor,
   //                            &mesh_lor,
   //                            0,
   //                            0.0,
   //                            "LOR");

   if (Mpi::Root()){
     mfem::out << "\n";
     mfem::out << "Peforming low order refined to higher order operation." << "\n";
   }

   // LOR projections
   direction = "LOR -> HO @ LOR";
   u_lor.ProjectCoefficient(u_ex_coeff);
   ParGridFunction u_lor_prev = u_lor;
   real_t lor_ke = compute_ke(&u_lor_prev, "P(R(HO)   ");
   if (vis) { visualize(LOR_dc,"velocity", "LOR", Wx, Wy, visport); Wx += offx; }

   if (gt->SupportsBackwardsOperator())
   {
      const Operator &P = gt->BackwardOperator();
      // Prolongate to HO space
      direction = "LOR -> HO @ HO";
      P.Mult(u_lor, u);
      lor_ke = compute_ke(&u_lor, "P(LOR)   ");
      if (vis) { visualize(HO_dc,"velocity", "P(LOR)", Wx, Wy, visport); Wx += offx; }

      // Restrict back to LOR space. This won't give the original function because
      // the rho_lor doesn't necessarily live in the range of R.
      direction = "LOR -> HO @ LOR";
      R.Mult(u, u_lor);
      lor_ke = compute_ke(&u_lor, "R(P(R(HO))   ");
      if (vis) { visualize(LOR_dc,"velocity", "R(P(LOR))", Wx, Wy, visport); }

      u_lor_prev -= u_lor;
      Vector u_lor_prev_true(fespace_lor.GetTrueVSize());
      u_lor_prev.GetTrueDofs(u_lor_prev_true);
      real_t l_inf = global_max(u_lor_prev_true);
      if (Mpi::Root())
      {
         cout.precision(12);
         cout << "|LOR - R(P(LOR))|_∞ = " << l_inf << endl;
      }
   }

   
   if (Mpi::Root()){
     mfem::out << "\n";
     mfem::out << "Same experiment as before, but with a scalar field." << "\n";
   }

   ParFiniteElementSpace scal_fes     (&mesh,     fec, 1);
   ParFiniteElementSpace scal_fes_lor (&mesh_lor, fec_lor, 1);
   
   ParGridFunction ke     (&scal_fes);        
   ParGridFunction rho    (&scal_fes);        
   ParGridFunction rho_lor(&scal_fes_lor);    

   ComputeKeGridFunction(u, ke);
   
   // Data collections for vis/analysis
   VisItDataCollection HO_dc_rho(MPI_COMM_WORLD, "HO", &mesh);
   HO_dc.RegisterField("density", &rho);
   VisItDataCollection LOR_dc_rho(MPI_COMM_WORLD, "LOR", &mesh_lor);
   LOR_dc.RegisterField("density", &rho_lor);

   ParBilinearForm M_ho(&scal_fes);
   M_ho.AddDomainIntegrator(new MassIntegrator);
   M_ho.Assemble();
   M_ho.Finalize();
   HypreParMatrix* M_ho_tdof = M_ho.ParallelAssemble();

   ParBilinearForm M_lor(&scal_fes_lor);
   M_lor.AddDomainIntegrator(new MassIntegrator);
   M_lor.Assemble();
   M_lor.Finalize();
   HypreParMatrix* M_lor_tdof = M_lor.ParallelAssemble();

   if (Mpi::Root()){
     mfem::out << "\n";
     mfem::out << "Peforming high to low order refined operation." << "\n";
   }

   // HO projections
   direction = "HO -> LOR @ HO";
   GridFunctionCoefficient RHO(&ke);
   rho.ProjectCoefficient(RHO);
   // Make sure AMR constraints are satisfied
   rho.SetTrueVector();
   rho.SetFromTrueVector();

   real_t ho_mass = compute_mass(&scal_fes, -1.0, HO_dc, "HO       ");
   if (vis) { visualize(HO_dc,"density", "HO", Wx, Wy, visport); Wx += offx; }

   if (use_pointwise_transfer)
   {
      gt = new InterpolationGridTransfer(scal_fes, scal_fes_lor);
   }
   else
   {
      gt = new L2ProjectionGridTransfer(scal_fes, scal_fes_lor);
   }

   // Configure element assembly for device acceleration
   gt->UseEA(use_ea);

   const Operator &Rscl = gt->ForwardOperator();


   // HO->LOR restriction
   direction = "HO -> LOR @ LOR";
   Rscl.Mult(rho, rho_lor);
   compute_mass(&scal_fes_lor, ho_mass, LOR_dc, "R(HO)    ");
   if (vis) { visualize(LOR_dc,"density", "R(HO)", Wx, Wy, visport); Wx += offx; }

   if (gt->SupportsBackwardsOperator())
   {
      const Operator &Pscl = gt->BackwardOperator();
      // LOR->HO prolongation
      direction = "HO -> LOR @ HO";
      ParGridFunction rho_prev = rho;
      Pscl.Mult(rho_lor, rho);
      compute_mass(&scal_fes, ho_mass, HO_dc, "P(R(HO)) ");
      if (vis) { visualize(HO_dc,"density", "P(R(HO))", Wx, Wy, visport); Wx = 0; Wy += offy; }

      rho_prev -= rho;
      Vector rho_prev_true(scal_fes.GetTrueVSize());
      rho_prev.GetTrueDofs(rho_prev_true);
      real_t l_inf = global_max(rho_prev_true);
      if (Mpi::Root())
      {
         cout.precision(12);
         cout << "|HO - P(R(HO))|_∞   = " << l_inf << endl;
      }
   }

   // HO* to LOR* dual fields
   ParLinearForm M_rho(&scal_fes), M_rho_lor(&scal_fes_lor);
   auto global_sum = [](const Vector& v)
   {
      real_t sum = v.Sum();
      MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM, MPI_COMM_WORLD);
      return sum;
   };
   if (!use_pointwise_transfer && gt->SupportsBackwardsOperator())
   {
      Vector M_rho_true(scal_fes.GetTrueVSize());
      M_ho_tdof->Mult(rho.GetTrueVector(), M_rho_true);
      scal_fes.GetRestrictionOperator()->MultTranspose(M_rho_true, M_rho);
      const Operator &Pscl = gt->BackwardOperator();
      Pscl.MultTranspose(M_rho, M_rho_lor);
      real_t ho_dual_mass = global_sum(M_rho);
      real_t lor_dual_mass = global_sum(M_rho_lor);
      if (Mpi::Root())
      {
         cout << "HO -> LOR dual field: " << abs(ho_dual_mass - lor_dual_mass) << "\n\n";
      }
   }

   if (Mpi::Root()){
     mfem::out << "\n";
     mfem::out << "Peforming low order refined to higher order operation." << "\n";
   }
   // LOR projections
   direction = "LOR -> HO @ LOR";
   rho_lor.ProjectCoefficient(RHO);
   ParGridFunction rho_lor_prev = rho_lor;
   real_t lor_mass = compute_mass(&scal_fes_lor, -1.0, LOR_dc, "LOR      ");
   if (vis) { visualize(LOR_dc,"density", "LOR", Wx, Wy, visport); Wx += offx; }

   if (gt->SupportsBackwardsOperator())
   {
      const Operator &Pscl = gt->BackwardOperator();
      // Prolongate to HO space
      direction = "LOR -> HO @ HO";
      Pscl.Mult(rho_lor, rho);
      compute_mass(&scal_fes, lor_mass, HO_dc, "P(LOR)   ");
      if (vis) { visualize(HO_dc,"density", "P(LOR)", Wx, Wy, visport); Wx += offx; }

      // Restrict back to LOR space. This won't give the original function because
      // the rho_lor doesn't necessarily live in the range of R.
      direction = "LOR -> HO @ LOR";
      Rscl.Mult(rho, rho_lor);
      compute_mass(&scal_fes_lor, lor_mass, LOR_dc, "R(P(LOR))");
      if (vis) { visualize(LOR_dc,"density", "R(P(LOR))", Wx, Wy, visport); }

      rho_lor_prev -= rho_lor;
      Vector rho_lor_prev_true(scal_fes_lor.GetTrueVSize());
      rho_lor_prev.GetTrueDofs(rho_lor_prev_true);
      real_t l_inf = global_max(rho_lor_prev_true);
      if (Mpi::Root())
      {
         cout.precision(12);
         cout << "|LOR - R(P(LOR))|_∞ = " << l_inf << endl;
      }
   }

   // LOR* to HO* dual fields
   if (!use_pointwise_transfer)
   {
      Vector M_rho_lor_true(scal_fes_lor.GetTrueVSize());
      M_lor_tdof->Mult(rho_lor.GetTrueVector(), M_rho_lor_true);
      scal_fes_lor.GetRestrictionOperator()->MultTranspose(M_rho_lor_true,
                                                          M_rho_lor);
      Rscl.MultTranspose(M_rho_lor, M_rho);
      real_t ho_dual_mass = global_sum(M_rho);
      real_t lor_dual_mass = global_sum(M_rho_lor);

      if (Mpi::Root())
      {
         cout << "lor dual mass = " << lor_dual_mass << '\n';
         cout << "ho dual mass = " << ho_dual_mass << '\n';
         cout << "LOR -> HO dual field: " << abs(ho_dual_mass - lor_dual_mass) << '\n';
      }
   }

   delete M_ho_tdof;
   delete M_lor_tdof;




   delete fec;
   delete fec_lor;
   delete gt;

   return 0;
}


void visualize(VisItDataCollection &dc,string field_name, string prefix, int x, int y,
               int visport)
{
   int w = Ww, h = Wh;

   char vishost[] = "localhost";

   socketstream sol_sockL2(vishost, visport);
   sol_sockL2 << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() <<
              "\n";
   sol_sockL2.precision(8);
   sol_sockL2 << "solution\n" << *dc.GetMesh() << *dc.GetField(field_name.c_str())
              << "window_geometry " << x << " " << y << " " << w << " " << h
              << "plot_caption '" << space << " " << prefix << field_name << "'"
              << "window_title '" << direction << "'" 
              << flush;
}

// Compute L2 ke
real_t compute_ke(ParGridFunction *gf, string prefix)
{

  real_t ke =  ComputeKineticEnergy(*gf);

  if (Mpi::Root())
  {
     cout.precision(18);
     cout << space << " " << prefix << " Kinetic energy = " << ke<< endl;
  }

  return ke;
   
}

// Modified exact solution to include higher frequencies
double u_exact(const Vector &x, Vector &u)
{
    real_t xi = x(0);
    real_t yi = x(1);
    real_t zi = x(2);

    u(0) =  sin(2*M_PI*xi) * cos(2*M_PI*yi) * cos(2*M_PI*zi);
    u(1) = -cos(2*M_PI*xi) * sin(2*M_PI*yi) * cos(2*M_PI*zi);
    u(2) = 0.0;
    
}

real_t ComputeKineticEnergy(ParGridFunction &v)
{
   /*
   Vector velx, vely, velz;
   real_t integ = 0.0;
   const FiniteElement *fe;
   ElementTransformation *T;
   FiniteElementSpace *fes = v.FESpace();
      
   real_t summedVolume = 0.0;
   real_t globalVolume   = 0.0;
   real_t global_integral = 0.0;

   for (int i = 0; i < fes->GetNE(); i++)
   {
      double volume_per_cell = 0.0;

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
         volume_per_cell += ip.weight*T->Weight();
      }
      summedVolume +=volume_per_cell;
   }

   MPI_Allreduce(&summedVolume,
                 &globalVolume,
                 1,
                 MPITypeMap<real_t>::mpi_type,
                 MPI_SUM,
                 MPI_COMM_WORLD);

   MPI_Allreduce(&integ,
                 &global_integral,
                 1,
                 MPITypeMap<real_t>::mpi_type,
                 MPI_SUM,
                 MPI_COMM_WORLD);

   return 0.5 * global_integral/globalVolume;
   */
    auto *fes = dynamic_cast<ParFiniteElementSpace*>(v.FESpace());
    ParBilinearForm mass(fes);
    mass.AddDomainIntegrator(new VectorMassIntegrator());

    // Doesn't work?
    // if (ctx.pa){
    //     mass.SetAssemblyLevel(AssemblyLevel::PARTIAL);
    // }
  
    mass.Assemble();
    mass.Finalize();

    // Assuming volume = 1
    const double ke = 0.5*mass.ParInnerProduct(v,v);
    return ke;

};

/*
void ComputeElementCenterValues(mfem::ParGridFunction* sol,
                                mfem::ParMesh* pmesh,
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
   std::string main_dir = "ElementCenters" + suffix;

   // Create subdirectory for this cycle step
   std::string cycle_dir = main_dir + "/cycle_" + std::to_string(step);
   // Construct the filename inside the cycle directory
   std::string fname = cycle_dir + "/element_centers_" + std::to_string(step) + ".txt";

   // Create directories on rank 0
   if (rank == 0)
   {
      if (system(("mkdir -p " + main_dir).c_str()) != 0)
         std::cerr << "Error creating " << main_dir << " directory!" << std::endl;
      if (system(("mkdir -p " + cycle_dir).c_str()) != 0)
         std::cerr << "Error creating " << cycle_dir << " directory!" << std::endl;
   }

   // Synchronize all ranks before proceeding
   MPI_Barrier(MPI_COMM_WORLD);

   // Local arrays to store data from the local elements
   std::vector<double> local_x, local_y, local_z;
   std::vector<double> local_velx, local_vely, local_velz;

   real_t local_tke = 0.0;
   real_t summed_tke = 0.0;
   real_t global_tke = 0.0;

   mfem::FiniteElementSpace *fes = sol->FESpace();
   int vdim = fes->GetVDim();

   // Loop over local elements
   for (int e = 0; e < fes->GetNE(); e++)
   {
      // Get element transformation for element e
      mfem::ElementTransformation *Trans = pmesh->GetElementTransformation(e);
      
      IntegrationPoint ip;
      ip.Set3(0.5,0.5,0.5); // Sample point in reference element

      // Get the physical coordinates for this sample point
      mfem::Vector phys_coords(Trans->GetSpaceDim());
      Trans->Transform(ip, phys_coords);

      double x_physical = phys_coords(0);
      double y_physical = phys_coords(1);
      double z_physical = phys_coords(2);

      // Evaluate the solution at the sample point
      mfem::Vector u_val(vdim);
      sol->GetVectorValue(*Trans, ip, u_val);
      double u_x = u_val(0);
      double u_y = u_val(1);
      double u_z = u_val(2);

      // Append sample point data to local arrays
      local_x.push_back(x_physical);
      local_y.push_back(y_physical);
      local_z.push_back(z_physical);
      local_velx.push_back(u_x);
      local_vely.push_back(u_y);
      local_velz.push_back(u_z);
   
      // Compute kinetic energy
      local_tke = 0.5*(u_x*u_x + u_y*u_y + u_z*u_z);
      summed_tke += local_tke;
   } // for each local element

   // Do a reduction to compute tke for all processers
   MPI_Allreduce(&summed_tke,
                 &global_tke,
                 1,
                 MPITypeMap<real_t>::mpi_type,
                 MPI_SUM,
                 MPI_COMM_WORLD);

   int local_elements = fes->GetNE();
   int total_elements = 0;

   MPI_Allreduce(&local_elements, &total_elements, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   global_tke = global_tke / total_elements;

   if (rank == 0)
       std::cout << "The computed tke from element centers is: " << global_tke << std::endl;


   // Prepare the data string, including the header on rank 0
   std::string data_str;
   if (rank == 0)
   {
      std::ostringstream header_stream;
      header_stream << "PLOR Transfer\n"
                    << "Step = " << step << "\n"
                    << "Time = " << std::scientific << std::setprecision(16) << time << "\n"
                    << "==================================================================="
                    << "==========================================================================\n"
                    << "            x                      y                      z                   vecx                   vecy                   vecz\n";
      data_str = header_stream.str();
   }

   // Append local data to data_str
   std::ostringstream local_data_stream;
   for (size_t i = 0; i < local_x.size(); i++)
   {
      local_data_stream << std::scientific << std::setprecision(16)
                        << std::setw(20) << local_x[i] << " "
                        << std::setw(20) << local_y[i] << " "
                        << std::setw(20) << local_z[i] << " "
                        << std::setw(20) << local_velx[i] << " "
                        << std::setw(20) << local_vely[i] << " "
                        << std::setw(20) << local_velz[i] << "\n";
   }
   data_str += local_data_stream.str();

   // Open the file collectively with MPI I/O
   MPI_File fh;
   int err = MPI_File_open(comm, fname.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
   if (err != MPI_SUCCESS)
   {
      if (rank == 0) std::cerr << "Error opening file " << fname << " with MPI I/O" << std::endl;
      MPI_Abort(comm, 1);
   }

   // All ranks write their data (including header on rank 0) in order using the shared file pointer
   MPI_File_write_ordered(fh, data_str.c_str(), data_str.size(), MPI_CHAR, MPI_STATUS_IGNORE);

   // Clear memory
   local_x.clear(); local_y.clear(); local_z.clear();
   local_velx.clear(); local_vely.clear(); local_velz.clear();
   data_str.clear();

   // Close the file
   MPI_File_close(&fh);

   // Output confirmation on rank 0
   if (rank == 0)
      std::cout << "Output element sample file saved: " << fname << std::endl;

   // Final synchronization
   MPI_Barrier(MPI_COMM_WORLD);
}
*/


void ComputeElementCenterValues(mfem::ParGridFunction* sol,
                                mfem::ParMesh* pmesh,
                                int step,
                                double time,
                                const std::string &suffix){
   // MPI setup
   MPI_Comm comm = pmesh->GetComm();
   int rank, size;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &size);

   // Construct the main directory name with suffix
   std::string main_dir = "ElementCenters" + suffix;

   // Create subdirectory for this cycle step
   std::string cycle_dir = main_dir + "/cycle_" + std::to_string(step);
   // Construct the filename inside the cycle directory
   std::string fname = cycle_dir + "/element_centers_" + std::to_string(step) + ".txt";

   // Create directories on rank 0
   if (rank == 0)
   {
      if (system(("mkdir -p " + main_dir).c_str()) != 0)
         std::cerr << "Error creating " << main_dir << " directory!" << std::endl;
      if (system(("mkdir -p " + cycle_dir).c_str()) != 0)
         std::cerr << "Error creating " << cycle_dir << " directory!" << std::endl;
   }

   // Synchronize all ranks before proceeding
   MPI_Barrier(MPI_COMM_WORLD);


    std::vector<double> local_x, local_y, local_z;
    std::vector<double> local_velx, local_vely, local_velz;

    mfem::FiniteElementSpace *fes = sol->FESpace();
    int vdim = fes->GetVDim();
    const FiniteElement *fe;

   real_t integ = 0.0;
   real_t summedVolume = 0.0;
   real_t globalVolume   = 0.0;
   real_t global_integral = 0.0;

    for (int i = 0; i < fes->GetNE(); i++)
    {
      
        double volume_per_cell = 0.0;
        mfem::ElementTransformation *Trans = pmesh->GetElementTransformation(i);
        fe = fes->GetFE(i);
        int intorder = 2 * fe->GetOrder();
        const IntegrationRule *ir = &IntRules.Get(fe->GetGeomType(), intorder);

        // Pre-allocate mfem::Vectors for efficiency
        mfem::Vector velx(ir->GetNPoints());
        mfem::Vector vely(ir->GetNPoints());
        mfem::Vector velz(ir->GetNPoints());

        sol->GetValues(i, *ir, velx, 1);
        sol->GetValues(i, *ir, vely, 2);
        sol->GetValues(i, *ir, velz, 3);

        // Loop over integration points
        for (int j = 0; j < ir->GetNPoints(); j++)
        {
            const IntegrationPoint &ip = ir->IntPoint(j);
            mfem::Vector phys_coords(Trans->GetSpaceDim());
            Trans->Transform(ip, phys_coords);
            double x_physical = phys_coords(0);
            double y_physical = phys_coords(1);
            double z_physical = phys_coords(2);

            // Store coordinates and velocity components
            local_x.push_back(x_physical);
            local_y.push_back(y_physical);
            local_z.push_back(z_physical);
            local_velx.push_back(velx(j));
            local_vely.push_back(vely(j));
            local_velz.push_back(velz(j));

            real_t vel2 = velx(j) * velx(j) + vely(j) * vely(j)
                          + velz(j) * velz(j);

            integ += ip.weight * Trans->Weight() * vel2;
            volume_per_cell += ip.weight*Trans->Weight();
        }
        summedVolume += volume_per_cell;
    }

   MPI_Allreduce(&summedVolume,
                 &globalVolume,
                 1,
                 MPITypeMap<real_t>::mpi_type,
                 MPI_SUM,
                 MPI_COMM_WORLD);

   MPI_Allreduce(&integ,
                 &global_integral,
                 1,
                 MPITypeMap<real_t>::mpi_type,
                 MPI_SUM,
                 MPI_COMM_WORLD);

   if (rank == 0)
     std::cout << "KE: " << 0.5*global_integral/globalVolume << std::endl;


   // Prepare the data string, including the header on rank 0
   std::string data_str;
   if (rank == 0)
   {
      std::ostringstream header_stream;
      header_stream << "PLOR Transfer\n"
                    << "Step = " << step << "\n"
                    << "Time = " << std::scientific << std::setprecision(16) << time << "\n"
                    << "==================================================================="
                    << "==========================================================================\n"
                    << "            x                      y                      z                   vecx                   vecy                   vecz\n";
      data_str = header_stream.str();
   }

   // Append local data to data_str
   std::ostringstream local_data_stream;
   for (size_t i = 0; i < local_x.size(); i++)
   {
      local_data_stream << std::scientific << std::setprecision(16)
                        << std::setw(20) << local_x[i] << " "
                        << std::setw(20) << local_y[i] << " "
                        << std::setw(20) << local_z[i] << " "
                        << std::setw(20) << local_velx[i] << " "
                        << std::setw(20) << local_vely[i] << " "
                        << std::setw(20) << local_velz[i] << "\n";
   }
   data_str += local_data_stream.str();

   // Open the file collectively with MPI I/O
   MPI_File fh;
   int err = MPI_File_open(comm, fname.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
   if (err != MPI_SUCCESS)
   {
      if (rank == 0) std::cerr << "Error opening file " << fname << " with MPI I/O" << std::endl;
      MPI_Abort(comm, 1);
   }

   // All ranks write their data (including header on rank 0) in order using the shared file pointer
   MPI_File_write_ordered(fh, data_str.c_str(), data_str.size(), MPI_CHAR, MPI_STATUS_IGNORE);

   // Clear memory
   local_x.clear(); local_y.clear(); local_z.clear();
   local_velx.clear(); local_vely.clear(); local_velz.clear();
   data_str.clear();

   // Close the file
   MPI_File_close(&fh);

   // Output confirmation on rank 0
   if (rank == 0)
      std::cout << "Output element sample file saved: " << fname << std::endl;

   // Final synchronization
   MPI_Barrier(MPI_COMM_WORLD);
}


/*
void ComputeElementCenterValues(mfem::ParGridFunction* sol,
                                mfem::ParMesh* pmesh,
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
   std::string main_dir = "ElementCenters" + suffix;

   // Create subdirectory for this cycle step
   std::string cycle_dir = main_dir + "/cycle_" + std::to_string(step);
   // Construct the filename inside the cycle directory
   std::string fname = cycle_dir + "/element_centers_" + std::to_string(step) + ".txt";

   // Create directories on rank 0
   if (rank == 0)
   {
      if (system(("mkdir -p " + main_dir).c_str()) != 0)
         std::cerr << "Error creating " << main_dir << " directory!" << std::endl;
      if (system(("mkdir -p " + cycle_dir).c_str()) != 0)
         std::cerr << "Error creating " << cycle_dir << " directory!" << std::endl;
   }

   // Synchronize all ranks before proceeding
   MPI_Barrier(MPI_COMM_WORLD);

   // Local arrays to store data from the local elements
   std::vector<double> local_x, local_y, local_z;
   std::vector<double> local_velx, local_vely, local_velz;

   real_t local_tke = 0.0;
   real_t summed_tke = 0.0;
   real_t global_tke = 0.0;

   mfem::FiniteElementSpace *fes = sol->FESpace();
   int vdim = fes->GetVDim();

   // Sampling setup
   int npts =  2;  // Number of sample points per coordinate direction

   // Loop over local elements
   for (int e = 0; e < pmesh->GetNE(); e++)
   {
      // Get element transformation for element e
      mfem::ElementTransformation *Trans = pmesh->GetElementTransformation(e);
      
      // For each element, loop over a uniform grid of points in the reference element [0,1]^d
      for (int iz = 0; iz < npts; iz++)
      {
         double z_ref = (npts == 1) ? 0.5 : static_cast<double>(iz) / (npts - 1);
         for (int iy = 0; iy < npts; iy++)
         {
            double y_ref = (npts == 1) ? 0.5 : static_cast<double>(iy) / (npts - 1);
            for (int ix = 0; ix < npts; ix++)
            {
               double x_ref = (npts == 1) ? 0.5 : static_cast<double>(ix) / (npts - 1);
               mfem::IntegrationPoint ip;
               ip.Set3(x_ref, y_ref, z_ref); // Sample point in reference element

               // Get the physical coordinates for this sample point
               mfem::Vector phys_coords(Trans->GetSpaceDim());
               Trans->Transform(ip, phys_coords);

               double x_physical = phys_coords(0);
               double y_physical = phys_coords(1);
               double z_physical = phys_coords(2);

               // Evaluate the solution at the sample point
               mfem::Vector u_val(vdim);
               sol->GetVectorValue(*Trans, ip, u_val);
               double u_x = u_val(0);
               double u_y = u_val(1);
               double u_z = u_val(2);

               // Append sample point data to local arrays
               local_x.push_back(x_physical);
               local_y.push_back(y_physical);
               local_z.push_back(z_physical);
               local_velx.push_back(u_x);
               local_vely.push_back(u_y);
               local_velz.push_back(u_z);
      
               local_tke = 0.5*(u_x*u_x + u_y*u_y + u_z*u_z);
            } // ix
         } // iy
      } // iz
      // Compute kinetic energy
      summed_tke += local_tke;
   } // for each local element


   // Do a reduction to compute tke for all processers
   MPI_Allreduce(&summed_tke,
                 &global_tke,
                 1,
                 MPITypeMap<real_t>::mpi_type,
                 MPI_SUM,
                 MPI_COMM_WORLD);

   int local_elements = fes->GetNE();
   int total_elements = 0;

   MPI_Allreduce(&local_elements, &total_elements, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   global_tke = global_tke / total_elements;

   if (rank == 0)
       std::cout << "The computed tke from element centers is: " << global_tke << std::endl;


   // Prepare the data string, including the header on rank 0
   std::string data_str;
   if (rank == 0)
   {
      std::ostringstream header_stream;
      header_stream << "PLOR Transfer\n"
                    << "Step = " << step << "\n"
                    << "Time = " << std::scientific << std::setprecision(16) << time << "\n"
                    << "==================================================================="
                    << "==========================================================================\n"
                    << "            x                      y                      z                   vecx                   vecy                   vecz\n";
      data_str = header_stream.str();
   }

   // Append local data to data_str
   std::ostringstream local_data_stream;
   for (size_t i = 0; i < local_x.size(); i++)
   {
      local_data_stream << std::scientific << std::setprecision(16)
                        << std::setw(20) << local_x[i] << " "
                        << std::setw(20) << local_y[i] << " "
                        << std::setw(20) << local_z[i] << " "
                        << std::setw(20) << local_velx[i] << " "
                        << std::setw(20) << local_vely[i] << " "
                        << std::setw(20) << local_velz[i] << "\n";
   }
   data_str += local_data_stream.str();

   // Open the file collectively with MPI I/O
   MPI_File fh;
   int err = MPI_File_open(comm, fname.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
   if (err != MPI_SUCCESS)
   {
      if (rank == 0) std::cerr << "Error opening file " << fname << " with MPI I/O" << std::endl;
      MPI_Abort(comm, 1);
   }

   // All ranks write their data (including header on rank 0) in order using the shared file pointer
   MPI_File_write_ordered(fh, data_str.c_str(), data_str.size(), MPI_CHAR, MPI_STATUS_IGNORE);

   // Clear memory
   local_x.clear(); local_y.clear(); local_z.clear();
   local_velx.clear(); local_vely.clear(); local_velz.clear();
   data_str.clear();

   // Close the file
   MPI_File_close(&fh);

   // Output confirmation on rank 0
   if (rank == 0)
      std::cout << "Output element sample file saved: " << fname << std::endl;

   // Final synchronization
   MPI_Barrier(MPI_COMM_WORLD);
}*/

real_t compute_mass(ParFiniteElementSpace *L2, real_t massL2,
                    VisItDataCollection &dc, string prefix)
{
   ConstantCoefficient one(1.0);
   ParLinearForm lf(L2);
   lf.AddDomainIntegrator(new DomainLFIntegrator(one));
   lf.Assemble();

   real_t newmass = lf(*dc.GetParField("density"));
   if (Mpi::Root())
   {
      cout.precision(18);
      cout << space << " " << prefix << " mass   = " << newmass;
      if (massL2 >= 0)
      {
         cout.precision(4);
         cout << " ("  << fabs(newmass-massL2)*100/massL2 << "%)";
      }
      cout << endl;
   }
   return newmass;
}

// Computes KE = 1/2 * (u1**2 + u2**2 + u3**3) 
// Will not work for L2 in the current form
void ComputeKeGridFunction(ParGridFunction &u, ParGridFunction &ke)
{
   FiniteElementSpace *v_fes = u.FESpace();
   FiniteElementSpace *fes = ke.FESpace();

   // AccumulateAndCountZones
   Array<int> zones_per_vdof;
   zones_per_vdof.SetSize(fes->GetVSize());
   zones_per_vdof = 0;

   ke = 0.0;

   // Local interpolation
   int elndofs;
   Array<int> v_dofs, dofs;
   Vector vals;
   Vector loc_data;
   int vdim = v_fes->GetVDim();

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      fes->GetElementVDofs(e, dofs);
      v_fes->GetElementVDofs(e, v_dofs);
      u.GetSubVector(v_dofs, loc_data);
      vals.SetSize(dofs.Size());
      ElementTransformation *tr = fes->GetElementTransformation(e);
      const FiniteElement *el = fes->GetFE(e);
      elndofs = el->GetDof();

      for (int dof = 0; dof < elndofs; ++dof)
      {
         // Project
         const IntegrationPoint &ip = el->GetNodes().IntPoint(dof);
         tr->SetIntPoint(&ip);

         // Evaluate the solution at the sample point
         Vector u_val(vdim);
         u.GetVectorValue(*tr, ip, u_val);

         real_t ke_val = u_val(0)*u_val(0) + u_val(1)*u_val(1) + u_val(2)*u_val(2); 
         vals(dof) = 0.5*ke_val;

      }

      // Accumulate values in all dofs, count the zones.
      for (int j = 0; j < dofs.Size(); j++)
      {
         int ldof = dofs[j];
         ke(ldof) += vals[j];
         zones_per_vdof[ldof]++;
      }
   }

   // Count the zones globally.
   GroupCommunicator &gcomm = ke.ParFESpace()->GroupComm();
   gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
   gcomm.Bcast(zones_per_vdof);

   // Accumulate for all vdofs.
   gcomm.Reduce<real_t>(ke.GetData(), GroupCommunicator::Sum);
   gcomm.Bcast<real_t>(ke.GetData());

   // Compute means
   for (int i = 0; i < ke.Size(); i++)
   {
      const int nz = zones_per_vdof[i];
      if (nz)
      {
         ke(i) /= nz;
      }
   }
}
