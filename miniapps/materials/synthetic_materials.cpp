// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details

// ===========================================================================
//
//        Mini-App: surrogate model for imperfect materials.
//
//  Details: refer to README
//
//  Runs:
//    mpirun -np 4 ./miniapps/materials/synthetic_materials
//
// ===========================================================================

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <math.h>
#include <string>

#include "material_metrics.hpp"
#include "solvers.hpp"
#include "util.hpp"
#include "visualizer.hpp"

using namespace std;
using namespace mfem;

enum TopologicalSupport { kParticles, kOctetTruss };

int main(int argc, char *argv[]) {
  // 0. Initialize MPI.
  Mpi::Init(argc, argv);
  int num_procs = Mpi::WorldSize();
  int myid = Mpi::WorldRank();
  Hypre::Init();

  // 1. Parse command-line options.
  const char *mesh_file = "../../data/ref-cube.mesh";
  int order = 1;
  int num_refs = 3;
  int num_parallel_refs = 3;
  int number_of_particles = 3;
  int topological_support = TopologicalSupport::kParticles;
  double nu = 1.0;
  double tau = 1.0;
  double zeta = 1.0;
  double l1 = 1;
  double l2 = 1;
  double l3 = 1;
  double e1 = 0;
  double e2 = 0;
  double e3 = 0;
  double pl1 = 1.0;
  double pl2 = 1.0;
  double pl3 = 1.0;
  bool paraview_export = true;
  bool glvis_export = true;

  OptionsParser args(argc, argv);
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree) or -1 for"
                 " isoparametric space.");
  args.AddOption(&num_refs, "-r", "--refs", "Number of uniform refinements");
  args.AddOption(&num_parallel_refs, "-rp", "--refs-parallel",
                 "Number of uniform refinements");
  args.AddOption(&topological_support, "-top", "--topology",
                 "Topological support. 0 particles, 1 octet-truss");
  args.AddOption(&nu, "-nu", "--nu", "Fractional exponent nu (smoothness)");
  args.AddOption(&tau, "-t", "--tau", "Parameter for topology generation");
  args.AddOption(&zeta, "-z", "--zeta",
                 "Parameter to scale the mixing of topology and randomness");
  args.AddOption(&l1, "-l1", "--l1",
                 "First component of diagonal core of theta");
  args.AddOption(&l2, "-l2", "--l2",
                 "Second component of diagonal core of theta");
  args.AddOption(&l3, "-l3", "--l3",
                 "Third component of diagonal core of theta");
  args.AddOption(&e1, "-e1", "--e1", "First euler angle for rotation of theta");
  args.AddOption(&e2, "-e2", "--e2",
                 "Second euler angle for rotation of theta");
  args.AddOption(&e3, "-e3", "--e3", "Third euler angle for rotation of theta");
  args.AddOption(&pl1, "-pl1", "--pl1", "Length scale 1 of particles");
  args.AddOption(&pl2, "-pl2", "--pl2", "Length scale 2 of particles");
  args.AddOption(&pl3, "-pl3", "--pl3", "Length scale 3 of particles");
  args.AddOption(&number_of_particles, "-n", "--number-of-particles",
                 "Number of particles");
  args.AddOption(&paraview_export, "-pvis", "--paraview-visualization",
                 "-no-pvis", "--no-paraview-visualization",
                 "Enable or disable ParaView visualization.");
  args.AddOption(&glvis_export, "-gvis", "--glvis-visualization", "-no-gvis",
                 "--no-glvis-visualization",
                 "Enable or disable GLVis visualization.");
  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
  if (Mpi::Root()) {
    args.PrintOptions(cout);
  }

  // 2. Read the mesh from the given mesh file.
  Mesh mesh(mesh_file, 1, 1);
  int dim = mesh.Dimension();

  // 4. Refine the mesh to increase the resolution.
  for (int i = 0; i < num_refs; i++) {
    mesh.UniformRefinement();
  }
  ParMesh pmesh(MPI_COMM_WORLD, mesh);
  mesh.Clear();
  for (int i = 0; i < num_parallel_refs; i++) {
    pmesh.UniformRefinement();
  }

  // 5. Define a finite element space on the mesh.
  H1_FECollection fec(order, dim);
  ParFiniteElementSpace fespace(&pmesh, &fec);
  HYPRE_BigInt size = fespace.GlobalTrueVSize();
  if (Mpi::Root()) {
    cout << "Number of finite element unknowns: " << size << endl;
  }

  // 6. Boundary conditions
  const Array<int> ess_tdof_list;

  // ========================================================================
  // II. Generate topological support
  // ========================================================================

  // II.1 Define the metric for the topological support.
  MaterialTopology *mdm = nullptr;
  if (topological_support == TopologicalSupport::kOctetTruss) {
    mdm = new OctetTrussTopology();
  } else {
    // Create the same random particles on all processors.
    std::vector<double> random_positions(3 * number_of_particles);
    std::vector<double> random_rotations(9 * number_of_particles);
    if (Mpi::Root()) {
      if (topological_support != TopologicalSupport::kParticles) {
        mfem::out << "Warning: Selected topological support not valid.\n"
                  << "         Fall back to kParticles." << std::endl;
      }
      // Generate random positions and rotations. We generate them on the root
      // process and then broadcast them to all processes because we need the
      // same random positions and rotations on all processes.
      FillWithRandomNumbers(random_positions, 0.2, 0.8);
      FillWithRandomRotations(random_rotations);
    }

    // Broadcast the random positions and rotations to all processes.
    MPI_Bcast(random_positions.data(), 3 * number_of_particles, MPI_DOUBLE, 0,
              MPI_COMM_WORLD);
    MPI_Bcast(random_rotations.data(), 9 * number_of_particles, MPI_DOUBLE, 0,
              MPI_COMM_WORLD);

    mdm =
        new ParticleTopology(pl1, pl2, pl3, random_positions, random_rotations);
  }

  // II.2 Define lambda to wrap the call to the distance metric.
  auto topo = [&mdm, &tau, &zeta](const Vector &x) {
    return (tau - mdm->ComputeMetric(x));
  };

  // II.1 Create a grid funtion for the topological support.
  FunctionCoefficient topo_coeff(topo);
  ParGridFunction v(&fespace);
  v.ProjectCoefficient(topo_coeff);

  // ========================================================================
  // III. Generate random imperfections via fractional PDE
  // ========================================================================

  /// III.1 Define the fractional PDE solution
  ParGridFunction u(&fespace);
  u = 0.0;

  // III.2 Define Diffusion Tensor for the anisotropic SPDE method. The function
  // below creates a diagonal matrix (l1, l2, l3)^2 and rotates it by the Euler
  // angles (e1, e2, e3). nu and dim normalize.
  auto diffusion_tensor =
      ConstructMatrixCoefficient(l1, l2, l3, e1, e2, e3, nu, dim);
  MatrixConstantCoefficient diffusion_coefficient(diffusion_tensor);

  // III.3 Define the right hand side, for us this is a normalized white noise.
  ParLinearForm b(&fespace);
  auto *WhiteNoise = new WhiteGaussianNoiseDomainLFIntegrator(4000);
  b.AddDomainIntegrator(WhiteNoise);
  b.Assemble();
  double normalization = ConstructNormalizationCoefficient(nu, l1, l2, l3, dim);
  b *= normalization;

  // III.4 Solve the SPDE problem
  materials::SPDESolver solver(diffusion_coefficient, nu, ess_tdof_list,
                               &fespace);
  solver.Solve(b, u);

  // ========================================================================
  // III. Combine topological support and random field
  // ========================================================================

  ParGridFunction w(&fespace);
  w = 0.0;
  w.Add(zeta, u);
  w.Add(1.0 - zeta, v);

  // ========================================================================
  // VI. Export visualization to ParaView and GLVis
  // ========================================================================

  materials::Visualizer vis(pmesh, order, u, v, w);
  if (paraview_export) {
    vis.ExportToParaView();
  }
  if (glvis_export) {
    vis.SendToGLVis();
  }

  delete mdm;
  return 0;
}
