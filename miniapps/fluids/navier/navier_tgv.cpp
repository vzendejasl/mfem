// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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
// AMR + VisIt quick start
// -----------------------
// Build from this directory:
//
//   make navier_tgv
//
// Then run this small, checked two-rank case (the mesh changes at t=0.01):
//
//   mpirun -np 2 ./navier_tgv -es 1 -o 3 -dt 0.002 -tf 0.05 \
//     -amr -amr-t 0.01 -amr-th 0.5 -amr-i 0 -amr-max-events 1 \
//     -amr-rebalance -amr-check -mi 5 -run tgv_amr_try
//
// This always writes native MFEM VisIt collections; -vis controls only the
// separate GLVis socket visualization. The command above creates, at minimum,
// the initial, AMR-event, and final snapshots:
//
//   tgv_amr_try_visit_000000.mfem_root
//   tgv_amr_try_visit_000005.mfem_root
//   tgv_amr_try_visit_000025.mfem_root
//
// Open a *.mfem_root file in VisIt (or use `visit <file>.mfem_root`). The
// collection contains velocity, pressure, vorticity, and qcriterion. The
// root-rank scalar history is tgv_amr_try_metrics_p_3.txt. For a resolution-
// driven study, add `-amr-keta 0.15`; add `-amr-enforce-keta` only when the
// selected global DOF budget is sufficient to meet that conservative target.

#include "navier_solver.hpp"
#include <algorithm>
#include <climits>
#include <fstream>
#include <limits>
#include <vector>

using namespace mfem;
using namespace navier;

struct s_NavierContext
{
   int element_subdivisions = 1;
   int order = 4;
   real_t kinvis = 1.0 / 1600.0;
   real_t t_final = 10 * 1e-3;
   real_t dt = 1e-3;
   bool pa = true;
   bool ni = false;
   bool visualization = false;
   bool checkres = false;
   bool amr = false;
   real_t amr_time = 1e-3;
   bool amr_enstrophy = true;
   int amr_stride = 8;
   real_t amr_threshold = 0.25;
   real_t amr_kmax_eta_target = 0.0;
   bool amr_enforce_kmax_eta = false;
   int amr_interval = 0;
   int amr_max_events = 1;
   int amr_max_dofs = 0;
   int amr_max_marked_per_rank = 0;
   int amr_max_marked_global = 0;
   int metrics_interval = 0;
   bool amr_preserve_history = true;
   bool amr_rebalance = false;
   bool amr_check = false;
   std::string run_name = "tgv";
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
   QuantitiesOfInterest(ParMesh *mesh) : pmesh(mesh)
   {
      h1fec = new H1_FECollection(1, pmesh->Dimension());
      h1fes = new ParFiniteElementSpace(pmesh, h1fec);
      BuildVolumeForm();
   };

   void UpdateAfterMeshChange()
   {
      h1fes->Update();
      delete mass_lf;
      mass_lf = nullptr;
      BuildVolumeForm();
   }

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
      real_t integral = 0.0;
      FiniteElementSpace *fes = w.FESpace();
      for (int e = 0; e < fes->GetNE(); ++e)
      {
         const FiniteElement *fe = fes->GetFE(e);
         const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                   2 * fe->GetOrder());
         w.GetValues(e, ir, wx, 1);
         w.GetValues(e, ir, wy, 2);
         w.GetValues(e, ir, wz, 3);

         ElementTransformation *T = fes->GetElementTransformation(e);
         for (int q = 0; q < ir.GetNPoints(); ++q)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            T->SetIntPoint(&ip);
            integral += 0.5 * ip.weight * T->Weight()
                        * (wx(q) * wx(q) + wy(q) * wy(q) + wz(q) * wz(q));
         }
      }

      real_t global_integral = 0.0;
      MPI_Allreduce(&integral, &global_integral, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
      return global_integral / volume;
   }

   ~QuantitiesOfInterest()
   {
      delete mass_lf;
      delete h1fes;
      delete h1fec;
   };

private:
   void BuildVolumeForm()
   {
      onecoeff.constant = 1.0;
      mass_lf = new ParLinearForm(h1fes);
      mass_lf->AddDomainIntegrator(new DomainLFIntegrator(onecoeff));
      mass_lf->Assemble();

      ParGridFunction one_gf(h1fes);
      one_gf.ProjectCoefficient(onecoeff);
      volume = mass_lf->operator()(one_gf);
   }

   ParMesh *pmesh;
   ConstantCoefficient onecoeff;
   H1_FECollection *h1fec = nullptr;
   ParFiniteElementSpace *h1fes = nullptr;
   ParLinearForm *mass_lf = nullptr;
   real_t volume;
};

struct ResolutionMetrics
{
   long long elements = 0;
   long long velocity_true_dofs = 0;
   long long pressure_true_dofs = 0;
   real_t kinetic_energy = 0.0;
   real_t enstrophy = 0.0;
   real_t dissipation = 0.0;
   real_t eta = 0.0;
   real_t h_min_over_p = 0.0;
   real_t h_max_over_p = 0.0;
   real_t kmax_eta_coarsest = 0.0;
   real_t kmax_eta_finest = 0.0;
};

ResolutionMetrics ComputeResolutionMetrics(ParMesh &pmesh,
                                           ParGridFunction &u,
                                           ParGridFunction &p,
                                           ParGridFunction &w,
                                           QuantitiesOfInterest &qoi,
                                           real_t kinvis)
{
   ResolutionMetrics metrics;
   metrics.elements = pmesh.GetGlobalNE();
   metrics.velocity_true_dofs = u.ParFESpace()->GlobalTrueVSize();
   metrics.pressure_true_dofs = p.ParFESpace()->GlobalTrueVSize();
   metrics.kinetic_energy = qoi.ComputeKineticEnergy(u);
   metrics.enstrophy = qoi.ComputeEnstrophy(w);
   metrics.dissipation = 2.0 * kinvis * metrics.enstrophy;
   metrics.eta = metrics.dissipation > 0.0
                 ? pow(kinvis * kinvis * kinvis / metrics.dissipation, 0.25)
                 : 0.0;

   const ParFiniteElementSpace *fes = u.ParFESpace();
   real_t local_h_min = std::numeric_limits<real_t>::max();
   real_t local_h_max = 0.0;
   for (int e = 0; e < fes->GetNE(); ++e)
   {
      const int order = std::max(1, fes->GetElementOrder(e));
      const real_t h_over_p = pmesh.GetElementSize(e, 1) / order;
      local_h_min = std::min(local_h_min, h_over_p);
      local_h_max = std::max(local_h_max, h_over_p);
   }
   MPI_Allreduce(&local_h_min, &metrics.h_min_over_p, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_MIN, pmesh.GetComm());
   MPI_Allreduce(&local_h_max, &metrics.h_max_over_p, 1,
                 MPITypeMap<real_t>::mpi_type, MPI_MAX, pmesh.GetComm());

   metrics.kmax_eta_coarsest = M_PI * metrics.eta / metrics.h_max_over_p;
   metrics.kmax_eta_finest = M_PI * metrics.eta / metrics.h_min_over_p;
   return metrics;
}

void WriteResolutionMetrics(FILE *file, real_t time, int cycle,
                            const ResolutionMetrics &metrics,
                            int marked_elements = 0,
                            real_t amr_setup_seconds = 0.0)
{
   if (file == nullptr) { return; }
   fprintf(file,
           "%20.16e %8d %12lld %16lld %16lld %20.16e %20.16e %20.16e "
           "%20.16e %20.16e %20.16e %20.16e %20.16e %12d %20.16e\n",
           time, cycle, metrics.elements, metrics.velocity_true_dofs,
           metrics.pressure_true_dofs, metrics.kinetic_energy,
           metrics.enstrophy, metrics.dissipation, metrics.eta,
           metrics.h_min_over_p, metrics.h_max_over_p,
           metrics.kmax_eta_coarsest, metrics.kmax_eta_finest,
           marked_elements, amr_setup_seconds);
   fflush(file);
}

struct MarkCandidate
{
   real_t priority;
   int element;
};

struct LoadBalanceMetrics
{
   int elements_min = 0;
   real_t elements_mean = 0.0;
   int elements_max = 0;
   int velocity_dofs_min = 0;
   real_t velocity_dofs_mean = 0.0;
   int velocity_dofs_max = 0;

   real_t ElementImbalance() const
   {
      return elements_mean > 0.0 ? elements_max / elements_mean : 0.0;
   }

   real_t VelocityDofImbalance() const
   {
      return velocity_dofs_mean > 0.0
             ? velocity_dofs_max / velocity_dofs_mean : 0.0;
   }
};

LoadBalanceMetrics ComputeLoadBalanceMetrics(ParMesh &pmesh,
                                              ParGridFunction &u)
{
   LoadBalanceMetrics metrics;
   int nranks = 0;
   MPI_Comm_size(pmesh.GetComm(), &nranks);

   const int local_elements = pmesh.GetNE();
   const int local_velocity_dofs = u.ParFESpace()->GetTrueVSize();
   int elements_sum = 0;
   int velocity_dofs_sum = 0;
   MPI_Allreduce(&local_elements, &elements_sum, 1, MPI_INT, MPI_SUM,
                 pmesh.GetComm());
   MPI_Allreduce(&local_elements, &metrics.elements_min, 1, MPI_INT, MPI_MIN,
                 pmesh.GetComm());
   MPI_Allreduce(&local_elements, &metrics.elements_max, 1, MPI_INT, MPI_MAX,
                 pmesh.GetComm());
   MPI_Allreduce(&local_velocity_dofs, &velocity_dofs_sum, 1, MPI_INT, MPI_SUM,
                 pmesh.GetComm());
   MPI_Allreduce(&local_velocity_dofs, &metrics.velocity_dofs_min, 1, MPI_INT,
                 MPI_MIN, pmesh.GetComm());
   MPI_Allreduce(&local_velocity_dofs, &metrics.velocity_dofs_max, 1, MPI_INT,
                 MPI_MAX, pmesh.GetComm());
   metrics.elements_mean = static_cast<real_t>(elements_sum) / nranks;
   metrics.velocity_dofs_mean = static_cast<real_t>(velocity_dofs_sum) / nranks;
   return metrics;
}

Array<int> SelectGlobalCandidates(const std::vector<MarkCandidate> &local,
                                  int global_limit, MPI_Comm comm)
{
   int rank = 0;
   int nranks = 0;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &nranks);

   const int local_count = static_cast<int>(local.size());
   std::vector<int> counts(nranks), displacements(nranks);
   MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);

   int global_count = 0;
   for (int r = 0; r < nranks; ++r)
   {
      displacements[r] = global_count;
      global_count += counts[r];
   }

   std::vector<real_t> local_priorities(local_count);
   std::vector<int> local_elements(local_count);
   for (int i = 0; i < local_count; ++i)
   {
      local_priorities[i] = local[i].priority;
      local_elements[i] = local[i].element;
   }

   std::vector<real_t> all_priorities(global_count);
   std::vector<int> all_elements(global_count);
   MPI_Allgatherv(local_priorities.data(), local_count,
                  MPITypeMap<real_t>::mpi_type, all_priorities.data(),
                  counts.data(), displacements.data(),
                  MPITypeMap<real_t>::mpi_type, comm);
   MPI_Allgatherv(local_elements.data(), local_count, MPI_INT,
                  all_elements.data(), counts.data(), displacements.data(),
                  MPI_INT, comm);

   struct GlobalCandidate
   {
      real_t priority;
      int rank;
      int element;
   };
   std::vector<GlobalCandidate> all_candidates;
   all_candidates.reserve(global_count);
   for (int r = 0; r < nranks; ++r)
   {
      for (int i = 0; i < counts[r]; ++i)
      {
         const int j = displacements[r] + i;
         all_candidates.push_back({all_priorities[j], r, all_elements[j]});
      }
   }
   std::sort(all_candidates.begin(), all_candidates.end(),
             [](const GlobalCandidate &a, const GlobalCandidate &b)
   {
      if (a.priority != b.priority) { return a.priority > b.priority; }
      if (a.rank != b.rank) { return a.rank < b.rank; }
      return a.element < b.element;
   });

   const int selected_count = global_limit > 0
                              ? std::min(global_limit, global_count)
                              : global_count;
   Array<int> selected;
   for (int i = 0; i < selected_count; ++i)
   {
      if (all_candidates[i].rank == rank)
      {
         selected.Append(all_candidates[i].element);
      }
   }
   return selected;
}

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

// Elementwise, volume-normalized enstrophy: int_K |curl u|^2 / |K|.
void ComputeElementEnstrophy(ParGridFunction &vorticity, Vector &indicator)
{
   FiniteElementSpace *fes = vorticity.FESpace();
   indicator.SetSize(fes->GetNE());

   Vector wx, wy, wz;
   for (int e = 0; e < fes->GetNE(); ++e)
   {
      const FiniteElement *fe = fes->GetFE(e);
      const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(),
                                                2 * fe->GetOrder());
      vorticity.GetValues(e, ir, wx, 1);
      vorticity.GetValues(e, ir, wy, 2);
      vorticity.GetValues(e, ir, wz, 3);

      ElementTransformation *T = fes->GetElementTransformation(e);
      real_t enstrophy = 0.0;
      real_t volume = 0.0;
      for (int q = 0; q < ir.GetNPoints(); ++q)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T->SetIntPoint(&ip);
         const real_t weight = ip.weight * T->Weight();
         enstrophy += weight * (wx(q) * wx(q) + wy(q) * wy(q) + wz(q) * wz(q));
         volume += weight;
      }
      indicator(e) = enstrophy / volume;
   }
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   OptionsParser args(argc, argv);
   args.AddOption(&ctx.element_subdivisions,
                  "-es",
                  "--element-subdivisions",
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
   args.AddOption(
      &ctx.checkres,
      "-cr",
      "--checkresult",
      "-no-cr",
      "--no-checkresult",
      "Enable or disable checking of the result. Returns -1 on failure.");
   args.AddOption(&ctx.amr,
                  "-amr",
                  "--enable-amr",
                  "-no-amr",
                  "--disable-amr",
                  "Enable one refine-only AMR event.");
   args.AddOption(&ctx.amr_time,
                  "-amr-t",
                  "--amr-time",
                  "Time at which to perform the one AMR event.");
   args.AddOption(&ctx.amr_enstrophy,
                  "-amr-enstrophy",
                  "--amr-enstrophy",
                  "-amr-index",
                  "--amr-index",
                  "Mark with normalized element enstrophy instead of indices.");
   args.AddOption(&ctx.amr_stride,
                  "-amr-s",
                  "--amr-stride",
                  "Refine local elements whose index is divisible by this value.");
   args.AddOption(&ctx.amr_threshold,
                  "-amr-th",
                  "--amr-threshold",
                  "Refine elements with indicator greater than this fraction of the global maximum.");
   args.AddOption(&ctx.amr_kmax_eta_target,
                  "-amr-keta",
                  "--amr-kmax-eta-target",
                  "Require selected elements to have local kmax*eta below this target; zero disables the resolution filter.");
   args.AddOption(&ctx.amr_enforce_kmax_eta,
                  "-amr-enforce-keta",
                  "--amr-enforce-kmax-eta",
                  "-no-amr-enforce-keta",
                  "--no-amr-enforce-kmax-eta",
                  "Select all under-resolved elements, not only high-enstrophy ones, until the kmax*eta target is met or the budget is exhausted.");
   args.AddOption(&ctx.amr_interval,
                  "-amr-i",
                  "--amr-interval",
                  "Repeat AMR every this many timesteps after the first event; zero performs one event.");
   args.AddOption(&ctx.amr_max_events,
                  "-amr-max-events",
                  "--amr-max-events",
                  "Maximum number of AMR events.");
   args.AddOption(&ctx.amr_max_dofs,
                  "-amr-max-dofs",
                  "--amr-max-dofs",
                  "Do not start another AMR event once velocity true DOFs reach this global ceiling; zero disables it.");
   args.AddOption(&ctx.amr_max_marked_per_rank,
                  "-amr-max-marked",
                  "--amr-max-marked-per-rank",
                  "Cap the number of enstrophy-marked elements per rank; zero disables the cap.");
   args.AddOption(&ctx.amr_max_marked_global,
                  "-amr-max-marked-global",
                  "--amr-max-marked-global",
                  "Globally select at most this many highest-priority AMR elements; zero disables the cap.");
   args.AddOption(&ctx.metrics_interval,
                  "-mi",
                  "--metrics-interval",
                  "Write resolution metrics every this many timesteps; zero records only AMR events and the endpoints.");
   args.AddOption(&ctx.amr_preserve_history,
                  "-amr-history",
                  "--amr-preserve-history",
                  "-amr-restart-bdf",
                  "--amr-restart-bdf",
                  "Transfer BDF velocity history instead of restarting BDF1 after AMR.");
   args.AddOption(&ctx.amr_rebalance,
                  "-amr-rebalance",
                  "--amr-rebalance",
                  "-no-amr-rebalance",
                  "--no-amr-rebalance",
                  "Rebalance the refined mesh and transfer all physical state a second time.");
   args.AddOption(&ctx.amr_check,
                  "-amr-check",
                  "--amr-check",
                  "-no-amr-check",
                  "--no-amr-check",
                  "Fail if AMR transfer or the first post-AMR linear solves do not pass checks.");
   args.AddOption(&ctx.run_name,
                  "-run",
                  "--run-name",
                  "Prefix for VisIt output and scalar diagnostic files.");
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
   MFEM_VERIFY(ctx.metrics_interval >= 0,
               "The metrics interval cannot be negative.");

   Mesh orig_mesh("../../../data/periodic-cube.mesh");
   Mesh mesh = Mesh::MakeRefined(orig_mesh, ctx.element_subdivisions,
                                 BasisType::ClosedUniform);
   orig_mesh.Clear();

   mesh.EnsureNodes();
   GridFunction *nodes = mesh.GetNodes();
   *nodes *= M_PI;

   if (ctx.amr)
   {
      // Local hex refinement requires MFEM's nonconforming mesh support.
      mesh.EnsureNCMesh();
   }

   int nel = mesh.GetNE();
   if (Mpi::Root())
   {
      mfem::out << "Number of elements: " << nel << std::endl;
   }

   auto *pmesh = new ParMesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

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
   int amr_events = 0;
   int last_amr_step = -1;
   bool saw_post_amr_step = false;
   bool post_amr_solves_converged = true;
   bool amr_dof_cap_reached = false;
   real_t max_amr_ke_change = 0.0;

   flowsolver.Setup(dt);

   ParGridFunction *u_gf = flowsolver.GetCurrentVelocity();
   ParGridFunction *p_gf = flowsolver.GetCurrentPressure();

   ParGridFunction w_gf(*u_gf);
   ParGridFunction q_gf(*p_gf);
   flowsolver.ComputeCurl3D(*u_gf, w_gf);
   ComputeQCriterion(*u_gf, q_gf);

   QuantitiesOfInterest kin_energy(pmesh);

   VisItDataCollection visit_dc(MPI_COMM_WORLD, ctx.run_name + "_visit", pmesh);
   visit_dc.SetLevelsOfDetail(ctx.order);
   visit_dc.SetCycle(0);
   visit_dc.SetTime(t);
   visit_dc.RegisterField("velocity", u_gf);
   visit_dc.RegisterField("pressure", p_gf);
   visit_dc.RegisterField("vorticity", &w_gf);
   visit_dc.RegisterField("qcriterion", &q_gf);
   visit_dc.Save();

   real_t u_inf_loc = u_gf->Normlinf();
   real_t p_inf_loc = p_gf->Normlinf();
   real_t u_inf = GlobalLpNorm(infinity(), u_inf_loc, MPI_COMM_WORLD);
   real_t p_inf = GlobalLpNorm(infinity(), p_inf_loc, MPI_COMM_WORLD);
   real_t ke = kin_energy.ComputeKineticEnergy(*u_gf);

   std::string fname = ctx.run_name + "_out_p_" + std::to_string(ctx.order)
                       + ".txt";
   std::string metrics_fname = ctx.run_name + "_metrics_p_"
                               + std::to_string(ctx.order) + ".txt";
   FILE *f = NULL;
   FILE *metrics_file = NULL;

   if (Mpi::Root())
   {
      int nel1d = static_cast<int>(std::round(pow(nel, 1.0 / 3.0)));
      int ngridpts = p_gf->ParFESpace()->GlobalVSize();
      printf("%11s %11s %11s %11s %11s\n", "Time", "dt", "u_inf", "p_inf", "ke");
      printf("%.5E %.5E %.5E %.5E %.5E\n", t, dt, u_inf, p_inf, ke);

      f = fopen(fname.c_str(), "w");
      fprintf(f, "3D Taylor Green Vortex\n");
      fprintf(f, "order = %d\n", ctx.order);
      fprintf(f, "grid = %d x %d x %d\n", nel1d, nel1d, nel1d);
      fprintf(f, "dofs per component = %d\n", ngridpts);
      fprintf(f, "=================================================\n");
      fprintf(f, "        time                   kinetic energy\n");
      fprintf(f, "%20.16e     %20.16e\n", t, ke);
      fflush(f);
      fflush(stdout);

      metrics_file = fopen(metrics_fname.c_str(), "w");
      fprintf(metrics_file,
              "# time cycle elements velocity_true_dofs pressure_true_dofs "
              "kinetic_energy enstrophy dissipation eta hmin_over_p hmax_over_p "
              "kmax_eta_coarsest kmax_eta_finest marked_elements amr_setup_seconds\n");
   }

   const ResolutionMetrics initial_metrics =
      ComputeResolutionMetrics(*pmesh, *u_gf, *p_gf, w_gf, kin_energy, ctx.kinvis);
   if (Mpi::Root())
   {
      WriteResolutionMetrics(metrics_file, t, 0, initial_metrics);
   }

   int global_step = 0;
   int bdf_step = 0;
   for (; !last_step; )
   {
      bool amr_this_step = false;
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      flowsolver.Step(t, dt, bdf_step);
      ++global_step;
      ++bdf_step;

      if (last_amr_step >= 0 && global_step > last_amr_step)
      {
         saw_post_amr_step = true;
         post_amr_solves_converged = post_amr_solves_converged
                                     && flowsolver.LastStepConverged();
      }

      // Record the pre-AMR state as well as AMR-event states. This keeps the
      // history complete when a scheduled event correctly selects no elements.
      if (ctx.metrics_interval > 0 && global_step % ctx.metrics_interval == 0
          && !last_step)
      {
         flowsolver.ComputeCurl3D(*u_gf, w_gf);
         const ResolutionMetrics periodic_metrics =
            ComputeResolutionMetrics(*pmesh, *u_gf, *p_gf, w_gf, kin_energy,
                                     ctx.kinvis);
         if (Mpi::Root())
         {
            WriteResolutionMetrics(metrics_file, t, global_step,
                                   periodic_metrics);
         }
      }

      const bool first_amr_event = amr_events == 0;
      const bool recurring_amr_event = ctx.amr_interval > 0
                                       && global_step % ctx.amr_interval == 0;
      const bool amr_is_scheduled = ctx.amr && !amr_dof_cap_reached
                                    && amr_events < ctx.amr_max_events
                                    && t >= ctx.amr_time
                                    && (first_amr_event || recurring_amr_event);
      if (amr_is_scheduled)
      {
         MFEM_VERIFY(ctx.amr_max_events > 0,
                     "The maximum number of AMR events must be positive.");
         MFEM_VERIFY(!ctx.amr_enstrophy ||
                     (ctx.amr_threshold > 0.0 && ctx.amr_threshold <= 1.0),
                     "The enstrophy threshold must be in (0, 1].");
         MFEM_VERIFY(ctx.amr_enstrophy || ctx.amr_stride > 0,
                     "The index-based AMR marking stride must be positive.");
         MFEM_VERIFY(ctx.amr_max_marked_per_rank >= 0,
                     "The per-rank AMR marking cap cannot be negative.");
         MFEM_VERIFY(ctx.amr_max_marked_global >= 0,
                     "The global AMR marking cap cannot be negative.");
         MFEM_VERIFY(ctx.amr_kmax_eta_target >= 0.0,
                     "The kmax*eta target cannot be negative.");
         MFEM_VERIFY(!ctx.amr_enforce_kmax_eta
                     || ctx.amr_kmax_eta_target > 0.0,
                     "Strict kmax*eta enforcement requires a positive target.");

         const int current_velocity_dofs =
            u_gf->ParFESpace()->GlobalTrueVSize();
         if (ctx.amr_max_dofs > 0 && current_velocity_dofs >= ctx.amr_max_dofs)
         {
            amr_dof_cap_reached = true;
            if (Mpi::Root())
            {
               mfem::out << "Skipping AMR: velocity true-DOF ceiling ("
                         << ctx.amr_max_dofs << ") has been reached.\n";
            }
            continue;
         }

         const long long elements_before = pmesh->GetGlobalNE();
         const int velocity_dofs_before =
            u_gf->ParFESpace()->GlobalTrueVSize();
         const int pressure_dofs_before =
            p_gf->ParFESpace()->GlobalTrueVSize();
         const real_t ke_before = kin_energy.ComputeKineticEnergy(*u_gf);
         const real_t div_before = flowsolver.ComputeDiscreteDivergenceNorm();

         std::vector<MarkCandidate> local_candidates;
         real_t global_indicator_max = 0.0;
         real_t eta_before = 0.0;
         if (ctx.amr_enstrophy)
         {
            flowsolver.ComputeCurl3D(*u_gf, w_gf);
            Vector indicator;
            ComputeElementEnstrophy(w_gf, indicator);
            const ResolutionMetrics metrics_before =
               ComputeResolutionMetrics(*pmesh, *u_gf, *p_gf, w_gf,
                                        kin_energy, ctx.kinvis);
            eta_before = metrics_before.eta;
            const real_t local_indicator_max = indicator.Max();
            MPI_Allreduce(&local_indicator_max, &global_indicator_max, 1,
                          MPITypeMap<real_t>::mpi_type, MPI_MAX,
                          pmesh->GetComm());
            const real_t threshold = ctx.amr_threshold * global_indicator_max;
            for (int e = 0; e < pmesh->GetNE(); ++e)
            {
               const int order = std::max(1, u_gf->ParFESpace()->GetElementOrder(e));
               const real_t h_over_p = pmesh->GetElementSize(e, 1) / order;
               const real_t local_kmax_eta = M_PI * eta_before / h_over_p;
               const bool needs_resolution = ctx.amr_kmax_eta_target == 0.0
                                             || local_kmax_eta
                                                < ctx.amr_kmax_eta_target;
               const bool enstrophy_selected = indicator(e) >= threshold;
               if (needs_resolution
                   && (enstrophy_selected || ctx.amr_enforce_kmax_eta))
               {
                  const real_t resolution_deficit =
                     ctx.amr_kmax_eta_target > 0.0
                     ? (ctx.amr_kmax_eta_target - local_kmax_eta)
                       / ctx.amr_kmax_eta_target
                     : 1.0;
                  const real_t enstrophy_weight = indicator(e) / global_indicator_max;
                  const real_t priority = ctx.amr_enforce_kmax_eta
                                          ? resolution_deficit * (1.0 + enstrophy_weight)
                                          : enstrophy_weight * resolution_deficit;
                  local_candidates.push_back({priority, e});
               }
            }
         }
         else
         {
            for (int e = 0; e < pmesh->GetNE(); ++e)
            {
               if (e % ctx.amr_stride == 0)
               {
                  local_candidates.push_back({-static_cast<real_t>(e), e});
               }
            }
         }

         std::sort(local_candidates.begin(), local_candidates.end(),
                   [](const MarkCandidate &a, const MarkCandidate &b)
         {
            return a.priority > b.priority;
         });
         if (ctx.amr_max_marked_per_rank > 0
             && static_cast<int>(local_candidates.size())
                > ctx.amr_max_marked_per_rank)
         {
            local_candidates.resize(ctx.amr_max_marked_per_rank);
         }

         int local_candidates_count = static_cast<int>(local_candidates.size());
         int global_candidates_count = 0;
         MPI_Allreduce(&local_candidates_count, &global_candidates_count, 1,
                       MPI_INT, MPI_SUM, pmesh->GetComm());
         if (global_candidates_count == 0)
         {
            if (Mpi::Root())
            {
               mfem::out << "AMR marker selected no elements satisfying the "
                         << "enstrophy/resolution policy.\n";
            }
            continue;
         }

         int global_mark_limit = ctx.amr_max_marked_global;
         if (ctx.amr_max_dofs > 0)
         {
            const int order = std::max(1, u_gf->ParFESpace()->GetElementOrder(0));
            long long safe_velocity_dofs_per_element = 7 * u_gf->ParFESpace()->GetVDim();
            for (int d = 0; d < pmesh->Dimension(); ++d)
            {
               safe_velocity_dofs_per_element *= order + 1;
            }
            const long long remaining_dofs = ctx.amr_max_dofs - current_velocity_dofs;
            const long long safe_mark_limit = remaining_dofs
                                              / safe_velocity_dofs_per_element;
            if (safe_mark_limit <= 0)
            {
               amr_dof_cap_reached = true;
               if (Mpi::Root())
               {
                  mfem::out << "Skipping AMR: the conservative DOF budget "
                            << "does not permit another refined element.\n";
               }
               continue;
            }
            const int safe_mark_limit_int = static_cast<int>(
                                           std::min(safe_mark_limit,
                                                    static_cast<long long>(INT_MAX)));
            global_mark_limit = global_mark_limit > 0
                                ? std::min(global_mark_limit, safe_mark_limit_int)
                                : safe_mark_limit_int;
         }

         Array<int> refine_elements = SelectGlobalCandidates(local_candidates,
                                                              global_mark_limit,
                                                              pmesh->GetComm());
         int local_marked = refine_elements.Size();
         int global_marked = 0;
         MPI_Allreduce(&local_marked, &global_marked, 1, MPI_INT, MPI_SUM,
                       pmesh->GetComm());
         if (global_marked == 0)
         {
            if (Mpi::Root()) { mfem::out << "AMR marker selected no elements.\n"; }
            continue;
         }

         StopWatch amr_setup_timer;
         amr_setup_timer.Start();
         flowsolver.PrepareForMeshChange(ctx.amr_preserve_history);
         pmesh->GeneralRefinement(refine_elements);
         flowsolver.UpdateAfterMeshChange(dt, !ctx.amr_preserve_history);
         kin_energy.UpdateAfterMeshChange();

         if (ctx.amr_rebalance)
         {
            flowsolver.PrepareForMeshChange(ctx.amr_preserve_history);
            pmesh->Rebalance();
            flowsolver.UpdateAfterMeshChange(dt, !ctx.amr_preserve_history);
            kin_energy.UpdateAfterMeshChange();
         }
         amr_setup_timer.Stop();
         real_t amr_setup_seconds = amr_setup_timer.RealTime();
         real_t amr_setup_seconds_max = 0.0;
         MPI_Allreduce(&amr_setup_seconds, &amr_setup_seconds_max, 1,
                       MPITypeMap<real_t>::mpi_type, MPI_MAX, pmesh->GetComm());
         visit_dc.SetMesh(MPI_COMM_WORLD, pmesh);

         // These fields are derived quantities, so recompute them instead of
         // transferring their old samples to the new mesh.
         w_gf.SetSpace(u_gf->ParFESpace());
         w_gf = 0.0;
         q_gf.SetSpace(p_gf->ParFESpace());
         q_gf = 0.0;
         flowsolver.ComputeCurl3D(*u_gf, w_gf);
         ComputeQCriterion(*u_gf, q_gf);

         const real_t ke_after = kin_energy.ComputeKineticEnergy(*u_gf);
         const real_t div_after = flowsolver.ComputeDiscreteDivergenceNorm();
         const real_t rel_ke_change = fabs(ke_after - ke_before)
                                      / std::max(fabs(ke_before), real_t(1e-30));
         max_amr_ke_change = std::max(max_amr_ke_change, rel_ke_change);
         const ResolutionMetrics event_metrics =
            ComputeResolutionMetrics(*pmesh, *u_gf, *p_gf, w_gf, kin_energy,
                                     ctx.kinvis);
         const LoadBalanceMetrics load_balance =
            ComputeLoadBalanceMetrics(*pmesh, *u_gf);

         if (Mpi::Root())
         {
            mfem::out << "\n================ AMR EVENT ================\n"
                      << "Time: " << t << "\n"
                      << "Elements: " << elements_before << " -> "
                      << event_metrics.elements << " (global)\n"
                      << "Elements marked: " << global_marked << " ("
                      << 100.0 * global_marked / elements_before << "%)\n"
                      << "Eligible elements: " << global_candidates_count << "\n"
                      << "Global marking limit: "
                      << (global_mark_limit > 0 ? global_mark_limit
                                                 : global_candidates_count)
                      << "\n"
                      << "Indicator: "
                      << (ctx.amr_enstrophy ? "normalized enstrophy" : "index")
                      << (ctx.amr_enstrophy ? ", global maximum " : "")
                      << (ctx.amr_enstrophy ? global_indicator_max : 0.0) << "\n"
                      << "kmax*eta target: " << ctx.amr_kmax_eta_target << "\n"
                      << "Strict kmax*eta enforcement: "
                      << (ctx.amr_enforce_kmax_eta ? "enabled" : "disabled")
                      << "\n"
                      << "Velocity true DOFs: " << velocity_dofs_before
                      << " -> " << u_gf->ParFESpace()->GlobalTrueVSize() << "\n"
                      << "Pressure true DOFs: " << pressure_dofs_before
                      << " -> " << p_gf->ParFESpace()->GlobalTrueVSize() << "\n"
                      << "KE before/after: " << ke_before << " / " << ke_after
                      << " (relative change " << rel_ke_change << ")\n"
                      << "||D u|| before/after: " << div_before << " / "
                      << div_after << "\n"
                      << "AMR setup time (max rank): " << amr_setup_seconds_max
                      << " s\n"
                      << "Load balance, elements (min/mean/max): "
                      << load_balance.elements_min << " / "
                      << load_balance.elements_mean << " / "
                      << load_balance.elements_max << " (max/mean "
                      << load_balance.ElementImbalance() << ")\n"
                      << "Load balance, velocity true DOFs (min/mean/max): "
                      << load_balance.velocity_dofs_min << " / "
                      << load_balance.velocity_dofs_mean << " / "
                      << load_balance.velocity_dofs_max << " (max/mean "
                      << load_balance.VelocityDofImbalance() << ")\n"
                      << "kmax*eta (coarsest/finest): "
                      << event_metrics.kmax_eta_coarsest << " / "
                      << event_metrics.kmax_eta_finest << "\n"
                      << (ctx.amr_preserve_history ?
                          "Transferred BDF history\n" : "Restarting BDF at order 1\n")
                      << "===========================================\n";
            WriteResolutionMetrics(metrics_file, t, global_step, event_metrics,
                                   global_marked, amr_setup_seconds_max);
         }

         if (!ctx.amr_preserve_history)
         {
            bdf_step = 0;
         }
         ++amr_events;
         last_amr_step = global_step;
         amr_this_step = true;
      }

      if (amr_this_step || global_step % 100 == 0 || last_step)
      {
         flowsolver.ComputeCurl3D(*u_gf, w_gf);
         ComputeQCriterion(*u_gf, q_gf);
         visit_dc.SetCycle(global_step);
         visit_dc.SetTime(t);
         visit_dc.Save();
      }

      u_inf_loc = u_gf->Normlinf();
      p_inf_loc = p_gf->Normlinf();
      u_inf = GlobalLpNorm(infinity(), u_inf_loc, MPI_COMM_WORLD);
      p_inf = GlobalLpNorm(infinity(), p_inf_loc, MPI_COMM_WORLD);
      ke = kin_energy.ComputeKineticEnergy(*u_gf);
      if (Mpi::Root())
      {
         printf("%.5E %.5E %.5E %.5E %.5E\n", t, dt, u_inf, p_inf, ke);
         fprintf(f, "%20.16e     %20.16e\n", t, ke);
         fflush(f);
         fflush(stdout);
      }
   }

   flowsolver.PrintTimingData();

   flowsolver.ComputeCurl3D(*u_gf, w_gf);
   const ResolutionMetrics final_metrics =
      ComputeResolutionMetrics(*pmesh, *u_gf, *p_gf, w_gf, kin_energy, ctx.kinvis);
   if (Mpi::Root())
   {
      WriteResolutionMetrics(metrics_file, t, global_step, final_metrics);
      fclose(metrics_file);
      fclose(f);
      mfem::out << "Final kmax*eta (coarsest/finest): "
                << final_metrics.kmax_eta_coarsest << " / "
                << final_metrics.kmax_eta_finest << std::endl;
   }

   if (ctx.amr_check)
   {
      const bool amr_check_passed = amr_events > 0 && saw_post_amr_step
                                    && max_amr_ke_change <= 1e-10
                                    && post_amr_solves_converged
                                    && (!ctx.amr_enforce_kmax_eta
                                        || final_metrics.kmax_eta_coarsest
                                           >= ctx.amr_kmax_eta_target);
      if (!amr_check_passed)
      {
         if (Mpi::Root())
         {
            mfem::err << "AMR check failed: events=" << amr_events
                      << ", max relative KE transfer change=" << max_amr_ke_change
                      << ", post-AMR step observed=" << saw_post_amr_step
                      << ", post-AMR solves converged="
                      << post_amr_solves_converged << std::endl;
         }
         return -1;
      }
   }

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
