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
// 1. Conduit deos not return back time step.Something is wrong with
// pressure.  Visit reload works fine. 
// Should I also implement sidre? I think so. 
// 2. Store Element data at center in binary for effiecnecy?
// 3. Compute fft of data directly?

#include "navier_solver.hpp"
#include "navier_utils.hpp"
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
   bool conduit = false;
   bool restart = true;
   int element_center_cycle = 100;
   int data_dump_cycle = 100;
   bool filter = false;
   bool oversample = true;

} ctx;


bool GetVisit(const s_NavierContext* ctx) { return ctx->visit; }
bool GetConduit(const s_NavierContext* ctx) { return ctx->conduit; }
real_t GetReynum(const s_NavierContext* ctx) { return ctx->reynum; }
int GetNumPts(const s_NavierContext* ctx) { return ctx->num_pts; }
int GetElementSubdivisions(const s_NavierContext* ctx) { return ctx->element_subdivisions; }
int GetElementSubdivisionsParallel(const s_NavierContext* ctx) { return ctx->element_subdivisions_parallel; }
int GetOrder(const s_NavierContext* ctx) { return ctx->order; }
real_t GetKinvis(const s_NavierContext* ctx) { return ctx->kinvis; }
bool GetPA(const s_NavierContext* ctx) { return ctx->pa; }
bool GetNI(const s_NavierContext* ctx) { return ctx->ni; }
real_t GetDt(const s_NavierContext* ctx) { return ctx->dt; }
bool GetOverSample(const s_NavierContext* ctx) { return ctx->oversample; }


// -----------------------------------------------------------------------------
// Linear forcing:  f = α (u − <u>)
// -----------------------------------------------------------------------------
class LinearForcingCoefficient : public VectorCoefficient
{
public:
   /** @param[in] alpha   dimensional‑less forcing constant (≈ 0.1 – 0.2 in
    *                     Rosales & Meneveau, 2005)
    *  @param[in] U       current velocity field (ParGridFunction, nodal or DG) */
   LinearForcingCoefficient(double alpha, ParGridFunction *U)
      : VectorCoefficient(U->ParFESpace()->GetParMesh()->SpaceDimension()),
        alpha_(alpha), U_(U),
        dim_(U->ParFESpace()->GetParMesh()->SpaceDimension()),
        ubar_(dim_) { }

   /** **Must be called once per time step _before_ assembling the RHS**
    *  (exactly the role of the `event acceleration(i++)` in Basilisk). */
   void UpdateMean()
   {
      // --- accumulate the global momentum -----------------------------------
      Vector local(dim_); local = 0.0;

      const double *u_dofs = U_->Read();     // raw pointer to all velocity DoFs
      const int     nd     = U_->Size();

      // components are interleaved (xxxx yyyy zzzz) in an H1 space,
      // still safe for DG because we only need the sum.
      for (int i = 0; i < nd; ++i) { local[i % dim_] += u_dofs[i]; }

      MPI_Allreduce(local.GetData(), ubar_.GetData(),
                    dim_, MPI_DOUBLE, MPI_SUM,
                    U_->ParFESpace()->GetComm());

      // --- divide by the global volume of the cubic 2π box -------------------
      const double L = 2.0 * M_PI;
      const double vol = L * L * L;
      ubar_ *= (1.0 / vol);
   }

   /** Point‑wise evaluation used by MFEM during assembly. */
   void Eval(Vector &F, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      Vector u_loc(dim_);
      U_->GetVectorValue(T, ip, u_loc);   // velocity at quadrature point

      F.SetSize(dim_);
      for (int d = 0; d < dim_; ++d)
         F(d) = alpha_ * (u_loc(d) - ubar_(d));
   }

private:
   const double      alpha_;
   ParGridFunction  *U_;     // not owned
   const int         dim_;
   Vector            ubar_;  // volume‑average velocity
};


// -----------------------------------------------------------------------------
// quick sanity‑check: project the forcing and integrate it
// -----------------------------------------------------------------------------
auto CheckForcing = [&](ParGridFunction &u,
                        LinearForcingCoefficient &force,
                        double  volume) /* volume = (2π)^3 once */
{
   // 1) project coefficient -> ParGridFunction f
   ParFiniteElementSpace *vfes = u.ParFESpace();
   ParGridFunction        f(vfes);
   f.ProjectCoefficient(force);   // expensive but fine for a check

   // 2) assemble the vector mass matrix (lumps all components)
   ParBilinearForm M(vfes);
   M.AddDomainIntegrator(new VectorMassIntegrator);
   M.Assemble();
   M.Finalize();

   // L2‑norm² of f
   const double f2 = M.ParInnerProduct(f,f);

   // 3) instantaneous power input  ∫ u·f dV
   const double P  = M.ParInnerProduct(u,f);

   if (Mpi::Root())
   {
      std::cout << " ‖f‖_L2² = " << f2
                << "   Power = " << P
                << "   P / vol = " << P/volume << '\n';
   }
};





// --- ABC base flow for forced‑HIT -----------------------------------
void vel_abc (const Vector &x, double /*t*/, Vector &u)
{
   const double u0 = 1.0, k = 1.0;          // Rosales & Meneveau (2005)
   const double X = x(0), Y = x(1), Z = x(2);
   u.SetSize(3);
   u(0) = u0*(  cos(k*Y) + sin(k*Z) );
   u(1) = u0*(  sin(k*X) + cos(k*Z) );
   u(2) = u0*(  cos(k*X) + sin(k*Y) );
}
// --------------------------------------------------------------------


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
      /*
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

    const double ke = 0.5*mass.ParInnerProduct(v,v);
    return ke / volume;
   };

   // This is the version we want to work because 
   // it satisfies H1 continuity. But right now 
   // the dofs and quad points are not the same.
   // D = 1/V \int_V u \cdot (-nabla \cross \w) dv
   // Gives indiciation of inertial range
   real_t ComputeInertialRangeEnergy(ParGridFunction &u)
   {

     FiniteElementSpace *fes = u.FESpace();

     Array<int> v_dofs;
     Vector loc_data;

     DenseMatrix grad_hat;
     DenseMatrix dshape;
     DenseMatrix grad;

     int elndofs;
     real_t integ = 0.0;
     int vdim = fes->GetVDim();

     for (int e = 0; e < fes->GetNE(); ++e)
     {
        fes->GetElementVDofs(e, v_dofs);
        const FiniteElement *el = fes->GetFE(e);
        ElementTransformation *tr = fes->GetElementTransformation(e);

        int dim = el->GetDim();
        elndofs = el->GetDof();
        dshape.SetSize(elndofs, dim);
        u.GetSubVector(v_dofs, loc_data);
           
        int intorder = 2 * el->GetOrder();
        const IntegrationRule *ir = &IntRules.Get(el->GetGeomType(), intorder);

        for (int j = 0; j < ir->GetNPoints(); j++){
          const IntegrationPoint &ip = ir->IntPoint(j);
          tr->SetIntPoint(&ip);
        
          el->CalcDShape(ip, dshape);
          grad_hat.SetSize(vdim, dim);
          DenseMatrix loc_data_mat(loc_data.GetData(), elndofs, vdim);
          MultAtB(loc_data_mat, dshape, grad_hat);

          const DenseMatrix &Jinv = tr->InverseJacobian();
          grad.SetSize(grad_hat.Height(), Jinv.Width());
          Mult(grad_hat, Jinv, grad);

          real_t grad_vel_norm = 0.0;
          for (int i=0; i < dim; i++){
            for (int j=0; j < dim; j++){
              grad_vel_norm += grad(i,j)*grad(i,j);
            }
          }

          integ += ip.weight * tr->Weight() * grad_vel_norm;

        }
     }

   real_t global_integral = 0.0;
   MPI_Allreduce(&integ,
                 &global_integral,
                 1,
                 MPITypeMap<real_t>::mpi_type,
                 MPI_SUM,
                 MPI_COMM_WORLD);
  
   return global_integral/volume;

   };

   /*
   // D = 1/V \int_V u \cdot (-nabla \cross \w) dv
   // Gives indiciation of inertial range
   real_t ComputeInertialRangeEnergy(ParGridFunction &v, ParGridFunction &curlw)
   {
      Vector velx, vely, velz;
      Vector curlwx, curlwy, curlwz;

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

         curlw.GetValues(i, *ir, curlwx, 1);
         curlw.GetValues(i, *ir, curlwy, 2);
         curlw.GetValues(i, *ir, curlwz, 3);

         T = fes->GetElementTransformation(i);
         for (int j = 0; j < ir->GetNPoints(); j++)
         {
            const IntegrationPoint &ip = ir->IntPoint(j);
            T->SetIntPoint(&ip);

            // u \cdot (- \nabla \cross \w)
            real_t vel_curl = -(velx(j) * curlwx(j) + vely(j) * curlwy(j)
                               + velz(j) * curlwz(j));

            integ += ip.weight * T->Weight() * vel_curl;
         }
      }

      real_t global_integral = 0.0;
      MPI_Allreduce(&integ,
                    &global_integral,
                    1,
                    MPITypeMap<real_t>::mpi_type,
                    MPI_SUM,
                    MPI_COMM_WORLD);

      // We want the magnitude contribution so we 
      // add the minus sign.
      return -global_integral / volume;
   };
   */

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

            // // Reference position (in the reference element)
            // int ref_dim = fe->GetDim(); // Dimension of the reference element
            // Vector ref_pos(ref_dim);
            // if (ref_dim >= 1) ref_pos(0) = ip.x; // x-coordinate
            // if (ref_dim >= 2) ref_pos(1) = ip.y; // y-coordinate
            // if (ref_dim >= 3) ref_pos(2) = ip.z; // z-coordinate

            // // Physical position (mapped to the physical element)
            // Vector phys_pos(T->GetSpaceDim()); // Physical space dimension
            // T->Transform(ip, phys_pos); // Maps reference -> physical

            // // Print reference and physical positions
            // mfem::out << "Integration Point " << j << " in Element " << i << ":\n";
            // mfem::out << "  Reference Position(0): " << ref_pos(0)<< "\n";
            // mfem::out << "  Physical Position(0):  " << phys_pos(0) << "\n";
            // mfem::out << "  Reference Position(1): " << ref_pos(1)<< "\n";
            // mfem::out << "  Physical Position(1):  " << phys_pos(1) << "\n";
            // mfem::out << "  Reference Position(2): " << ref_pos(2)<< "\n";
            // mfem::out << "  Physical Position(2):  " << phys_pos(2) << "\n";

  
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

  void ComputeGridPtsRequirementsTurb(ParGridFunction &u, real_t Kolmogorov_length, real_t *hmin_eta , real_t *kmax_eta)
  {

      ParMesh *pmesh_u = u.ParFESpace()->GetParMesh();
      FiniteElementSpace *fes = u.FESpace();
      int vdim = fes->GetVDim();

      real_t local_hmin_eta = 0.0; 
      real_t local_kmax_eta = 0.0;

      for (int e = 0; e < fes->GetNE(); ++e)
      {
         real_t hmin = pmesh_u->GetElementSize(e, 1) /
                            (real_t) fes->GetElementOrder(0);
         real_t kmax = M_PI/hmin;

         // For a resolved simulatin hmin/eta should be < 2.1 (Pope)
         local_hmin_eta = fmax(local_hmin_eta, hmin/Kolmogorov_length); 

         // For a resolved simulatin kmax*eta should be > 1.5 (Pope)
         local_kmax_eta = fmax(local_kmax_eta,kmax*Kolmogorov_length);
      }

      real_t hmin_eta_global = 0.0;
      MPI_Allreduce(&local_hmin_eta,
                    &hmin_eta_global,
                    1,
                    MPITypeMap<real_t>::mpi_type,
                    MPI_MAX,
                    pmesh_u->GetComm());

      real_t kmax_eta_global = 0.0;
      MPI_Allreduce(&local_kmax_eta,
                    &kmax_eta_global,
                    1,
                    MPITypeMap<real_t>::mpi_type,
                    MPI_MAX,
                    pmesh_u->GetComm());

      *hmin_eta = hmin_eta_global;
      *kmax_eta = kmax_eta_global;
  }

  void ComputeKolmogorovAndTaylorMicroLength(ParGridFunction &d_gf, real_t vol_avg_dissipation, real_t *kolmogorov_length, 
                                                                   real_t *avg_lambda, real_t *avg_kolmogorov_length, 
                                                                   real_t *kolmogorov_time_scale,
                                                                   real_t *avg_kolmogorov_time_scale, real_t *max_diss, real_t ke)
  {
      double max_dissipation = 0.0;
  
      // Iterate over the entire grid function to find the maximum dissipation value
      for (int i = 0; i < d_gf.Size(); ++i) {
          max_dissipation = std::max(max_dissipation, d_gf(i));
      }
  
      // Reduce across all MPI processes to ensure global maximum
      double global_max_dissipation;
      MPI_Allreduce(&max_dissipation, &global_max_dissipation, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  
      // Compute the smallest Kolmogorov length scale using the maximum dissipation
      // eta = (nu^3/diss_max)^0.25
      *kolmogorov_length = pow((ctx.kinvis * ctx.kinvis * ctx.kinvis) / global_max_dissipation, 0.25);

      // eta = (nu^3/<diss>)^0.25
      *avg_kolmogorov_length = pow((ctx.kinvis * ctx.kinvis * ctx.kinvis) / vol_avg_dissipation, 0.25);

      // Compute the smallest Taylor Micro scale using the maximum dissipation
      // lambda = sqrt(10*<ke>/<diss>), < > means volume average
      *avg_lambda = pow(10.0*ke/vol_avg_dissipation/ctx.reynum, 0.50);

      // Kolmogorov time scale
      // Tau_eta = sqrt(\nu/diss_max)
      *kolmogorov_time_scale = pow(ctx.kinvis/global_max_dissipation,0.50);

      // Kolmogorov time scale
      // Tau_eta = sqrt(\nu/diss_max)
      *avg_kolmogorov_time_scale = pow(ctx.kinvis/vol_avg_dissipation,0.50);

      // Assign the max dissipation
      *max_diss = global_max_dissipation;
  }

  real_t ComputeAveragedDissipation(ParGridFunction &d)
  {
      Vector d_vec;
      const FiniteElement *fe;
      ElementTransformation *T;
      FiniteElementSpace *fes = d.FESpace();
      real_t integ = 0.0;

      double totalVolume = 0.0;
      double totalDissipation = 0.0;
  
      for (int i = 0; i < fes->GetNE(); i++)
      {
          fe = fes->GetFE(i);
          int intorder = 2 * fe->GetOrder();
          const IntegrationRule *ir = &IntRules.Get(fe->GetGeomType(), intorder);

          d.GetValues(i, *ir, d_vec);
          T = fes->GetElementTransformation(i);

          // Prepare to compute the integral and volume over this element
          double volume_per_cell = 0.0;
          double elem_diss = 0.0;
  
          for (int j = 0; j < ir->GetNPoints(); j++)
          {
              const IntegrationPoint &ip = ir->IntPoint(j);
              T->SetIntPoint(&ip);
  
              // Evaluate the solution at this integration point
              real_t local_diss = d_vec(j);
     
              // Compute the Jacobian determinant at the current integration point
              real_t detJ = T->Weight();
              real_t weight = ip.weight;

              volume_per_cell += weight*detJ;
              elem_diss   += local_diss * detJ * weight;

          }
          totalVolume +=volume_per_cell;
          totalDissipation += elem_diss;
      }
  
      double globalDissipation = 0.0;
      double globalVolume   = 0.0;

      MPI_Allreduce(&totalVolume,
                    &globalVolume,
                    1,
                    MPITypeMap<real_t>::mpi_type,
                    MPI_SUM,
                    MPI_COMM_WORLD);

      MPI_Allreduce(&totalDissipation,
                    &globalDissipation,
                    1,
                    MPITypeMap<real_t>::mpi_type,
                    MPI_SUM,
                    MPI_COMM_WORLD);

      return globalDissipation/globalVolume;
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

// Computes \eta = 2*\nu*(\nabla u + trans(\nabla u))^2
void ComputeDissipation(ParGridFunction &u, ParGridFunction &d)
{
   FiniteElementSpace *v_fes = u.FESpace();
   FiniteElementSpace *fes = d.FESpace();

   // AccumulateAndCountZones
   Array<int> zones_per_vdof;
   zones_per_vdof.SetSize(fes->GetVSize());
   zones_per_vdof = 0;

   d = 0.0;

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

         real_t d_val =   sq(grad(0, 0)) + sq(grad(1, 1)) + sq(grad(2, 2))
                        + 0.5*(sq(grad(0,1) + grad(1,0)) + sq(grad(0,2) + grad(2,0))  
                        + sq(grad(1,2) + grad(2,1))); 

         vals(dof) = 2.0*ctx.kinvis*d_val;
      }

      // Accumulate values in all dofs, count the zones.
      for (int j = 0; j < dofs.Size(); j++)
      {
         int ldof = dofs[j];
         d(ldof) += vals[j];
         zones_per_vdof[ldof]++;
      }
   }

   // Count the zones globally.
   GroupCommunicator &gcomm = d.ParFESpace()->GroupComm();
   gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
   gcomm.Bcast(zones_per_vdof);

   // Accumulate for all vdofs.
   gcomm.Reduce<real_t>(d.GetData(), GroupCommunicator::Sum);
   gcomm.Bcast<real_t>(d.GetData());

   // Compute means
   for (int i = 0; i < d.Size(); i++)
   {
      const int nz = zones_per_vdof[i];
      if (nz)
      {
         d(i) /= nz;
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

void SamplePoints(ParGridFunction *sol, ParMesh *pmesh, int step, double time, const std::string &suffix);

void ComputeElementCenterValuesScalar(ParGridFunction *sol, ParMesh *pmesh,int step, double time);

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
   args.AddOption(&ctx.element_center_cycle, "-ecc", "--Element-Center-Cycle", "Element Center Cycle.");
   args.AddOption(&ctx.data_dump_cycle, "-ddc", "--Data-Dump-Cycle", "Data Dump Cycle.");
   args.AddOption(
       &ctx.filter, 
       "-flt", 
       "--Filter-Alias-Error",
       "-no-flt",
       "--no-Filter-Alias-Error",
       "Enable or disiable filter to controal alias error.");
   args.AddOption(
       &ctx.oversample, 
       "-ovs", 
       "--Over-Sample",
       "-no-ovs",
       "--no-Over-Sample",
       "Enable or disable oversampling of solution.");
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
   // Update kinematic viscosity
   ctx.kinvis = 1.0 / ctx.reynum;

   ParMesh *pmesh = nullptr;
   Mesh *mesh = nullptr;
   ParGridFunction *u_gf = nullptr;
   ParGridFunction *p_gf = nullptr;
   NavierSolver *flowsolver = nullptr;
   LinearForcingCoefficient *lin_force = nullptr;   // <‑‑ add just after flowsolver ptr

   double t = 0.0;
   int step = 0;
   int global_cycle = 0;

   bool restart_files_found = false;

   if (ctx.restart)
   {
      // Try to load the checkpoint files
      restart_files_found = LoadCheckpoint(pmesh, u_gf, p_gf, flowsolver, t, step, myid, &ctx);
      if (restart_files_found)
      {
         if (Mpi::Root())
         {
            std::cout << "Restart files found. Continuing from checkpoint at time t = " << t << std::endl;
         }
         // Store the initial step number at restart
         global_cycle = step;

         // Reset step from restart for flow solver
         step = 0;
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

     if (Mpi::Root())
     {
        std::cout << "Creating the mesh..." << std::endl;
     }

      // Initialize as mesh
      Mesh *init_mesh;

      real_t length = 2*M_PI;
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

      // Define a translation function for the mesh nodes
      VectorFunctionCoefficient scale(mesh->Dimension(), [&](const Vector &x_in, Vector &x_out)
                                          {
         double scale = 1.0;

         x_out[0] = x_in[0]/scale ; // Translate x-coordinate
         x_out[1] = x_in[1]/scale ; // Translate y-coordinate
         if (mesh->Dimension() == 3){
           x_out[2] = x_in[2]/scale; // Translate z-coordinate
         } });

      // mesh->Transform(translate);
      // mesh->Transform(scale);

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

      // Create the flow solver
      flowsolver = new NavierSolver(pmesh, ctx.order, ctx.kinvis);
      flowsolver->EnablePA(ctx.pa);
      flowsolver->EnableNI(ctx.ni);

      // Set the initial condition
      u_gf = flowsolver->GetCurrentVelocity();
      VectorFunctionCoefficient u_excoeff(pmesh->Dimension(), vel_abc);

      u_gf->ProjectCoefficient(u_excoeff);

      p_gf = flowsolver->GetCurrentPressure();

      // after flowsolver->Setup(dt);
      const double alpha_forcing = 0.10;                 // same as Basilisk example
      lin_force = new LinearForcingCoefficient(alpha_forcing, u_gf);
      CheckForcing(*u_gf, *lin_force, 8.*M_PI*M_PI*M_PI);   // new line
      
      Array<int> all_attr(pmesh->attributes.Max());
      all_attr = 1;                                      // whole periodic box
      flowsolver->AddAccelTerm(lin_force, all_attr);

      // Set up the flow solver
      flowsolver->Setup(ctx.dt);



      // --------------------------------------------------------------------
      

      if (Mpi::Root())
      {
         mfem::out << "Done setting up the flowsolver. " << std::endl;
      }
   }

   int nel = pmesh->GetGlobalNE();
   if (Mpi::Root())
   {
      mfem::out << "Number of elements: " << nel << std::endl;
   }

   ParFiniteElementSpace *velocity_fespace = u_gf->ParFESpace();
   ParFiniteElementSpace *pressure_fespace = p_gf->ParFESpace();
   
   // Initialize w_gf and q_gf using the finite element spaces
   ParGridFunction w_gf(velocity_fespace);
   ParGridFunction q_gf(pressure_fespace);
   ParGridFunction d_gf(pressure_fespace);

   flowsolver->ComputeCurl3D(*u_gf, w_gf);
   ComputeQCriterion(*u_gf, q_gf);
   ComputeDissipation(*u_gf, d_gf);
   QuantitiesOfInterest kin_energy(pmesh);

   ParaViewDataCollection *pvdc = NULL;
   if (ctx.paraview)
   {
      std::string paraview_dir = std::string("ParaviewData_") 
                                               + "Re" + std::to_string(static_cast<int>(ctx.reynum)) 
                                               + "NumPtsPerDir" +std::to_string(ctx.num_pts) 
                                               + "RefLv" + std::to_string(
                                                   ctx.element_subdivisions 
                                                 + ctx.element_subdivisions_parallel) 
                                               + "Order" + std::to_string(ctx.order)
                                               + "/tgv_output_paraview";

      pvdc = new ParaViewDataCollection(paraview_dir, pmesh);
      pvdc->SetDataFormat(VTKFormat::BINARY32);
      pvdc->SetHighOrderOutput(true);
      pvdc->SetLevelsOfDetail(ctx.order);
      pvdc->SetCycle(global_cycle + step);
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
         std::string visit_dir = std::string("VisitData_") 
                                                  + "Re" + std::to_string(static_cast<int>(ctx.reynum)) 
                                                  + "NumPtsPerDir" +std::to_string(ctx.num_pts) 
                                                  + "RefLv" + std::to_string(
                                                      ctx.element_subdivisions 
                                                    + ctx.element_subdivisions_parallel) 
                                                  + "P" + std::to_string(ctx.order)
                                                  + "/tgv_output_visit";

         dc = new VisItDataCollection(MPI_COMM_WORLD,visit_dir, pmesh);
      }
      int precision = 16;
      dc->SetPrecision(precision);
      dc->SetCycle(global_cycle + step);
      dc->SetTime(t);
      dc->SetFormat(DataCollection::PARALLEL_FORMAT);
      dc->RegisterField("velocity", u_gf);
      dc->RegisterField("pressure", p_gf);
      dc->RegisterField("vorticity", &w_gf);
      dc->RegisterField("qcriterion", &q_gf);
      // dc->RegisterField("dissipation", &d_gf);
      dc->Save();
   }


   ConduitDataCollection *cdc = NULL;
   if (ctx.conduit)
   {
#ifdef MFEM_USE_CONDUIT
         // // Create a parallel ConduitDataCollection
         // cdc = new ConduitDataCollection(MPI_COMM_WORLD, collection_name, pmesh);

         std::string conduit_dir = std::string("ConduitData_") 
                                                  + "Re" + std::to_string(static_cast<int>(ctx.reynum)) 
                                                  + "NumPtsPerDir" +std::to_string(ctx.num_pts) 
                                                  + "RefLv" + std::to_string(
                                                      ctx.element_subdivisions 
                                                    + ctx.element_subdivisions_parallel) 
                                                  + "P" + std::to_string(ctx.order)
                                                  + "/tgv_output_conduit";

         cdc = new ConduitDataCollection(MPI_COMM_WORLD,conduit_dir, pmesh);

         // Set the Conduit relay protocol (options include "hdf5", "json", "conduit_json", "conduit_bin")
         cdc->SetProtocol("hdf5"); // Using "json" for human-readable output
         {
           int precision = 16;
           cdc->SetPrecision(precision);
           cdc->SetFormat(DataCollection::PARALLEL_FORMAT);
           cdc->SetCycle(global_cycle + step);
           cdc->SetTime(t);
           cdc->RegisterField("velocity", u_gf);
           cdc->RegisterField("pressure", p_gf);
           cdc->RegisterField("vorticity", &w_gf);
           cdc->RegisterField("qcriterion", &q_gf);
           // cdc->RegisterField("dissipation", &d_gf);
           cdc->Save();
         }
#else
         MFEM_ABORT("Must build with MFEM_USE_CONDUIT=YES for binary output.");
#endif

   }

   real_t u_inf_loc = u_gf->Normlinf();
   real_t p_inf_loc = p_gf->Normlinf();

   real_t u_inf = GlobalLpNorm(infinity(), u_inf_loc, MPI_COMM_WORLD);
   real_t p_inf = GlobalLpNorm(infinity(), p_inf_loc, MPI_COMM_WORLD);

   real_t ke = kin_energy.ComputeKineticEnergy(*u_gf);
   real_t vel_curl_ke = kin_energy.ComputeInertialRangeEnergy(*u_gf);
   real_t enstrophy = kin_energy.ComputeEnstrophy(w_gf);

   real_t kolmLenScl = 0.0;
   real_t avg_kolmLenScl = 0.0;
   real_t avg_lambda = 0.0;
   real_t kolmTimeScl = 0.0;
   real_t avg_kolmTimeScl = 0.0;
   real_t hmin_eta = 0.0;
   real_t kmax_eta = 0.0;
   real_t u_rms =  pow(2.0/3.0*ke,0.5);
   real_t max_diss = 0.0;

   real_t avg_diss = kin_energy.ComputeAveragedDissipation(d_gf);
   kin_energy.ComputeKolmogorovAndTaylorMicroLength(d_gf, avg_diss, &kolmLenScl, &avg_lambda, &avg_kolmLenScl, &kolmTimeScl, &avg_kolmTimeScl, &max_diss, ke);
   kin_energy.ComputeGridPtsRequirementsTurb(*u_gf, kolmLenScl, &hmin_eta, &kmax_eta);

   // This computes how resolved our grid is.
   // See Aspen 2008 Implicit LES Anaylsis
   real_t PI_nu = pow(avg_diss,0.5)/(avg_kolmLenScl*pow(vel_curl_ke,0.75));
   real_t PI_nu_min = pow(max_diss,0.5)/(kolmLenScl*pow(vel_curl_ke,0.75));

   // Taylor Reynolds Number
   // Re_lambda = u' lambda/nu
   real_t Re_taylor = u_rms*avg_lambda/ctx.kinvis;

   // Compute the cfl
   real_t cfl;
   cfl = flowsolver->ComputeCFL(*u_gf, ctx.dt);

   std::string fname = std::string("tgv_out_") 
                                            + "Re" + std::to_string(static_cast<int>(ctx.reynum)) 
                                            + "NumPtsPerDir" +std::to_string(ctx.num_pts) 
                                            + "RefLv" + std::to_string(
                                                ctx.element_subdivisions 
                                              + ctx.element_subdivisions_parallel) 
                                            + "P" + std::to_string(ctx.order)
                                            + ".txt";

   std::string fname_turb = std::string("tgv_out_turb_") 
                                            + "Re" + std::to_string(static_cast<int>(ctx.reynum)) 
                                            + "NumPtsPerDir" +std::to_string(ctx.num_pts) 
                                            + "RefLv" + std::to_string(
                                                ctx.element_subdivisions 
                                              + ctx.element_subdivisions_parallel) 
                                            + "P" + std::to_string(ctx.order)
                                            + ".txt";
   std::string fname_turb_grid = std::string("tgv_out_turb_grid_") 
                                            + "Re" + std::to_string(static_cast<int>(ctx.reynum)) 
                                            + "NumPtsPerDir" +std::to_string(ctx.num_pts) 
                                            + "RefLv" + std::to_string(
                                                ctx.element_subdivisions 
                                              + ctx.element_subdivisions_parallel) 
                                            + "P" + std::to_string(ctx.order)
                                            + ".txt";
   FILE *f = NULL;
   FILE *f_turb = NULL;
   FILE *f_turb_grid = NULL;

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
      f_turb = fopen(fname_turb.c_str(), file_mode);
      f_turb_grid = fopen(fname_turb_grid.c_str(), file_mode);

      if (!f)
      {
        std::cerr << "Error opening file " << fname << std::endl;
        MPI_Abort(MPI_COMM_WORLD,1);
      }

      if (!f_turb)
      {
        std::cerr << "Error opening file " << fname_turb << std::endl;
        MPI_Abort(MPI_COMM_WORLD,1);
      }

      if (!f_turb_grid)
      {
        std::cerr << "Error opening file " << fname_turb_grid << std::endl;
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
          fprintf(f, "=========================================================================================\n");
          fprintf(f, "        time                      cycle                 kinetic energy               enstrophy\n");

          // Write the initial data point
           fprintf(f, "%20.16e     %20.16e     %20.16e     %20.16e\n", t, static_cast<real_t>(global_cycle + step), ke, enstrophy);

          // Write header only if not restarting
          fprintf(f_turb, "3D Taylor Green Vortex (turbulence metrics)\n");
          fprintf(f_turb, "Reynolds Number = %d\n", static_cast<int>(ctx.reynum));
          fprintf(f_turb, "order = %d\n", ctx.order);
          fprintf(f_turb, "grid = %d x %d x %d\n", nel1d, nel1d, nel1d);
          fprintf(f_turb, "dofs per component = %d\n", ngridpts);
          fprintf(f_turb, "===============================================================================");
          fprintf(f_turb, "===============================================================================");
          fprintf(f_turb, "===============================================================================");
          fprintf(f_turb, "=================================================================\n");
          fprintf(f_turb, "        time                        cycle                Max Dissipation       Average Dissipation     Min Kolmogorov Length Scale    Taylor Length Scale");
          fprintf(f_turb, "        Average Kolm Len          Kolmogorov Time Scale            Average Kolm Time Scale       Taylor Re (Avg)");
          fprintf(f_turb, "               u_rms    \n");

          // Write the initial data point
           fprintf(f_turb, "%20.16e     %20.16e     %20.16e     %20.16e     %20.16e     %20.16e    %20.16e     %20.16e      %20.16e      %20.16e      %20.16e\n",
                       t, static_cast<real_t>(global_cycle + step), max_diss, avg_diss, kolmLenScl, 
                       avg_lambda, avg_kolmLenScl, kolmTimeScl, avg_kolmTimeScl,
                       Re_taylor, u_rms);

          // Write header only if not restarting
          fprintf(f_turb_grid, "3D Taylor Green Vortex (turbulence grid metrics)\n");
          fprintf(f_turb_grid, "Reynolds Number = %d\n", static_cast<int>(ctx.reynum));
          fprintf(f_turb_grid, "order = %d\n", ctx.order);
          fprintf(f_turb_grid, "grid = %d x %d x %d\n", nel1d, nel1d, nel1d);
          fprintf(f_turb_grid, "dofs per component = %d\n", ngridpts);
          fprintf(f_turb_grid, "===============================================================================");
          fprintf(f_turb_grid, "=================================================================\n");
          fprintf(f_turb_grid, "        time                       cycle                  K_max*eta (>1.5)              hmin/eta (<2.1)");
          fprintf(f_turb_grid, "        Average PI_NU              Min PI_NU        \n");

          // Write the initial data point
           fprintf(f_turb_grid, "%20.16e     %20.16e     %20.16e     %20.16e     %20.16e    %20.16e\n",
                       t, static_cast<real_t>(global_cycle + step), kmax_eta, hmin_eta, PI_nu, PI_nu_min);
      } 

      fflush(f);
      fflush(f_turb);
      fflush(f_turb_grid);
      fflush(stdout);
   }

   real_t dt = ctx.dt;
   real_t t_final = ctx.t_final;
   bool last_step = false;

   if(ctx.filter){
     // NOT WORKING!!
     flowsolver->SetFilterAlpha(0.03); // Enable sharp cutoff
     flowsolver->SetCutoffModes(3);   // Cut off highest mode
   }

   for (; !last_step; ++step)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

       double KE_before = kin_energy.ComputeKineticEnergy(*u_gf);
      lin_force->UpdateMean();
      CheckForcing(*u_gf, *lin_force, 8.*M_PI*M_PI*M_PI);   // new line
// ------------ sanity check: instantaneous power injected by the forcing
{
   using std::cout; using std::endl;
   ParFiniteElementSpace *vfes = u_gf->ParFESpace();

   // assemble rhs = ∫_Ω f · v_test  (the same integrator NavierSolver uses)
   ParLinearForm rhs(vfes);
   rhs.AddDomainIntegrator(new VectorDomainLFIntegrator(*lin_force));
   rhs.Assemble();

   // Power = ∫ u · f  =  rhs(u)
   const double power_chk = rhs(*u_gf);

   // Optional: ∫ f · 1  (should be ~0 for linear forcing)
   Vector one_vec(vfes->GetParMesh()->SpaceDimension());
   one_vec = 1.0;
   VectorConstantCoefficient one_coeff(one_vec);

   ParGridFunction one_gf(vfes);
   one_gf.ProjectCoefficient(one_coeff);

   const double fsum_chk = rhs(one_gf);

   if (Mpi::Root())
   {
      cout.setf(std::ios::scientific);
      cout << "   [check]  Power(u,f) = " << power_chk
           << "   Σf·1 = " << fsum_chk << endl;
   }
}
// -----------------------------------------------------------------------







      flowsolver->Step(t, dt, step);

       // === Sanity check ends ===========================================
       double KE_after  = kin_energy.ComputeKineticEnergy(*u_gf);
       if (Mpi::Root())
           std::cout << "   ΔKE = " << KE_after - KE_before << std::endl;



      cfl = flowsolver->ComputeCFL(*u_gf, ctx.dt);

      if ((global_cycle + step) % ctx.data_dump_cycle == 0 || last_step)
      {
         // If restarting, skip the first saved checkpoint
         if (!(ctx.restart && step == 0 && restart_files_found))
         {
            ComputeQCriterion(*u_gf, q_gf);
            flowsolver->ComputeCurl3D(*u_gf, w_gf);

            if (ctx.paraview)
            {
               pvdc->SetCycle(global_cycle + step);
               pvdc->SetTime(t);
               pvdc->Save();
               if (Mpi::Root())
               {
                  std::cout << "\nParaview file saved." << std::endl;
               }
            }

            if (ctx.visit)
            {
               dc->SetCycle(global_cycle + step);
               dc->SetTime(t);
               dc->Save();

               if (Mpi::Root())
               {
                  std::cout << "\nVisit file saved at cycle " << global_cycle + step << "." << std::endl;
               }

               real_t u_inf_loc = dc->GetField("velocity")->Normlinf();
               real_t p_inf_loc = dc->GetField("pressure")->Normlinf();

               real_t u_inf = mfem::GlobalLpNorm(mfem::infinity(), 
                                                       u_inf_loc, 
                                                       MPI_COMM_WORLD);
               real_t p_inf = mfem::GlobalLpNorm(mfem::infinity(), 
                                                             p_inf_loc, 
                                                             MPI_COMM_WORLD);
               if (Mpi::Root())
               {
                   std::cout << "After loading from checkpoint in LoadCheckpoint: u_gf Norml2 = "
                             << u_inf << ", p_gf Norml2 = " << p_inf << std::endl;
               }
            }

            if (ctx.conduit)
            {
               cdc->SetCycle(global_cycle + step);
               cdc->SetTime(t);
               cdc->Save();

               if (Mpi::Root())
               {
                  std::cout << "\nConduit file saved at cycle " << global_cycle + step << "." << std::endl;
               }

               real_t u_inf_loc = cdc->GetField("velocity")->Normlinf();
               real_t p_inf_loc = cdc->GetField("pressure")->Normlinf();

               real_t u_inf = mfem::GlobalLpNorm(mfem::infinity(), 
                                                       u_inf_loc, 
                                                       MPI_COMM_WORLD);
               real_t p_inf = mfem::GlobalLpNorm(mfem::infinity(), 
                                                             p_inf_loc, 
                                                             MPI_COMM_WORLD);
                  
               if (Mpi::Root())
               {
                   std::cout << "After loading from checkpoint in LoadCheckpoint: u_gf Norml2 = "
                             << u_inf << ", p_gf Norml2 = " << p_inf << std::endl;
               }
            }
         }
      }

      if ((global_cycle + step) % ctx.element_center_cycle == 0 || last_step)
      {
         // If restarting, skip the first saved checkpoint
         if (!(ctx.restart && step == 0 && restart_files_found))
         {
            SamplePoints( u_gf, pmesh, global_cycle + step, t, "Velocity", &ctx);
            // SamplePointsAdios( u_gf, pmesh, global_cycle + step, t, "Velocity",ctx.oversample, &ctx);
            // ComputeElementCenterValues(&w_gf, pmesh, global_cycle + step, t, "Vorticity");

            if (Mpi::Root())
            {
               std::cout << "\nOutput element center file saved at cycle " << global_cycle + step << "." << std::endl;
            }

         }
      }
            
      u_inf_loc = u_gf->Normlinf();
      p_inf_loc = p_gf->Normlinf();

      u_inf = GlobalLpNorm(infinity(), u_inf_loc, MPI_COMM_WORLD);
      p_inf = GlobalLpNorm(infinity(), p_inf_loc, MPI_COMM_WORLD);

      flowsolver->ComputeCurl3D(*u_gf, w_gf);

      ke = kin_energy.ComputeKineticEnergy(*u_gf);
      vel_curl_ke = kin_energy.ComputeInertialRangeEnergy(*u_gf);
      enstrophy = kin_energy.ComputeEnstrophy(w_gf);

      ComputeDissipation(*u_gf, d_gf);
      avg_diss = kin_energy.ComputeAveragedDissipation(d_gf);
      kin_energy.ComputeKolmogorovAndTaylorMicroLength(d_gf, avg_diss, &kolmLenScl, &avg_lambda, &avg_kolmLenScl, &kolmTimeScl, &avg_kolmTimeScl, &max_diss, ke);
      kin_energy.ComputeGridPtsRequirementsTurb(*u_gf, kolmLenScl, &hmin_eta, &kmax_eta);
      Re_taylor = u_rms*avg_lambda/ctx.kinvis;
      u_rms =  pow(2.0/3.0*ke,0.5);

      PI_nu = pow(avg_diss,0.5)/(avg_kolmLenScl*pow(vel_curl_ke,0.75));
      PI_nu_min = pow(max_diss,0.5)/(kolmLenScl*pow(vel_curl_ke,0.75));

      if (Mpi::Root())
      {
         // If restarting, skip the first saved checkpoint
         if (!(ctx.restart && step == 0 && restart_files_found))
         {
           printf("%.5E %.5E %.5E %.5E %.5E %.5E %.5E\n", t, ctx.dt, u_inf, p_inf, ke, enstrophy, cfl);
           fprintf(f, "%20.16e     %20.16e     %20.16e     %20.16e\n", t, static_cast<real_t>(step), ke, enstrophy);
           fprintf(f_turb, "%20.16e     %20.16e     %20.16e     %20.16e     %20.16e     %20.16e    %20.16e     %20.16e      %20.16e      %20.16e      %20.16e\n",
                       t, static_cast<real_t>(global_cycle + step), max_diss, avg_diss, kolmLenScl, 
                       avg_lambda, avg_kolmLenScl, kolmTimeScl, avg_kolmTimeScl,
                       Re_taylor, u_rms);
           fprintf(f_turb_grid, "%20.16e     %20.16e     %20.16e     %20.16e     %20.16e    %20.16e\n",
                       t, static_cast<real_t>(global_cycle + step), kmax_eta, hmin_eta, PI_nu, PI_nu_min);
           fflush(f);
           fflush(f_turb);
           fflush(f_turb_grid);
           fflush(stdout);
         }
      }
   }

   // flowsolver->PrintTimingData();

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

/*
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
   std::string main_dir = "ElementCenters" + suffix +
                            "_Re" + std::to_string(static_cast<int>(ctx.reynum)) +
                            "NumPtsPerDir" + std::to_string(ctx.num_pts) +
                            "RefLv" + std::to_string(ctx.element_subdivisions + ctx.element_subdivisions_parallel) +
                            "P" + std::to_string(ctx.order);

   // Create subdirectory for this cycle step
   std::string cycle_dir = main_dir + "/cycle_" + std::to_string(step);
   // Construct the filename inside the cycle directory
   std::string fname = cycle_dir + "/element_centers_" + std::to_string(step) + ".txt";

   if (rank == 0)
   {
      // Create main and cycle directories
      if (system(("mkdir -p " + main_dir).c_str()) != 0)
         std::cerr << "Error creating " << main_dir << " directory!" << std::endl;
      if (system(("mkdir -p " + cycle_dir).c_str()) != 0)
         std::cerr << "Error creating " << cycle_dir << " directory!" << std::endl;
   }

   MPI_Barrier(MPI_COMM_WORLD);

   // Instead of one integration point (the element center), we will sample each element
   // on an N x N x N grid, where N = ctx.order + 1.
   // int npts = ctx.order + 1;  // number of sample points per coordinate direction
   int npts = ctx.order + 1;  // number of sample points per coordinate direction

   // Local arrays to store data from the local elements
   std::vector<double> local_x, local_y, local_z;
   std::vector<double> local_velx, local_vely, local_velz;

   FiniteElementSpace *fes = sol->FESpace();
   int vdim = fes->GetVDim();

   // Loop over local elements
   for (int e = 0; e < pmesh->GetNE(); e++)
   {
      // Print reference and physical positions
      // mfem::out << "In Element " << e << ":\n";
      // Get element transformation for element e
      ElementTransformation *Trans = pmesh->GetElementTransformation(e);
      
      // For each element, loop over a uniform grid of points in the reference element [0,1]^d.
      for (int iz = 0; iz < npts; iz++)
      {
         double z_ref = (npts == 1) ? 0.5 : static_cast<double>(iz) / (npts - 1);
         // double z_ref = static_cast<double>(iz) / npts;
         for (int iy = 0; iy < npts; iy++)
         {
            double y_ref = (npts == 1) ? 0.5 : static_cast<double>(iy) / (npts - 1);
            // double y_ref = static_cast<double>(iy) / npts;
            for (int ix = 0; ix < npts; ix++)
            {
               double x_ref = (npts == 1) ? 0.5 : static_cast<double>(ix) / (npts - 1);
               // double x_ref = static_cast<double>(ix) / npts;
               IntegrationPoint ip;
               ip.Set3(x_ref, y_ref, z_ref); // sample point in reference element

               // Get the physical coordinates for this sample point
               Vector phys_coords(Trans->GetSpaceDim());
               Trans->Transform(ip, phys_coords);

               double x_physical = phys_coords(0);
               double y_physical = phys_coords(1);
               double z_physical = phys_coords(2);

               // Evaluate the solution at the sample point
               Vector u_val(vdim);
               sol->GetVectorValue(*Trans, ip, u_val);
               double u_x = u_val(0);
               double u_y = u_val(1);
               double u_z = u_val(2);

               // Physical position (mapped to the physical element)
               Vector phys_pos(Trans->GetSpaceDim()); // Physical space dimension
               Trans->Transform(ip, phys_pos); // Maps reference -> physical

               // Append sample point data to local arrays
               local_x.push_back(x_physical);
               local_y.push_back(y_physical);
               local_z.push_back(z_physical);
               local_velx.push_back(u_x);
               local_vely.push_back(u_y);
               local_velz.push_back(u_z);
            } // ix
         } // iy
      } // iz
   } // for each local element
   if (rank == 0)
     std::cout << "Done looping over all elements" << std::endl;

   // Gather local element sample counts
   int local_num = local_x.size();
   std::vector<int> all_num_elements(size);
   std::vector<int> displs(size);
   MPI_Gather(&local_num, 1, MPI_INT,
              all_num_elements.data(), 1, MPI_INT, 0, comm);

   std::vector<double> all_x, all_y, all_z;
   std::vector<double> all_velx, all_vely, all_velz;
   if (rank == 0)
   {
      int total = 0;
      displs[0] = 0;
      for (int i = 0; i < size; i++)
      {
         total += all_num_elements[i];
         if (i > 0)
            displs[i] = displs[i - 1] + all_num_elements[i - 1];
      }
      all_x.resize(total);
      all_y.resize(total);
      all_z.resize(total);
      all_velx.resize(total);
      all_vely.resize(total);
      all_velz.resize(total);
   }

   if (rank == 0)
     std::cout << "Starting to set the sizes." << std::endl;

   MPI_Gatherv(local_x.data(), local_num, MPI_DOUBLE,
               all_x.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
   MPI_Gatherv(local_y.data(), local_num, MPI_DOUBLE,
               all_y.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
   MPI_Gatherv(local_z.data(), local_num, MPI_DOUBLE,
               all_z.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
   MPI_Gatherv(local_velx.data(), local_num, MPI_DOUBLE,
               all_velx.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
   MPI_Gatherv(local_vely.data(), local_num, MPI_DOUBLE,
               all_vely.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);
   MPI_Gatherv(local_velz.data(), local_num, MPI_DOUBLE,
               all_velz.data(), all_num_elements.data(), displs.data(), MPI_DOUBLE, 0, comm);

   if (rank == 0)
     std::cout << "Done gather all the data." << std::endl;

   if (rank == 0)
   {
      FILE *f = fopen(fname.c_str(), "w");
      if (!f)
      {
         std::cerr << "Error opening file " << fname << std::endl;
         MPI_Abort(MPI_COMM_WORLD,1);
      }

      // Write header information
      fprintf(f, "3D Taylor Green Vortex\n");
      fprintf(f, "Order = %d\n", ctx.order);
      fprintf(f, "Step = %d\n", step);
      fprintf(f, "Time = %e\n", time);
      fprintf(f, "===================================================================");
      fprintf(f, "==========================================================================\n");
      fprintf(f, "            x                      y                      z         ");
      fprintf(f, "            vecx                   vecy                   vecz\n");

      // Write data for each sample point
      for (size_t i = 0; i < all_x.size(); i++)
      {
         fprintf(f, "%20.16e %20.16e %20.16e %20.16e %20.16e %20.16e\n",
                 all_x[i], all_y[i], all_z[i],
                 all_velx[i], all_vely[i], all_velz[i]);
      }
      fflush(f);
      fclose(f);
      std::cout << "Output element sample file saved: " << fname << std::endl;
   }

   MPI_Barrier(MPI_COMM_WORLD);
}
*/


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


