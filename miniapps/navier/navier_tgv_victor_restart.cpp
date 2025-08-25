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
   int num_pts = 8;
   bool visit = true;
   bool paraview = false;
   bool binary = false;
   bool conduit = false;
   bool restart = true;
   int element_center_cycle = 100;
   int data_dump_cycle = 100;
   bool filter = false;
   bool oversample = true;
   real_t alpha = 0.3;
   real_t delta_const = 1e-8;
   bool problem1 = true;

} ctx;

void project_Hdiv_to_L2(ParGridFunction &result,
                        ParGridFunction &u_hdiv,
                        ParFiniteElementSpace *test_fes,   // vector L2(DG) target
                        bool pa);
void compute_Curl_Hcurl_to_Hdiv(ParGridFunction &result, ParGridFunction &gftrial, ParFiniteElementSpace *trial_fes,  
                        ParGridFunction &gftest, ParFiniteElementSpace *test_fes, bool pa);

void project_Hcurl_Hdiv(ParGridFunction &result, ParGridFunction &gftrial, ParFiniteElementSpace *trial_fes,  ParGridFunction &gftest, ParFiniteElementSpace *test_fes, bool pa);
void project_H1_to_Hcurl(ParGridFunction &result, 
   ParGridFunction &gftrial, 
   ParFiniteElementSpace *trial_fes,  
   ParFiniteElementSpace *test_fes, bool pa);

void project_H1_to_L2(ParGridFunction &result,          // in L2 space (output)
                      ParGridFunction &u_h1,            // in H1 vector space (input)
                      ParFiniteElementSpace *fes_h1,    // H1 trial space
                      ParFiniteElementSpace *fes_l2,    // L2 test/target space
                      bool pa);

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
bool GetFilter(const s_NavierContext* ctx) { return ctx->filter; }
bool GetOverSample(const s_NavierContext* ctx) {return ctx->oversample;};
real_t GetAlpha(const s_NavierContext* ctx) {return ctx->alpha;};

void vel_tgv(const Vector &x, real_t t, Vector &u)
{
   real_t xi = x(0);
   real_t yi = x(1);
   real_t zi = x(2);

   if (!ctx.problem1)
   {
     xi = 2*M_PI*x(0);
     yi = 2*M_PI*x(1);
     zi = 2*M_PI*x(2);
   }

   u(0) = sin(xi) * cos(yi) * cos(zi);
   u(1) = -cos(xi) * sin(yi) * cos(zi);
   u(2) = 0.0;
}

// // For verifying the velocity decomposition
// void vel_tgv(const Vector &x, real_t t, Vector &u)
// {
//       u(0) = sin(2*M_PI*x(0)) + sin(4*M_PI*x(1)) + sin(6*M_PI*x(2));
//       u(1) = sin(6*M_PI*x(0)) + sin(2*M_PI*x(1)) + sin(4*M_PI*x(2));
//       u(2) = sin(4*M_PI*x(0)) + sin(6*M_PI*x(1)) + sin(2*M_PI*x(2));
// }

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

   real_t ComputeKineticEnergy(ParGridFunction &v, ParGridFunction &ke_gf)
   // real_t ComputeKineticEnergy(ParGridFunction &v)
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

    ParFiniteElementSpace *vfes = v.ParFESpace();
    ParBilinearForm mass(vfes);
    ConstantCoefficient ones(1.0);
    mass.AddDomainIntegrator(new VectorMassIntegrator(ones));
  
    mass.Assemble();
    mass.Finalize();

    // Create KE grid function
    VectorGridFunctionCoefficient U(&v);     

    InnerProductCoefficient uu(U, U);          

    ConstantCoefficient half(0.5);

    ProductCoefficient kcoeff(half, uu);       

    ke_gf.ProjectCoefficient(kcoeff);

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

  void ComputeGridPtsRequirementsTurb(ParGridFunction &u, real_t Kolmogorov_length, real_t *hmin_eta , real_t *kmax_eta, real_t *kmax_return, real_t *hmin_return)
  {

      ParMesh *pmesh_u = u.ParFESpace()->GetParMesh();
      FiniteElementSpace *fes = u.FESpace();
      int vdim = fes->GetVDim();

      real_t local_hmin_eta = 0.0; 
      real_t local_kmax_eta = 0.0;

      real_t local_hmin = 0.0; 
      real_t local_kmax = 0.0;

      for (int e = 0; e < fes->GetNE(); ++e)
      {
         real_t hmin = pmesh_u->GetElementSize(e, 1) /
                            (real_t) fes->GetElementOrder(0);
         real_t kmax = M_PI/hmin;

         local_hmin = fmax(local_hmin,hmin); 
         local_kmax = fmax(local_kmax,kmax);

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

      real_t hmin_global = 0.0;
      MPI_Allreduce(&local_hmin,
                    &hmin_global,
                    1,
                    MPITypeMap<real_t>::mpi_type,
                    MPI_MAX,
                    pmesh_u->GetComm());

      real_t kmax_global = 0.0;
      MPI_Allreduce(&local_kmax,
                    &kmax_global,
                    1,
                    MPITypeMap<real_t>::mpi_type,
                    MPI_MAX,
                    pmesh_u->GetComm());

      *hmin_return = hmin_global;
      *kmax_return = kmax_global;
  }

  void ComputeKolmogorovAndTaylorMicroLength(ParGridFunction &d_gf,real_t vol_avg_dissipation, real_t *kolmogorov_length, 
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

         real_t q_val = -(0.5 * (sq(grad(0, 0)) + sq(grad(1, 1)) + sq(grad(2, 2)))
                        + grad(0, 1) * grad(1, 0) 
                        + grad(0, 2) * grad(2, 0)
                        + grad(1, 2) * grad(2, 1));
         // real_t q_val =   grad(0,0)*grad(1,1) + grad(1,1)*grad(2,2) + grad(0,0)*grad(2,2)
         //                - grad(0,1)*grad(1,0) 
         //                - grad(0,2)*grad(2,0)
         //                - grad(1,2)*grad(2,1);

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

void ComputeDivergence3D(ParGridFunction &u, ParGridFunction &du)
{

   FiniteElementSpace *v_fes = u.FESpace();
   FiniteElementSpace *fes = du.FESpace();

   // AccumulateAndCountZones
   Array<int> zones_per_vdof;
   zones_per_vdof.SetSize(fes->GetVSize());
   zones_per_vdof = 0;

   du = 0.0;

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

         const IntegrationPoint &ip = el->GetNodes().IntPoint(dof);
         tr->SetIntPoint(&ip);

         // MFEM does all the heavy lifting here:
         // GetDivergence computes ∇·u at the current integration point
         vals(dof) = u.GetDivergence(*tr);
      }

      // Accumulate values in all dofs, count the zones.
      for (int j = 0; j < dofs.Size(); j++)
      {
         int ldof = dofs[j];
         du(ldof) += vals[j];
         zones_per_vdof[ldof]++;
      }
   }

   // Count the zones globally.
   GroupCommunicator &gcomm = du.ParFESpace()->GroupComm();
   gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
   gcomm.Bcast(zones_per_vdof);

   // Accumulate for all vdofs.
   gcomm.Reduce<real_t>(du.GetData(), GroupCommunicator::Sum);
   gcomm.Bcast<real_t>(du.GetData());

   // Compute means
   for (int i = 0; i < du.Size(); i++)
   {
      const int nz = zones_per_vdof[i];
      if (nz)
      {
         du(i) /= nz;
      }
   }
}

void ComputeVorticalPart( NavierSolver *solver, 
                          ParGridFunction &u,
                          ParGridFunction &w_gf,
                          ParGridFunction &u_vort)
{

    ParFiniteElementSpace *vfes = u.ParFESpace();

    Array<int> ess_tdof_list;

    VectorGridFunctionCoefficient w_coeff(&w_gf);
    ParLinearForm b(vfes);
    b.AddDomainIntegrator(new VectorDomainLFIntegrator(w_coeff));
    b.Assemble();

    ParBilinearForm vLap(vfes);
    ConstantCoefficient one(1.0); 
    vLap.AddDomainIntegrator(new VectorDiffusionIntegrator(one));
    vLap.SetAssemblyLevel(AssemblyLevel::PARTIAL);
    vLap.Assemble();

    OperatorPtr A;
    Vector B,X;
    ParGridFunction x(vfes);
    x = 0.0;

    vLap.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
       
    Solver *prec = new OperatorJacobiSmoother(vLap, ess_tdof_list);
    
    CGSolver cg(MPI_COMM_WORLD);
    cg.SetRelTol(1e-12);
    cg.SetMaxIter(2000);
    cg.SetPrintLevel(1);
    cg.SetOperator(*A);
    cg.Mult(B, X);

    vLap.RecoverFEMSolution(X, b, x);

    solver->ComputeCurl3D(x, u_vort);

    delete prec;

}

void VelocityDecomposition(
                          ParGridFunction &u,
                          ParGridFunction &curl_Ah,
                          ParGridFunction &grad_phi,
                          ParMesh *pmesh, int order, bool pa, real_t delta_const)
{

   int dim = pmesh->Dimension();
   int sdim = pmesh->SpaceDimension();
   FiniteElementCollection *fec    = new ND_FECollection(order, dim);
   FiniteElementCollection *nd_fec = new ND_FECollection(order, dim);   // H(curl)
   FiniteElementCollection *rt_fec = new RT_FECollection(order-1, dim); // H(div)
   FiniteElementCollection *l2_fec = new L2_FECollection(order-1, dim);
   FiniteElementCollection *h1_fec = new H1_FECollection(order, dim);

   ParFiniteElementSpace *l2_fespace_scalar = new ParFiniteElementSpace(pmesh, l2_fec);
   ParFiniteElementSpace *l2_fespace_vector = new ParFiniteElementSpace(pmesh, l2_fec, dim);

   ParFiniteElementSpace *nd_fespace = new ParFiniteElementSpace(pmesh, nd_fec);
   ParFiniteElementSpace *rt_fespace = new ParFiniteElementSpace(pmesh, rt_fec);
   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
   ParFiniteElementSpace *h1_fespace_scalar = new ParFiniteElementSpace(pmesh, h1_fec);
   ParFiniteElementSpace *h1_fespace_vector = new ParFiniteElementSpace(pmesh, h1_fec, dim);


   ParGridFunction u_l2(l2_fespace_vector);
   ParGridFunction u_hcurl_l2_project(nd_fespace);
   ParGridFunction curl_u_hdiv(rt_fespace);
   ParGridFunction curl_u_hcurl_l2_project(nd_fespace);
   ParGridFunction temp_hdiv_test(rt_fespace);
   ParGridFunction temp_hcurl_test(nd_fespace);

   // ParDiscreteLinearOperator H1_to_L2_op(h1_fespace_vector, l2_fespace_vector);
   // H1_to_L2_op.AddDomainInterpolator(new IdentityInterpolator);
   // H1_to_L2_op.Assemble();
   // H1_to_L2_op.Finalize();
   // H1_to_L2_op.Mult(u, u_l2);

   // Reference approach (simpler):
   VectorGridFunctionCoefficient u_coeff(&u);
   u_l2.ProjectCoefficient(u_coeff);

   // Use this:
   // project_H1_to_L2(u_l2, u, h1_fespace_vector, l2_fespace_vector, pa);

   u_hcurl_l2_project = 0.0;
   curl_u_hdiv = 0.0;
   curl_u_hcurl_l2_project = 0.0;
   temp_hdiv_test = 0.0;
   temp_hcurl_test = 0.0;
   u_hcurl_l2_project = 0.0;
   
   // 1. Project u to Hcurl
   project_H1_to_Hcurl(u_hcurl_l2_project, u, h1_fespace_vector, nd_fespace, pa);

   // 2. Compute the curl of u which will now be in H(div)
   compute_Curl_Hcurl_to_Hdiv(curl_u_hdiv, u_hcurl_l2_project, nd_fespace, temp_hdiv_test, rt_fespace, pa);

   // 3. Project u H(div) to u H(curl), this will the rhs of the solver
   project_Hcurl_Hdiv(curl_u_hcurl_l2_project, curl_u_hdiv, rt_fespace, 
      u_hcurl_l2_project, nd_fespace, pa);

   // 8. Determine the list of true (i.e. parallel conforming) essential //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   Array<int> ess_bdr;

   if (pmesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr = 0;
      // nd_fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 9. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (f,phi_i) where f is given by the function f_exact and phi_i are the
   //    basis functions in the finite element fespace.
   VectorGridFunctionCoefficient f(&curl_u_hcurl_l2_project);
   ParLinearForm *b = new ParLinearForm(nd_fespace);
   b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
   b->Assemble();

   // 10. Define the solution vector x as a parallel finite element grid function
   //     corresponding to fespace. Initialize x by projecting the exact
   //     solution. Note that only values from the boundary edges will be used
   //     when eliminating the non-homogeneous boundary condition to modify the
   //     r.h.s. vector b.
   ParGridFunction x(nd_fespace);
   x = 0.0;

   // 11. Set up the parallel bilinear form corresponding to the EM diffusion
   //     operator curl muinv curl + sigma I, by adding the curl-curl and the
   //     mass domain integrators.
   Coefficient *muinv = new ConstantCoefficient(1.0);
   Coefficient *sigma = new ConstantCoefficient(delta_const);
   ParBilinearForm *a = new ParBilinearForm(nd_fespace);
   if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a->AddDomainIntegrator(new CurlCurlIntegrator(*muinv));
   a->AddDomainIntegrator(new VectorFEMassIntegrator(*sigma));

   // 12. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   a->Assemble();

   OperatorPtr A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   // 13. Solve the system AX=B using PCG with an AMS preconditioner.
   if (pa)
   {
#ifdef MFEM_USE_AMGX
      MatrixFreeAMS ams(*a, *A, *fespace, muinv, sigma, NULL, ess_bdr, useAmgX);
#else
      MatrixFreeAMS ams(*a, *A, *nd_fespace, muinv, sigma, NULL, ess_bdr);
#endif
      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(1000);
      cg.SetPrintLevel(1);
      cg.SetOperator(*A);
      cg.SetPreconditioner(ams);
      cg.Mult(B, X);
   }
   else
   {
      if (Mpi::Root())
      {
         std::cout << "Size of linear system: "
              << A.As<HypreParMatrix>()->GetGlobalNumRows() << std::endl;
      }

      ParFiniteElementSpace *prec_ndfespace =
         (a->StaticCondensationIsEnabled() ? a->SCParFESpace() : nd_fespace);
      HypreAMS ams(*A.As<HypreParMatrix>(), prec_ndfespace);
      // ams.SetSingularProblem();
      HyprePCG pcg(*A.As<HypreParMatrix>());
      pcg.SetTol(1e-12);
      pcg.SetMaxIter(500);
      pcg.SetPrintLevel(2);
      pcg.SetPreconditioner(ams);
      pcg.Mult(B, X);
   }

   // 14. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a->RecoverFEMSolution(X, *b, x);

   // The test space which is being projected to is
   // H(div) from the trial space H(curl)
   ParGridFunction Ah_hdiv(rt_fespace);
   ParGridFunction Ah_hcurl(nd_fespace);
   // I think this is redundant
   Ah_hcurl = x;
   project_Hcurl_Hdiv(Ah_hdiv, x, nd_fespace, Ah_hdiv, rt_fespace, pa);

   // Compute curl of Ah in H(div)
   ParGridFunction curl_Ah_hdiv(rt_fespace);
   compute_Curl_Hcurl_to_Hdiv(curl_Ah_hdiv, x, nd_fespace, temp_hdiv_test, rt_fespace, pa);

   ParGridFunction curl_Ah_l2(l2_fespace_vector);
   project_Hdiv_to_L2(curl_Ah_l2, curl_Ah_hdiv,l2_fespace_vector,pa);

   ParGridFunction grad_phi_l2(l2_fespace_vector);
   grad_phi_l2 = u_l2;
   grad_phi_l2 -= curl_Ah_l2;

   // Create coefficients from the L2 grid functions
   VectorGridFunctionCoefficient grad_phi_l2_coeff(&grad_phi_l2);
   VectorGridFunctionCoefficient curl_Ah_l2_coeff(&curl_Ah_l2);

   // Project to grad phi_l2 and curl_Ah_l2 to H1
   ParGridFunction grad_phi_h1(h1_fespace_vector);
   ParGridFunction curl_Ah_h1(h1_fespace_vector);
   
   // Use ProjectDiscCoefficient for averaging-based projection from L2 to H1
   grad_phi_h1.ProjectDiscCoefficient(grad_phi_l2_coeff);
   curl_Ah_h1.ProjectDiscCoefficient(curl_Ah_l2_coeff);

   curl_Ah = curl_Ah_h1;
   grad_phi = grad_phi_h1;

   // Set \nabla \cdot (\nabla \times Ah) to be in L2
   ParGridFunction div_curl_Ah(l2_fespace_scalar);
   ParDiscreteLinearOperator div_op(rt_fespace, l2_fespace_scalar);
   div_op.AddDomainInterpolator(new DivergenceInterpolator);
   div_op.Assemble();
   div_op.Finalize();

   // Compute \nabla \cdot (\nabla \times Ah) in H(div)
   ParGridFunction div_curl_Ah_l2(l2_fespace_scalar);
   div_op.Mult(curl_Ah_hdiv, div_curl_Ah_l2);

   ParGridFunction div_Ah_l2(l2_fespace_scalar);
   div_op.Mult(Ah_hdiv, div_Ah_l2);

   {
      ConstantCoefficient zero(0.0);
      double div_curl_A_error = div_curl_Ah_l2.ComputeL2Error(zero);
      double div_A_error = div_Ah_l2.ComputeL2Error(zero);

      if (Mpi::Root())
      {
         std::cout << "div(curl A) L2 norm: " << div_curl_A_error << std::endl;
         std::cout << "div(A) L2 norm: " << div_A_error << std::endl;
      }
   }

   ParGridFunction u_reconstructed(l2_fespace_vector);
   u_reconstructed = curl_Ah_l2;
   u_reconstructed += grad_phi_l2;
   
   double reconstruction_error = u_reconstructed.ComputeL2Error(u_coeff);
   if (Mpi::Root()){
       std::cout << "Reconstruction error: " << reconstruction_error << std::endl;
   }

   DataCollection *dc = NULL;
   std::string visit_dir = std::string("Decomposition_VisitData_") 
                                               + "Re" + std::to_string(static_cast<int>(ctx.reynum)) 
                                               + "NumPtsPerDir" +std::to_string(ctx.num_pts) 
                                               + "RefLv" + std::to_string(
                                                   ctx.element_subdivisions 
                                                 + ctx.element_subdivisions_parallel) 
                                               + "P" + std::to_string(ctx.order)
                                               + "/output_visit";

   dc = new VisItDataCollection(MPI_COMM_WORLD,visit_dir, pmesh);
   int precision = 16;
   dc->SetPrecision(precision);
   dc->SetCycle(0);
   dc->SetTime(0);
   dc->SetFormat(DataCollection::PARALLEL_FORMAT);
   dc->RegisterField("u", &u);
   dc->RegisterField("u_l2", &u_l2);
   dc->RegisterField("u_hcurl", &u_hcurl_l2_project);
   dc->RegisterField("curl_u_hdiv", &curl_u_hdiv);
   dc->RegisterField("curl_u_hcurl", &curl_u_hcurl_l2_project);
   dc->RegisterField("Ah", &x);
   dc->RegisterField("Ah_hdiv", &Ah_hdiv);
   dc->RegisterField("curl_Ah_hdiv", &curl_Ah_hdiv);
   dc->RegisterField("curl_Ah_l2", &curl_Ah_l2);
   dc->RegisterField("curl_Ah_h1", &curl_Ah_h1);
   dc->RegisterField("grad_phi_l2", &grad_phi_l2);
   dc->RegisterField("grad_phi_h1", &grad_phi_h1);
   dc->Save();


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
   args.AddOption(&ctx.alpha, "-alpha", "--Filter-Amplitude", "Filter Amplitude, filter must be true");
   args.AddOption(&ctx.delta_const, "-regscl", "--Regularization-Scale", "Small constant for velocity decomposition solver");
   args.AddOption(&ctx.problem1, "-problem1", "--Problem-1", "-no-problem1",
                  "--no-Problem-1",
                  "Domain length will be 2pi, otherwise 1.0");
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
      if (!ctx.problem1)
      {
        length = 1.0;
      }
      init_mesh = new Mesh(Mesh::MakeCartesian3D(ctx.num_pts,
                                                 ctx.num_pts,
                                                 ctx.num_pts,
                                                 Element::HEXAHEDRON,
                                                 length,
                                                 length,
                                                 length, false));

      // init_mesh = new Mesh(Mesh::MakeCartesian3DWith24TetsPerHex(ctx.num_pts,
      //                                            ctx.num_pts,
      //                                            ctx.num_pts,
      //                                            length,
      //                                            length,
      //                                            length));

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
      VectorFunctionCoefficient u_excoeff(pmesh->Dimension(), vel_tgv);

      u_gf->ProjectCoefficient(u_excoeff);
      p_gf = flowsolver->GetCurrentPressure();

      if(ctx.filter){
        flowsolver->SetFilterAlpha(ctx.alpha); // Enable sharp cutoff
        flowsolver->SetCutoffModes(ctx.order-1);   // Cut off highest mode
      }

      // Set up the flow solver
      flowsolver->Setup(ctx.dt);

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
   ParGridFunction ke_gf(pressure_fespace);


   ParGridFunction curl_Ah(velocity_fespace);
   ParGridFunction grad_phi(velocity_fespace);

   // VelocityDecomposition(*u_gf, curl_Ah, grad_phi,pmesh,ctx.order,ctx.pa,ctx.delta_const);
   // SamplePoints( u_gf, pmesh,0, 0, "Velocity", &ctx);
   // ParGridFunction divu_gf(pressure_fespace);

   // ParGridFunction u_comp(velocity_fespace);
   // ParGridFunction u_vort(velocity_fespace);

   flowsolver->ComputeCurl3D(*u_gf, w_gf);
   ComputeQCriterion(*u_gf, q_gf);
   ComputeDissipation(*u_gf, d_gf);

   // ComputeDivergence3D(*u_gf, divu_gf);
   // ComputeVorticalPart(flowsolver, *u_gf, w_gf, u_vort);

   QuantitiesOfInterest kin_energy(pmesh);
   real_t ke = kin_energy.ComputeKineticEnergy(*u_gf, ke_gf);

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
                                               + "/output_paraview";

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
         dc = new SidreDataCollection("output_sidre", pmesh);
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
                                                  + "/output_visit";

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
      dc->RegisterField("grad_phi", &grad_phi);
      dc->RegisterField("curl_Ah", &curl_Ah);
      // dc->RegisterField("dissipation", &d_gf);
      // dc->RegisterField("divu", &divu_gf);
      // dc->RegisterField("ke", &ke_gf);
      // dc->RegisterField("u_vort", &u_vort);
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
                                                  + "/output_conduit";

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

   // real_t ke = kin_energy.ComputeKineticEnergy(*u_gf);
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

   real_t kmax = 0.0;
   real_t hmin = 0.0;

   real_t avg_diss = kin_energy.ComputeAveragedDissipation(d_gf);
   kin_energy.ComputeKolmogorovAndTaylorMicroLength(d_gf, avg_diss, &kolmLenScl, &avg_lambda, &avg_kolmLenScl, &kolmTimeScl, &avg_kolmTimeScl, &max_diss, ke);
   kin_energy.ComputeGridPtsRequirementsTurb(*u_gf, kolmLenScl, &hmin_eta, &kmax_eta, &kmax, &hmin);

   // Pope definetion of grid resolution
   real_t avg_hmin_eta = hmin/avg_kolmLenScl;
   real_t avg_kmax_eta = kmax*avg_kolmLenScl;

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
          fprintf(f_turb_grid, "        Average PI_NU              Min PI_NU        ");
          fprintf(f_turb_grid, "        K_max*eta(Avg)             hmin/eta(Avg)    \n");

          // Write the initial data point
           fprintf(f_turb_grid, "%20.16e     %20.16e     %20.16e     %20.16e     %20.16e    %20.16e    %20.16e    %20.16e\n",
                       t, static_cast<real_t>(global_cycle + step), kmax_eta, hmin_eta, PI_nu, PI_nu_min, avg_kmax_eta, avg_hmin_eta);
      } 

      fflush(f);
      fflush(f_turb);
      fflush(f_turb_grid);
      fflush(stdout);
   }

   real_t dt = ctx.dt;
   real_t t_final = ctx.t_final;
   bool last_step = false;

   for (; !last_step; ++step)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      // Adjust alpha for restart
      real_t effective_alpha = ctx.alpha;  // Default to the original alpha
      if (ctx.restart && restart_files_found && step <= 500)  // Ramp over first 10 steps
      {
   
         // Gradual ramp up
         real_t ramp_factor = 0.05 + 0.95 * (step / 500.0);
         effective_alpha = ctx.alpha * ramp_factor;
         
         if (Mpi::Root())
         {
            std::cout << "Restart ramp: using alpha = " << effective_alpha 
                      << " (step " << step << "/500)" << std::endl;
         }
      }

      if(ctx.filter){
        // Update the filter amplification
        flowsolver->SetFilterAlpha(effective_alpha);
      }
   
      flowsolver->Step(t, dt, step);

      cfl = flowsolver->ComputeCFL(*u_gf, ctx.dt);

      if ((global_cycle + step) % ctx.data_dump_cycle == 0 || last_step)
      {
         // If restarting, skip the first saved checkpoint
         if (!(ctx.restart && step == 0 && restart_files_found))
         {
            ComputeQCriterion(*u_gf, q_gf);
            flowsolver->ComputeCurl3D(*u_gf, w_gf);
            // VelocityDecomposition(*u_gf,curl_Ah,grad_phi,pmesh,ctx.order,ctx.pa,ctx.delta_const);

            // ComputeDivergence3D(*u_gf, divu_gf);
            // ComputeVorticalPart(flowsolver, *u_gf, w_gf, u_vort);

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
            SamplePointsAtDoFs(u_gf, pmesh, global_cycle + step, t, "Velocity", &ctx);
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

      ke = kin_energy.ComputeKineticEnergy(*u_gf, ke_gf);
      // ke = kin_energy.ComputeKineticEnergy(*u_gf);
      vel_curl_ke = kin_energy.ComputeInertialRangeEnergy(*u_gf);
      enstrophy = kin_energy.ComputeEnstrophy(w_gf);

      ComputeDissipation(*u_gf, d_gf);
      avg_diss = kin_energy.ComputeAveragedDissipation(d_gf);
      kin_energy.ComputeKolmogorovAndTaylorMicroLength(d_gf, avg_diss, &kolmLenScl, &avg_lambda, &avg_kolmLenScl, &kolmTimeScl, &avg_kolmTimeScl, &max_diss, ke);
      kin_energy.ComputeGridPtsRequirementsTurb(*u_gf, kolmLenScl, &hmin_eta, &kmax_eta, &kmax, &hmin);
      Re_taylor = u_rms*avg_lambda/ctx.kinvis;
      u_rms =  pow(2.0/3.0*ke,0.5);

      PI_nu = pow(avg_diss,0.5)/(avg_kolmLenScl*pow(vel_curl_ke,0.75));
      PI_nu_min = pow(max_diss,0.5)/(kolmLenScl*pow(vel_curl_ke,0.75));

      avg_hmin_eta = hmin/avg_kolmLenScl;
      avg_kmax_eta = kmax*avg_kolmLenScl;


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
           // fprintf(f_turb_grid, "%20.16e     %20.16e     %20.16e     %20.16e     %20.16e    %20.16e\n",
           //             t, static_cast<real_t>(global_cycle + step), kmax_eta, hmin_eta, PI_nu, PI_nu_min);
           fprintf(f_turb_grid, "%20.16e     %20.16e     %20.16e     %20.16e     %20.16e    %20.16e    %20.16e    %20.16e\n",
                       t, static_cast<real_t>(global_cycle + step), kmax_eta, hmin_eta, PI_nu, PI_nu_min, avg_kmax_eta, avg_hmin_eta);
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



// The test space is what you are projecting to and the trial space is where you are projecting from
void project_Hcurl_Hdiv(ParGridFunction &result, ParGridFunction &gftrial, ParFiniteElementSpace *trial_fes,  
                        ParGridFunction &gftest, ParFiniteElementSpace *test_fes, bool pa)
{
   ParBilinearForm *a = new ParBilinearForm(test_fes);
   if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a->AddDomainIntegrator(new VectorFEMassIntegrator());
   ParMixedBilinearForm *a_mixed = new ParMixedBilinearForm(trial_fes, test_fes);
   if (pa) {a_mixed->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a_mixed->AddDomainIntegrator(new VectorFEMassIntegrator());

   // a_mixed->AddDomainIntegrator(new MixedVectorMassIntegrator());  // More explicit

   a->Assemble();
   if(!pa){a->Finalize();}

   a_mixed->Assemble();
   if(!pa){a_mixed->Finalize();}

   Vector B(test_fes->GetTrueVSize());
   Vector X(test_fes->GetTrueVSize());

   if (pa)
   {
      ParLinearForm b(test_fes); // used as a vector
      a_mixed->Mult(gftrial, b); // process-local multiplication
      b.ParallelAssemble(B);
   }
   else
   {
      HypreParMatrix *mixed = a_mixed->ParallelAssemble();

      Vector P(trial_fes->GetTrueVSize());
      gftrial.GetTrueDofs(P);

      mixed->Mult(P,B);

      delete mixed;
   }

    // 11. Define and apply a parallel PCG solver for AX=B with Jacobi
   //     preconditioner.
   if (pa)
   {
      Array<int> ess_tdof_list; // empty

      OperatorPtr A;
      a->FormSystemMatrix(ess_tdof_list, A);

      OperatorJacobiSmoother Jacobi(*a, ess_tdof_list);

      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(1000);
      cg.SetPrintLevel(1);
      cg.SetOperator(*A);
      cg.SetPreconditioner(Jacobi);
      X = 0.0;
      cg.Mult(B, X);
   }
   else
   {
      HypreParMatrix *Amat = a->ParallelAssemble();
      HypreDiagScale Jacobi(*Amat);
      HyprePCG pcg(*Amat);
      pcg.SetTol(1e-12);
      pcg.SetMaxIter(1000);
      pcg.SetPrintLevel(2);
      pcg.SetPreconditioner(Jacobi);
      X = 0.0;
      pcg.Mult(B, X);

      delete Amat;
   }

   result.SetFromTrueDofs(X);
}

// The test space is what you are projecting to and the trial space is where you are projecting from
void compute_Curl_Hcurl_to_Hdiv(ParGridFunction &result, ParGridFunction &gftrial, ParFiniteElementSpace *trial_fes,  
                        ParGridFunction &gftest, ParFiniteElementSpace *test_fes, bool pa)
{
   ParBilinearForm *a = new ParBilinearForm(test_fes);
   if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a->AddDomainIntegrator(new VectorFEMassIntegrator());
   ParMixedBilinearForm *a_mixed = new ParMixedBilinearForm(trial_fes, test_fes);
   if (pa) {a_mixed->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a_mixed->AddDomainIntegrator(new MixedVectorCurlIntegrator());

   a->Assemble();
   if(!pa){a->Finalize();}

   a_mixed->Assemble();
   if(!pa){a_mixed->Finalize();}

   Vector B(test_fes->GetTrueVSize());
   Vector X(test_fes->GetTrueVSize());

   if (pa)
   {
      ParLinearForm b(test_fes); // used as a vector
      a_mixed->Mult(gftrial, b); // process-local multiplication
      b.ParallelAssemble(B);
   }
   else
   {
      HypreParMatrix *mixed = a_mixed->ParallelAssemble();

      Vector P(trial_fes->GetTrueVSize());
      gftrial.GetTrueDofs(P);

      mixed->Mult(P,B);

      delete mixed;
   }

    // 11. Define and apply a parallel PCG solver for AX=B with Jacobi
   //     preconditioner.
   if (pa)
   {
      Array<int> ess_tdof_list; // empty

      OperatorPtr A;
      a->FormSystemMatrix(ess_tdof_list, A);

      OperatorJacobiSmoother Jacobi(*a, ess_tdof_list);

      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(1000);
      cg.SetPrintLevel(1);
      cg.SetOperator(*A);
      cg.SetPreconditioner(Jacobi);
      X = 0.0;
      cg.Mult(B, X);
   }
   else
   {
      HypreParMatrix *Amat = a->ParallelAssemble();
      HypreDiagScale Jacobi(*Amat);
      HyprePCG pcg(*Amat);
      pcg.SetTol(1e-12);
      pcg.SetMaxIter(1000);
      pcg.SetPrintLevel(2);
      pcg.SetPreconditioner(Jacobi);
      X = 0.0;
      pcg.Mult(B, X);

      delete Amat;
   }

   result.SetFromTrueDofs(X);
}

// Project H(div) field (u_hdiv) into vector L2(DG) (result) in the true L2 sense: M y = b.
// test_fes must be a vector L2/DG space with vdim = mesh dim.
void project_Hdiv_to_L2(ParGridFunction &result,
                        ParGridFunction &u_hdiv,
                        ParFiniteElementSpace *test_fes,   // vector L2 target
                        bool pa)
{
   // 1) L2 mass operator on target
   ParBilinearForm a(test_fes);
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a.AddDomainIntegrator(new VectorMassIntegrator());
   a.Assemble();
   if (!pa) { a.Finalize(); }

   // 2) RHS: b_i = (u_hdiv, phi_i)
   VectorGridFunctionCoefficient ucoeff(&u_hdiv);
   ParLinearForm b(test_fes);
   b.AddDomainIntegrator(new VectorDomainLFIntegrator(ucoeff));
   b.Assemble();

   // true-dof vectors
   Vector B(test_fes->GetTrueVSize());
   Vector X(test_fes->GetTrueVSize());
   b.ParallelAssemble(B);
   X = 0.0;

   if (pa)
   {
      Array<int> ess_tdof_list;
      OperatorPtr Aop;
      a.FormSystemMatrix(ess_tdof_list, Aop);

      OperatorJacobiSmoother Jacobi(a, ess_tdof_list); 
      CGSolver cg(test_fes->GetComm());
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(200);
      cg.SetPrintLevel(0);
      cg.SetOperator(*Aop);
      cg.SetPreconditioner(Jacobi);
      cg.Mult(B, X);
   }
   else
   {
      // Fully assembled Hypre path
      std::unique_ptr<HypreParMatrix> A(a.ParallelAssemble());
      HypreDiagScale Jacobi(*A);
      HyprePCG pcg(*A);
      pcg.SetTol(1e-12);
      pcg.SetMaxIter(200);
      pcg.SetPrintLevel(2);
      pcg.SetPreconditioner(Jacobi);
      pcg.Mult(B, X);
   }

   // 3) Scatter to result in L2(DG)
   result = 0.0;
   result.SetFromTrueDofs(X);
}

// Project H1 (vector) → H(curl) (ND) in L2-sense.
void project_H1_to_Hcurl(ParGridFunction &result,          // in ND space (output)
                         ParGridFunction &u_h1,            // in H1 vector space (input)
                         ParFiniteElementSpace *fes_h1,    // not used, but keep for symmetry
                         ParFiniteElementSpace *fes_nd,    // ND test/target
                         bool pa)
{
   // Mass matrix on the ND space
   ParBilinearForm M(fes_nd);
   if (pa) { M.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   M.AddDomainIntegrator(new VectorFEMassIntegrator()); // <- ND/RT mass
   M.Assemble();
   if (!pa) { M.Finalize(); }

   // RHS: b_i = (u_h1, w_i) with w_i in ND
   VectorGridFunctionCoefficient ucoeff(&u_h1);
   ParLinearForm b(fes_nd);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(ucoeff)); // <- ND/RT RHS
   b.Assemble();

   Vector B(fes_nd->GetTrueVSize()), X(fes_nd->GetTrueVSize());
   b.ParallelAssemble(B);
   X = 0.0;

   if (pa)
   {
      Array<int> ess_tdof_list; // none for pure L2 projection
      OperatorPtr Mop;
      M.FormSystemMatrix(ess_tdof_list, Mop);
      OperatorJacobiSmoother Jacobi(M, ess_tdof_list);
      CGSolver cg(fes_nd->GetComm());
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(500);
      cg.SetPrintLevel(0);
      cg.SetOperator(*Mop);
      cg.SetPreconditioner(Jacobi);
      cg.Mult(B, X);
   }
   else
   {
      std::unique_ptr<HypreParMatrix> Mpar(M.ParallelAssemble());
      HypreDiagScale Jacobi(*Mpar);
      HyprePCG pcg(*Mpar);
      pcg.SetTol(1e-12);
      pcg.SetMaxIter(500);
      pcg.SetPrintLevel(2);
      pcg.SetPreconditioner(Jacobi);
      pcg.Mult(B, X);
   }

   result = 0.0;
   result.SetFromTrueDofs(X);
}

// Project H1 (vector) → L2 (vector) in the L2 sense: M_L2 * result = M_mixed * u_h1
void project_H1_to_L2(ParGridFunction &result,          // in L2 space (output)
                      ParGridFunction &u_h1,            // in H1 vector space (input)
                      ParFiniteElementSpace *fes_h1,    // H1 trial space
                      ParFiniteElementSpace *fes_l2,    // L2 test/target space
                      bool pa)
{
   // L2 mass matrix on the target space
   ParBilinearForm M_L2(fes_l2);
   if (pa) { M_L2.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   M_L2.AddDomainIntegrator(new VectorFEMassIntegrator()); // L2 mass for vectors
   M_L2.Assemble();
   if (!pa) { M_L2.Finalize(); }

   // Mixed mass matrix: (u_h1, v_l2) for u_h1 in H1, v_l2 in L2
   ParMixedBilinearForm M_mixed(fes_h1, fes_l2);
   if (pa) { M_mixed.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   M_mixed.AddDomainIntegrator(new VectorMassIntegrator()); // Mixed mass integrator
   M_mixed.Assemble();
   if (!pa) { M_mixed.Finalize(); }

   // Set up RHS: B = M_mixed * u_h1
   Vector B(fes_l2->GetTrueVSize());
   Vector X(fes_l2->GetTrueVSize());

   if (pa)
   {
      ParLinearForm b(fes_l2); // used as a vector
      M_mixed.Mult(u_h1, b); // process-local multiplication
      b.ParallelAssemble(B);
   }
   else
   {
      HypreParMatrix *mixed = M_mixed.ParallelAssemble();
      Vector P(fes_h1->GetTrueVSize());
      u_h1.GetTrueDofs(P);
      mixed->Mult(P, B);
      delete mixed;
   }

   // Solve M_L2 * X = B
   X = 0.0;
   if (pa)
   {
      Array<int> ess_tdof_list; // empty for pure L2 projection
      OperatorPtr M_op;
      M_L2.FormSystemMatrix(ess_tdof_list, M_op);
      
      OperatorJacobiSmoother Jacobi(M_L2, ess_tdof_list);
      CGSolver cg(fes_l2->GetComm());
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(500);
      cg.SetPrintLevel(0);
      cg.SetOperator(*M_op);
      cg.SetPreconditioner(Jacobi);
      cg.Mult(B, X);
   }
   else
   {
      std::unique_ptr<HypreParMatrix> M_par(M_L2.ParallelAssemble());
      HypreDiagScale Jacobi(*M_par);
      HyprePCG pcg(*M_par);
      pcg.SetTol(1e-12);
      pcg.SetMaxIter(500);
      pcg.SetPrintLevel(2);
      pcg.SetPreconditioner(Jacobi);
      pcg.Mult(B, X);
   }

   // Set result from true DOFs
   result = 0.0;
   result.SetFromTrueDofs(X);
}