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

// Example runs
// mpirun -n 16 ./navier_tgv_victor_restart -no-ovs -o 4 -num_pts_per_dir 8 -es 1 -esp 1 -Re 1600 -tf 20 -time-out -nsnap 25 -no-problem1 -u0_from_mach -mach0 0.2
// mpirun -n 16 ./navier_tgv_victor_restart -no-ovs -o 4 -num_pts_per_dir 8 -es 1 -esp 1 -Re 1600 -tf 20 -time-out -nsnap 25 -no-problem1
// mpirun -n 16 ./navier_tgv_victor_restart -no-ovs -o 4 -num_pts_per_dir 8 -es 1 -esp 1 -Re 1600 -tf 20 -ddc 1000

#include "navier_solver.hpp"
#include "navier_utils.hpp"
#include <fstream>
#include <algorithm>
#include <iostream>
#include <string>

using namespace mfem;
using namespace navier;

real_t delta_const = 1e-8;
bool static_cond = false;
bool snapshot_dumped = false;
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
   bool oversample = false;
   real_t alpha = 0.3;
   real_t delta_const = 1e-8;
   bool problem1 = true;
   double u0 = 1.0;

   // Add these for time-based output
   bool time_based_output = false;  
   int num_snapshots = 25;          
   real_t snapshot_interval = 0.0;  
   int snapshot_index = 0;          // Which snapshot we're looking for next
   std::vector<real_t> snapshot_times; // Pre-computed target times

   bool u0_based_on_mach = false;
   double Mach0 = 0.1;
   double time_snapshot_dump = 1.43239;
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

   u(0) =  ctx.u0*sin(xi) * cos(yi) * cos(zi);
   u(1) = -ctx.u0*cos(xi) * sin(yi) * cos(zi);
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

   // real_t ComputeKineticEnergy(ParGridFunction &v, ParGridFunction &ke_gf)
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

    // ParFiniteElementSpace *vfes = v.ParFESpace();
    // ParBilinearForm mass(vfes);
    // ConstantCoefficient ones(1.0);
    // mass.AddDomainIntegrator(new VectorMassIntegrator(ones));
  
    // mass.Assemble();
    // mass.Finalize();

    // // Create KE grid function
    // VectorGridFunctionCoefficient U(&v);     

    // InnerProductCoefficient uu(U, U);          

    // ConstantCoefficient half(0.5);

    // ProductCoefficient kcoeff(half, uu);       

    // ke_gf.ProjectCoefficient(kcoeff);

    // const double ke = 0.5*mass.ParInnerProduct(v,v);
    // return ke / volume;
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
           
        int intorder = 2 * el->GetOrder() + 2;
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

   /*
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
  }*/


  real_t ComputeEnstrophy(ParGridFunction &u)
  {

   const ParFiniteElementSpace *pfes = u.ParFESpace();

   double local_half_w2_int = 0.0; 
   double local_vol        = 0.0;  

   Array<int> vdofs;
   Vector loc_data;                 
   DenseMatrix dshape;              
   DenseMatrix grad_hat;            
   DenseMatrix grad;                

   const int ne = pfes->GetNE();
   for (int e = 0; e < ne; ++e)
   {
      pfes->GetElementVDofs(e, vdofs);
      u.GetSubVector(vdofs, loc_data);

      ElementTransformation *T = pfes->GetElementTransformation(e);
      const FiniteElement   *el = pfes->GetFE(e);

      const int elndofs = el->GetDof();
      const int vdim    = pfes->GetVDim();
      const int dim     = 3;

      const int ir_order = 2*el->GetOrder() + 2;
      const IntegrationRule &ir = IntRules.Get(el->GetGeomType(), ir_order);

      dshape.SetSize(elndofs, dim);
      DenseMatrix loc_data_mat(loc_data.GetData(), elndofs, vdim);

      for (int i = 0; i < ir.GetNPoints(); ++i)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         T->SetIntPoint(&ip);

         el->CalcDShape(ip, dshape);

         grad_hat.SetSize(vdim, dim);
         MultAtB(loc_data_mat, dshape, grad_hat);

         const DenseMatrix &Jinv = T->InverseJacobian();

         grad.SetSize(vdim, dim);
         Mult(grad_hat, Jinv, grad); 

         const double wx = grad(2,1) - grad(1,2);
         const double wy = grad(0,2) - grad(2,0);
         const double wz = grad(1,0) - grad(0,1);
         const double w2 = wx*wx + wy*wy + wz*wz;

         const double dV = ip.weight * T->Weight();

         local_half_w2_int += 0.5 * w2 * dV;
         local_vol         += dV;        
      }
   }

   double global_half_w2_int = 0.0;
   double global_vol         = 0.0;
   MPI_Comm comm = pfes->GetComm();
   MPI_Allreduce(&local_half_w2_int, &global_half_w2_int, 1, MPI_DOUBLE, MPI_SUM, comm);
   MPI_Allreduce(&local_vol,         &global_vol,         1, MPI_DOUBLE, MPI_SUM, comm);

   return global_half_w2_int / global_vol;

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

   double ComputeTaylorMicroscale(const mfem::ParGridFunction &u,
                                                      double lambda_components[3])
   {

      MFEM_VERIFY(u.ParFESpace() != nullptr, "Velocity ParGridFunction has no FESpace.");
      const ParFiniteElementSpace &fes = *u.ParFESpace();
      MFEM_VERIFY(fes.GetParMesh() && fes.GetParMesh()->Dimension() == 3, "Expect 3D mesh.");
      MFEM_VERIFY(fes.GetVDim() == 3, "Expect vdim=3 velocity.");

      MPI_Comm comm = fes.GetComm();
      ParMesh &pmesh = *fes.GetParMesh();

      double local_vol = 0.0;
      double local_u2[3]  = {0.0, 0.0, 0.0};  // ∫ u_b^2 dV
      double local_du2[3] = {0.0, 0.0, 0.0};  // ∫ (∂u_b/∂x_b)^2 dV

      Vector uval(3);
      DenseMatrix grad(3,3); // grad(i,j) = ∂u_i/∂x_j

      const int NE = pmesh.GetNE();
      for (int e = 0; e < NE; ++e)
      {
         const FiniteElement &fe = *fes.GetFE(e);
         ElementTransformation &T = *fes.GetElementTransformation(e);

         const int ir_order = 2*fe.GetOrder() + 2;
         const IntegrationRule &ir = IntRules.Get(fe.GetGeomType(), ir_order);

         for (int q = 0; q < ir.GetNPoints(); ++q)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            T.SetIntPoint(&ip);
            const double w = ip.weight * T.Weight(); // physical dV

            // ParGridFunction API: value via (elem, ip, vec)
            u.GetVectorValue(e, ip, uval);
            // Physical gradient via transformation
            u.GetVectorGradient(T, grad); // size vdim x dim

            local_vol += w;
            for (int b = 0; b < 3; ++b)
            {
               const double ub  = uval(b);
               const double dub = grad(b,b); // longitudinal derivative ∂u_b/∂x_b
               local_u2[b]  += ub*ub * w;
               local_du2[b] += dub*dub * w;
            }
         }
      }

      double vol = 0.0, u2[3], du2[3];
      MPI_Allreduce(&local_vol, &vol, 1, MPI_DOUBLE, MPI_SUM, comm);
      MPI_Allreduce(local_u2,  u2,  3, MPI_DOUBLE, MPI_SUM, comm);
      MPI_Allreduce(local_du2, du2, 3, MPI_DOUBLE, MPI_SUM, comm);

      double lambda_sum = 0.0;
      double lam_local[3] = {0.0, 0.0, 0.0};
      for (int b = 0; b < 3; ++b)
      {
         const double u2_avg  = (vol > 0.0) ? (u2[b]  / vol) : 0.0; // ⟨u_b^2⟩
         const double du2_avg = (vol > 0.0) ? (du2[b] / vol) : 0.0; // ⟨(∂u_b/∂x_b)^2⟩
         lam_local[b] = (du2_avg > 0.0) ? std::sqrt(u2_avg / du2_avg) : 0.0;
         lambda_sum  += lam_local[b];
      }

      // If caller provided storage, fill λx, λy, λz (order = x,y,z = components 0,1,2)
      if (lambda_components)
      {
         lambda_components[0] = lam_local[0];
         lambda_components[1] = lam_local[1];
         lambda_components[2] = lam_local[2];
      }

      return lambda_sum / 3.0; // component-averaged λ
   }

   template <typename T>
   T sq(T x)
   {
      return x * x;
   }

   // 3D, vdim=3.
   // Computes (by reference):
   //   skewness    = \displaystyle \frac{\left\langle \frac{1}{3}\left[(\partial_x u)^3 + (\partial_y v)^3 + (\partial_z w)^3\right]\right\rangle}
   //                           {\left(\left\langle \frac{1}{3}\left[(\partial_x u)^2 + (\partial_y v)^2 + (\partial_z w)^2\right]\right\rangle\right)^{3/2}}
   //
   //   flatness    = \displaystyle \frac{\left\langle \frac{1}{3}\left[(\partial_x u)^4 + (\partial_y v)^4 + (\partial_z w)^4\right]\right\rangle}
   //                           {\left(\left\langle \frac{1}{3}\left[(\partial_x u)^2 + (\partial_y v)^2 + (\partial_z w)^2\right]\right\rangle\right)^{2}}
   //
   //   D3_over_D1  = \displaystyle \frac{\left\langle \sum_{j=1}^{3}\left(\partial_{x_j} u_3\right)^2 \right\rangle}
   //                           {\left\langle \sum_{j=1}^{3}\left(\partial_{x_j} u_1\right)^2 \right\rangle}
   //
   //   E3_over_E1  = \displaystyle \frac{\left\langle u_3^2 \right\rangle}{\left\langle u_1^2 \right\rangle}
   //
   // All angle brackets \langle \cdot \rangle denote volume averages: \langle \phi \rangle = \frac{1}{V}\int \phi \, dV,
   // with V = \text{global\_volume}. We first form the averages, then take ratios/powers.
   void ComputeSkewFlat_D3D1_E3E1(const mfem::ParGridFunction &u,
                                  double &skewness,
                                  double &flatness,
                                  double &D3_over_D1,
                                  double &E3_over_E1)
   {
      using namespace mfem;

      MFEM_VERIFY(u.ParFESpace() != nullptr, "Velocity ParGridFunction has no FESpace.");
      const ParFiniteElementSpace &fes = *u.ParFESpace();
      MFEM_VERIFY(fes.GetParMesh() && fes.GetParMesh()->Dimension() == 3, "Expect 3D mesh.");
      MFEM_VERIFY(fes.GetVDim() == 3, "Expect vdim=3 velocity.");

      MPI_Comm comm = fes.GetComm();
      ParMesh &pmesh = *fes.GetParMesh();

      // ---------------- Local accumulators ----------------
      double local_volume = 0.0;

      // \int u_i^2 \, dV  (for E3/E1)
      double local_u2[3]  = {0.0, 0.0, 0.0};

      // Longitudinal moments: \int (\partial_{x}u)^p, (\partial_{y}v)^p, (\partial_{z}w)^p \, dV for p=2,3,4
      double local_d2[3]  = {0.0, 0.0, 0.0};
      double local_d3[3]  = {0.0, 0.0, 0.0};
      double local_d4[3]  = {0.0, 0.0, 0.0};

      // \int \sum_{j=1}^{3} (\partial_{x_j} u_i)^2 \, dV  (for D3/D1)
      double local_gsq[3] = {0.0, 0.0, 0.0};

      mfem::Vector      local_u_val(3);
      mfem::DenseMatrix local_grad(3, 3); // local_grad(i,j) = \partial_{x_j} u_i

      const int NE = pmesh.GetNE();

      for (int e = 0; e < NE; ++e)
      {
         const FiniteElement &fe = *fes.GetFE(e);
         ElementTransformation &T = *fes.GetElementTransformation(e);

         // Integration rule order sufficient for up to 4th moments
         const int ir_order = 2 * fe.GetOrder() + 2;
         const IntegrationRule &ir = IntRules.Get(fe.GetGeomType(), ir_order);

         for (int q = 0; q < ir.GetNPoints(); ++q)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            T.SetIntPoint(&ip);

            const double dV = ip.weight * T.Weight();

            u.GetVectorValue(e, ip, local_u_val);
            u.GetVectorGradient(T, local_grad);

            local_volume += dV;

            // --- \int u_i^2 dV ---
            local_u2[0] += sq(local_u_val(0)) * dV;
            local_u2[1] += sq(local_u_val(1)) * dV;
            local_u2[2] += sq(local_u_val(2)) * dV;

            // --- Longitudinal derivatives: (\partial_x u), (\partial_y v), (\partial_z w) ---
            local_d2[0] += sq(local_grad(0,0)) * dV;
            local_d2[1] += sq(local_grad(1,1)) * dV;
            local_d2[2] += sq(local_grad(2,2)) * dV;

            local_d3[0] +=  local_grad(0,0) * sq(local_grad(0,0)) * dV; // (\partial_x u)^3
            local_d3[1] +=  local_grad(1,1) * sq(local_grad(1,1)) * dV; // (\partial_y v)^3
            local_d3[2] +=  local_grad(2,2) * sq(local_grad(2,2)) * dV; // (\partial_z w)^3

            local_d4[0] += sq(sq(local_grad(0,0))) * dV; // (\partial_x u)^4
            local_d4[1] += sq(sq(local_grad(1,1))) * dV; // (\partial_y v)^4
            local_d4[2] += sq(sq(local_grad(2,2))) * dV; // (\partial_z w)^4

            // --- \int \sum_{j=1}^{3} (\partial_{x_j} u_i)^2 dV ---
            local_gsq[0] += ( sq(local_grad(0,0)) + sq(local_grad(0,1)) + sq(local_grad(0,2)) ) * dV;
            local_gsq[1] += ( sq(local_grad(1,0)) + sq(local_grad(1,1)) + sq(local_grad(1,2)) ) * dV;
            local_gsq[2] += ( sq(local_grad(2,0)) + sq(local_grad(2,1)) + sq(local_grad(2,2)) ) * dV;
         }
      }

      // ---------------- Global reductions ----------------
      double global_volume = 0.0;

      double global_u2[3];
      double global_d2[3];
      double global_d3[3];
      double global_d4[3];
      double global_gsq[3];

      MPI_Allreduce(&local_volume, &global_volume, 1, MPI_DOUBLE, MPI_SUM, comm);
      MPI_Allreduce(local_u2,      global_u2,      3, MPI_DOUBLE, MPI_SUM, comm);
      MPI_Allreduce(local_d2,      global_d2,      3, MPI_DOUBLE, MPI_SUM, comm);
      MPI_Allreduce(local_d3,      global_d3,      3, MPI_DOUBLE, MPI_SUM, comm);
      MPI_Allreduce(local_d4,      global_d4,      3, MPI_DOUBLE, MPI_SUM, comm);
      MPI_Allreduce(local_gsq,     global_gsq,     3, MPI_DOUBLE, MPI_SUM, comm);

      // ---------------- Form averages FIRST ----------------
      const double one_third = 1.0 / 3.0;

      // \left\langle \frac{1}{3}\big[(\partial_x u)^p + (\partial_y v)^p + (\partial_z w)^p\big] \right\rangle
      double avg_long_d2 = 0.0;
      double avg_long_d3 = 0.0;
      double avg_long_d4 = 0.0;

      if (global_volume > 0.0)
      {
         avg_long_d2 = one_third * ( (global_d2[0] + global_d2[1] + global_d2[2]) / global_volume );
         avg_long_d3 = one_third * ( (global_d3[0] + global_d3[1] + global_d3[2]) / global_volume );
         avg_long_d4 = one_third * ( (global_d4[0] + global_d4[1] + global_d4[2]) / global_volume );
      }

      // \left\langle \sum_{j=1}^{3}(\partial_{x_j} u_i)^2 \right\rangle for i=1 and i=3
      double avg_gsq_comp1 = (global_volume > 0.0) ? (global_gsq[0] / global_volume) : 0.0;
      double avg_gsq_comp3 = (global_volume > 0.0) ? (global_gsq[2] / global_volume) : 0.0;

      // \left\langle u_i^2 \right\rangle for i=1 and i=3
      double avg_u2_comp1  = (global_volume > 0.0) ? (global_u2[0] / global_volume) : 0.0;
      double avg_u2_comp3  = (global_volume > 0.0) ? (global_u2[2] / global_volume) : 0.0;

      // ---------------- Final ratios/powers ----------------
      // S = \frac{\langle \frac{1}{3}[(\partial_x u)^3 + (\partial_y v)^3 + (\partial_z w)^3] \rangle}
      //          { \langle \frac{1}{3}[(\partial_x u)^2 + (\partial_y v)^2 + (\partial_z w)^2] \rangle^{3/2} }
      skewness = (avg_long_d2 > 0.0) ? (avg_long_d3 / std::pow(avg_long_d2, 1.5)) : 0.0;

      // F = \frac{\langle \frac{1}{3}[(\partial_x u)^4 + (\partial_y v)^4 + (\partial_z w)^4] \rangle}
      //          { \langle \frac{1}{3}[(\partial_x u)^2 + (\partial_y v)^2 + (\partial_z w)^2] \rangle^{2} }
      flatness = (avg_long_d2 > 0.0) ? (avg_long_d4 / sq(avg_long_d2)) : 0.0;

      // \frac{D_3}{D_1} = \frac{\langle \sum_{j=1}^{3}(\partial_{x_j} u_3)^2 \rangle}{\langle \sum_{j=1}^{3}(\partial_{x_j} u_1)^2 \rangle}
      D3_over_D1 = (avg_gsq_comp1 > 0.0) ? (avg_gsq_comp3 / avg_gsq_comp1) : 0.0;

      // \frac{E_3}{E_1} = \frac{\langle u_3^2 \rangle}{\langle u_1^2 \rangle}
      E3_over_E1 = (avg_u2_comp1 > 0.0) ? (avg_u2_comp3 / avg_u2_comp1) : 0.0;
   }




  void ComputeKolmogorovAndTaylorMicroLength(const real_t *max_dissipation,real_t vol_avg_dissipation, real_t *kolmogorov_length, 
                                                                   real_t *avg_lambda, real_t *avg_kolmogorov_length, 
                                                                   real_t *kolmogorov_time_scale,
                                                                   real_t *avg_kolmogorov_time_scale, real_t ke)
  {
  
      // Compute the smallest Kolmogorov length scale using the maximum dissipation
      // eta = (nu^3/diss_max)^0.25
      *kolmogorov_length = pow((ctx.kinvis * ctx.kinvis * ctx.kinvis) / *max_dissipation, 0.25);

      // eta = (nu^3/<diss>)^0.25
      *avg_kolmogorov_length = pow((ctx.kinvis * ctx.kinvis * ctx.kinvis) / vol_avg_dissipation, 0.25);

      // Compute the smallest Taylor Micro scale using the maximum dissipation
      // lambda = sqrt(10*<ke>/<diss>), < > means volume average
      *avg_lambda = pow(10.0*ke*ctx.kinvis/vol_avg_dissipation, 0.50);

      // Kolmogorov time scale
      // Tau_eta = sqrt(\nu/diss_max)
      *kolmogorov_time_scale = pow(ctx.kinvis/ *max_dissipation,0.50);

      // Kolmogorov time scale
      // Tau_eta = sqrt(\nu/diss_max)
      *avg_kolmogorov_time_scale = pow(ctx.kinvis/vol_avg_dissipation,0.50);

  }

  // Computes \eta = 2*\nu*(\nabla u + trans(\nabla u))^2
  void ComputeAveragedDissipation(ParGridFunction &u, double *dissipation_ave, double *SijSij_ave, real_t *max_dissipation)
  {
     
    const ParFiniteElementSpace *vfes = u.ParFESpace();
  
     double local_diss = 0.0; 
     double local_vol  = 0.0;  
     double local_SijSij = 0.0; 
  
     Array<int> vdofs;
     Vector loc_data;                 
     DenseMatrix dshape, grad_hat, grad, S;              

     real_t local_max_dissipation = 0.0;
  
     const int ne = vfes->GetNE();
     for (int e = 0; e < ne; ++e)
     {
        vfes->GetElementVDofs(e, vdofs);
        u.GetSubVector(vdofs, loc_data);
  
        ElementTransformation *T = vfes->GetElementTransformation(e);
        const FiniteElement   *el = vfes->GetFE(e);
  
        const int elndofs = el->GetDof();
        const int vdim    = vfes->GetVDim();
        const int dim     = vfes->GetMesh()->Dimension();
  
        const int ir_order = 2*el->GetOrder() + 2;
        const IntegrationRule &ir = IntRules.Get(el->GetGeomType(), ir_order);
  
        dshape.SetSize(elndofs, dim);
        DenseMatrix loc_data_mat(loc_data.GetData(), elndofs, vdim);
  
        for (int i = 0; i < ir.GetNPoints(); ++i)
        {
           const IntegrationPoint &ip = ir.IntPoint(i);
           T->SetIntPoint(&ip);
  
           el->CalcDShape(ip, dshape);
  
           grad_hat.SetSize(vdim, dim);
           MultAtB(loc_data_mat, dshape, grad_hat);
  
           const DenseMatrix &Jinv = T->InverseJacobian();
  
           grad.SetSize(vdim, dim);
           Mult(grad_hat, Jinv, grad); 

           S.SetSize(dim,dim);
           for (int a = 0; a < dim; ++a)
           {
              for (int b = 0; b < dim; ++b)
              {
                 S(a,b) = 0.5*(grad(a,b) + grad(b,a));
              }
           }

           // Frobenius norm squared: S:S
           double S2 = 0.0;
           for (int a = 0; a < dim; ++a)
           {
              for (int b = 0; b < dim; ++b)
              {
                 S2 += S(a,b)*S(a,b);
              }
           }
         
           const double dV = ip.weight * T->Weight();
           local_diss   += 2.0*ctx.kinvis * S2 * dV;
           local_SijSij += S2*dV;
           local_vol    += dV;        

           // Point wise max
           local_max_dissipation = std::max(local_max_dissipation, static_cast<real_t>( 2.0 * ctx.kinvis * S2) );
        }
     }
  
     MPI_Comm comm = vfes->GetComm();

     real_t global_max_dissipation = 0.0;
     MPI_Allreduce(&local_max_dissipation, &global_max_dissipation,
                   1, MPITypeMap<real_t>::mpi_type, MPI_MAX, comm);

      if (max_dissipation) { *max_dissipation = global_max_dissipation; }


     double global_diss   = 0.0;
     double global_SijSij = 0.0;
     double global_vol    = 0.0;

     MPI_Allreduce(&local_diss,   &global_diss  , 1, MPI_DOUBLE, MPI_SUM, comm);
     MPI_Allreduce(&local_SijSij, &global_SijSij, 1, MPI_DOUBLE, MPI_SUM, comm);
     MPI_Allreduce(&local_vol,    &global_vol   , 1, MPI_DOUBLE, MPI_SUM, comm);
  
     *SijSij_ave      = global_SijSij/ global_vol;
     *dissipation_ave = global_diss / global_vol;
  
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

// ---- small helper: symmetric 3x3 eigenvalues (Jacobi) ----
static inline void SymmetricEigenvalues3x3(const mfem::DenseMatrix &A, double ev[3])
{
   double a00=A(0,0), a01=A(0,1), a02=A(0,2);
   double a11=A(1,1), a12=A(1,2), a22=A(2,2);

   auto sweep = [&](int p,int q, double &app,double &aqq,double &apq,
                    double &ar1,double &ar2)
   {
      if (std::abs(apq) <= 0.0) return;
      double tau = (aqq - app)/(2.0*apq);
      double t = (tau >= 0.0) ? 1.0/(tau + std::sqrt(1.0 + tau*tau))
                              : 1.0/(tau - std::sqrt(1.0 + tau*tau));
      double c = 1.0/std::sqrt(1.0 + t*t);
      double s = t*c;
      double appn = c*c*app - 2.0*s*c*apq + s*s*aqq;
      double aqqn = s*s*app + 2.0*s*c*apq + c*c*aqq;
      double apqn = 0.0;
      double ar1n = c*ar1 - s*ar2;
      double ar2n = s*ar1 + c*ar2;
      app=appn; aqq=aqqn; apq=apqn; ar1=ar1n; ar2=ar2n;
   };

   for (int it=0; it<8; ++it) // few sweeps suffice
   {
      sweep(0,1, a00,a11,a01, a02,a12);
      sweep(0,2, a00,a22,a02, a01,a12);
      sweep(1,2, a11,a22,a12, a01,a02);
   }
   ev[0]=a00; ev[1]=a11; ev[2]=a22;
   // sort ascending
   if (ev[0]>ev[1]) std::swap(ev[0],ev[1]);
   if (ev[1]>ev[2]) std::swap(ev[1],ev[2]);
   if (ev[0]>ev[1]) std::swap(ev[0],ev[1]);
}

// ---- main routine: nodal λ2 like your Q code ----
void ComputeLambda2Nodal(mfem::ParGridFunction &u, mfem::ParGridFunction &lambda2)
{
   using namespace mfem;
   FiniteElementSpace *v_fes = u.FESpace();
   FiniteElementSpace *s_fes = lambda2.FESpace();

   MFEM_VERIFY(v_fes->GetVDim() >= v_fes->GetMesh()->Dimension(),
               "Expect vdim >= dim for velocity.");

   // Count per vdof for averaging (like your Q routine)
   Array<int> zones_per_vdof(s_fes->GetVSize());
   zones_per_vdof = 0;
   lambda2 = 0.0;

   Array<int> v_dofs, s_dofs;
   Vector loc_vec;
   DenseMatrix dshape, grad_hat, grad; // grad: vdim x dim

   for (int e = 0; e < s_fes->GetNE(); ++e)
   {
      s_fes->GetElementVDofs(e, s_dofs);
      v_fes->GetElementVDofs(e, v_dofs);

      ElementTransformation *T = s_fes->GetElementTransformation(e);
      const FiniteElement *el_s = s_fes->GetFE(e);
      const FiniteElement *el_v = v_fes->GetFE(e);

      const int nd_s = el_s->GetDof();
      const int nd_v = el_v->GetDof();
      const int dim  = T->GetSpaceDim();
      const int vdim = v_fes->GetVDim();

      // local velocity dofs as (nd_v x vdim)
      u.GetSubVector(v_dofs, loc_vec);
      DenseMatrix Ue(loc_vec.GetData(), nd_v, vdim);

      // storage for element values written to scalar dofs
      Vector vals(nd_s);

      // gradient buffers
      dshape.SetSize(nd_v, dim);
      grad_hat.SetSize(vdim, dim);
      grad.SetSize(vdim, dim);

      // Loop interpolation points = element nodes of scalar space
      const IntegrationRule &nodes = el_s->GetNodes();
      for (int i = 0; i < nd_s; ++i)
      {
         const IntegrationPoint &ip = nodes.IntPoint(i);
         T->SetIntPoint(&ip);

         // Compute ∇u at ip
         el_v->CalcDShape(ip, dshape);                 // dφ/dξ
         const DenseMatrix &Jinv = T->InverseJacobian();
         DenseMatrix dshape_phys(dshape.Height(), dshape.Width());
         Mult(dshape, Jinv, dshape_phys);              // dφ/dx

         MultAtB(Ue, dshape_phys, grad);               // grad(u): vdim x dim

         // Build S and W (3x3 padded), then M = S^2 + W^2
         DenseMatrix S(3), W(3);
         S = 0.0; W = 0.0;
         const int n = std::min({3, vdim, dim});
         for (int a=0; a<n; ++a)
         {
            for (int b=0; b<n; ++b)
            {
               const double aab = grad(a,b);
               const double aba = grad(b,a);
               S(a,b) = 0.5*(aab + aba);
               W(a,b) = 0.5*(aab - aba);
            }
         }
         DenseMatrix SS(3), WW(3), M(3);
         Mult(S,S,SS);
         Mult(W,W,WW);
         Add(1.0, SS, 1.0, WW, M);

         // λ2 = middle eigenvalue of M
         double ev[3]; SymmetricEigenvalues3x3(M, ev);
         vals(i) = ev[1];
      }

      // Accumulate to scalar DOFs and count
      for (int j = 0; j < s_dofs.Size(); ++j)
      {
         int ldof = s_dofs[j];
         lambda2(ldof) += vals[j];
         zones_per_vdof[ldof] += 1;
      }
   }

   // Communicate & average over shared vdofs
   GroupCommunicator &gcomm = lambda2.ParFESpace()->GroupComm();
   gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
   gcomm.Bcast(zones_per_vdof);

   gcomm.Reduce<real_t>(lambda2.GetData(), GroupCommunicator::Sum);
   gcomm.Bcast<real_t>(lambda2.GetData());

   for (int i = 0; i < lambda2.Size(); ++i)
   {
      const int nz = zones_per_vdof[i];
      if (nz) { lambda2(i) /= nz; }
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

FiniteElementCollection *nd_fec = nullptr;
FiniteElementCollection *rt_fec = nullptr;
FiniteElementCollection *l2_fec = nullptr;
FiniteElementCollection *h1_fec = nullptr;

ParFiniteElementSpace *l2_fespace_scalar = nullptr; 
ParFiniteElementSpace *l2_fespace_vector = nullptr;

ParFiniteElementSpace *nd_fespace = nullptr;
ParFiniteElementSpace *rt_fespace = nullptr;

ParFiniteElementSpace *h1_fespace_scalar = nullptr;
ParFiniteElementSpace *h1_fespace_vector = nullptr;

class H1ToL2OrHdivProjector
{
private:
    ParFiniteElementSpace *test_fes;
    bool pa;
    
    // Store the mass matrix and solver for reuse
    std::unique_ptr<ParBilinearForm> mass_form;
    std::unique_ptr<CGSolver> cg_solver;
    std::unique_ptr<OperatorJacobiSmoother> jacobi_prec;
    std::unique_ptr<HypreParMatrix> mass_matrix;
    std::unique_ptr<HypreDiagScale> hypre_prec;
    std::unique_ptr<HyprePCG> hypre_solver;
    OperatorPtr mass_op;

    void SetUpOperator();
    
public:
    H1ToL2OrHdivProjector(ParFiniteElementSpace *test_space, 
                       bool partial_assembly); 

    void Apply(ParGridFunction &result, const ParGridFunction &u_h1) const;
};

class H1ToHdivOrHcurlProjector
{
private:
    ParFiniteElementSpace *test_fes;
    bool pa;
    
    // Store the mass matrix and solver for reuse
    std::unique_ptr<ParBilinearForm> mass_form;
    std::unique_ptr<CGSolver> cg_solver;
    std::unique_ptr<OperatorJacobiSmoother> jacobi_prec;
    std::unique_ptr<HypreParMatrix> mass_matrix;
    std::unique_ptr<HypreDiagScale> hypre_prec;
    std::unique_ptr<HyprePCG> hypre_solver;
    OperatorPtr mass_op;

    void SetUpOperator();
    
public:
   H1ToHdivOrHcurlProjector(ParFiniteElementSpace *test_space, 
                      bool partial_assembly); 

    void Apply(ParGridFunction &result, const ParGridFunction &u_h1) const;
};

class HcurlHdivProjector 
{
private:
    ParFiniteElementSpace *trial_fes;
    ParFiniteElementSpace *test_fes;
    bool pa;
    
    // Store the mass matrix and solver for reuse
    std::unique_ptr<ParBilinearForm> mass_form;
    std::unique_ptr<ParMixedBilinearForm> mixed_form;
    std::unique_ptr<CGSolver> cg_solver;
    std::unique_ptr<OperatorJacobiSmoother> jacobi_prec;
    std::unique_ptr<HypreParMatrix> mass_matrix;
    std::unique_ptr<HypreParMatrix> mixed_matrix;
    std::unique_ptr<HypreDiagScale> hypre_prec;
    std::unique_ptr<HyprePCG> hypre_solver;
    OperatorPtr mass_op;
    
    void SetUpOperator();
public:
    HcurlHdivProjector(ParFiniteElementSpace *trial_space, 
                       ParFiniteElementSpace *test_space, 
                       bool partial_assembly);

    void Apply(ParGridFunction &result, const ParGridFunction &gftrial) const;
};

class ComputeDivergenceHdivToL2
{
private:
    ParFiniteElementSpace *trial_fes;
    ParFiniteElementSpace *test_fes;
    bool pa;
    
    // Store the mass matrix and solver for reuse
    std::unique_ptr<ParBilinearForm> mass_form;
    std::unique_ptr<ParMixedBilinearForm> mixed_form;
    std::unique_ptr<CGSolver> cg_solver;
    std::unique_ptr<OperatorJacobiSmoother> jacobi_prec;
    std::unique_ptr<HypreParMatrix> mass_matrix;
    std::unique_ptr<HypreParMatrix> mixed_matrix;
    std::unique_ptr<HypreDiagScale> hypre_prec;
    std::unique_ptr<HyprePCG> hypre_solver;
    OperatorPtr mass_op;
    
    void SetUpOperator();
public:
    ComputeDivergenceHdivToL2(ParFiniteElementSpace *trial_space, 
                       ParFiniteElementSpace *test_space, 
                       bool partial_assembly);
    
    void Apply(ParGridFunction &result, const ParGridFunction &gftrial) const;
};

class ComputeCurlHcurlToHdiv
{
private:
    ParFiniteElementSpace *trial_fes;
    ParFiniteElementSpace *test_fes;
    bool pa;
    
    // Store the mass matrix and solver for reuse
    std::unique_ptr<ParBilinearForm> mass_form;
    std::unique_ptr<ParMixedBilinearForm> mixed_form;
    std::unique_ptr<CGSolver> cg_solver;
    std::unique_ptr<OperatorJacobiSmoother> jacobi_prec;
    std::unique_ptr<HypreParMatrix> mass_matrix;
    std::unique_ptr<HypreParMatrix> mixed_matrix;
    std::unique_ptr<HypreDiagScale> hypre_prec;
    std::unique_ptr<HyprePCG> hypre_solver;
    OperatorPtr mass_op;
    
    void SetUpOperator();

public:
    ComputeCurlHcurlToHdiv(ParFiniteElementSpace *trial_space, 
                       ParFiniteElementSpace *test_space, 
                       bool partial_assembly); 
    
    void Apply(ParGridFunction &result, const ParGridFunction &gftrial) const;
};

class ComputeGradientH1ScalarToHcurl
{
private:
    ParFiniteElementSpace *trial_fes;
    ParFiniteElementSpace *test_fes;
    bool pa;
    
    // Store the mass matrix and solver for reuse
    std::unique_ptr<ParBilinearForm> mass_form;
    std::unique_ptr<ParMixedBilinearForm> mixed_form;
    std::unique_ptr<CGSolver> cg_solver;
    std::unique_ptr<OperatorJacobiSmoother> jacobi_prec;
    std::unique_ptr<HypreParMatrix> mass_matrix;
    std::unique_ptr<HypreParMatrix> mixed_matrix;
    std::unique_ptr<HypreDiagScale> hypre_prec;
    std::unique_ptr<HyprePCG> hypre_solver;
    OperatorPtr mass_op;
        
    void SetUpOperator();
    
public:
    ComputeGradientH1ScalarToHcurl(ParFiniteElementSpace *trial_space, 
                       ParFiniteElementSpace *test_space, 
                       bool partial_assembly);
    
    void Apply(ParGridFunction &result, const ParGridFunction &gftrial) const;
};

struct ProjectorOps {

   H1ToHdivOrHcurlProjector projectorH1ToHdiv;
   H1ToL2OrHdivProjector projectorL2ToH1Scalar;
   ComputeGradientH1ScalarToHcurl projectorComputeGradientH1ScalarToHcurl;
   H1ToL2OrHdivProjector projectorH1ToL2;
   H1ToHdivOrHcurlProjector projectorH1ToHcurl;
   HcurlHdivProjector projectorHcurlToHdiv;
   HcurlHdivProjector projectorHdivToHcurl;
   ComputeDivergenceHdivToL2 projectorDivHdivToL2;
   ComputeCurlHcurlToHdiv projectorCurlHcurlToHdiv;
   H1ToL2OrHdivProjector projectorHdivToL2;
   H1ToL2OrHdivProjector projectorL2ToH1;

   ProjectorOps(ParFiniteElementSpace *h1_fespace_vector,
                ParFiniteElementSpace *h1_fespace_scalar,
                ParFiniteElementSpace *nd_fespace,
                ParFiniteElementSpace *rt_fespace,
                ParFiniteElementSpace *l2_fespace_vector, 
                ParFiniteElementSpace *l2_fespace_scalar, 
                bool pa)

   : projectorH1ToHdiv(rt_fespace, pa),
     projectorL2ToH1Scalar(h1_fespace_scalar, pa),
     projectorComputeGradientH1ScalarToHcurl(h1_fespace_scalar, nd_fespace, pa),
     projectorH1ToL2(l2_fespace_vector, pa),
     projectorH1ToHcurl(nd_fespace, pa),
     projectorHcurlToHdiv(rt_fespace, nd_fespace, pa),
     projectorHdivToHcurl(nd_fespace, rt_fespace, pa),
     projectorDivHdivToL2(rt_fespace,l2_fespace_scalar, pa),
     projectorCurlHcurlToHdiv(nd_fespace, rt_fespace, pa),
     projectorHdivToL2(l2_fespace_vector, pa),
     projectorL2ToH1(h1_fespace_vector, pa)
     {

     }
};

void solve_scalar_potential( const ProjectorOps& ops,
                             const ParGridFunction &u_h1,
                             ParGridFunction &grad_phi_h1,
                             ParMesh *pmesh, bool pa);

void solve_vector_potential( const ProjectorOps& ops,
                             const ParGridFunction &u_h1,
                             ParGridFunction &curl_Ah_h1,
                             ParMesh *pmesh, bool pa);


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
   args.AddOption(&ctx.reynum, "-Re", "--Reynolds-number", "Reynolds Number.");
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
   args.AddOption(&ctx.time_based_output,
                  "-time-out",
                  "--time-based-output", 
                  "-no-time-out",
                  "--no-time-based-output",
                  "Enable time-based output instead of cycle-based.");
   args.AddOption(&ctx.num_snapshots,
                  "-nsnap",
                  "--num-snapshots",
                  "Number of evenly-spaced snapshots to output.");   
   args.AddOption(&ctx.u0_based_on_mach, "-u0_from_mach", "--U0-from-mach", "-no-u0_from_mach",
                  "--no-u0_from_mach",
                  "Compute u0 based on a mach number?");
   args.AddOption(&ctx.time_snapshot_dump, "-time_snapshot", "--Time-Snapshot", 
      "Dump single data snap shot at this time.");
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

   ctx.num_snapshots += 1;
   if (ctx.time_based_output)
   {
      ctx.snapshot_interval = ctx.t_final / (ctx.num_snapshots - 1);
      ctx.snapshot_times.resize(ctx.num_snapshots);

      for (int i = 0; i < ctx.num_snapshots; i++)
      {
         ctx.snapshot_times[i] = i * ctx.snapshot_interval;
      }

      if (Mpi::Root())
      {
         std::cout << "Time-based output enabled:" << std::endl;
         std::cout << "  Number of snapshots: " << ctx.num_snapshots << std::endl;
         std::cout << "  Time interval: " << ctx.snapshot_interval << std::endl;
         std::cout << "  Target times: ";
         for (auto t : ctx.snapshot_times) std::cout << t << " ";
         std::cout << std::endl;
      }
   }

   if (ctx.time_based_output && ctx.snapshot_times.empty()) 
   {
       ctx.snapshot_interval = ctx.t_final / (ctx.num_snapshots - 1);
       ctx.snapshot_times.resize(ctx.num_snapshots);
       for (int i = 0; i < ctx.num_snapshots; i++) 
       {
           ctx.snapshot_times[i] = i * ctx.snapshot_interval;
       }
   }

   // This is only for setting up the initial velocity to compare with 
   // compressible codes!!
   // Can adjust this for different Mach number comparisions
   // Specify manually on purpose
   if (ctx.u0_based_on_mach)
   {
      double gamma = 5.0/3.0;
      double p0    = 100.0;
      double rho0  = 1.0;
      ctx.u0 = ctx.Mach0*sqrt(gamma*p0/rho0);
      if (Mpi::Root())
      {
         std::cout << "Mach0: " << ctx.Mach0 << std::endl;  
      }
   }

   // K0 = 1.0/L0
   double L0 = (ctx.problem1) ? 1.0 : 1.0/(2.0*M_PI);

   // t*=u0/L*t
   double t_star_final = L0/ctx.u0*ctx.t_final;

   // t = u0/L*t* (we are solving the for rescaled time based on velocity)
   double dt_scale = ctx.u0/L0;

   // Update kinematic viscosity
   ctx.kinvis = ctx.u0 * L0 / (ctx.reynum);

   // Update the time scales accordingly
   ctx.dt /=dt_scale;
   ctx.t_final = t_star_final*dt_scale;

   if (Mpi::Root())
   {
      double Re_eff = ctx.u0*L0 / (ctx.kinvis);
      std::cout << "Configured L0 =" << L0 
                << ", u0 = " << ctx.u0
                << ", nu=" << ctx.kinvis
                << ", dt=" << ctx.dt
                << ", t_final=" << ctx.t_final
                << ", effective Re=" << Re_eff << std::endl;
   }

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
            std::cout << "Restart files found. Continuing from checkpoint at time t = " 
                      << t << ", step = " << step << std::endl;
         }

         // Store the initial step number at restart
         global_cycle = step + 1; // Acount for the cycle shift when restarting

         // Reset step counter for the new run segment
         step = 0;

         if (ctx.time_based_output)
         {
             ctx.snapshot_index = 0;
             // Find the first snapshot time that's GREATER than current time
             // (we've already passed any that are <= current time)
             while (ctx.snapshot_index < ctx.num_snapshots && 
                    t >= ctx.snapshot_times[ctx.snapshot_index] - ctx.dt * 0.01)
             {
                 ctx.snapshot_index++;
             }

             if (Mpi::Root())
             {
                 std::cout << "Restart: starting snapshot index = " << ctx.snapshot_index;
                 if (ctx.snapshot_index < ctx.num_snapshots)
                 {
                     std::cout << " (next target time = " << ctx.snapshot_times[ctx.snapshot_index] << ")";
                 }
                 std::cout << std::endl;
             }
         }
      }
      else
      {
         if (Mpi::Root())
         {
            std::cout << "Restart files not found. Starting from initial conditions." << std::endl;
         }
         global_cycle = 0;
      }
   }
   else
   {
      global_cycle = 0;
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

      // VectorFunctionCoefficient translate_set_mesh(mesh->Dimension(), [&](const Vector &x_in, Vector &x_out){

      //    x_out[0] = x_in[0]*0.5; // Translate x-coordinate
      //    x_out[1] = x_in[1]*x_in[1]*2.0; // Translate y-coordinate
      //    if (mesh->Dimension() == 3)
      //    {
      //       x_out[2] = x_in[2]*0.75; // Translate z-coordinate
      //    }
      // });

      // // Smooth wave-like perturbations
      // VectorFunctionCoefficient wave_perturb(mesh->Dimension(), [&](const Vector &x_in, Vector &x_out){
      //     double amplitude = 0.1; // Small amplitude
      //     double frequency = 4.0;  // Number of waves across domain
      //     
      //     x_out[0] = x_in[0] + amplitude * sin(frequency * M_PI * x_in[1]);
      //     x_out[1] = x_in[1] + amplitude * sin(frequency * M_PI * x_in[0]);
      //     if (mesh->Dimension() == 3) {
      //         x_out[2] = x_in[2] + amplitude * sin(frequency * M_PI * (x_in[0] + x_in[1]));
      //     }
      // });

      // // // Apply translation to the mesh
      // mesh->Transform(wave_perturb);

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

      // // Define a translation function for the mesh nodes
      // VectorFunctionCoefficient scale(mesh->Dimension(), [&](const Vector &x_in, Vector &x_out)
      //                                     {
      //    double scale = 1.0;

      //    x_out[0] = x_in[0]/scale ; // Translate x-coordinate
      //    x_out[1] = x_in[1]/scale ; // Translate y-coordinate
      //    if (mesh->Dimension() == 3){
      //      x_out[2] = x_in[2]/scale; // Translate z-coordinate
      //    } });

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

   int dim = pmesh->Dimension();

   ParFiniteElementSpace *velocity_fespace = u_gf->ParFESpace();
   ParFiniteElementSpace *pressure_fespace = p_gf->ParFESpace();

   // Initialize w_gf and q_gf using the finite element spaces
   ParGridFunction w_gf(velocity_fespace);
   ParGridFunction q_gf(pressure_fespace);
   ParGridFunction lambda2_gf(pressure_fespace);
   // ParGridFunction ke_gf(pressure_fespace);

   flowsolver->ComputeCurl3D(*u_gf, w_gf);
   ComputeQCriterion(*u_gf, q_gf);
   ComputeLambda2Nodal(*u_gf, lambda2_gf);

   // ComputeDivergence3D(*u_gf, divu_gf);
   // ComputeVorticalPart(flowsolver, *u_gf, w_gf, u_vort);

   /*
   nd_fec = new ND_FECollection(ctx.order, dim);
   rt_fec = new RT_FECollection(ctx.order-1, dim); // H(div)
   l2_fec = new L2_FECollection(ctx.order-1, dim);
   h1_fec = new H1_FECollection(ctx.order, dim);

   l2_fespace_scalar = new ParFiniteElementSpace(pmesh, l2_fec);
   l2_fespace_vector = new ParFiniteElementSpace(pmesh, l2_fec, dim);

   nd_fespace = new ParFiniteElementSpace(pmesh, nd_fec);
   rt_fespace = new ParFiniteElementSpace(pmesh, rt_fec);

   h1_fespace_scalar = new ParFiniteElementSpace(pmesh, h1_fec);
   h1_fespace_vector = new ParFiniteElementSpace(pmesh, h1_fec, dim);

   ProjectorOps ops(h1_fespace_vector,
                    h1_fespace_scalar,
                    nd_fespace,
                    rt_fespace,
                    l2_fespace_vector, 
                    l2_fespace_scalar, 
                    ctx.pa);



   // \nabla \times Ah in H(div)
   // 1. Define velocity spaces
   ParGridFunction curl_Ah_h1(h1_fespace_vector);
   solve_vector_potential(ops, *u_gf, curl_Ah_h1, pmesh, ctx.pa);

   ParGridFunction curl_Ah_l2(l2_fespace_vector);
   ops.projectorH1ToL2.Apply(curl_Ah_l2, curl_Ah_h1);

   // Solve for scalar potential

   // 4. Solve Poisson problem \nabla^2 \phi = div(u)
   ParGridFunction grad_phi_h1(h1_fespace_vector);
   solve_scalar_potential(ops, *u_gf, grad_phi_h1, pmesh, ctx.pa);

   ParGridFunction grad_phi_l2(l2_fespace_vector);
   ops.projectorH1ToL2.Apply(grad_phi_l2, grad_phi_h1);

   // Sanity Checks
   ParGridFunction u_l2(l2_fespace_vector);
   ops.projectorH1ToL2.Apply(u_l2, *u_gf);

   ParGridFunction vel_error(l2_fespace_vector);
   vel_error = grad_phi_l2;
   vel_error += curl_Ah_l2;
   vel_error -= u_l2;

   // Subtract grad phi from u -- do we get a better curl Ah field?
   curl_Ah_l2 = u_l2;
   curl_Ah_l2 -= grad_phi_l2;

   ParGridFunction curl_Ah_h1_from_grad_phi(h1_fespace_vector);
   ops.projectorL2ToH1.Apply(curl_Ah_h1_from_grad_phi, curl_Ah_l2);

   // 15. Compute and print the L^2 norm of the error.
   {

      ConstantCoefficient zero(0.0);

      Vector zero_v(dim);
      zero_v = 0.0;
      VectorConstantCoefficient zero_vec(zero_v);

      // double curl_grad_phi_computed_error_project = curl_grad_phi_hdiv.ComputeL2Error(zero_vec);
      // double div_curl_A_error_l2 = div_curl_Ah_l2.ComputeL2Error(zero);
      double total_vel_error = vel_error.ComputeL2Error(zero);
   

      if (myid == 0)
      {
         // cout << "curl(grad phi) project L2 error (should be ~0): " << curl_grad_phi_computed_error_project << endl;
         std::cout << "vel error from reconstruction: " << total_vel_error << std::endl;
      }
   }

   */

   QuantitiesOfInterest kin_energy(pmesh);
   // jreal_t ke = kin_energy.ComputeKineticEnergy(*u_gf, ke_gf);
   real_t ke = kin_energy.ComputeKineticEnergy(*u_gf);

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
      dc->RegisterField("lambda2", &lambda2_gf);
      // dc->RegisterField("curl_Ah", &curl_Ah_h1);
      // dc->RegisterField("curl_Ah_from_grad_phi", &curl_Ah_h1_from_grad_phi);
      // dc->RegisterField("grad_phi", &grad_phi_h1);
      // dc->RegisterField("divu", &divu_gf);
      // dc->RegisterField("ke", &ke_gf);
      // dc->RegisterField("u_vort", &u_vort);
      dc->Save();
   }


   /*
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
           cdc->Save();
         }
#else
         MFEM_ABORT("Must build with MFEM_USE_CONDUIT=YES for binary output.");
#endif

   }*/

   real_t u_inf_loc = u_gf->Normlinf();
   real_t p_inf_loc = p_gf->Normlinf();

   real_t u_inf = GlobalLpNorm(infinity(), u_inf_loc, MPI_COMM_WORLD);
   real_t p_inf = GlobalLpNorm(infinity(), p_inf_loc, MPI_COMM_WORLD);

   // real_t ke = kin_energy.ComputeKineticEnergy(*u_gf);
   real_t vel_curl_ke = kin_energy.ComputeInertialRangeEnergy(*u_gf);
   real_t enstrophy = kin_energy.ComputeEnstrophy(*u_gf);

   real_t kolmLenScl = 0.0;
   real_t avg_kolmLenScl = 0.0;
   real_t avg_lambda = 0.0;
   real_t kolmTimeScl = 0.0;
   real_t avg_kolmTimeScl = 0.0;
   real_t hmin_eta = 0.0;
   real_t kmax_eta = 0.0;
   real_t u_rms =  pow(2.0/3.0*ke,0.5);

   real_t kmax = 0.0;
   real_t hmin = 0.0;

   real_t avg_diss = 0.0;
   real_t avg_SijSij = 0.0;
   real_t max_dissipation = 0.0;

   real_t S, F, D31, E31;
   kin_energy.ComputeSkewFlat_D3D1_E3E1(*u_gf, S, F, D31, E31);

   kin_energy.ComputeAveragedDissipation(*u_gf, &avg_diss, &avg_SijSij,&max_dissipation);
   kin_energy.ComputeKolmogorovAndTaylorMicroLength(&max_dissipation, avg_diss, &kolmLenScl, &avg_lambda, &avg_kolmLenScl, &kolmTimeScl, &avg_kolmTimeScl, ke);
   kin_energy.ComputeGridPtsRequirementsTurb(*u_gf, kolmLenScl, &hmin_eta, &kmax_eta, &kmax, &hmin);

   // double lambdas[3];
   double avg_lambda_iso = kin_energy.ComputeTaylorMicroscale(*u_gf, nullptr);

   // if (Mpi::Root())
   // {
   //    std::printf("Taylor microscale components: "
   //                "lambda_x=%.6e  lambda_y=%.6e  lambda_z=%.6e  |  avg=%.6e\n",
   //                lambdas[0], lambdas[1], lambdas[2], lambda_avg);
   // }

   // Pope definetion of grid resolution
   real_t avg_hmin_eta = hmin/avg_kolmLenScl;
   real_t avg_kmax_eta = kmax*avg_kolmLenScl;

   // This computes how resolved our grid is.
   // See Aspen 2008 Implicit LES Anaylsis
   real_t PI_nu = pow(avg_diss,0.5)/(avg_kolmLenScl*pow(vel_curl_ke,0.75));
   real_t PI_nu_min = pow(max_dissipation,0.5)/(kolmLenScl*pow(vel_curl_ke,0.75));

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
   std::string fname_turb_continued = std::string("tgv_out_turb_continued_") 
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
   FILE *f_turb_continued = NULL;
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
      f_turb_continued = fopen(fname_turb_continued.c_str(), file_mode);
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

      if (!f_turb_continued)
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
          fprintf(f, "==============================================================================================================================\n");
          fprintf(f, "        time                      cycle                 kinetic energy               enstrophy               cfl\n");

          // Write the initial data point
           fprintf(f, "%20.16e     %20.16e     %20.16e     %20.16e      %20.16e\n", t, static_cast<real_t>(global_cycle + step), ke, enstrophy, cfl);

          // Write header only if not restarting
          fprintf(f_turb, "3D Taylor Green Vortex (turbulence metrics)\n");
          fprintf(f_turb, "Reynolds Number = %d\n", static_cast<int>(ctx.reynum));
          fprintf(f_turb, "order = %d\n", ctx.order);
          fprintf(f_turb, "grid = %d x %d x %d\n", nel1d, nel1d, nel1d);
          fprintf(f_turb, "dofs per component = %d\n", ngridpts);
          fprintf(f_turb, "===============================================================================");
          fprintf(f_turb, "===============================================================================");
          fprintf(f_turb, "===============================================================================");
          fprintf(f_turb, "===================================================================================\n");
          fprintf(f_turb, "        time                        cycle                Max Dissipation       Average Dissipation     Min Kolmogorov Length Scale    Taylor Length Scale");
          fprintf(f_turb, "        Taylor Length Scale (aniso)      Average Kolm Len    Kolmogorov Time Scale       Average Kolm Time Scale       Taylor Re (Avg)");
          fprintf(f_turb, "               u_rms    \n");

          // Write the initial data point
           fprintf(f_turb, "%20.16e     %20.16e     %20.16e     %20.16e     %20.16e     %20.16e     %20.16e    %20.16e     %20.16e      %20.16e      %20.16e      %20.16e\n",
                       t, static_cast<real_t>(global_cycle + step), max_dissipation, avg_diss, kolmLenScl, 
                       avg_lambda, avg_lambda_iso, avg_kolmLenScl, kolmTimeScl, avg_kolmTimeScl,
                       Re_taylor, u_rms);

          // Write header only if not restarting
          fprintf(f_turb_continued, "3D Taylor Green Vortex (turbulence metrics)\n");
          fprintf(f_turb_continued, "Reynolds Number = %d\n", static_cast<int>(ctx.reynum));
          fprintf(f_turb_continued, "order = %d\n", ctx.order);
          fprintf(f_turb_continued, "grid = %d x %d x %d\n", nel1d, nel1d, nel1d);
          fprintf(f_turb_continued, "dofs per component = %d\n", ngridpts);
          fprintf(f_turb_continued, "===================================================================================");
          fprintf(f_turb_continued, "===================================================================================\n");
          fprintf(f_turb_continued, "        time                        cycle                     avg_SijSij                 Skewness");
          fprintf(f_turb_continued, "            Flatness                    D31                       E31\n");

          // Write the initial data point
           fprintf(f_turb_continued, "%20.16e     %20.16e      %20.16e     %20.16e     %20.16e     %20.16e     %20.16e\n",
                       t, static_cast<real_t>(global_cycle + step), avg_SijSij, S, F, D31, E31); 

          // Write header only if not restarting
          fprintf(f_turb_grid, "3D Taylor Green Vortex (turbulence grid metrics)\n");
          fprintf(f_turb_grid, "Reynolds Number = %d\n", static_cast<int>(ctx.reynum));
          fprintf(f_turb_grid, "order = %d\n", ctx.order);
          fprintf(f_turb_grid, "grid = %d x %d x %d\n", nel1d, nel1d, nel1d);
          fprintf(f_turb_grid, "dofs per component = %d\n", ngridpts);
          fprintf(f_turb_grid, "===============================================================================");
          fprintf(f_turb_grid, "==============================================================================");
          fprintf(f_turb_grid, "==============================================================================\n");
          fprintf(f_turb_grid, "        time                       cycle                  K_max*eta (>1.5)              hmin/eta (<2.1)");
          fprintf(f_turb_grid, "        Average PI_NU              Min PI_NU        ");
          fprintf(f_turb_grid, "        K_max*eta(Avg)             hmin/eta(Avg)    \n");

          // Write the initial data point
           fprintf(f_turb_grid, "%20.16e     %20.16e     %20.16e     %20.16e     %20.16e    %20.16e    %20.16e    %20.16e\n",
                       t, static_cast<real_t>(global_cycle + step), kmax_eta, hmin_eta, PI_nu, PI_nu_min, avg_kmax_eta, avg_hmin_eta);
      } 

      fflush(f);
      fflush(f_turb);
      fflush(f_turb_continued);
      fflush(f_turb_grid);
      fflush(stdout);
   }

   real_t dt = ctx.dt;
   real_t t_final = ctx.t_final;
   bool last_step = false;

   for (; !last_step; ++step)
   {
      bool should_dump_data = false;
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
         should_dump_data=true;
      }

      // Adjust alpha for restart
      real_t effective_alpha = ctx.alpha;  // Default to the original alpha
      if (ctx.filter && ctx.restart && restart_files_found && step <= 500)  // Ramp over first 10 steps
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

      if (ctx.time_based_output)
      {
         // Check if we should output based on current time and snapshot schedule
         if (ctx.snapshot_index < ctx.num_snapshots)
         {
            // Check if current time is within a small tolerance of the target snapshot time
            if (t >= ctx.snapshot_times[ctx.snapshot_index] - ctx.dt * 0.01)
            {
               should_dump_data = true;
               if (Mpi::Root())
               {
                  std::cout << "Time-based output triggered at t = " << t 
                            << ", target time = " << ctx.snapshot_times[ctx.snapshot_index] 
                            << std::endl;
               }
            }
         }
         // Also output on last step
         if (last_step && !should_dump_data)
         {
            should_dump_data = true;
         }
      }
      else
      {
         should_dump_data = ((global_cycle + step) % ctx.data_dump_cycle == 0) || last_step;
      }
            
      if (!snapshot_dumped && t >= ctx.time_snapshot_dump - ctx.dt * 0.01 )
      {
        if (Mpi::Root())
        {
           std::cout << "Dumping single data snap shot = " << t 
                     << ", target time = " << ctx.time_snapshot_dump 
                     << std::endl;
        }
         should_dump_data = true;
         snapshot_dumped = true;
      }

      // Skip output on the very first step after restart to avoid duplicates
      if (should_dump_data)
      {
         // If restarting, skip the first saved checkpoint
         if (!(ctx.restart && step == 0 && restart_files_found))
         {
            ComputeQCriterion(*u_gf, q_gf);
            ComputeLambda2Nodal(*u_gf, lambda2_gf);
            flowsolver->ComputeCurl3D(*u_gf, w_gf);

            // For all output types, use this consistent output_cycle calculation:
            int output_cycle = global_cycle + step;

            if (ctx.paraview)
            {
               pvdc->SetCycle(output_cycle);
               pvdc->SetTime(t);
               pvdc->Save();
               if (Mpi::Root())
               {
                  std::cout << "\nParaview file saved." << std::endl;
               }
            }

            if (ctx.visit)
            {
               dc->SetCycle(output_cycle);
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

            /*
            if (ctx.conduit)
            {
               cdc->SetCycle(output_cycle);
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

            }*/
                
            if (ctx.time_based_output && ctx.snapshot_index < ctx.num_snapshots)
            {
               ctx.snapshot_index++;
            }
         }
      }


      bool should_dump_element_centers = false;
      if (ctx.time_based_output)
      {
         should_dump_element_centers = should_dump_data;
      }
      else
      {
         should_dump_element_centers = ((global_cycle + step) % ctx.element_center_cycle == 0) || last_step;
      }
   
      if (should_dump_element_centers)
      {
         // If restarting, skip the first saved checkpoint
         if (!(ctx.restart && step == 0 && restart_files_found))
         {
            SamplePoints(u_gf, pmesh, global_cycle + step, t, "Velocity", &ctx);
            SamplePointsAtDoFs(u_gf, pmesh, global_cycle + step, t, "Velocity", &ctx);
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

      // ke = kin_energy.ComputeKineticEnergy(*u_gf, ke_gf);
      ke = kin_energy.ComputeKineticEnergy(*u_gf);
      vel_curl_ke = kin_energy.ComputeInertialRangeEnergy(*u_gf);
      enstrophy = kin_energy.ComputeEnstrophy(*u_gf);

      kin_energy.ComputeAveragedDissipation(*u_gf, &avg_diss, &avg_SijSij, &max_dissipation);
      kin_energy.ComputeKolmogorovAndTaylorMicroLength(&max_dissipation, avg_diss, &kolmLenScl, &avg_lambda, &avg_kolmLenScl, &kolmTimeScl, &avg_kolmTimeScl, ke);
      kin_energy.ComputeGridPtsRequirementsTurb(*u_gf, kolmLenScl, &hmin_eta, &kmax_eta, &kmax, &hmin);
      avg_lambda_iso = kin_energy.ComputeTaylorMicroscale(*u_gf, nullptr);
      u_rms =  pow(2.0/3.0*ke, 0.5);
      // Re_taylor = u_rms*avg_lambda/ctx.kinvis;
      Re_taylor = u_rms*avg_lambda_iso/ctx.kinvis;

      PI_nu = pow(avg_diss,0.5)/(avg_kolmLenScl*pow(vel_curl_ke,0.75));
      PI_nu_min = pow(max_dissipation,0.5)/(kolmLenScl*pow(vel_curl_ke,0.75));

      avg_hmin_eta = hmin/avg_kolmLenScl;
      avg_kmax_eta = kmax*avg_kolmLenScl;

      kin_energy.ComputeSkewFlat_D3D1_E3E1(*u_gf, S, F, D31, E31);

      // if (Mpi::Root())
      // {
      //    std::printf("Taylor microscale components: "
      //                "lambda_x=%.6e  lambda_y=%.6e  lambda_z=%.6e  |  avg=%.6e\n",
      //                lambdas[0], lambdas[1], lambdas[2], lambda_avg);
      // }


      if (Mpi::Root())
      {
         // If restarting, skip the first saved checkpoint
         if (!(ctx.restart && step == 0 && restart_files_found))
         {
           printf("%.5E %.5E %.5E %.5E %.5E %.5E %.5E\n", t, ctx.dt, u_inf, p_inf, ke, enstrophy, cfl);
           fprintf(f, "%20.16e     %20.16e     %20.16e     %20.16e      %20.16e\n", t, static_cast<real_t>(step + global_cycle), ke, enstrophy, cfl);
           fprintf(f_turb, "%20.16e     %20.16e     %20.16e     %20.16e     %20.16e     %20.16e     %20.16e    %20.16e     %20.16e      %20.16e      %20.16e      %20.16e\n",
                       t, static_cast<real_t>(global_cycle + step), max_dissipation, avg_diss, kolmLenScl, 
                       avg_lambda, avg_lambda_iso, avg_kolmLenScl, kolmTimeScl, avg_kolmTimeScl,
                       Re_taylor, u_rms);
           fprintf(f_turb_continued, "%20.16e     %20.16e      %20.16e     %20.16e     %20.16e     %20.16e     %20.16e\n",
                       t, static_cast<real_t>(global_cycle + step), avg_SijSij, S, F, D31, E31); 
           fprintf(f_turb_grid, "%20.16e     %20.16e     %20.16e     %20.16e     %20.16e    %20.16e    %20.16e    %20.16e\n",
                       t, static_cast<real_t>(global_cycle + step), kmax_eta, hmin_eta, PI_nu, PI_nu_min, avg_kmax_eta, avg_hmin_eta);
           fflush(f);
           fflush(f_turb);
           fflush(f_turb_continued);
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

   // delete nd_fespace;
   // delete rt_fespace;
   // delete h1_fespace_vector;
   // delete h1_fespace_scalar;
   // delete l2_fespace_vector;
   // delete l2_fespace_scalar;
   // delete nd_fec;
   // delete rt_fec;
   // delete h1_fec;
   // delete l2_fec;

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

void solve_vector_potential( const ProjectorOps& ops,
                             const ParGridFunction &u_h1,
                             ParGridFunction &curl_Ah_h1,
                             ParMesh *pmesh, bool pa)
{
   int myid = Mpi::WorldRank();

   int dim = pmesh->Dimension();

   // 2. Peform the needed projections
   // Project u in H1 to Hcurl
   ParGridFunction u_hcurl(nd_fespace);

   ops.projectorH1ToHcurl.Apply(u_hcurl, u_h1);

   // Compute the curl of u
   ParGridFunction curl_u(rt_fespace);
   curl_u = 0.0;

   // We can also solve a linear system to move form one space to another
   ops.projectorCurlHcurlToHdiv.Apply(curl_u, u_hcurl);

   // The test space which is being projected to is
   // H(curl) from the trial space H(div)
   // Note that the trial space needs to not be empyt ie.
   // be projected to

   ParGridFunction curl_u_hcurl(nd_fespace);

   ops.projectorHcurlToHdiv.Apply(curl_u_hcurl, curl_u);

   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the boundary attributes from the mesh as essential
   //    (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   Array<int> ess_bdr;

   if (pmesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmesh->bdr_attributes.Max());
      ess_bdr = 0;
   }

   // 9. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (f,phi_i) where f is given by the function f_exact and phi_i are the
   //    basis functions in the finite element fespace.
   VectorGridFunctionCoefficient f(&curl_u_hcurl);
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
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   OperatorPtr A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   // 13. Solve the system AX=B using PCG with an AMS preconditioner.
   if (pa)
   {
#ifdef MFEM_USE_AMGX
      MatrixFreeAMS ams(*a, *A, *nd_fespace, muinv, sigma, NULL, ess_bdr, useAmgX);
#else
      MatrixFreeAMS ams(*a, *A, *nd_fespace, muinv, sigma, NULL, ess_bdr);

      // std::unique_ptr<Solver> solver;
      // solver.reset(new LORSolver<HypreBoomerAMG>(*a, ess_tdof_list));
      // solver.reset(new OperatorJacobiSmoother(*a, ess_tdof_list));
#endif
      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(1000);
      cg.SetPrintLevel(0);
      cg.SetOperator(*A);
      cg.SetPreconditioner(ams);
      // cg.SetPreconditioner(*solver);
      cg.Mult(B, X);
   }
   else
   {
      ParFiniteElementSpace *prec_ndfespace =
         (a->StaticCondensationIsEnabled() ? a->SCParFESpace() : nd_fespace);
      HypreAMS ams(*A.As<HypreParMatrix>(), prec_ndfespace);
      // ams.SetSingularProblem();
      HyprePCG pcg(*A.As<HypreParMatrix>());
      pcg.SetTol(1e-12);
      pcg.SetMaxIter(500);
      pcg.SetPrintLevel(0);
      pcg.SetPreconditioner(ams);
      pcg.Mult(B, X);
   }

   // 14. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a->RecoverFEMSolution(X, *b, x);

   // 3. Solve for the vector potential
   ParGridFunction Ah(nd_fespace);
   Ah = x;

   // Compute the curl of the vector potential which is the divergence 
   // free part of the velocity field

   // Compute curl of Ah in H(div)
   ParGridFunction curl_Ah(rt_fespace);
   ops.projectorCurlHcurlToHdiv.Apply(curl_Ah, Ah);

   // 4. Verification part to make sure field is divergence free

   // Verification of divergence free field
   // The test space which is being projected to is
   // H(div) from the trial space H(curl)
   ParGridFunction Ah_hdiv(rt_fespace);
   ops.projectorHdivToHcurl.Apply(Ah_hdiv, Ah);

   // Set \nabla \cdot (\nabla \times Ah) to be in L2
   // Compute \nabla \cdot (\nabla \times Ah) in H(div)
   ParGridFunction div_curl_Ah(l2_fespace_scalar);
   ParGridFunction div_Ah(l2_fespace_scalar);

   ops.projectorDivHdivToL2.Apply(div_curl_Ah, curl_Ah);
   ops.projectorDivHdivToL2.Apply(div_Ah, Ah_hdiv);

   // 5. Move curl of vector potential to H1 for visualization for later
   ParGridFunction curl_Ah_l2(l2_fespace_vector);
   ops.projectorHdivToL2.Apply(curl_Ah_l2, curl_Ah);

   // Project from L2 to H1 by solving linear system
   ops.projectorL2ToH1.Apply(curl_Ah_h1, curl_Ah_l2);

   {
      ConstantCoefficient zero(0.0);
      Vector zero_v(dim);
      zero_v = 0.0;
      VectorConstantCoefficient zero_vec(zero_v);

      double div_curl_A_error = div_curl_Ah.ComputeL2Error(zero);
      double div_A_error = div_Ah.ComputeL2Error(zero);
   

      if (myid == 0)
      {
         std::cout << "div(curl A) L2 norm (should be ~0): " << div_curl_A_error << std::endl;
         std::cout << "div(A) L2 norm (should be ~0): " << div_A_error << std::endl;
      }
   }


   // 18. Free the used memory.
   delete a;
   delete sigma;
   delete muinv;
   delete b;

}

void solve_scalar_potential( const ProjectorOps& ops,
                             const ParGridFunction &u_h1,
                             ParGridFunction &grad_phi_h1,
                             ParMesh *pmesh, bool pa)
{
   int myid = Mpi::WorldRank();

   // 1. Project u from H1 to Hdiv
   ParGridFunction u_hdiv(rt_fespace);
   ops.projectorH1ToHdiv.Apply(u_hdiv, u_h1);
   
   // 2. Divergence of u in L2 space
   ParGridFunction div_u_l2(l2_fespace_scalar);
   ops.projectorDivHdivToL2.Apply(div_u_l2, u_hdiv);
   
   // 2. Project div u from L2 to H1 for decomposition
   ParGridFunction div_u_h1(h1_fespace_scalar);
   ops.projectorL2ToH1Scalar.Apply(div_u_h1, div_u_l2);
   
   // 4. Solve Poisson problem \nabla^2 \phi = div(u)
   
   // Set up Laplacian operator in H1 space
   ParBilinearForm laplacian(h1_fespace_scalar);
   if (pa) { laplacian.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   laplacian.AddDomainIntegrator(new DiffusionIntegrator());
   laplacian.Assemble();
   if (!pa) { laplacian.Finalize(); }
   
   // Set up RHS using div_u_h1 (both already in same H1 space)
   GridFunctionCoefficient div_u_coeff(&div_u_h1);
   ParLinearForm rhs(h1_fespace_scalar);
   rhs.AddDomainIntegrator(new DomainLFIntegrator(div_u_coeff));
   rhs.Assemble();
   
   Vector RHS(h1_fespace_scalar->GetTrueVSize());
   Vector PHI(h1_fespace_scalar->GetTrueVSize());
   rhs.ParallelAssemble(RHS);
   RHS *= -1.0;
   PHI = 0.0;
   
   // Use OrthoSolver to handle null space (constant functions)
   Array<int> empty_ess_tdof;  // No essential BC for periodic problem
   
   if (pa)
   {
      OperatorPtr laplacian_op;
      laplacian.FormSystemMatrix(empty_ess_tdof, laplacian_op);
      
      // Create preconditioner
      OperatorJacobiSmoother jac(laplacian, empty_ess_tdof);
      
      // Set up base solver
      CGSolver base_solver(h1_fespace_scalar->GetComm());
      base_solver.SetRelTol(1e-12);
      base_solver.SetMaxIter(1000);
      base_solver.SetPrintLevel(0);  // Reduce output since OrthoSolver will print
      base_solver.SetOperator(*laplacian_op);
      base_solver.SetPreconditioner(jac);
      
      // Create OrthoSolver to handle null space
      OrthoSolver ortho_solver(h1_fespace_scalar->GetComm());
      ortho_solver.SetSolver(base_solver);
      ortho_solver.SetOperator(*laplacian_op);
      
      ortho_solver.Mult(RHS, PHI);
   }
   else
   {
      std::unique_ptr<HypreParMatrix> A(laplacian.ParallelAssemble());
      
      // Create base preconditioner  
      HypreBoomerAMG amg(*A);
      amg.SetPrintLevel(0);
      
      // Set up base solver
      HyprePCG base_solver(*A);
      base_solver.SetTol(1e-12);
      base_solver.SetMaxIter(1000);
      base_solver.SetPrintLevel(0);  // Reduce output
      base_solver.SetPreconditioner(amg);
      
      // Create OrthoSolver to handle null space
      OrthoSolver ortho_solver(h1_fespace_scalar->GetComm());
      ortho_solver.SetSolver(base_solver);
      ortho_solver.SetOperator(*A);
      
      ortho_solver.Mult(RHS, PHI);
   }
   
   // Set the solution
   ParGridFunction phi_scalar(h1_fespace_scalar);
   phi_scalar = 0.0;
   
   phi_scalar.SetFromTrueDofs(PHI);

   // 5. Compute compressive part of velocify field
   ParGridFunction grad_phi(nd_fespace);
   ops.projectorComputeGradientH1ScalarToHcurl.Apply(grad_phi, phi_scalar);
   
   // 6. Compute curl of grad_phi for verification for later
   ParGridFunction curl_grad_phi(rt_fespace);
   ops.projectorCurlHcurlToHdiv.Apply(curl_grad_phi, grad_phi);

   // 7. Project grad phi from from Hcurl to H1
   ParGridFunction grad_phi_hdiv(rt_fespace);
   ops.projectorHdivToHcurl.Apply(grad_phi_hdiv,grad_phi);

   ParGridFunction grad_phi_l2(l2_fespace_vector);

   ops.projectorHdivToL2.Apply(grad_phi_l2, grad_phi_hdiv);

   ops.projectorL2ToH1.Apply(grad_phi_h1, grad_phi_l2);

   int dim = pmesh->Dimension();
   {

      ConstantCoefficient zero(0.0);

      Vector zero_v(dim);
      zero_v = 0.0;
      VectorConstantCoefficient zero_vec(zero_v);

      double curl_grad_phi_computed_error = curl_grad_phi.ComputeL2Error(zero_vec);
   

      if (myid == 0)
      {
         std::cout << "curl(grad phi) L2 error (should be ~0): " << curl_grad_phi_computed_error << std::endl;
      }
   }

   // 18. Free the used memory.
}

H1ToL2OrHdivProjector::H1ToL2OrHdivProjector(ParFiniteElementSpace *test_space, 
                   bool partial_assembly) 
    : test_fes(test_space), pa(partial_assembly)
{
    SetUpOperator();
}

void H1ToL2OrHdivProjector::SetUpOperator()
{
    // Create mass matrix on ND space (this is the expensive part)
    mass_form = std::make_unique<ParBilinearForm>(test_fes);
    if (pa) { mass_form->SetAssemblyLevel(AssemblyLevel::PARTIAL); }

    if (test_fes->GetVDim() == 1){
        mass_form->AddDomainIntegrator(new MassIntegrator());
    } else {
        mass_form->AddDomainIntegrator(new VectorMassIntegrator());
    }

    mass_form->Assemble();
    if (!pa) { mass_form->Finalize(); }
    
    // Setup solver for the mass matrix
    if (pa)
    {
        Array<int> ess_tdof_list; // empty for L2 projection
        mass_form->FormSystemMatrix(ess_tdof_list, mass_op);
        
        jacobi_prec = std::make_unique<OperatorJacobiSmoother>(*mass_form, ess_tdof_list);
        cg_solver = std::make_unique<CGSolver>(test_fes->GetComm());
        cg_solver->SetRelTol(1e-12);
        cg_solver->SetMaxIter(500);
        cg_solver->SetPrintLevel(0);
        cg_solver->SetOperator(*mass_op);
        cg_solver->SetPreconditioner(*jacobi_prec);
    }
    else
    {
        mass_matrix = std::unique_ptr<HypreParMatrix>(mass_form->ParallelAssemble());
        hypre_prec = std::make_unique<HypreDiagScale>(*mass_matrix);
        hypre_solver = std::make_unique<HyprePCG>(*mass_matrix);
        hypre_solver->SetTol(1e-12);
        hypre_solver->SetMaxIter(500);
        hypre_solver->SetPrintLevel(0);
        hypre_solver->SetPreconditioner(*hypre_prec);
    }
}

void H1ToL2OrHdivProjector::Apply(ParGridFunction &result, const ParGridFunction &u_h1) const
{
    ParLinearForm b(test_fes);

    if (test_fes->GetVDim() == 1){
        GridFunctionCoefficient ucoeff(&u_h1);
        b.AddDomainIntegrator(new DomainLFIntegrator(ucoeff));
    } else {
        VectorGridFunctionCoefficient ucoeff(&u_h1);
        b.AddDomainIntegrator(new VectorDomainLFIntegrator(ucoeff));
    }

    b.Assemble();
    
    Vector B(test_fes->GetTrueVSize()), X(test_fes->GetTrueVSize());
    b.ParallelAssemble(B);
    X = 0.0;
    
    if (pa)
    {
        cg_solver->Mult(B, X);
    }
    else
    {
        hypre_solver->Mult(B, X);
    }
    
    result = 0.0;
    result.SetFromTrueDofs(X);
}

 H1ToHdivOrHcurlProjector::H1ToHdivOrHcurlProjector(ParFiniteElementSpace *test_space, 
                       bool partial_assembly) 
        : test_fes(test_space), pa(partial_assembly)
 {
    SetUpOperator();
 }

 void H1ToHdivOrHcurlProjector::SetUpOperator()
{
    // Create mass matrix on ND space (this is the expensive part)
    mass_form = std::make_unique<ParBilinearForm>(test_fes);
    if (pa) { mass_form->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
    mass_form->AddDomainIntegrator(new VectorFEMassIntegrator());
    mass_form->Assemble();
    if (!pa) { mass_form->Finalize(); }
    
    // Setup solver for the mass matrix
    if (pa)
    {
        Array<int> ess_tdof_list; // empty for L2 projection
        mass_form->FormSystemMatrix(ess_tdof_list, mass_op);
        
        jacobi_prec = std::make_unique<OperatorJacobiSmoother>(*mass_form, ess_tdof_list);
        cg_solver = std::make_unique<CGSolver>(test_fes->GetComm());
        cg_solver->SetRelTol(1e-12);
        cg_solver->SetMaxIter(500);
        cg_solver->SetPrintLevel(0);
        cg_solver->SetOperator(*mass_op);
        cg_solver->SetPreconditioner(*jacobi_prec);
    }
    else
    {
        mass_matrix = std::unique_ptr<HypreParMatrix>(mass_form->ParallelAssemble());
        hypre_prec = std::make_unique<HypreDiagScale>(*mass_matrix);
        hypre_solver = std::make_unique<HyprePCG>(*mass_matrix);
        hypre_solver->SetTol(1e-12);
        hypre_solver->SetMaxIter(500);
        hypre_solver->SetPrintLevel(0);
        hypre_solver->SetPreconditioner(*hypre_prec);
    }
}

void H1ToHdivOrHcurlProjector::Apply(ParGridFunction &result, const ParGridFunction &u_h1) const
{
    VectorGridFunctionCoefficient ucoeff(&u_h1);
    ParLinearForm b(test_fes);
    b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(ucoeff));
    b.Assemble();
    
    Vector B(test_fes->GetTrueVSize()), X(test_fes->GetTrueVSize());
    b.ParallelAssemble(B);  // This B depends on u_h1!
    X = 0.0;
    
    if (pa)
    {
        cg_solver->Mult(B, X);  // Solve M*X = B
    }
    else
    {
        hypre_solver->Mult(B, X);  // Solve M*X = B
    }
    
    result = 0.0;
    result.SetFromTrueDofs(X);
}

 HcurlHdivProjector::HcurlHdivProjector(ParFiniteElementSpace *trial_space, 
                       ParFiniteElementSpace *test_space, 
                       bool partial_assembly) 
        : trial_fes(trial_space), test_fes(test_space), pa(partial_assembly)
 {
     SetUpOperator();
 }

 void HcurlHdivProjector::SetUpOperator()
 {
     mass_form = std::make_unique<ParBilinearForm>(test_fes);
     if (pa) { mass_form->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
     mass_form->AddDomainIntegrator(new VectorFEMassIntegrator());
     mass_form->Assemble();
     if (!pa) { mass_form->Finalize(); }

     mixed_form = std::make_unique<ParMixedBilinearForm>(trial_fes, test_fes);
     if (pa) { mixed_form->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
     mixed_form->AddDomainIntegrator(new VectorFEMassIntegrator());
     mixed_form->Assemble();
     if (!pa) { mixed_form->Finalize(); }
     
     // Setup solver for the mass matrix
     if (pa)
     {
         Array<int> ess_tdof_list; // empty for L2 projection
         mass_form->FormSystemMatrix(ess_tdof_list, mass_op);

         jacobi_prec = std::make_unique<OperatorJacobiSmoother>(*mass_form, ess_tdof_list);
         cg_solver = std::make_unique<CGSolver>(test_fes->GetComm());
         cg_solver->SetRelTol(1e-12);
         cg_solver->SetMaxIter(500);
         cg_solver->SetPrintLevel(0);
         cg_solver->SetOperator(*mass_op);
         cg_solver->SetPreconditioner(*jacobi_prec);
     }
     else
     {
         mixed_matrix = std::unique_ptr<HypreParMatrix>(mixed_form->ParallelAssemble());
         mass_matrix = std::unique_ptr<HypreParMatrix>(mass_form->ParallelAssemble());
         hypre_prec = std::make_unique<HypreDiagScale>(*mass_matrix);
         hypre_solver = std::make_unique<HyprePCG>(*mass_matrix);
         hypre_solver->SetTol(1e-12);
         hypre_solver->SetMaxIter(500);
         hypre_solver->SetPrintLevel(0);
         hypre_solver->SetPreconditioner(*hypre_prec);
     }
 }

void HcurlHdivProjector::Apply(ParGridFunction &result, const ParGridFunction &gftrial) const
{
    Vector B(test_fes->GetTrueVSize());
    Vector X(test_fes->GetTrueVSize());

    if (pa)
    {
       ParLinearForm b(test_fes); // used as a vector
       mixed_form->Mult(gftrial, b); // process-local multiplication
       b.ParallelAssemble(B);
    }
    else
    {

       Vector P(trial_fes->GetTrueVSize());
       gftrial.GetTrueDofs(P);
       mixed_matrix->Mult(P,B);
    }

    X = 0.0;
    if(pa)
    {
     cg_solver->Mult(B,X);
    }else{
     hypre_solver->Mult(B,X);
    }
    result.SetFromTrueDofs(X);
}

ComputeDivergenceHdivToL2::ComputeDivergenceHdivToL2(ParFiniteElementSpace *trial_space, 
                      ParFiniteElementSpace *test_space, 
                      bool partial_assembly) 
       : trial_fes(trial_space), test_fes(test_space), pa(partial_assembly)
{
    SetUpOperator();
}

void ComputeDivergenceHdivToL2::SetUpOperator()
{
    mass_form = std::make_unique<ParBilinearForm>(test_fes);
    if (pa) { mass_form->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
    mass_form->AddDomainIntegrator(new MassIntegrator());
    mass_form->Assemble();
    if (!pa) { mass_form->Finalize(); }

    mixed_form = std::make_unique<ParMixedBilinearForm>(trial_fes, test_fes);
    if (pa) { mixed_form->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
    mixed_form->AddDomainIntegrator(new VectorFEDivergenceIntegrator());
    mixed_form->Assemble();
    if (!pa) { mixed_form->Finalize(); }
    
    // Setup solver for the mass matrix
    if (pa)
    {
        Array<int> ess_tdof_list; // empty for L2 projection
        mass_form->FormSystemMatrix(ess_tdof_list, mass_op);

        jacobi_prec = std::make_unique<OperatorJacobiSmoother>(*mass_form, ess_tdof_list);
        cg_solver = std::make_unique<CGSolver>(test_fes->GetComm());
        cg_solver->SetRelTol(1e-12);
        cg_solver->SetMaxIter(500);
        cg_solver->SetPrintLevel(0);
        cg_solver->SetOperator(*mass_op);
        cg_solver->SetPreconditioner(*jacobi_prec);
    }
    else
    {
        mixed_matrix = std::unique_ptr<HypreParMatrix>(mixed_form->ParallelAssemble());
        mass_matrix = std::unique_ptr<HypreParMatrix>(mass_form->ParallelAssemble());
        hypre_prec = std::make_unique<HypreDiagScale>(*mass_matrix);
        hypre_solver = std::make_unique<HyprePCG>(*mass_matrix);
        hypre_solver->SetTol(1e-12);
        hypre_solver->SetMaxIter(500);
        hypre_solver->SetPrintLevel(0);
        hypre_solver->SetPreconditioner(*hypre_prec);
    }
}
void ComputeDivergenceHdivToL2::Apply(ParGridFunction &result, const ParGridFunction &gftrial) const
{
    Vector B(test_fes->GetTrueVSize());
    Vector X(test_fes->GetTrueVSize());

    if (pa)
    {
       ParLinearForm b(test_fes); // used as a vector
       mixed_form->Mult(gftrial, b); // process-local multiplication
       b.ParallelAssemble(B);
    }
    else
    {

       Vector P(trial_fes->GetTrueVSize());
       gftrial.GetTrueDofs(P);
       mixed_matrix->Mult(P,B);
    }

    X = 0.0;
    if(pa)
    {
     cg_solver->Mult(B,X);
    }else{
     hypre_solver->Mult(B,X);
    }
    result.SetFromTrueDofs(X);
}

ComputeCurlHcurlToHdiv::ComputeCurlHcurlToHdiv(ParFiniteElementSpace *trial_space, 
                       ParFiniteElementSpace *test_space, 
                       bool partial_assembly) 
        : trial_fes(trial_space), test_fes(test_space), pa(partial_assembly)
{
    SetUpOperator();
}

void ComputeCurlHcurlToHdiv::SetUpOperator()
{
    mass_form = std::make_unique<ParBilinearForm>(test_fes);
    if (pa) { mass_form->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
    mass_form->AddDomainIntegrator(new VectorFEMassIntegrator());
    mass_form->Assemble();
    if (!pa) { mass_form->Finalize(); }

    mixed_form = std::make_unique<ParMixedBilinearForm>(trial_fes, test_fes);
    if (pa) { mixed_form->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
    mixed_form->AddDomainIntegrator(new MixedVectorCurlIntegrator());
    mixed_form->Assemble();
    if (!pa) { mixed_form->Finalize(); }
    
    // Setup solver for the mass matrix
    if (pa)
    {
        Array<int> ess_tdof_list; // empty for L2 projection
        mass_form->FormSystemMatrix(ess_tdof_list, mass_op);

        jacobi_prec = std::make_unique<OperatorJacobiSmoother>(*mass_form, ess_tdof_list);
        cg_solver = std::make_unique<CGSolver>(test_fes->GetComm());
        cg_solver->SetRelTol(1e-12);
        cg_solver->SetMaxIter(500);
        cg_solver->SetPrintLevel(0);
        cg_solver->SetOperator(*mass_op);
        cg_solver->SetPreconditioner(*jacobi_prec);
    }
    else
    {
        mixed_matrix = std::unique_ptr<HypreParMatrix>(mixed_form->ParallelAssemble());
        mass_matrix = std::unique_ptr<HypreParMatrix>(mass_form->ParallelAssemble());
        hypre_prec = std::make_unique<HypreDiagScale>(*mass_matrix);
        hypre_solver = std::make_unique<HyprePCG>(*mass_matrix);
        hypre_solver->SetTol(1e-12);
        hypre_solver->SetMaxIter(500);
        hypre_solver->SetPrintLevel(0);
        hypre_solver->SetPreconditioner(*hypre_prec);
    }
}
    
void ComputeCurlHcurlToHdiv::Apply(ParGridFunction &result, const ParGridFunction &gftrial) const
{
    Vector B(test_fes->GetTrueVSize());
    Vector X(test_fes->GetTrueVSize());

    if (pa)
    {
       ParLinearForm b(test_fes); // used as a vector
       mixed_form->Mult(gftrial, b); // process-local multiplication
       b.ParallelAssemble(B);
    }
    else
    {

       Vector P(trial_fes->GetTrueVSize());
       gftrial.GetTrueDofs(P);
       mixed_matrix->Mult(P,B);
    }

    X = 0.0;
    if(pa)
    {
     cg_solver->Mult(B,X);
    }else{
     hypre_solver->Mult(B,X);
    }
    result.SetFromTrueDofs(X);
}

ComputeGradientH1ScalarToHcurl::ComputeGradientH1ScalarToHcurl(ParFiniteElementSpace *trial_space, 
                       ParFiniteElementSpace *test_space, 
                       bool partial_assembly) 
        : trial_fes(trial_space), test_fes(test_space), pa(partial_assembly)
{
    SetUpOperator();
}
    
void ComputeGradientH1ScalarToHcurl::SetUpOperator()
{
    mass_form = std::make_unique<ParBilinearForm>(test_fes);
    if (pa) { mass_form->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
    mass_form->AddDomainIntegrator(new VectorFEMassIntegrator());
    mass_form->Assemble();
    if (!pa) { mass_form->Finalize(); }

    mixed_form = std::make_unique<ParMixedBilinearForm>(trial_fes, test_fes);
    if (pa) { mixed_form->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
    mixed_form->AddDomainIntegrator(new MixedVectorGradientIntegrator());
    mixed_form->Assemble();
    if (!pa) { mixed_form->Finalize(); }
    
    // Setup solver for the mass matrix
    if (pa)
    {
        Array<int> ess_tdof_list; // empty for L2 projection
        mass_form->FormSystemMatrix(ess_tdof_list, mass_op);

        jacobi_prec = std::make_unique<OperatorJacobiSmoother>(*mass_form, ess_tdof_list);
        cg_solver = std::make_unique<CGSolver>(test_fes->GetComm());
        cg_solver->SetRelTol(1e-12);
        cg_solver->SetMaxIter(500);
        cg_solver->SetPrintLevel(0);
        cg_solver->SetOperator(*mass_op);
        cg_solver->SetPreconditioner(*jacobi_prec);
    }
    else
    {
        mixed_matrix = std::unique_ptr<HypreParMatrix>(mixed_form->ParallelAssemble());
        mass_matrix = std::unique_ptr<HypreParMatrix>(mass_form->ParallelAssemble());
        hypre_prec = std::make_unique<HypreDiagScale>(*mass_matrix);
        hypre_solver = std::make_unique<HyprePCG>(*mass_matrix);
        hypre_solver->SetTol(1e-12);
        hypre_solver->SetMaxIter(500);
        hypre_solver->SetPrintLevel(0);
        hypre_solver->SetPreconditioner(*hypre_prec);
    }
}
    
void ComputeGradientH1ScalarToHcurl::Apply(ParGridFunction &result, const ParGridFunction &gftrial) const
{
    Vector B(test_fes->GetTrueVSize());
    Vector X(test_fes->GetTrueVSize());

    if (pa)
    {
       ParLinearForm b(test_fes); // used as a vector
       mixed_form->Mult(gftrial, b); // process-local multiplication
       b.ParallelAssemble(B);
    }
    else
    {

       Vector P(trial_fes->GetTrueVSize());
       gftrial.GetTrueDofs(P);
       mixed_matrix->Mult(P,B);
    }

    X = 0.0;
    if(pa)
    {
     cg_solver->Mult(B,X);
    }else{
     hypre_solver->Mult(B,X);
    }
    result.SetFromTrueDofs(X);
}
