// pfindpts_victor.cpp — batch pfindpts following MFEM miniapp pattern (older API friendly)

#include "mfem.hpp"
#include "fem/gslib.hpp"
#include <cmath>
#include <iostream>

using namespace mfem;
using std::cout;
using std::endl;

static real_t Lx=1.0, Ly=1.0, Lz=1.0;
static bool use_vector = true;

// -------------------------------
// Smooth periodic fields for testing
// -------------------------------
double ScalarInitFun(const Vector &x)
{
   return std::sin(2.0*M_PI*x[0]/Lx)*
          std::cos(2.0*M_PI*x[1]/Ly)*
          std::cos(2.0*M_PI*x[2]/Lz);
}

void VectorInit(const Vector &x, Vector &v)
{
   v.SetSize(3);
   v[0] =  std::sin(2.0*M_PI*x[1]/Ly)*std::cos(2.0*M_PI*x[2]/Lz);
   v[1] =  std::sin(2.0*M_PI*x[2]/Lz)*std::cos(2.0*M_PI*x[0]/Lx);
   v[2] =  std::sin(2.0*M_PI*x[0]/Lx)*std::cos(2.0*M_PI*x[1]/Ly);
}

// -----------------------
// Simple mesh perturbation
// -----------------------
class SmallSmoothPerturbation : public VectorCoefficient
{
   real_t amp;
public:
   SmallSmoothPerturbation(real_t a) : VectorCoefficient(3), amp(a) {}
   virtual void Eval(Vector &dx, ElementTransformation &T,
                     const IntegrationPoint &ip) override
   {
      Vector X(3); T.Transform(ip, X);
      dx.SetSize(3);
      dx[0] = amp * std::sin(2.0*M_PI*X[1]/Ly) * std::sin(2.0*M_PI*X[2]/Lz);
      dx[1] = amp * std::sin(2.0*M_PI*X[2]/Lz) * std::sin(2.0*M_PI*X[0]/Lx);
      dx[2] = amp * std::sin(2.0*M_PI*X[0]/Lx) * std::sin(2.0*M_PI*X[1]/Ly);
   }
};

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   int nx=8, ny=8, nz=8, order=2;
   real_t amp=0.05;
   use_vector = true;

   OptionsParser args(argc, argv);
   args.AddOption(&nx, "-nx", "--nx", "Elements in x.");
   args.AddOption(&ny, "-ny", "--ny", "Elements in y.");
   args.AddOption(&nz, "-nz", "--nz", "Elements in z.");
   args.AddOption(&order, "-o", "--order", "FE order.");
   args.AddOption(&Lx, "-Lx", "--Lx", "Domain length in x.");
   args.AddOption(&Ly, "-Ly", "--Ly", "Domain length in y.");
   args.AddOption(&Lz, "-Lz", "--Lz", "Domain length in z.");
   args.AddOption(&amp, "-amp", "--amplitude", "Mesh perturbation amplitude.");
   args.AddOption(&use_vector, "-v", "--vector", "-s", "--scalar",
                  "Interpolate vector field (true) or scalar field (false).");
   args.Parse();
   if (!args.Good()) { if (Mpi::Root()) args.PrintUsage(cout); return 1; }
   if (Mpi::Root())  { args.PrintOptions(cout); }

   const int dim = 3;
   H1_FECollection h1c(order, dim);

   // ----------------------------
   // 1) Source: perturbed periodic mesh + field
   // ----------------------------
   Mesh src_serial = Mesh::MakeCartesian3D(nx, ny, nz,
      Element::HEXAHEDRON, Lx, Ly, Lz, /*sfc_ordering=*/true);
   ParMesh src_mesh(MPI_COMM_WORLD, src_serial);
   src_serial.Clear();

   // nodal space for coordinates (vector H1, byNODES to simplify packing)
   ParFiniteElementSpace src_nodes_fes(&src_mesh, &h1c, dim, Ordering::byNODES);
   ParGridFunction src_nodes(&src_nodes_fes);
   src_mesh.SetNodalFESpace(&src_nodes_fes);
   src_mesh.GetNodes(src_nodes);

   // perturb coordinates
   SmallSmoothPerturbation disp(amp);
   ParGridFunction disp_h1(&src_nodes_fes);
   disp_h1.ProjectCoefficient(disp);
   src_nodes += disp_h1;
   src_mesh.NewNodes(src_nodes, /*make_owner=*/true);

   // source field on perturbed mesh
   const int vdim = use_vector ? 3 : 1;
   ParFiniteElementSpace src_fes(&src_mesh, &h1c, vdim, Ordering::byNODES);
   ParGridFunction u_src(&src_fes); u_src = 0.0;
   if (use_vector) { VectorFunctionCoefficient vcoef(3, VectorInit);
                     u_src.ProjectCoefficient(vcoef); }
   else            { FunctionCoefficient      scoef(ScalarInitFun);
                     u_src.ProjectCoefficient(scoef); }

   // ----------------------------
   // 2) Target: uniform periodic mesh
   //    (we'll sample at its H1 nodal coordinates)
   // ----------------------------
   Mesh tgt_serial = Mesh::MakeCartesian3D(nx, ny, nz,
      Element::HEXAHEDRON, Lx, Ly, Lz, /*sfc_ordering=*/true);
   ParMesh tgt_mesh(MPI_COMM_WORLD, tgt_serial);
   tgt_serial.Clear();

   // Set target nodal FES (byNODES) and get nodal coordinates
   ParFiniteElementSpace tgt_nodes_fes(&tgt_mesh, &h1c, dim, Ordering::byNODES);
   ParGridFunction tgt_nodes(&tgt_nodes_fes);
   tgt_mesh.SetNodalFESpace(&tgt_nodes_fes);
   tgt_mesh.GetNodes(tgt_nodes);

   // H1 field space on target (continuous output)
   ParFiniteElementSpace tgt_fes_h1(&tgt_mesh, &h1c, vdim, Ordering::byNODES);
   ParGridFunction u_h1(&tgt_fes_h1); u_h1 = 0.0;

   // (optional) DG/L2 "honeycomb"
   L2_FECollection l2c(order, dim);
   ParFiniteElementSpace tgt_fes_l2(&tgt_mesh, &l2c, vdim);
   ParGridFunction u_dg(&tgt_fes_l2); u_dg = 0.0;

   // ----------------------------
   // 3) Batch FindPoints / Interpolate like the miniapp
   //    Query points = target H1 true nodes
   // ----------------------------
   // Pack target nodal true dofs into vxyz with point_ordering = byNODES
   Vector Xtrue; tgt_nodes.GetTrueDofs(Xtrue);
   MFEM_VERIFY((int)Ordering::byNODES == 0, "Assumed enum value changed.");
   const int pts_cnt = Xtrue.Size()/dim;

   Vector vxyz(pts_cnt * dim);
   // tgt_nodes_fes is byNODES, so we can copy directly
   vxyz = Xtrue;
   const int point_ordering = Ordering::byNODES;

   // Set up pfindpts on the SOURCE mesh
   FindPointsGSLIB finder(MPI_COMM_WORLD);
   finder.Setup(src_mesh);
   const double L = std::max(Lx, std::max(Ly, Lz));
   const double tol = std::max(1e-1 * L, 10.0 * amp);
   finder.SetDistanceToleranceForPointsFoundOnBoundary(tol);

   // Find computational locations for all target nodes, then interpolate u_src there
   finder.FindPoints(vxyz, point_ordering);

   Vector interp_vals(pts_cnt * vdim);
   finder.Interpolate(u_src, interp_vals);

   // ----------------------------
   // 4) Load results into target GridFunctions
   // ----------------------------
   // H1 continuous: directly set from true dofs (orders/ordering match)
   u_h1.SetFromTrueDofs(interp_vals);

   // Optional DG "honeycomb": element-local projection of u_h1
   if (vdim == 1)
   {
      GridFunctionCoefficient gc(&u_h1);
      u_dg.ProjectDiscCoefficient(gc, GridFunction::ARITHMETIC);
   }
   else
   {
      VectorGridFunctionCoefficient vgc(&u_h1);
      u_dg.ProjectDiscCoefficient(vgc, GridFunction::ARITHMETIC);
   }

   // ----------------------------
   // 5) Save VisIt outputs
   // ----------------------------
   {
      VisItDataCollection visit_src("PeriodicInterp_Source", &src_mesh);
      visit_src.SetPrefixPath("visit_src");
      visit_src.RegisterField("u_src", &u_src);
      visit_src.Save();
   }
   {
      VisItDataCollection visit_tgt("PeriodicInterp_Target", &tgt_mesh);
      visit_tgt.SetPrefixPath("visit_tgt");
      visit_tgt.RegisterField("u_h1", &u_h1);
      visit_tgt.RegisterField("u_dg", &u_dg);
      visit_tgt.Save();
   }

   if (Mpi::Root())
   {
      cout << "\nWrote:\n"
           << "  visit_src/PeriodicInterp_Source.visit   (perturbed mesh + u_src)\n"
           << "  visit_tgt/PeriodicInterp_Target.visit   (uniform mesh + u_h1, u_dg)\n"
           << "Open u_h1 for the smooth continuous field, u_dg for the honeycomb view.\n";
   }

   finder.FreeData();
   return 0;
}
