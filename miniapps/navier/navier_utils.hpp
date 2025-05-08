// navier_utils.hpp
#ifndef NAVIER_UTILS_HPP
#define NAVIER_UTILS_HPP

#include "mfem.hpp"
#include "navier_solver.hpp"

struct s_NavierContext;

bool GetVisit(const s_NavierContext* ctx);
bool GetConduit(const s_NavierContext* ctx);
mfem::real_t GetReynum(const s_NavierContext* ctx);
int GetNumPts(const s_NavierContext* ctx);
int GetElementSubdivisions(const s_NavierContext* ctx);
int GetElementSubdivisionsParallel(const s_NavierContext* ctx);
int GetOrder(const s_NavierContext* ctx);
mfem::real_t GetKinvis(const s_NavierContext* ctx);
bool GetPA(const s_NavierContext* ctx);
bool GetNI(const s_NavierContext* ctx);
mfem::real_t GetDt(const s_NavierContext* ctx);

bool LoadCheckpoint(mfem::ParMesh *&pmesh,
                    mfem::ParGridFunction *&u_gf,
                    mfem::ParGridFunction *&p_gf,
                    mfem::navier::NavierSolver *&flowsolver,
                    double &t,
                    int &step,
                    int myid,
                    const s_NavierContext* ctx);

void SamplePoints(mfem::ParGridFunction* sol,
                                mfem::ParMesh* pmesh,
                                int step,
                                double time,
                                const std::string &suffix,
                                const s_NavierContext* ctx);

#endif // NAVIER_UTILS_HPP

