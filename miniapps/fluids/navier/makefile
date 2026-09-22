# Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

# Use the MFEM build directory
MFEM_DIR ?= ../../..
MFEM_BUILD_DIR ?= ../../..
MFEM_INSTALL_DIR ?= ../../../mfem
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/fluids/navier/,)
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

NAVIER_COMMON_SRC = navier_solver.cpp navier_utils.cpp
NAVIER_COMMON_OBJ = $(NAVIER_COMMON_SRC:.cpp=.o)

ifeq ($(MFEM_USE_GSLIB),YES)
   NAVIER_PARTICLE_COMMON_SRC = navier_particles.cpp
   NAVIER_PARTICLE_COMMON_OBJ = $(NAVIER_PARTICLE_COMMON_SRC:.cpp=.o)

   NAVIER_COMMON_SRC += $(NAVIER_PARTICLE_COMMON_SRC)
   NAVIER_COMMON_OBJ += $(NAVIER_PARTICLE_COMMON_OBJ)
endif

COMMON_LIB = -L$(MFEM_BUILD_DIR)/miniapps/common -lmfem-common

# If MFEM_SHARED is set, add the ../common rpath
COMMON_LIB += $(if $(MFEM_SHARED:YES=),,\
   $(MFEM_XLINKER)-rpath,$(abspath $(MFEM_BUILD_DIR)/miniapps/common))

PAR_MINIAPPS = navier_mms navier_kovasznay navier_kovasznay_vs navier_tgv\
 navier_shear navier_shear_layer navier_3dfoc navier_turbchan navier_bifurcation
ifeq ($(MFEM_USE_GSLIB),YES)
   PAR_MINIAPPS += navier_cht
endif

ifeq ($(MFEM_USE_MPI),NO)
   MINIAPPS =
else
   MINIAPPS = $(PAR_MINIAPPS)
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all lib-common clean clean-build clean-exec
# Keeping the *.o files fixes an issue with the MacOS version of 'make'.
.PRECIOUS: %.o

# Remove built-in rules
%: %.cpp
%.o: %.cpp

%: %.o $(NAVIER_COMMON_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $^ -o $@ $(MFEM_LIBS) $(COMMON_LIB)

# navier_tgv uses only NavierSolver. Linking navier_utils.cpp here introduces
# unrelated context-helper symbols that this standalone driver does not define.
navier_tgv: navier_tgv.o navier_solver.o
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $^ -o $@ $(MFEM_LIBS) $(COMMON_LIB)

%.o: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK) | lib-common
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@

all: $(MINIAPPS)

# Rule for building lib-common
lib-common:
	$(MAKE) -C $(MFEM_BUILD_DIR)/miniapps/common

PAR_TEST_EXECS = navier_mms navier_kovasznay navier_tgv
MFEM_TESTS = TEST_EXECS
include $(MFEM_TEST_MK)

RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)

navier_mms-test-par: navier_mms
	@$(call mfem-test,$<, $(RUN_MPI), Navier,-cr)

navier_kovasznay-test-par: navier_kovasznay
	@$(call mfem-test,$<, $(RUN_MPI), Navier,-cr)

navier_tgv-test-par: navier_tgv
	@$(call mfem-test,$<, $(RUN_MPI), Navier,-cr)

navier_tgv-amr-test-par: navier_tgv
	@$(call mfem-test,$<, $(RUN_MPI), Navier AMR,-es 1 -o 2 -tf 0.007 -dt 0.001 -amr -amr-t 0.004 -amr-th 0.8 -amr-check)

navier_shear_layer-amr-test-par: navier_shear_layer
	@$(call mfem-test,$<, $(RUN_MPI), Shear-layer AMR,-dim 2 -nx 16 -ny 8 -o 2 -Re 100 -dt 0.001 -tf 0.02 -amr -amr-t 0.005 -amr-th 0.5 -amr-check -amr-rebalance -no-visit)

navier_shear_layer-dye-amr-test-par: navier_shear_layer
	@$(call mfem-test,$<, $(RUN_MPI), Shear-layer dye AMR,-dim 2 -nx 16 -ny 8 -o 2 -Re 100 -dt 0.001 -tf 0.02 -amr -amr-t 0.005 -amr-dye-mixing -amr-dye-min 0.3 -amr-dye-max 0.7 -amr-check -amr-rebalance -no-visit)

navier_shear_layer-dye-amr-cap-test-par: navier_shear_layer
	@$(call mfem-test,$<, $(RUN_MPI), Shear-layer capped dye AMR,-dim 2 -nx 16 -ny 8 -o 2 -Re 500 -dt 0.001 -tf 0.02 -pa -amr -amr-t 0.005 -amr-i 5 -amr-max-events 0 -amr-max-marked 4 -amr-dye-mixing -amr-dye-min 0.1 -amr-dye-max 0.8 -amr-check -amr-rebalance -no-visit)

navier_shear_layer-dye-amr-coarsen-test-par: navier_shear_layer
	@$(call mfem-test,$<, $(RUN_MPI), Shear-layer moving dye AMR,-dim 2 -nx 16 -ny 8 -o 2 -Re 500 -dt 0.001 -tf 0.02 -pa -amr -amr-t 0.005 -amr-i 5 -amr-max-events 0 -amr-max-marked 4 -amr-dye-mixing -amr-dye-min 0.1 -amr-dye-max 0.8 -amr-coarsen -amr-check -amr-rebalance -no-visit)

navier_shear_layer-dye-dynamic-test-par: navier_shear_layer
	@$(call mfem-test,$<, $(RUN_MPI), Shear-layer simple moving dye AMR,-dim 2 -nx 16 -ny 8 -o 2 -Re 500 -dt 0.001 -tf 0.04 -pa -amr-dye-dynamic -amr-t 0.005 -amr-i 5 -amr-max-level 1 -amr-check -amr-rebalance -no-visit)

navier_shear_layer-dye-enstrophy-amr-test-par: navier_shear_layer
	@$(call mfem-test,$<, $(RUN_MPI), Shear-layer dye or enstrophy AMR,-dim 2 -nx 16 -ny 8 -o 2 -Re 500 -dt 0.001 -tf 0.04 -pa -amr-dye-dynamic -amr-dye-enstrophy -amr-th 0.5 -amr-t 0.005 -amr-i 5 -amr-max-level 1 -amr-check -amr-rebalance -no-visit)

navier_shear_layer-dye-gradient-amr-test-par: navier_shear_layer
	@$(call mfem-test,$<, $(RUN_MPI), Shear-layer moving gradient AMR,-dim 2 -nx 16 -ny 8 -o 2 -Re 500 -dt 0.001 -tf 0.04 -pa -amr -amr-t 0.005 -amr-i 5 -amr-max-events 0 -amr-max-marked 4 -amr-dye-gradient -amr-dye-grad-th 0.15 -amr-dye-grad-coarsen-th 0.05 -amr-dye-grad-inlet-buffer 0.75 -amr-coarsen -amr-check -amr-rebalance -no-visit)

navier_shear_layer-3d-dye-amr-test-par: navier_shear_layer
	@$(call mfem-test,$<, $(RUN_MPI), Shear-layer 3D dye AMR,-dim 3 -nx 8 -ny 4 -nz 4 -lx 4 -ly 1 -lz 0.5 -no-per-z -o 2 -Re 100 -dt 0.0005 -tf 0.005 -amr -amr-t 0.002 -amr-dye-mixing -amr-dye-min 0.05 -amr-dye-max 0.95 -amr-check -amr-rebalance -no-visit)

navier_shear_layer-3d-dye-amr-pa-test-par: navier_shear_layer
	@$(call mfem-test,$<, $(RUN_MPI), Shear-layer 3D dye AMR PA,-dim 3 -nx 8 -ny 4 -nz 4 -lx 4 -ly 1 -lz 0.5 -no-per-z -o 2 -Re 100 -dt 0.0005 -tf 0.005 -pa -amr -amr-t 0.002 -amr-dye-mixing -amr-dye-min 0.05 -amr-dye-max 0.95 -amr-check -amr-rebalance -no-visit)

navier_shear_layer-3d-enstrophy-amr-test-par: navier_shear_layer
	@$(call mfem-test,$<, $(RUN_MPI), Shear-layer 3D enstrophy AMR,-dim 3 -nx 8 -ny 4 -nz 4 -lx 4 -ly 1 -lz 0.5 -no-per-z -o 2 -Re 100 -dt 0.0005 -tf 0.005 -amr -amr-t 0.002 -amr-th 0.5 -amr-check -amr-rebalance -no-visit)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(PAR_MINIAPPS)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -rf tgv_out*.txt *_output ParaView Navier_Bifurcation_*
