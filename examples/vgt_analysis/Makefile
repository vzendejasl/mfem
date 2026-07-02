# VGT analysis — built against the MFEM parallel library.
# Follows the same pattern as examples/makefile.
#
# Usage (from this directory):
#   make
#   make test_vgt_mfem_units
#   make VGT_DEPS_DIR=/absolute/path/to/third_party/vgt_deps test_vgt_mfem_units
#   make clean

THIS_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
VGT_DEPS_DIR ?= $(abspath $(THIS_DIR)/../../../third_party/vgt_deps)
VGT_DEPS_MK := $(VGT_DEPS_DIR)/vgt_deps.mk

# ── MFEM paths (mirrors examples/makefile exactly) ────────────────────────────
MFEM_DIR       ?= ../..
MFEM_BUILD_DIR  ?= ../..
MFEM_INSTALL_DIR ?= ../../mfem
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)
include $(VGT_DEPS_MK)

# MFEM_FLAGS and MFEM_LIBS are now set by config.mk.
# If MFEM does not provide a default MPI compiler in PATH, fall back to a
# common Homebrew/MPICH install location.
MFEM_CXX := $(or $(shell command -v $(MFEM_CXX) 2>/dev/null),\
                $(shell command -v mpicxx 2>/dev/null),\
                /usr/local/bin/mpicxx)

# ── Shared repo-owned VGT dependencies ────────────────────────────────────────
ifeq ($(wildcard $(VGT_SHARED_INCLUDE_DIR)/vgt_lapack.hpp),)
$(error Missing VGT headers under $(VGT_SHARED_INCLUDE_DIR). Populate third_party/vgt_deps first.)
endif
ifeq ($(wildcard $(VGT_SHARED_EIGEN_DIR)/Eigen/Core),)
$(error Missing Eigen headers under $(VGT_SHARED_EIGEN_DIR). Populate third_party/vgt_deps first.)
endif
ifeq ($(wildcard $(VGT_SHARED_LAPACK_LIB_DIR)/liblapacke.a),)
$(error Missing LAPACK/LAPACKE libraries under $(VGT_SHARED_LAPACK_LIB_DIR). Populate third_party/vgt_deps first.)
endif

# The current MFEM config.mk was generated while this branch still pointed at an
# external VGT bundle for LAPACK/LAPACKE. Strip those inherited external paths
# and replace them with the repo-owned shared prefix above.
VGT_EXTERNAL_LAPACK_INC_PATTERNS := \
  %/vgt_analysis/vgt_all_bundle/cpp/build/_deps/lapack_ref-build/include \
  %/vgt_analysis/vgt_all_bundle/cpp/build/_deps/lapack_ref-src/LAPACKE/include

VGT_EXTERNAL_LAPACK_LIB_PATTERNS := \
  %/vgt_analysis/vgt_all_bundle/cpp/build/_deps/lapack_ref-build/lib/liblapacke.a \
  %/vgt_analysis/vgt_all_bundle/cpp/build/_deps/lapack_ref-build/lib/liblapack.a \
  %/vgt_analysis/vgt_all_bundle/cpp/build/_deps/lapack_ref-build/lib/libblas.a

MFEM_TPLFLAGS := $(filter-out $(VGT_EXTERNAL_LAPACK_INC_PATTERNS),$(MFEM_TPLFLAGS))
MFEM_INCFLAGS := -I$(MFEM_INC_DIR) $(MFEM_TPLFLAGS)
MFEM_FLAGS := $(MFEM_CPPFLAGS) $(MFEM_CXXFLAGS) $(MFEM_INCFLAGS)

MFEM_EXT_LIBS := $(filter-out $(VGT_EXTERNAL_LAPACK_LIB_PATTERNS),$(MFEM_EXT_LIBS)) \
                 $(VGT_SHARED_LAPACK_LIBS)
MFEM_LIBS := -L$(MFEM_LIB_DIR) -lmfem $(MFEM_EXT_LIBS)

VGT_INCLUDE = -I$(VGT_SHARED_INCLUDE_DIR) \
              -I$(VGT_SHARED_EIGEN_DIR) \
              -I$(VGT_SHARED_LAPACK_INCLUDE_DIR)
VGT_DEFINES = -DVGT_DATA_DIR=\"$(VGT_SHARED_DATA_DIR)\"

# ── Targets ───────────────────────────────────────────────────────────────────
TARGETS = test_vgt_mfem_units vgt_mfem_mesh vgt_mfem_convergence vgt_mfem_example vgt_mfem_benchmark vgt_mfem_periodic vgt_mfem_periodic_eigen vgt_mfem_periodic_convergence

.SUFFIXES:
.SUFFIXES: .o .cpp
.PHONY: all clean report report-refresh

# Remove built-in implicit rule for %.cpp → %
%: %.cpp

# Build rule — same as examples/makefile, but with repo-owned VGT deps.
%: %.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $(VGT_INCLUDE) $(VGT_DEFINES) $< -o $@ $(MFEM_LIBS)

all: $(TARGETS)

report: vgt_mfem_benchmark vgt_mfem_convergence
	python3 tools/update_vgt_mfem_summary.py --run

report-refresh:
	python3 tools/update_vgt_mfem_summary.py --skip-run

# Generate an error message if the MFEM library is not built
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built. Run 'make parallel' in $(MFEM_DIR) first.)

clean:
	rm -f $(TARGETS) *.o *~
	rm -rf *.dSYM
