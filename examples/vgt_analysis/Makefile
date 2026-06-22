# VGT analysis — built against the MFEM parallel library.
# Follows the same pattern as examples/makefile.
#
# Usage (from this directory):
#   make
#   make test_vgt_mfem_units
#   make VGT_DIR=/path/to/vgt_all_bundle test_vgt_mfem_units
#   make clean

THIS_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))

# ── MFEM paths (mirrors examples/makefile exactly) ────────────────────────────
MFEM_DIR       ?= ../..
MFEM_BUILD_DIR  ?= ../..
MFEM_INSTALL_DIR ?= ../../mfem
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

# MFEM_FLAGS and MFEM_LIBS are now set by config.mk.
# If MFEM does not provide a default MPI compiler in PATH, fall back to a
# common Homebrew/MPICH install location.
MFEM_CXX ?= $(or $(shell command -v mpicxx 2>/dev/null),/usr/local/bin/mpicxx)

# ── VGT bundle discovery ──────────────────────────────────────────────────────
VGT_DIR ?= $(firstword \
  $(wildcard $(THIS_DIR)/../../../../vgt_analysis/vgt_all_bundle) \
  $(wildcard $(HOME)/Documents/vgt_analysis/vgt_all_bundle))

ifeq ($(strip $(VGT_DIR)),)
$(error Could not locate vgt_all_bundle. Set VGT_DIR=/absolute/path/to/vgt_all_bundle)
endif

VGT_DIR := $(abspath $(VGT_DIR))
VGT_CPP_BUILD_DIR ?= $(VGT_DIR)/cpp/build
VGT_DATA_DIR      ?= $(VGT_DIR)/data
VGT_LAPACKE_ROOT  ?= $(VGT_CPP_BUILD_DIR)/_deps/lapack_ref-build
VGT_LAPACKE_SRC_INC ?= $(VGT_CPP_BUILD_DIR)/_deps/lapack_ref-src/LAPACKE/include
VGT_EIGEN_DIR ?= $(firstword $(foreach d,/usr/local/include/eigen3 /opt/homebrew/include/eigen3 /usr/include/eigen3,$(if $(wildcard $(d)/Eigen/Core),$(d))))

VGT_INCLUDE = -I$(VGT_DIR)/cpp/include
ifneq ($(strip $(VGT_EIGEN_DIR)),)
VGT_INCLUDE += -I$(VGT_EIGEN_DIR)
endif
ifneq ($(wildcard $(VGT_LAPACKE_ROOT)/include/lapacke.h),)
VGT_INCLUDE += -I$(VGT_LAPACKE_ROOT)/include
endif
ifneq ($(wildcard $(VGT_LAPACKE_SRC_INC)/lapacke.h),)
VGT_INCLUDE += -I$(VGT_LAPACKE_SRC_INC)
endif

VGT_DEFINES = -DVGT_DATA_DIR=\"$(VGT_DATA_DIR)\"

VGT_EXTRA_LIBS =
ifeq ($(findstring lapacke,$(MFEM_LIBS)),)
ifneq ($(wildcard $(VGT_LAPACKE_ROOT)/lib/liblapacke.a),)
VGT_EXTRA_LIBS += $(VGT_LAPACKE_ROOT)/lib/liblapacke.a \
                  $(VGT_LAPACKE_ROOT)/lib/liblapack.a \
                  $(VGT_LAPACKE_ROOT)/lib/libblas.a
endif
endif

# ── Targets ───────────────────────────────────────────────────────────────────
TARGETS = test_vgt_mfem_units vgt_mfem_mesh vgt_mfem_convergence vgt_mfem_example vgt_mfem_benchmark vgt_mfem_periodic vgt_mfem_periodic_convergence

.SUFFIXES:
.SUFFIXES: .o .cpp
.PHONY: all clean report report-refresh

# Remove built-in implicit rule for %.cpp → %
%: %.cpp

# Build rule — same as examples/makefile; VGT_INCLUDE is the only addition
%: %.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $(VGT_INCLUDE) $(VGT_DEFINES) $< -o $@ $(MFEM_LIBS) $(VGT_EXTRA_LIBS)

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
