# VGT analysis — built against the MFEM parallel library.
# Follows the same pattern as examples/makefile.
#
# Usage (from this directory):
#   make MFEM_CXX=/usr/local/bin/mpicxx
#   make MFEM_CXX=/usr/local/bin/mpicxx test_vgt_mfem_units
#   make clean

# ── MFEM paths (mirrors examples/makefile exactly) ────────────────────────────
MFEM_DIR       ?= ../..
MFEM_BUILD_DIR  ?= ../..
MFEM_INSTALL_DIR ?= ../../mfem
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

# MFEM_FLAGS and MFEM_LIBS are now set by config.mk.
# Because MFEM was rebuilt with MFEM_USE_LAPACK=YES pointing to the VGT
# pre-built LAPACK, MFEM_FLAGS already carries the LAPACKE -I paths and
# MFEM_LIBS already carries liblapacke.a / liblapack.a / libblas.a and the
# Intel Fortran runtime — no extra link flags needed here.

# ── VGT header-only libraries (the only addition vs examples/makefile) ─────────
VGT_DIR     ?= /Users/victorzendejaslopez/Documents/vgt_analysis/vgt_all_bundle
VGT_INCLUDE  = -I$(VGT_DIR)/cpp/include

# ── Targets ───────────────────────────────────────────────────────────────────
TARGETS = test_vgt_mfem_units vgt_mfem_mesh vgt_mfem_convergence vgt_mfem_example vgt_mfem_vis

.SUFFIXES:
.SUFFIXES: .o .cpp
.PHONY: all clean

# Remove built-in implicit rule for %.cpp → %
%: %.cpp

# Build rule — same as examples/makefile; VGT_INCLUDE is the only addition
%: %.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $(VGT_INCLUDE) $< -o $@ $(MFEM_LIBS)

all: $(TARGETS)

# Generate an error message if the MFEM library is not built
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built. Run 'make parallel' in $(MFEM_DIR) first.)

clean:
	rm -f $(TARGETS) *.o *~
	rm -rf *.dSYM
