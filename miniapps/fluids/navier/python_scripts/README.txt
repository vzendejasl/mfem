Python Scripts Setup and Run Guide
==================================

Scripts covered:
  ComputeSpectraCompressiveVorticalModesParallel.py
  convert_txt_to_hdf5_parallel.py

Key workflow:
  1. Convert MFEM TXT output to structured FFT-ready HDF5.
  2. Run the parallel spectra script on that HDF5 file.

Structured HDF5 schema written by convert_txt_to_hdf5_parallel.py:
  /grid/x
  /grid/y
  /grid/z
  /fields/vx
  /fields/vy
  /fields/vz

This schema is what ComputeSpectraCompressiveVorticalModesParallel.py reads
directly in parallel.


================================================================================
LOCAL MAC
================================================================================

Create env:

  /Users/victorzendejaslopez/anaconda3/bin/conda create -y -n heffte-py \
    -c conda-forge \
    python=3.11 cmake make cxx-compiler mpich mpi4py numpy scipy fftw git \
    pkg-config matplotlib pandas h5py

Activate:

  source /Users/victorzendejaslopez/anaconda3/etc/profile.d/conda.sh
  conda activate heffte-py

Build HeFFTe:

  git clone --depth 1 https://github.com/icl-utk-edu/heffte.git third_party/heffte

  cmake -S third_party/heffte -B third_party/heffte/build \
    -D CMAKE_BUILD_TYPE=Release \
    -D BUILD_SHARED_LIBS=ON \
    -D CMAKE_INSTALL_PREFIX=/Users/victorzendejaslopez/Documents/MFEM/third_party/heffte/install \
    -D CMAKE_C_COMPILER=$CONDA_PREFIX/bin/mpicc \
    -D CMAKE_CXX_COMPILER=$CONDA_PREFIX/bin/mpicxx \
    -D FFTW_ROOT=$CONDA_PREFIX \
    -D Heffte_ENABLE_FFTW=ON \
    -D Heffte_ENABLE_PYTHON=ON \
    -D Python_EXECUTABLE=$CONDA_PREFIX/bin/python

  cmake --build third_party/heffte/build -j4
  cmake --install third_party/heffte/build

Exports:

  export PYTHONPATH=/Users/victorzendejaslopez/Documents/MFEM/third_party/heffte/install/share/heffte/python:$PYTHONPATH

Verify:

  python -c "import heffte; print(heffte.__version__)"
  python -c "import h5py; print('h5py mpi =', h5py.get_config().mpi)"

Run:

  mpirun -n 4 python ComputeSpectraCompressiveVorticalModesParallel.py \
    your_data.h5 --backend fftw --no-plot

  mpirun -n 4 python convert_txt_to_hdf5_parallel.py your_data.txt


================================================================================
TUOLUMNE
================================================================================

This is the cleaned-up working path.

Important discovery:
  On Tuolumne, MPI-enabled h5py only worked correctly after mpi4py and h5py
  were both linked against the same GNU MPI stack.

Working linkage:
  mpi4py -> libmpi_gnu.so.12
  h5py   -> libmpi_gnu.so.12

Broken linkage we saw earlier:
  mpi4py -> libmpi_gnu.so.12
  h5py   -> libmpi_cray.so.12

That mismatch was the reason for:
  Attempting to use an MPI routine (internal_Comm_dup) before initializing or after finalizing MPICH


--------------------------------------------------------------------------------
1. Create the env
--------------------------------------------------------------------------------

  conda create -n heffte-py-tuo python=3.11 -y
  conda activate heffte-py-tuo
  conda install -c conda-forge numpy scipy matplotlib pandas cython cmake git -y


--------------------------------------------------------------------------------
2. Load modules
--------------------------------------------------------------------------------

Use the same stack for build and run:

  module purge
  module load craype-x86-trento
  module load gcc/13.3.1-magic
  module load cray-mpich/9.1.0
  module load cray-hdf5-parallel
  module load cray-fftw/3.3.10.11
  module load rocm/6.4.0
  module load cmake/3.29.2


--------------------------------------------------------------------------------
3. Build mpi4py against GNU MPI
--------------------------------------------------------------------------------

  pip uninstall -y mpi4py
  MPICC=$(which mpicc) pip install --no-cache --force-reinstall \
    --no-build-isolation --no-binary=mpi4py mpi4py

Verify:

  ldd $(python -c "import mpi4py.MPI as m; print(m.__file__)") | egrep 'mpi|mpich'

Expected:
  libmpi_gnu.so.12


--------------------------------------------------------------------------------
4. Build MPI-enabled h5py against GNU parallel HDF5
--------------------------------------------------------------------------------

Use the GNU HDF5 variant, not the Cray one:

  export HDF5_MPI=ON
  export HDF5_DIR=/opt/cray/pe/hdf5-parallel/1.14.3.7/gnu/12.2
  export LD_LIBRARY_PATH=/opt/cray/pe/lib64/cce:$HDF5_DIR/lib:$LD_LIBRARY_PATH

  pip uninstall -y h5py
  env CC=$(which mpicc) \
      HDF5_MPI=ON \
      HDF5_DIR=/opt/cray/pe/hdf5-parallel/1.14.3.7/gnu/12.2 \
      pip install --no-cache --force-reinstall --no-build-isolation --no-binary=h5py h5py

Verify:

  python -c "import h5py; print(h5py.__version__)"
  python -c "import h5py; print(h5py.get_config().mpi)"
  ldd $(python -c "import h5py.h5 as h; print(h.__file__)") | egrep 'mpi|mpich|hdf5'

Expected:
  h5py.get_config().mpi -> True
  libmpi_gnu.so.12
  /opt/cray/pe/hdf5-parallel/1.14.3.7/gnu/12.2/lib/libhdf5.so.310

If h5py resolves to the conda env's own libhdf5*.so instead of the GNU Cray HDF5:
  move the conflicting conda HDF5 libraries aside and keep the GNU HDF5 path
  first on LD_LIBRARY_PATH.


--------------------------------------------------------------------------------
5. Build HeFFTe
--------------------------------------------------------------------------------

  export FFTW_PATH=/opt/cray/pe/fftw/3.3.10.11/x86_trento
  export MPICC=$(which mpicc)
  export MPICXX=$(which mpicxx)

  git clone --depth 1 https://github.com/icl-utk-edu/heffte.git third_party/heffte

  cmake -S third_party/heffte -B third_party/heffte/build \
    -D CMAKE_BUILD_TYPE=Release \
    -D BUILD_SHARED_LIBS=ON \
    -D CMAKE_INSTALL_PREFIX=$PWD/third_party/heffte/install \
    -D CMAKE_C_COMPILER=${MPICC} \
    -D CMAKE_CXX_COMPILER=${MPICXX} \
    -D FFTW_ROOT=${FFTW_PATH} \
    -D Heffte_ENABLE_FFTW=ON \
    -D Heffte_ENABLE_PYTHON=ON \
    -D Python_EXECUTABLE=$(which python)

  cmake --build third_party/heffte/build -j8
  cmake --install third_party/heffte/build

If install fails because of missing CMakeRelink libheffte.so file, use:

  mkdir -p $PWD/third_party/heffte/install/lib64
  cp -P third_party/heffte/build/libheffte.so* $PWD/third_party/heffte/install/lib64/

Exports if HeFFTe is under the current working directory:

  export PYTHONPATH=$PWD/third_party/heffte/install/share/heffte/python:$PYTHONPATH
  export LD_LIBRARY_PATH=$PWD/third_party/heffte/install/lib64:$LD_LIBRARY_PATH

Exports for the installed path used on Tuolumne in our tests:

  export PYTHONPATH=/p/lustre5/zendejas/third_party/heffte/install/share/heffte/python:$PYTHONPATH
  export LD_LIBRARY_PATH=/p/lustre5/zendejas/third_party/heffte/install/lib64:$LD_LIBRARY_PATH

Verify:

  python -c "import heffte; print(heffte.__version__)"
  python -c "import heffte; print(heffte.heffte_config.libheffte_path)"


--------------------------------------------------------------------------------
6. Converter validation that worked
--------------------------------------------------------------------------------

After fixing mpi4py + h5py linkage, this worked on Tuolumne:

  srun -l -n 2 python ~/Documents/mfem_build/mfem/miniapps/fluids/navier/python_scripts/convert_txt_to_hdf5_parallel.py \
    blast_tgv3Dk2r3_SampledData/cycle_6358/velocity_sampled_data_uniform_interpolated_cycle_6358.txt

Observed result:
  - parallel TXT read worked
  - structured grid discovered correctly
  - parallel HDF5 write worked
  - parallel HDF5 verification read worked
  - TKE and row counts matched exactly


--------------------------------------------------------------------------------
7. Copy-paste batch script: converter
--------------------------------------------------------------------------------

  #!/bin/bash
  #SBATCH -N 1
  #SBATCH --ntasks-per-node 36
  #SBATCH -t 02:00:00

  module purge
  module load craype-x86-trento
  module load gcc/13.3.1-magic
  module load cray-mpich/9.1.0
  module load cray-hdf5-parallel
  module load cray-fftw/3.3.10.11
  module load rocm/6.4.0
  module load cmake/3.29.2

  source /g/g11/zendejas/anaconda3/etc/profile.d/conda.sh
  conda activate heffte-py-tuo

  export HDF5_MPI=ON
  export HDF5_DIR=/opt/cray/pe/hdf5-parallel/1.14.3.7/gnu/12.2
  export PYTHONPATH=/p/lustre5/zendejas/third_party/heffte/install/share/heffte/python:$PYTHONPATH
  export LD_LIBRARY_PATH=/p/lustre5/zendejas/third_party/heffte/install/lib64:/opt/cray/pe/lib64/cce:$HDF5_DIR/lib:$LD_LIBRARY_PATH
  export OMP_NUM_THREADS=1

  python -c "import h5py; print('h5py mpi =', h5py.get_config().mpi)"
  python -c "import mpi4py.MPI as m; print('mpi4py OK')"

  srun -l -n 36 python ~/Documents/mfem_build/mfem/miniapps/fluids/navier/python_scripts/convert_txt_to_hdf5_parallel.py \
    your_data.txt


--------------------------------------------------------------------------------
8. Copy-paste batch script: spectra
--------------------------------------------------------------------------------

  #!/bin/bash
  #SBATCH -N 2
  #SBATCH --ntasks-per-node 112
  #SBATCH -t 02:00:00

  module purge
  module load craype-x86-trento
  module load gcc/13.3.1-magic
  module load cray-mpich/9.1.0
  module load cray-hdf5-parallel
  module load cray-fftw/3.3.10.11
  module load rocm/6.4.0
  module load cmake/3.29.2

  source /g/g11/zendejas/anaconda3/etc/profile.d/conda.sh
  conda activate heffte-py-tuo

  export HDF5_MPI=ON
  export HDF5_DIR=/opt/cray/pe/hdf5-parallel/1.14.3.7/gnu/12.2
  export PYTHONPATH=/p/lustre5/zendejas/third_party/heffte/install/share/heffte/python:$PYTHONPATH
  export LD_LIBRARY_PATH=/p/lustre5/zendejas/third_party/heffte/install/lib64:/opt/cray/pe/lib64/cce:$HDF5_DIR/lib:$LD_LIBRARY_PATH
  export OMP_NUM_THREADS=1

  python -c "import h5py, heffte; print('h5py mpi =', h5py.get_config().mpi); print('HeFFTe =', heffte.__version__)"

  srun -l -n 224 python ~/Documents/mfem_build/mfem/miniapps/fluids/navier/python_scripts/ComputeSpectraCompressiveVorticalModesParallel.py \
    your_data.h5 --backend fftw --no-plot


================================================================================
NOTES
================================================================================

convert_txt_to_hdf5_parallel.py
  - Reads the TXT file in parallel after rank 0 builds the chunk index.
  - Writes structured FFT-ready HDF5.
  - Verifies TKE and row counts after writing.
  - H5 -> TXT conversion remains serial on rank 0.

ComputeSpectraCompressiveVorticalModesParallel.py
  - For the structured HDF5 schema above, each rank reads its local HDF5 slab directly.
  - FFTs and spectral decomposition are distributed through HeFFTe/MPI.
  - Legacy flat HDF5/TXT still uses rank 0 reconstruction plus scatter.
