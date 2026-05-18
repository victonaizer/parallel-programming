#!/bin/bash
#SBATCH --job-name=lab5_mpi_matrix
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=00:10:00
#SBATCH --output=results/slurm-%j.out

cd "$SLURM_SUBMIT_DIR"
if [ ! -f Makefile ]; then
    cd ..
fi

module purge
module load intel/mpi5

make
python3 scripts/generate_matrix.py -n "${MATRIX_SIZE:-2000}" -d data
mpirun -np "${SLURM_NTASKS:-40}" ./matrix_mul_mpi_super data/matrix_a.txt data/matrix_b.txt data/result.txt
python3 scripts/verify.py data/matrix_a.txt data/matrix_b.txt data/result.txt
