#!/bin/bash
#SBATCH --nodes=3
#SBATCH --ntasks=120
#SBATCH --time=00:30:00
#SBATCH --job-name python_analysis
#SBATCH --output=log_%j.txt

cd $SLURM_SUBMIT_DIR

module load NiaEnv/2019b intelpython3

source activate myCondaEnvironment

#mpirun -np 27 python 0_mpi_separator.py
#mpirun -np 108 python 1_compute_rmsd.py
#mpirun -np 108 python 2_compute_states.py
#mpirun -np 108 python 3_compute_h-bonds.py
#mpirun -np 108 python 5_compute_contacts.py
#mpirun -np 108 python 6_compute_rmsf.py
mpirun -np 108 python 7_compute_deviation.py
