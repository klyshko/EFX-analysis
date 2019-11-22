import warnings
warnings.filterwarnings('ignore')
import sys

import MDAnalysis as md
import matplotlib.pyplot as plt
import numpy as np

from MDAnalysis.analysis import rms, align
plt.style.use('seaborn-poster')

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()



'''
B = 8./3 * np.pi**2 * rmsf**2
'''


N = 108
N_atoms = 724

start = 65000

## Setting up local indexing and sizes

local_n = N // size

ind_start = rank * local_n
if rank != size - 1 :
    ind_end = (rank+1)*local_n
else:
    ind_end = N

### Allocate arrays and create data

if rank == 0:
    print('RMSF analysis...')
    rmsf = np.empty((N, N_atoms), dtype=np.float)
else:
    rmsf = None

local_rmsf = np.zeros((local_n, N_atoms), dtype=np.float)

#folder = 'data/charmm36m/2us/'
folder = 'data/amber/T3/'

for i, j in enumerate(range(ind_start, ind_end)):
    struct = folder + 'chain.pdb'
    traj = folder + 'chain_{}.xtc'.format(j+1)
    print(j, traj)
    sys.stdout.flush()
    system = md.Universe(struct, traj)
    reference = md.Universe(struct)
    aligner = align.AlignTraj(system, reference, select="all", in_memory=True).run()
    R = rms.RMSF(system.select_atoms('not name H*'))
    R.run(start = start, verbose = True)
    local_rmsf[i, :] = R.rmsf

comm.barrier()

## Gather data
comm.Gather(local_rmsf, rmsf, root=0)

## save or print
if rank == 0:
    print(rmsf.shape)
    #npy_file = '6_rmsf/charmm36m/rmsf_2us'
    npy_file = '6_rmsf/amber/T3_rmsf_800ns'
    np.save(npy_file, rmsf)
    print('RMSF are saved to ' + npy_file)
