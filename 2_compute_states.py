import warnings
warnings.filterwarnings('ignore')
import sys

import MDAnalysis as md
import matplotlib.pyplot as plt
import numpy as np

from MDAnalysis.analysis import rms, hbonds
plt.style.use('seaborn-poster')

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


N = 108
timesteps = 200001
N_ref = 16

## Setting up local indexing and sizes

local_n = N // size

ind_start = rank * local_n
if rank != size - 1 :
    ind_end = (rank+1)*local_n
else:
    ind_end = N

### Allocate arrays and create data

if rank == 0:
    print('RMSD (new state definition) analysis...')
    rmsd = np.empty((N, N_ref, timesteps), dtype=np.float)
else:
    rmsd = None

local_rmsd = np.zeros((local_n, N_ref, timesteps), dtype=np.float)

state_selection = 'protein and resid 350 367 368 369 381 382 384 413 414 415 416 417 418 and not type H'

structure_names = ['AAAA', 'AAAB', 'AABA', 'AABB', 'ABAA', 'ABAB', 'ABBA', 'ABBB', 'BAAA', 'BAAB', 'BABA', 'BABB', 'BBAA', 'BBAB', 'BBBA', 'BBBB']
references = []
for name in structure_names:
    u = md.Universe('isolated_289/new_' + name + '_fixed.pdb')
    references.append(u)
## Do smth

for k, ref in enumerate(references):
    ## change folder name here
    folder = 'data/charmm36m/2us/'
    for i, j in enumerate(range(ind_start, ind_end)):
        struct = folder + 'chain.pdb'
        traj = folder + 'chain_{}.xtc'.format(j+1)
        print(k, traj)
        sys.stdout.flush()
        system = md.Universe(struct, traj)
        #R = rms.RMSD(system, reference=ref, select='not name H*')
        R = rms.RMSD(system, reference=ref, select='backbone and not type H', groupselections=[state_selection])
        R.run() 
        rmsd_ = R.rmsd.T   # transpose makes it easier for plotting
        #local_rmsd[i, k] = rmsd_[2]
        local_rmsd[i, k] = rmsd_[3]
    
comm.barrier()

## Gather data
comm.Gather(local_rmsd, rmsd, root=0)


## save or print
if rank == 0:
    print(rmsd.shape)
    npy_file = '1_rmsd/charmm36m/new_state_rmsd_2us'
    np.save(npy_file, rmsd)
    print('RMSD are saved to ' + npy_file)
