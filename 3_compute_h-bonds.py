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
timesteps = 2001
step = 100

## Setting up local indexing and sizes

local_n = N // size

ind_start = rank * local_n
if rank != size - 1 :
    ind_end = (rank+1)*local_n
else:
    ind_end = N

### Allocate arrays and create data

if rank == 0:
    print('H-bonds analysis...')
    hbonds_array = np.empty((N, timesteps), dtype=np.float)
else:
    hbonds_array = None

local_hbonds_array = np.zeros((local_n, timesteps), dtype=np.float)


## Do smth

folder = 'data/charmm36m/2us/'
for i, j in enumerate(range(ind_start, ind_end)):
    struct = folder + 'chain.pdb'
    traj = folder + 'chain_{}.xtc'.format(j+1)
    print(traj)
    sys.stdout.flush()

    #local_hbonds_array[i] = np.zeros(timesteps, dtype=np.float) + rank
    
    system = md.Universe(struct, traj)
    ## compute information about hbonds and write it to the 'hb.timeseries'
    hb = hbonds.HydrogenBondAnalysis(system)
    hb.run(step = step)
    
    ## go through the 'hb.timeseries' file and calculate number of bonds for each time frame 
    ## (it's the length of array frame)
    hb_number = []
    for frame in hb.timeseries:
        hb_number.append(len(frame))
    hb_number = np.array(hb_number)    
    local_hbonds_array[i] = hb_number
    
comm.barrier()

## Gather data
comm.Gather(local_hbonds_array, hbonds_array, root=0)


## save or print
if rank == 0:
    print(hbonds_array.shape)
    npy_file = '3_h-bonds/charmm36m/hb_2us'
    np.save(npy_file, hbonds_array)
    print('H-bonds are saved to ' + npy_file)