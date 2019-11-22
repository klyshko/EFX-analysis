import warnings
warnings.filterwarnings('ignore')

import mdtraj as md
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-poster')

import sys
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def ss(trajectory):
    return md.compute_dssp(trajectory, simplified=True)

N = 108
folder = 'data/charmm36m/2us/'


# I guess you have to run with 27 processes???

print('SS analysis is crazy...')


## Setting up local indexing and sizes

local_n = N // size

ind_start = rank * local_n
if rank != size - 1 :
    ind_end = (rank+1)*local_n
else:
    ind_end = N

### Allocate arrays and create data

if rank == 0:
	struct = folder + 'chain.pdb'
	traj = folder + 'chain_1.xtc'
	t = md.load(traj, top=struct)
	
	time = t.xyz.shape[0]
	residues = t.n_residues
	type_ = ss(t).dtype

	ss_array = np.chararray((N, time,  residues))

else:

    ss_array = None
    time = None
    residues = None
    type_ = None


time = comm.bcast(time, root=0)
residues = comm.bcast(residues, root=0)
type_ = comm.bcast(type_, root=0)

print(time, residues, type_)
sys.stdout.flush()

local_ss_array = np.chararray((local_n, time,  residues))


for i, j in enumerate(range(ind_start, ind_end)):
    struct = folder + 'chain.pdb'
    traj = folder + 'chain_{}.xtc'.format(j+1)
    system = md.load(traj, top=struct)
    print(rank, traj)
    sys.stdout.flush()
    sec_str = ss(system)
    sec_str = sec_str.astype(local_ss_array.dtype)
    local_ss_array[i] = sec_str
    
comm.barrier()

## Gather data
comm.Gather(local_ss_array, ss_array, root=0)

## save or print
if rank == 0:
    print(ss_array.shape)
    npy_file = '4_ss/charmm36m/ss_array_2us'
    np.save(npy_file, ss_array)
    print('Secondary structures are saved to ' + npy_file)

