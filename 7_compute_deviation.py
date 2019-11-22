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


N = 108
N_ref = 16
N_atoms = 724

stride = 10

#N_total = 200001
N_total = 80713
timesteps = 8072
## Setting up local indexing and sizes

local_n = N // size

ind_start = rank * local_n
if rank != size - 1 :
    ind_end = (rank+1)*local_n
else:
    ind_end = N

### Allocate arrays and create data

if rank == 0:
    print('Deviation analysis')
    deviation = np.empty((N, timesteps, N_atoms), dtype=np.float)
else:
    deviation = None

local_deviation = np.zeros((local_n, timesteps, N_atoms), dtype=np.float)

#state_selection = 'protein and resid 350 367 368 369 381 382 384 413 414 415 416 417 418 and not type H'

structure_names = ['AAAA', 'AAAB', 'AABA', 'AABB', 'ABAA', 'ABAB', 'ABBA', 'ABBB', 'BAAA', 'BAAB', 'BABA', 'BABB', 'BBAA', 'BBAB', 'BBBA', 'BBBB']
references = []
for name in structure_names:
    #u = md.Universe('isolated_289/new_' + name + '_fixed.pdb')
    u = md.Universe('isolated_289/gromacs_amber/' + name + '_fixed_gmx.gro')
    references.append(u)

#folder = 'data/charmm36m/2us/'
folder = 'data/amber/T3/'

states = None
if rank == 0:
    rmsd = np.load('1_rmsd/amber/T3_min_rmsd_800ns.npy')
    states = np.argmin(rmsd, axis = 1)

local_states = np.empty(N_total, dtype=np.int)
comm.Scatter(states, local_states, root=0)
## Do smth


for i, j in enumerate(range(ind_start, ind_end)):
    struct = folder + 'chain.pdb'
    traj = folder + 'chain_{}.xtc'.format(j+1)
    print(j, traj)
    sys.stdout.flush()
    system = md.Universe(struct, traj)
    counter = 0
    selection = 'not name H*'
    system_heavy = system.select_atoms(selection)
    for t in system.trajectory[::stride]:
        ref_index = local_states[t.frame]
        ref = references[ref_index]
        align.alignto(system, ref, select="not name H*", weights="mass")
        diff = system_heavy.positions - ref.select_atoms(selection).positions
        local_deviation[i, counter] = np.linalg.norm(diff, axis = 1)
        counter += 1
        if rank == 0:
            print(counter)
    
comm.barrier()

## Gather data
comm.Gather(local_deviation, deviation, root=0)

## save or print
if rank == 0:
    print(deviation.shape)
    #npy_file = '7_deviation/charmm36m/deviation_2us_all'
    npy_file = '7_deviation/amber/T3_deviation_800ns_all'
    np.save(npy_file, deviation)
    print('Deviations are saved to ' + npy_file)
