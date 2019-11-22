import warnings
warnings.filterwarnings('ignore')

import MDAnalysis as md
import matplotlib.pyplot as plt
import numpy as np

import sys

from MDAnalysis.analysis import rms, hbonds, contacts, distances
plt.style.use('seaborn-poster')

from mpi4py import MPI

def contact_size(system):
    '''The function calculates the size of the vector needed to contain contact map '''
    counter = 0
    for res1 in system.residues:
        for res2 in system.residues:
            if abs(res1.resid - res2.resid) > 1 and res1.resid < res2.resid:
                counter += 1
    return counter


def contact_map(system, stride):
    '''The function creates contact map (0 and 1) for for every pair of aminoacids'''
    cutoff = 5.5
    sel_heavy = 'not name H*'
    
    num_data_points = (len(system.trajectory) // stride) + (0 if (len(system.trajectory) % stride == 0) else 1)
    size = contact_size(system)
    contacts = np.empty((num_data_points, size), dtype=np.float)
    index = 0
    for ts in system.trajectory[::stride]:
        if ts.frame % 100 == 0 :
            output = round(100 * ts.frame / len(system.trajectory), 1)
            print(output,"% complete",end='\r')
            sys.stdout.flush()
        contact_map = np.zeros(size, dtype=np.float)
        i = 0
        for res1 in system.residues:
            for res2 in system.residues:
                if abs(res1.resid - res2.resid) > 1 and res1.resid < res2.resid:
                    dist = distances.distance_array(res1.atoms.select_atoms(sel_heavy).positions, res2.atoms.select_atoms(sel_heavy).positions)
                    if np.amin(dist) < cutoff:
                        contact_map[i] = 1.
                    i += 1
        contacts[index,:] = contact_map
        index += 1
    return contacts


def contact_map2(system, stride):
    '''The function creates contact map (0 and 1) for for every pair of aminoacids'''
    cutoff = 5.5
    sel_heavy = 'not name H*'
    num_data_points = (len(system.trajectory) // stride) + (0 if (len(system.trajectory) % stride == 0) else 1)
    size = contact_size(system)
    contacts = np.empty((num_data_points, size), dtype=np.float)
    slices = []
    min_i = 0
    max_i = 0
    for res in system.residues:
        max_i = min_i + res.atoms.select_atoms(sel_heavy).n_atoms
        slices.append((min_i, max_i))
        min_i = max_i

    #print(slices)
    heavy_system = system.select_atoms(sel_heavy)
    index = 0
    for ts in system.trajectory[::stride]:
        if ts.frame % 100 == 0 :
            output = round(100 * ts.frame / len(system.trajectory), 1)
            print(output,"% complete",end='\r')
            sys.stdout.flush()
        contact_map = np.zeros(size, dtype=np.float)
        contact_matrix = distances.contact_matrix(heavy_system.atoms.positions, cutoff = 5.5)
        counter = 0
        for i, res1 in enumerate(slices):
            i1 = res1[0]
            i2 = res1[1]
            for j, res2 in enumerate(slices):
                if abs(i - j) > 1 and i < j:
                    j1 = res2[0]
                    j2 = res2[1]
                    if np.any(contact_matrix[i1:i2, j1:j2]):
                        contact_map[counter] = 1.
                    counter += 1
        contacts[index,:] = contact_map
        index += 1
    return contacts


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


N = 108
timesteps = 20001 #16688 #8072
stride = 10

## Setting up local indexing and sizes

local_n = N // size

ind_start = rank * local_n
if rank != size - 1 :
    ind_end = (rank+1)*local_n
else:
    ind_end = N

### Allocate arrays and create data

if rank == 0:
    q_array = np.empty((N, timesteps), dtype=np.float)
else:
    q_array = None

local_q_array = np.zeros((local_n, timesteps), dtype=np.float)

## Do smth

v1 = contact_map(md.Universe('isolated_289/new_AAAA_fixed.pdb'), 1)
s = np.sum(v1)
folder = 'data/charmm36m/2us/'

for i, j in enumerate(range(ind_start, ind_end)):
    struct = folder + 'chain.pdb'
    traj = folder + 'chain_{}.xtc'.format(j+1)
    print(rank, traj)
    sys.stdout.flush()
    system = md.Universe(struct, traj)
    ## compute information about native contacts
    v2 = contact_map2(system, stride)
    q = np.dot(v1, v2.T) / s    
    local_q_array[i] = q
    
comm.barrier()

## Gather data
comm.Gather(local_q_array, q_array, root=0)

## save or print
if rank == 0:
    print(q_array.shape)
    npy_file = '5_native-contacts/charmm36m/q_2us'
    np.save(npy_file, q_array)
    print('Native contacts are saved to ' + npy_file)