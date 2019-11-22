import sys
import warnings
warnings.filterwarnings('ignore')

import MDAnalysis as md
import matplotlib.pyplot as plt
import numpy as np
import os.path

from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.analysis.base import AnalysisFromFunction

from MDAnalysis.analysis import rms
plt.style.use('seaborn-poster')

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def find_chains(selection):
    N = 108 
    chains = [] 
    for i in range(N):
        start =  i * 1473
        end = start + 1473
        chain_i = selection[start:end]
        chains.append(chain_i)
    return chains


folder = 'data/charmm36m/2us/'

folder_data = '../289/MDRun/'
struct = folder_data + 'protein_firstframe.gro'
traj = folder_data + 'protein_traj.xtc'
print(folder, struct, traj)
sys.stdout.flush()
u = md.Universe(struct, traj, in_memory=False)
proteins = u.select_atoms('protein')
chains = find_chains(proteins)

N = 108

## Setting up local indexing and sizes

local_n = N // size

ind_start = rank * local_n
if rank != size - 1 :
    ind_end = (rank+1)*local_n
else:
    ind_end = N


if rank == 0:
    inds = np.arange(N)
else:
    inds = None

local_inds = np.empty(local_n, dtype = np.int)
comm.Scatter(inds, local_inds, root = 0)

#print(chains)


for i in local_inds:
    filename = folder + 'chain_{}.xtc'.format(i+1)
    print(filename)
    sys.stdout.flush()
    protein = chains[i]
    if not os.path.exists(filename):
        u2 = md.Merge(protein).load_new(AnalysisFromFunction(lambda ag: ag.positions.copy(), protein).run().results, format=MemoryReader)
        with md.Writer(filename, protein.n_atoms) as W:
            for ts in u2.trajectory:
                W.write(u2)

if rank == 0:
    chains[0].write(folder + 'chain.gro')
    chains[0].write(folder + 'chain.pdb')
