import numpy as np
import os
import sys
# import networkx as nx
import time
from qiskit.circuit import Parameter
from qiskit.utils import QuantumInstance
from qiskit import Aer
from qiskit.opflow import CircuitStateFn
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.opflow import AerPauliExpectation, CircuitSampler, StateFn, CircuitStateFn


from ansatz_lib import  makecirc_LTFIM, makecirc_LTFIM3
from hamiltonian_lib import termZ, makeZ, makeZZ, makeX, makeZZfirst_TFIM, makeZZfirstEO, makeXfirst
from BP_lib import CostNEW
from myfunctions import fidel, residual, makeinit

from qiskit.algorithms.optimizers import L_BFGS_B


### Part 1)

#Input___ MAIN Optimization
prep=2 # number of parameters per layer

g = float(sys.argv[1])
h= float(sys.argv[2])
maxfun=10**10 # Default is 100, but 1000 for educated guess (N=8) and BP
maxiter=int(sys.argv[3])
ans=str(sys.argv[4])
sign=int(sys.argv[5])
Lguess=int(sys.argv[6]) # 8
P = int(sys.argv[7]) # 10
N_random_inits = int(sys.argv[8]) #50

optnow=L_BFGS_B(maxfun=maxfun, maxiter=maxiter, ftol=2.220446049250313e-15, iprint=-1, eps=1e-08,  max_evals_grouped=10**10)

if(ans=="ans2"):
    prep=2
else:
    prep=3

def ansatz(P,L):
    if(ans=="ans2"):
        return makecirc_LTFIM(P,L,sign,h)
    else:
        return makecirc_LTFIM3(P,L) 



#End INPUT__


# quantum instance
qi=QuantumInstance(backend=Aer.get_backend('qasm_simulator'))


### Part 2)

# size bounds
Lmin=4
Lmax=24
# Load ED file
outfile_ED = f"temp_results_LTFIM/ED_LTFIM_g={g}_h={h}_{ans}_sign={sign}.npz"
if os.path.isfile(outfile_ED):
    outfile_ED_data = np.load(outfile_ED, allow_pickle=True) #load saved data
    eminn = outfile_ED_data["eminn"]
    emaxx = outfile_ED_data["emaxx"]
else:
    raise Exception("No ED results file for these XYZ parameters")

# output file with transferability results
file_TR = f"temp_results_LTFIM/transf_RANDPOINTS_LTFIM_g={g}_h={h}_{ans}_P={P}_sign={sign}.npz"


# trasferability bounds
LminT=10 
LmaxT=24

# results
narray=np.arange(LminT,LmaxT+1)
guess_res_en=np.zeros((N_random_inits, narray.size)) # on row jth you find the transf results for random init jth

for r in range(N_random_inits):

    pars_init = np.random.rand(prep*P)*2*np.pi # optimal parameters for r-th random init. P is fixed

    backend = Aer.get_backend('statevector_simulator')

    for (j,n) in enumerate(range(LminT,LmaxT+1)):
        # could be copied from the previous part of the code for increased efficiency
        # define the Hamiltonian
        edges=[]
        for i in range(n):
            edges.append((i,(i+1)%n))
        coeff = np.ones(len(edges))
        Hzz = makeZZ(edges, coeff, n)
        Hx = makeX(n)
        Hz = makeZ(n)
        H_tot = sign*Hzz - g * Hx - h * Hz # H_tot includes all edges
            
        emin=eminn[(n-Lmin)]
        emax=emaxx[(n-Lmin)]

        circTry = ansatz(P,n)

        # compute the expectation value
        qqq=circTry.assign_parameters(pars_init)
        psi=CircuitStateFn(qqq)
        measurable_expression = StateFn(H_tot, is_measurement=True).compose(psi)
        expectation = AerPauliExpectation().convert(measurable_expression)
        sampler = CircuitSampler(backend).convert(expectation)
        temp=(sampler.eval().real)
        #
        guess_res_en[r,j] = residual(temp,emin,emax)


np.savez(file_TR, narray=narray, guess_res_en=guess_res_en)

