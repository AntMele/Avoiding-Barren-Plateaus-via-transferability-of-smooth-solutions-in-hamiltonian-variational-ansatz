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

from ansatz_lib import makecirc_LTFIM, makecirc_LTFIM3
from hamiltonian_lib import makeXX, makeYY, makeZZ, termZ, makeZ, makeX, makeXXfirstEO, makeYYfirstEO, makeZZfirstEO
from BP_lib import CostNEW
from myfunctions import fidel, residual, makeinit

from qiskit.algorithms.optimizers import L_BFGS_B

##### part 1

# whether to run the ED code
run_ED = False
ComputeState=True

# model parameters
g = float(sys.argv[1])
h= float(sys.argv[2])
# size bounds
Lmin=4
Lmax=int(sys.argv[3])
# Initial guess size for transferability
Lguess=int(sys.argv[4])
Pmaxguess=int(sys.argv[5])
iterguess=int(sys.argv[6])
P=int(sys.argv[7])
ans=str(sys.argv[8])
sign=int(sys.argv[9])
randN=int(sys.argv[10])
# trasferability bounds
LminT=Lguess+2
LmaxT=Lmax

if(ans=="ans2"):
    prep=2
else:
    prep=3

def ansatz(P,L):
    if(ans=="ans2"):
        return makecirc_LTFIM(P,L,sign,h)
    else:
        return makecirc_LTFIM3(P,L) 

# output file
outfile_ED = f"temp_results_LTFIM/ED_LTFIM_g={g}_h={h}_{ans}_sign={sign}.npz"


if os.path.isfile(outfile_ED):
    outfile_ED_data = np.load(outfile_ED, allow_pickle=True) #load saved data
    eminn = outfile_ED_data["eminn"]
    emaxx = outfile_ED_data["emaxx"]
    #gsED = outfile_ED_data["gsED"]
else:
    eminn=[]
    emaxx=[]
    gsED=[]
    for L in range(Lmin,Lmax+1):
        
        # define the Hamiltonian
        edges=[]
        for i in range(L):
            edges.append((i,(i+1)%L))
        coeff = np.ones(len(edges))
        Hzz = makeZZ(edges, coeff, L)
        Hx = makeX(L)
        Hz = makeZ(L)
        H_tot = sign* Hzz - g * Hx - h * Hz # H_tot includes all edges

        result = NumPyMinimumEigensolver().compute_minimum_eigenvalue(operator=H_tot)
        eminn.append(result.eigenvalue.real)
        gsEDnow = result.eigenstate.to_matrix(massive=True)
        gsED.append(gsEDnow)
        print(eminn)
        result = NumPyMinimumEigensolver().compute_minimum_eigenvalue(operator=-H_tot)
        emaxx.append(-1*result.eigenvalue.real)
    
    # print(eminn)
    eminn=np.array(eminn)
    emaxx=np.array(emaxx)
    gsED=np.array(gsED)


    np.savez(outfile_ED, eminn=eminn, emaxx=emaxx, gsED=gsED)


##### part 2



# file with VQE results for Lguess
EdGuess_file = f"temp_results_LTFIM/file_Opt_NewStartRand_r{randN}_LTFIM_g={g}_h={h}_{ans}_L={Lguess}_P={P}_maxiter={iterguess}_sign={sign}.npz"

if os.path.isfile(EdGuess_file):
    EdGuess_file_data = np.load(EdGuess_file, allow_pickle=True) #load saved data
    optparlist=EdGuess_file_data["optparlist"]
    optparlist=optparlist
else:
    raise Exception("No VQE results file for these TFIM parameters")



# output file with transferability results
file_TR = f"temp_results_LTFIM/transf_LTFIM_NewStartRand_r{randN}_Lguess={Lguess}_g={g}_h={h}_P={P}_{ans}_maxiter={iterguess}_sign={sign}.npz"

# pars_init = optparlistALL[(Lguess-Lmin)//2][P-1] # optimal parameters for given Lguess and P
pars_init = optparlist # optimal parameters for given Lguess and P


narray=[]
guess_res_en=[]
guess_fid=[]
backend = Aer.get_backend('statevector_simulator')

for L in range(LminT,LmaxT+1):
    # could be copied from the previous part of the code for increased efficiency
    # define the Hamiltonian
    edges=[]
    for i in range(L):
        edges.append((i,(i+1)%L))
    coeff = np.ones(len(edges))
    Hzz = makeZZ(edges, coeff, L)
    Hx = makeX(L)
    Hz = makeZ(L)
    H_tot = sign*Hzz - g * Hx - h * Hz # H_tot includes all edges

    emin=eminn[(L-Lmin)]
    emax=emaxx[(L-Lmin)]

    circTry = ansatz(P,L)

    # compute the expectation value
    qqq=circTry.assign_parameters(pars_init)
    psi=CircuitStateFn(qqq)
    #if (ComputeState==True):
    #    fid=fidel(psi.to_matrix(massive=True),gsED[(L-Lmin)])
    #    print("FIDELITY="+str(fid))
    
    measurable_expression = StateFn(H_tot, is_measurement=True).compose(psi)
    expectation = AerPauliExpectation().convert(measurable_expression)
    sampler = CircuitSampler(backend).convert(expectation)
    temp=(sampler.eval().real)
    #
    narray.append(L)
    guess_res_en.append(residual(temp,emin,emax))
    #print(guess_res_en[-1])
    #guess_fid.append(fid)
    #print(guess_fid[-1])
    #print(gsED[(L-Lmin)])


np.savez(file_TR, narray=np.array(narray), guess_res_en=np.array(guess_res_en))#,guess_fid=np.array(guess_fid))
