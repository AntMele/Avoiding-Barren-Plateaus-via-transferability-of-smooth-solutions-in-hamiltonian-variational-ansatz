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

from ansatz_lib import makecircXXZ, makecircXXZS
from hamiltonian_lib import makeXX, makeYY, makeZZ, makeX, makeXXfirstEO, makeYYfirstEO, makeZZfirstEO
from BP_lib import CostNEW
from myfunctions import fidel, residual, makeinit

from qiskit.algorithms.optimizers import L_BFGS_B

##### part 1

# whether to run the ED code
run_ED = True
ComputeState=True

# model parameters
deltaY=float(sys.argv[1])
deltaZ=float(sys.argv[2])

# output file
outfile_ED = f"temp_results_XYZ/ED_XYZ_deltaY={deltaY}_deltaZ={deltaZ}.npz"

# size bounds
Lmin=4
Lmax=int(sys.argv[3])
# Initial guess size for transferability
Lguess=int(sys.argv[4])

Pmaxguess=int(sys.argv[5])
iterguess=int(sys.argv[6])
P=int(sys.argv[7])
# trasferability bounds
LminT=Lguess+2
LmaxT=Lmax
if run_ED:

    eminn=[]
    emaxx=[]
    gsED=[]
    for n in range(Lmin,Lmax+1,2):
        
        # define the Hamiltonian
        edges=[]
        for i in range(n):
            edges.append((i,(i+1)%n))
        coeff = np.ones(len(edges))
        Hzz = makeZZ(edges, coeff, n)
        Hyy = makeYY(edges, coeff, n)
        Hxx = makeXX(edges, coeff, n)
        H_tot = Hxx + deltaY * Hyy + deltaZ * Hzz # H_tot includes all edges

        result = NumPyMinimumEigensolver().compute_minimum_eigenvalue(operator=H_tot)
        eminn.append(result.eigenvalue.real)
        gsEDnow = result.eigenstate.to_matrix(massive=True)
        gsED.append(gsEDnow)
        result = NumPyMinimumEigensolver().compute_minimum_eigenvalue(operator=-H_tot)
        emaxx.append(-1*result.eigenvalue.real)

        ##### save exact GS for fidelity calculation
    	
    # print(eminn)
    eminn=np.array(eminn)
    emaxx=np.array(emaxx)
    gsED=np.array(gsED)

    np.savez(outfile_ED, eminn=eminn, emaxx=emaxx, gsED=gsED)

else:
    # file with ED results
    if os.path.isfile(outfile_ED):
        outfile_ED_data = np.load(outfile_ED, allow_pickle=True) #load saved data
        eminn = outfile_ED_data["eminn"]
        emaxx = outfile_ED_data["emaxx"]
        #gsED = outfile_ED_data["gsED"]
    else:
        raise Exception("No ED results file for these XYZ parameters")


##### part 2



# file with VQE results for Lguess
EdGuess_file = f"temp_results_XYZ/file_Opt_deltaY={deltaY}_deltaZ={deltaZ}_L={Lguess}_Pmax={Pmaxguess}_maxiter={iterguess}.npz" 
if os.path.isfile(EdGuess_file):
    EdGuess_file_data = np.load(EdGuess_file, allow_pickle=True) #load saved data
    optparlist=EdGuess_file_data["optparlist"]
    optparlist=optparlist
else:
    raise Exception("No VQE results file for these XYZ parameters")



# output file with transferability results
file_TR = f"temp_results_XYZ/transf_XYZ_Lguess={Lguess}_deltaY={deltaY}_deltaZ={deltaZ}_P={P}_maxiter={iterguess}.npz"

# pars_init = optparlistALL[(Lguess-Lmin)//2][P-1] # optimal parameters for given Lguess and P
pars_init = optparlist[P-1] # optimal parameters for given Lguess and P

narray=[]
guess_res_en=[]
guess_fid=[]
backend = Aer.get_backend('statevector_simulator')

for n in range(LminT,LmaxT+1,2):
    # could be copied from the previous part of the code for increased efficiency
    # define the Hamiltonian
    edges=[]
    for i in range(n):
        edges.append((i,(i+1)%n))
    coeff = np.ones(len(edges))
    Hzz = makeZZ(edges, coeff, n)
    Hyy = makeYY(edges, coeff, n)
    Hxx = makeXX(edges, coeff, n)
    H_tot = Hxx + deltaY * Hyy + deltaZ * Hzz # H_tot includes all edges
    
    emin=eminn[(n-Lmin)//2]
    emax=emaxx[(n-Lmin)//2]

    circTry = makecircXXZ(P,n,deltaY,deltaZ)

    ##### load exact GS, compute fidelity, save fidelity to the same file_TR

    ##### save the states psi too; they will be the starting points for a refined new
    ##### optimization, to be performed in a new file
    ##### DO the SAME for the TFIM

    # compute the expectation value
    qqq=circTry.assign_parameters(pars_init)
    psi=CircuitStateFn(qqq)
    if (ComputeState==True):
        fid=fidel(psi.to_matrix(massive=True),gsED[(n-Lmin)//2])
        print("FIDELITY="+str(fid))
    
    measurable_expression = StateFn(H_tot, is_measurement=True).compose(psi)
    expectation = AerPauliExpectation().convert(measurable_expression)
    sampler = CircuitSampler(backend).convert(expectation)
    temp=(sampler.eval().real)
    #
    narray.append(n)
    guess_res_en.append(residual(temp,emin,emax))
    guess_fid.append(fid)


np.savez(file_TR, narray=np.array(narray), guess_res_en=np.array(guess_res_en), guess_fid=np.array(guess_fid))
