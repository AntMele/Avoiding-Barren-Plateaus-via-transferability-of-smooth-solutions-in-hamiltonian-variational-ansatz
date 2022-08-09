import numpy as np
import sys
# import networkx as nx
import time
from qiskit.circuit import Parameter
from qiskit.utils import QuantumInstance
from qiskit import Aer
from qiskit.opflow import CircuitStateFn
from qiskit.algorithms import VQE, NumPyMinimumEigensolver

from ansatz_lib import  makecirc_LTFIM, makecirc_LTFIM3
from hamiltonian_lib import termZ, makeZ, makeZZ, makeX, makeZZfirst_TFIM, makeZZfirstEO, makeXfirst
from BP_lib import CostNEW
from myfunctions import fidel, residual, makeinit

from qiskit.algorithms.optimizers import L_BFGS_B


#Input___ MAIN Optimization
prep=2 # number of parameters per layer
INTERP=True

# model parameters
g = float(sys.argv[1])
h= float(sys.argv[2])
maxfun=10**10 # Default is 100, but 1000 for educated guess (N=8) and BP
maxiter=int(sys.argv[3])
Lmin=int(sys.argv[4])
Lmax=Lmin
L=Lmin
Pmax= int(sys.argv[5])
Pmin=Pmax
ans=str(sys.argv[6])
sign=int(sys.argv[7])

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

optnow=L_BFGS_B(maxfun=maxfun, maxiter=maxiter, ftol=2.220446049250313e-15, iprint=-1, eps=1e-08,  max_evals_grouped=10**10)
# output file
outfileOpt = f"temp_results_LTFIM/file_Opt_StartRand_LTFIM_g={g}_h={h}_{ans}_L={Lmin}_P={Pmax}_maxiter={maxiter}_sign={sign}.npz"

# for saving results


    
en=[]
resen=[]
fid=[]
optparlist=[]
first_res=[]

Hzz_first = makeZZfirst_TFIM(L)
Hx_first = makeXfirst(L)
Hz_first = termZ(0,L)
H_first = sign*Hzz_first - g * Hx_first - h * Hz_first

edges=[]
for i in range(L):
    edges.append((i,(i+1)%L))
coeff = np.ones(len(edges))
Hzz = makeZZ(edges, coeff, L)
Hx = makeX(L)
Hz = makeZ(L)
H_tot =  sign*Hzz - g * Hx - h * Hz # H_tot includes all edges


ed_result = NumPyMinimumEigensolver().compute_minimum_eigenvalue(operator=H_tot) # exact diag. gs energy
emin = ed_result.eigenvalue.real 
psi0 = ed_result.eigenstate.to_matrix(massive=True)
#
ed_result = NumPyMinimumEigensolver().compute_minimum_eigenvalue(operator=-H_tot)
emax= -ed_result.eigenvalue.real



# quantum instance
qi=QuantumInstance(backend=Aer.get_backend('qasm_simulator'))

for P in range(Pmin,Pmax+1):
    
    circ = ansatz(P,L)
    values = []
    def store_intermediate_result(eval_count, parameters, mean, std):
        """
        Alert: values is non-local in this scope
        """
        values.append(mean)
        if(eval_count%100==0):
            print(repr(np.array(parameters)))
            print(str(eval_count)+": residual_en="+str(residual((L)*mean,emin,emax)))
    
    init = np.random.rand(prep*P)*2*np.pi 
    vqe = VQE(ansatz=circ, optimizer=optnow, max_evals_grouped=1000000, initial_point=init,callback=store_intermediate_result,quantum_instance=qi,include_custom=True)

    result = vqe.compute_minimum_eigenvalue(operator=H_first)    
    circ=ansatz(P,L)
    qqq=circ.assign_parameters(result.optimal_point)
    psi=CircuitStateFn(qqq)
        
    en.append((L)*result.optimal_value)
    resen.append(residual( ((L)*result.optimal_value), emin, emax))
    fid.append(fidel(psi.to_matrix(massive= True),psi0))
    optparlist.append(result.optimal_point)
    first_res.append(residual((L)*values[0],emin,emax))
    


np.savez(outfileOpt, enALL=en, resen=resen, fidALL=fid, optparlist=optparlist, first_res=first_res, emin=emin, emax=emax)
   


