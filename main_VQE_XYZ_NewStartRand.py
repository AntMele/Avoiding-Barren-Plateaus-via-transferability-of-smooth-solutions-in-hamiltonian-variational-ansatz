import numpy as np
import sys
# import networkx as nx
import time
from qiskit.circuit import Parameter
from qiskit.utils import QuantumInstance
from qiskit import Aer
from qiskit.opflow import CircuitStateFn
from qiskit.algorithms import VQE, NumPyMinimumEigensolver

from ansatz_lib import makecircXXZ, makecircXXZS
from hamiltonian_lib import makeXX, makeYY, makeZZ, makeX, makeXXfirstEO, makeYYfirstEO, makeZZfirstEO
from BP_lib import CostNEW
from myfunctions import fidel, residual, makeinit

from qiskit.algorithms.optimizers import L_BFGS_B


#Input___ MAIN Optimization
prep=2 # number of parameters per layer
INTERP=True

# model parameters
deltaY=float(sys.argv[1])
deltaZ=float(sys.argv[2])

fact=10**3
maxfun=10**10 # Default is 100, but 1000 for educated guess (N=8) and BP
maxiter=int(sys.argv[3])
Lmin=int(sys.argv[4])
Lmax=Lmin
L=Lmin
P=int(sys.argv[5])
randN=int(sys.argv[6])
#End INPUT__

optnow=L_BFGS_B(maxfun=maxfun, maxiter=maxiter, ftol=2.220446049250313e-15, iprint=-1, eps=1e-08,  max_evals_grouped=10**10)
# output file
outfileOpt = f"temp_results_XYZ/file_Opt_NewStartRand_r{randN}_deltaY={deltaY}_deltaZ={deltaZ}_L={Lmin}_P={P}_maxiter={maxiter}.npz"

# for saving results
EnExact_ALL=[]
EnMaxExact_ALL=[]

 

Hzz_first = makeZZfirstEO(L)
Hyy_first = makeYYfirstEO(L)
Hxx_first = makeXXfirstEO(L)
#
H_first = Hxx_first + deltaY * Hyy_first + deltaZ * Hzz_first # first two edges (even and odd)

edges=[]
for i in range(L):
    edges.append((i,(i+1)%L))
coeff = np.ones(len(edges))
Hzz = makeZZ(edges, coeff, L)
Hyy = makeYY(edges, coeff, L)
Hxx = makeXX(edges, coeff, L)
H_tot = Hxx + deltaY * Hyy + deltaZ * Hzz # H_tot includes all edges

ed_result = NumPyMinimumEigensolver().compute_minimum_eigenvalue(operator=H_tot) # exact diag. gs energy
emin = ed_result.eigenvalue.real 
psi0 = ed_result.eigenstate.to_matrix(massive=True)
#
ed_result = NumPyMinimumEigensolver().compute_minimum_eigenvalue(operator=-H_tot)
emax= -ed_result.eigenvalue.real

# quantum instance
qi=QuantumInstance(backend=Aer.get_backend('qasm_simulator'))


circ = makecircXXZ(P,L,deltaY,deltaZ)
values = []
def store_intermediate_result(eval_count, parameters, mean, std):
    """
    Alert: values is non-local in this scope
    """
    values.append(mean)
    if(eval_count%100==0):
        print(repr(np.array(parameters)))
        print(str(eval_count)+": residual_en="+str(residual((L/2)*mean,emin,emax)))

init = np.random.rand(prep*P)*2*np.pi 
#L-BFGS-B scipy
time_start=time.time()
vqe = VQE(ansatz=circ, optimizer=optnow, max_evals_grouped=1000000, initial_point=init,callback=store_intermediate_result,quantum_instance=qi,include_custom=True)
time_end=time.time()
result = vqe.compute_minimum_eigenvalue(operator=H_first)    
# print("time="+str(time_end-time_start))
#
circ=makecircXXZ(P,L,deltaY,deltaZ)
qqq=circ.assign_parameters(result.optimal_point)
psi=CircuitStateFn(qqq)

# translational fidelity
circS=makecircXXZS(P,L,deltaY,deltaZ)
qqqS=circS.assign_parameters(result.optimal_point)
psiS=CircuitStateFn(qqqS)
#
transl_now=np.linalg.norm(np.vdot(psiS.to_matrix(massive=True),psi.to_matrix(massive=True)))

en=(L/2)*result.optimal_value
resen=residual( ((L/2)*result.optimal_value), emin, emax)
fid=fidel(psi.to_matrix(massive= True),psi0)
transl=transl_now
optparlist=result.optimal_point
first_res=residual((L/2)*values[0],emin,emax)





np.savez(outfileOpt, enALL=en, resen=resen, fidALL=fid, transl=transl,\
     optparlist=optparlist, first_res=first_res, emin=emin, emax=emax)
   


