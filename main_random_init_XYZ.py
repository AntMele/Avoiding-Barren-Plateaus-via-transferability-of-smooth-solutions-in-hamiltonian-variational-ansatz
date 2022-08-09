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


### Part 1)

#Input___ MAIN Optimization
prep=2 # number of parameters per layer

# model parameters
deltaY=float(sys.argv[1])
deltaZ=float(sys.argv[1])

maxfun=10**10 # Default is 100, but 1000 for educated guess (N=8) and BP
maxiter=int(sys.argv[3])
Lguess=int(sys.argv[4]) # 8
P = int(sys.argv[5]) # 10
N_random_inits = int(sys.argv[6]) #50
#End INPUT__

optnow=L_BFGS_B(maxfun=maxfun, maxiter=maxiter, ftol=2.220446049250313e-15, iprint=-1, eps=1e-08,  max_evals_grouped=10**10)


# output file
outfileOpt = f"temp_results/random_init_Lguess={Lguess}_XYZ_deltaY={deltaY}_deltaZ={deltaZ}.npz"

# for saving results: avoid useless dynamical allocation of memory
en=np.zeros(N_random_inits)
resen=np.zeros(N_random_inits)
fid=np.zeros(N_random_inits)
transl=np.zeros(N_random_inits)
optparlist=[]
first_res=np.zeros(N_random_inits)


Hzz_first = makeZZfirstEO(Lguess)
Hyy_first = makeYYfirstEO(Lguess)
Hxx_first = makeXXfirstEO(Lguess)
#
H_first = Hxx_first + deltaY * Hyy_first + deltaZ * Hzz_first # first two edges (even and odd)

edges=[]
for i in range(Lguess):
    edges.append((i,(i+1)%Lguess))
coeff = np.ones(len(edges))
Hzz = makeZZ(edges, coeff, Lguess)
Hyy = makeYY(edges, coeff, Lguess)
Hxx = makeXX(edges, coeff, Lguess)
H_tot = Hxx + deltaY * Hyy + deltaZ * Hzz # H_tot includes all edges

ed_result = NumPyMinimumEigensolver().compute_minimum_eigenvalue(operator=H_tot) # exact diag. gs energy
emin = ed_result.eigenvalue.real 
psi0 = ed_result.eigenstate.to_matrix(massive=True)
#
ed_result = NumPyMinimumEigensolver().compute_minimum_eigenvalue(operator=-H_tot)
emax= -ed_result.eigenvalue.real

EnExact = emin
EnMaxExact = emax

# quantum instance
qi=QuantumInstance(backend=Aer.get_backend('qasm_simulator'))

for r in range(N_random_inits):
    
    circ=makecircXXZ(P,Lguess,deltaY,deltaZ)

    values = []
    def store_intermediate_result(eval_count, parameters, mean, std):
        """
        Alert: values is non-local in this scope
        """
        values.append(mean)
        if(eval_count%100==0):
            print(repr(np.array(parameters)))
            print(str(eval_count)+": residual_en="+str(residual((Lguess/2)*mean,emin,emax)))
    
    init = np.random.rand(prep*P)*np.pi/2 
    
    #L-BFGS-B scipy
    vqe = VQE(ansatz=circ, optimizer=optnow,\
            max_evals_grouped=1000000, initial_point=init,callback=store_intermediate_result,quantum_instance=qi,include_custom=True)
    result = vqe.compute_minimum_eigenvalue(operator=H_first)    
    # print("time="+str(time_end-time_start))
    #
    circ=makecircXXZ(P,Lguess,deltaY,deltaZ)
    qqq=circ.assign_parameters(result.optimal_point)
    psi=CircuitStateFn(qqq)
    
    # translational fidelity
    circS=makecircXXZS(P,Lguess,deltaY,deltaZ)
    qqqS=circS.assign_parameters(result.optimal_point)
    psiS=CircuitStateFn(qqqS)
    #
    transl_now=np.linalg.norm(np.vdot(psiS.to_matrix(massive=True),psi.to_matrix(massive=True)))
    
    en[r] = ((Lguess/2)*result.optimal_value)
    resen[r] = (residual( ((Lguess/2)*result.optimal_value), emin, emax))
    fid[r] = (fidel(psi.to_matrix(massive= True),psi0))
    transl[r] = (transl_now)
    optparlist.append(result.optimal_point)
    first_res[r] = (residual((Lguess/2)*values[0],emin,emax))
    

np.savez(outfileOpt, en=en, resen=resen, fid=fid, transl=transl,\
     optparlist=optparlist, first_res=first_res, EnExact=EnExact, EnMaxExact=EnMaxExact)
   

### Part 2)

# size bounds
Lmin=4
Lmax=24
# Load ED file
outfile_ED = f"temp_results/ED_XYZ_deltaY={deltaY}_deltaZ={deltaZ}.npz"
if os.path.isfile(outfile_ED):
    outfile_ED_data = np.load(outfile_ED, allow_pickle=True) #load saved data
    eminn = outfile_ED_data["eminn"]
    emaxx = outfile_ED_data["emaxx"]
else:
    raise Exception("No ED results file for these XYZ parameters")

# output file with transferability results
file_TR = f"temp_results/transf_RAND_XYZ_Lguess={Lguess}_deltaY={deltaY}_deltaZ={deltaZ}_P={P}.npz"


# trasferability bounds
LminT=10 
LmaxT=24

# results
narray=np.arange(LminT,LmaxT+1,2)
guess_res_en=np.zeros((N_random_inits, narray.size)) # on row jth you find the transf results for random init jth

for r in range(N_random_inits):

    pars_init = optparlist[r] # optimal parameters for r-th random init. P is fixed

    backend = Aer.get_backend('statevector_simulator')

    for (j,n) in enumerate(range(LminT,LmaxT+1,2)):
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

