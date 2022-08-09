import numpy as np
import os
# import matplotlib.pyplot as plt
import time
import sys
from qiskit.circuit import Parameter

from ansatz_lib import makecircXXZ, makecircXXZS
from hamiltonian_lib import makeZZfirstEO, makeYYfirstEO, makeXXfirstEO
from BP_lib import CostNEW
 

# whether to run the first and/or second part
# barren plateau
### INPUT
deltaY=float(sys.argv[1])
deltaZ=float(sys.argv[2])

# rand input parameters
nrands=int(sys.argv[3]) # 1000 # random sampling points for gradient estimation
L=int(sys.argv[4]) #4

# educated input parameters
#from main_ansatz_BP import Lguess

eps_opt=float(sys.argv[5]) #Epsilon Educated Barren Plateaus

#from main_edguess_XYZ import Pmin, Pmax
P=int(sys.argv[6])
indexr=int(sys.argv[7])

Lguess=int(sys.argv[8])
Pmaxguess=int(sys.argv[9])
#
epsd= 0.000001 #epsilon of *finite difference gradient* estimation 
#

if ( (np.abs(deltaY-1) < 10**(-4)) and (np.abs(deltaZ-1) < 10**(-4)) ):
    Ht_BP=makeZZfirstEO(L)
else:
    Ht_BP= makeXXfirstEO(L) + deltaY*makeYYfirstEO(L) + deltaZ*makeZZfirstEO(L)

### END INPUT

outfile_BP_EdGuess = f"temp_results_XYZ/fileBP_IntornoSolRandom_{indexr}_XYZ_L={L}_P={P}_deltaY={deltaY}_deltaZ={deltaZ}_Lg={Lguess}_eps={eps_opt}.npz"

EdGuess_file = f"temp_results_XYZ/file_Opt_NewStartRand_r{indexr}_deltaY={deltaY}_deltaZ={deltaZ}_L={Lguess}_P={Pmaxguess}_maxiter=1000000.npz"

if os.path.isfile(EdGuess_file):
    EdGuess_file_data = np.load(EdGuess_file, allow_pickle=True) #load saved data
    optparlist=EdGuess_file_data["optparlist"]
else:
    raise Exception("No VQE results file for these XYZ parameters")


# circuit for given N and P
circ_BP=makecircXXZ(P,L,deltaY,deltaZ) 
sumcost=0
Grad_sample=np.zeros(nrands)
Grad_sample_sq=np.zeros(nrands)

parOtt=optparlist

for i in range(nrands):
    parR= parOtt + eps_opt*(2*np.random.rand(2*P) - 1*np.ones(2*P) )
    pardx=  np.copy(parR)
    pardx[0] = pardx[0] + epsd
    parsx= np.copy(parR)
    parsx[0]= parsx[0] - epsd
    #
    grad = (CostNEW(pardx, Ht_BP, circ_BP) - CostNEW(parsx, Ht_BP, circ_BP)) / (2*epsd) 
    #
    cost_now=CostNEW(parR, Ht_BP, circ_BP) #cost function evaluated at the point parR
    # print("Cost= "+str(cost_now))
    sumcost+= cost_now
    Grad_sample[i]=grad
    Grad_sample_sq[i]= grad**2

Meancost = sumcost/nrands
Meangrad = np.mean(Grad_sample)
Vargrad =  np.var(Grad_sample,ddof=1)
Mean_grad_sq= np.mean(Grad_sample_sq)
Mean_grad_abs = np.mean(np.abs(Grad_sample))

np.savez(outfile_BP_EdGuess, Meancost=Meancost, Meangrad=Meangrad, Vargrad=Vargrad, Mean_grad_sq=Mean_grad_sq, Mean_grad_abs=Mean_grad_abs)
        
