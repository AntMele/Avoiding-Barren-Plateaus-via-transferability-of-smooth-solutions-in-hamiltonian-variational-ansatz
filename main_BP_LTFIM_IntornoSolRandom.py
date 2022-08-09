import numpy as np
import os
# import matplotlib.pyplot as plt
import time
import sys
from qiskit.circuit import Parameter

from ansatz_lib import makecirc_LTFIM, makecirc_LTFIM3
from hamiltonian_lib import termZ, makeZ, makeZZ, makeX, makeZZfirst_TFIM, makeZZfirstEO, makeXfirst
from BP_lib import CostNEW


# barren plateau
### INPUT
g = float(sys.argv[1])
h = float(sys.argv[2])

# rand input parameters
nrands=int(sys.argv[3]) # 1000 # random sampling points for gradient estimation
L=int(sys.argv[4]) #4

# educated input parameters
#from main_ansatz_BP import Lguess

eps_opt=float(sys.argv[5]) #Epsilon Educated Barren Plateaus
P=int(sys.argv[6])
indexr=int(sys.argv[7])
ans=str(sys.argv[8])
onlyZZ=int(sys.argv[9])
sign=int(sys.argv[10])

Lguess=int(sys.argv[11])
Pmaxguess=int(sys.argv[12])

#
epsd= 0.000001 #epsilon of *finite difference gradient* estimation 
#

if(ans=="ans2"):
    prep=2
else:
    prep=3

def ansatz(P,L):
    if(ans=="ans2"):
        return makecirc_LTFIM(P,L,sign,h)
    else:
        return makecirc_LTFIM3(P,L) 

if ( onlyZZ == 0 ):
    Hzz_first = makeZZfirst_TFIM(L)
    Hx_first = makeXfirst(L)
    Hz_first = termZ(0,L)
    Ht_BP =  sign* Hzz_first - g * Hx_first - h * Hz_first
else:
    Hzz_first = makeZZfirst_TFIM(L)
    Ht_BP=  sign* Hzz_first

### END INPUT


##### 2) BP educated guessfileBP_EpsRand_{indexr}_XYZ_L={L}_P={P}_deltaY={deltaY}_deltaZ={deltaZ}_eps={eps_opt}.npz"
outfile_BP_EdGuess = f"temp_results_LTFIM/fileBP_IntornoSolRandom_{indexr}_LTFIM_L={L}_P={P}_g={g}_h={h}_Lg={Lguess}_eps={eps_opt}_ans={ans}_OnlyZZ={onlyZZ}_sign={sign}.npz"
#

EdGuess_file = f"temp_results_LTFIM/file_Opt_NewStartRand_r{indexr}_LTFIM_g={g}_h={h}_{ans}_L={Lguess}_P={Pmaxguess}_maxiter=1000000_sign=1.npz"
#transf_LTFIM_NewStartRand_r{indexr}_Lguess={Lguess}_g={g}_h={h}_P={Pmaxguess}_{ans}_maxiter=1000000_sign={sign}.npz"

if os.path.isfile(EdGuess_file):
    EdGuess_file_data = np.load(EdGuess_file, allow_pickle=True) #load saved data
    optparlist=EdGuess_file_data["optparlist"]
else:
    raise Exception("No VQE results file for these XYZ parameters")




# circuit for given N and P
circ_BP=ansatz(P,L)
sumcost=0
Grad_sample=np.zeros(nrands)
Grad_sample_sq=np.zeros(nrands)

parOtt=optparlist

for i in range(nrands):
    parR= parOtt + eps_opt*(2*np.random.rand(prep*P) - 1*np.ones(prep*P) )
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
        
