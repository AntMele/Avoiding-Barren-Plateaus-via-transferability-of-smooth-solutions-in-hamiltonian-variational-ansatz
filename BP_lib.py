import numpy as np
from qiskit import Aer
from qiskit.opflow import AerPauliExpectation, CircuitSampler, StateFn, CircuitStateFn

def CostNEW(par, Ht, circNEW): 
    """
    Compute the value of the cost function at parameters par
    NEW: Ht and circNEW provided from the main
    OLD: *args = Ht,deltaY,deltaZ,P,n are the fixed variables
    """
    #
    ''' OLD
    Ht=args[0]
    deltaY=args[1]
    deltaZ=args[2]
    P=args[3]
    n=args[4]
    '''
    # MODIFICA
    qqqNEW=circNEW.assign_parameters(par)
    # Choose backend
    backend = Aer.get_backend('statevector_simulator') 
    # Create a quantum state from the QC output
    psi = CircuitStateFn(qqqNEW)
    # "Apply Ht to psi"
    measurable_expression = StateFn(Ht, is_measurement=True).compose(psi) 
    # Compute the exp val using Qiskit
    expectation = AerPauliExpectation().convert(measurable_expression)
    sampler = CircuitSampler(backend).convert(expectation) 
    temp=(sampler.eval().real)
    return temp
