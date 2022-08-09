import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

"""
Create the ansatz circuit for the XYZ model
"""
def makecircXXZ(PP,dim,deltaY,deltaZ):

    # included this in the function
    parA=[]
    parB=[]
    '''
    parC=[]
    parD=[]
    parE=[]
    parF=[]
    '''
    for i in range(PP):
        parA.append(Parameter(f'a{i:010b}'))
        parB.append(Parameter(f'b{i:010b}'))
        '''
        parC.append(Parameter(f'c{i:010b}'))
        parD.append(Parameter(f'd{i:010b}'))
        parE.append(Parameter(f'e{i:010b}'))
        parF.append(Parameter(f'f{i:010b}'))
        '''
    ci = QuantumCircuit(dim)    
    for i in range(dim):
        ci.x(i)
    for i in range(dim):    
        if (i%2==0):
            ci.h(i)
            ci.cx(i,(i+1)%dim)
    for p in range(PP):
        for i in range((dim)//2):
            ci.rzz(2*deltaZ*parA[p],(2*i+1)%dim,(2*i+2)%dim)
        for i in range((dim)//2):
            ci.ryy(2*deltaY*parA[p],(2*i+1)%dim,(2*i+2)%dim)
        for i in range((dim)//2):
            ci.rxx(2*parA[p],(2*i+1)%dim,(2*i+2)%dim)
        for i in range((dim+1)//2):
            ci.rzz(2*deltaZ*parB[p],(2*i)%dim,(2*i+1)%dim)
        for i in range((dim+1)//2):
            ci.ryy(2*deltaY*parB[p],(2*i)%dim,(2*i+1)%dim)
        for i in range((dim+1)//2):
            ci.rxx(2*parB[p],(2*i)%dim,(2*i+1)%dim)
    return ci


"""
The shifted circuit for computing the translational invariance is equal to the previous one but with SWAP gates at the end
"""
def makecircXXZS(PP,dim,deltaY,deltaZ):
    parA=[]
    parB=[]
    '''
    parC=[]
    parD=[]
    parE=[]
    parF=[]
    '''
    for i in range(PP):
        parA.append(Parameter(f'a{i:010b}'))
        parB.append(Parameter(f'b{i:010b}'))
        '''
        parC.append(Parameter(f'c{i:010b}'))
        parD.append(Parameter(f'd{i:010b}'))
        parE.append(Parameter(f'e{i:010b}'))
        parF.append(Parameter(f'f{i:010b}'))
        '''
    ci = QuantumCircuit(dim)    
    for i in range(dim):
        ci.x(i)
    for i in range(dim):    
        if (i%2==0):
            ci.h(i)
            ci.cx(i,(i+1)%dim)
    for p in range(PP):
        for i in range((dim)//2):
            ci.rzz(2*deltaZ*parA[p],(2*i+1)%dim,(2*i+2)%dim)
        for i in range((dim)//2):
            ci.ryy(2*deltaY*parA[p],(2*i+1)%dim,(2*i+2)%dim)
        for i in range((dim)//2):
            ci.rxx(2*parA[p],(2*i+1)%dim,(2*i+2)%dim)
        for i in range((dim+1)//2):
            ci.rzz(2*deltaZ*parB[p],(2*i)%dim,(2*i+1)%dim)
        for i in range((dim+1)//2):
            ci.ryy(2*deltaY*parB[p],(2*i)%dim,(2*i+1)%dim)
        for i in range((dim+1)//2):
            ci.rxx(2*parB[p],(2*i)%dim,(2*i+1)%dim)
    for i in range(1,dim):
        ci.swap(0,i)    
    return ci


"""
Create the ansatz circuit for the TFIM model
"""
def makecirc_TFIM(PP,dim):

    parGam=[]
    parBet=[]
    for i in range(PP):
        parGam.append(Parameter(f'a{i:010b}'))
        parBet.append(Parameter(f'b{i:010b}'))
    
    ci = QuantumCircuit(dim)
    for i in range(dim):
        ci.h(i)
    #ci.barrier()
    for p in range(PP):
        for i in range(dim//2):
            ci.rzz(-2*parGam[p],2*i,2*i+1)
        for i in range(dim//2):
            ci.rzz(-2*parGam[p],2*i+1,(2*i+2)%dim)
        if (dim%2==1):
            ci.rzz(-2*parGam[p],dim-1,0)
        #circ.barrier()
        for ii in range(dim):
            ci.rx(-2*parBet[p],ii)     
            
    return ci        




def makecircXXZg(PP,dim,deltaY,deltaZ):
    
    # included this in the function
    parA=[]
    parB=[]
    parC=[]
    '''
    parD=[]
    parE=[]
    parF=[]
    '''
    for i in range(PP):
        parA.append(Parameter(f'a{i:010b}'))
        parB.append(Parameter(f'b{i:010b}'))
        parC.append(Parameter(f'c{i:010b}'))
        '''
        parD.append(Parameter(f'd{i:010b}'))
        parE.append(Parameter(f'e{i:010b}'))
        parF.append(Parameter(f'f{i:010b}'))
        '''
    ci = QuantumCircuit(dim)    
    for i in range(dim):
        ci.x(i)
    for i in range(dim):    
        if (i%2==0):
            ci.h(i)
            ci.cx(i,(i+1)%dim)
    for p in range(PP):
        for i in range((dim)//2):
            ci.rzz(2*deltaZ*parA[p],(2*i+1)%dim,(2*i+2)%dim)
        for i in range((dim)//2):
            ci.ryy(2*deltaY*parA[p],(2*i+1)%dim,(2*i+2)%dim)
        for i in range((dim)//2):
            ci.rxx(2*parA[p],(2*i+1)%dim,(2*i+2)%dim)
        for i in range((dim+1)//2):
            ci.rzz(2*deltaZ*parB[p],(2*i)%dim,(2*i+1)%dim)
        for i in range((dim+1)//2):
            ci.ryy(2*deltaY*parB[p],(2*i)%dim,(2*i+1)%dim)
        for i in range((dim+1)//2):
            ci.rxx(2*parB[p],(2*i)%dim,(2*i+1)%dim)
        for i in range(dim):
            ci.rx(2*parC[p],i)
    return ci


"""
The shifted circuit for computing the translational invariance is equal to the previous one but with SWAP gates at the end
"""
def makecircXXZSg(PP,dim,deltaY,deltaZ):
    parA=[]
    parB=[]
    parC=[]
    '''
    parD=[]
    parE=[]
    parF=[]
    '''
    for i in range(PP):
        parA.append(Parameter(f'a{i:010b}'))
        parB.append(Parameter(f'b{i:010b}'))
        parC.append(Parameter(f'c{i:010b}'))
        '''
        parD.append(Parameter(f'd{i:010b}'))
        parE.append(Parameter(f'e{i:010b}'))
        parF.append(Parameter(f'f{i:010b}'))
        '''
    ci = QuantumCircuit(dim)    
    for i in range(dim):
        ci.x(i)
    for i in range(dim):    
        if (i%2==0):
            ci.h(i)
            ci.cx(i,(i+1)%dim)
    for p in range(PP):
        for i in range((dim)//2):
            ci.rzz(2*deltaZ*parA[p],(2*i+1)%dim,(2*i+2)%dim)
        for i in range((dim)//2):
            ci.ryy(2*deltaY*parA[p],(2*i+1)%dim,(2*i+2)%dim)
        for i in range((dim)//2):
            ci.rxx(2*parA[p],(2*i+1)%dim,(2*i+2)%dim)
        for i in range((dim+1)//2):
            ci.rzz(2*deltaZ*parB[p],(2*i)%dim,(2*i+1)%dim)
        for i in range((dim+1)//2):
            ci.ryy(2*deltaY*parB[p],(2*i)%dim,(2*i+1)%dim)
        for i in range((dim+1)//2):
            ci.rxx(2*parB[p],(2*i)%dim,(2*i+1)%dim)
        for i in range(dim):
            ci.rx(2*parC[p],i)
    for i in range(1,dim):
        ci.swap(0,i)    
    return ci





def makecirc_LTFIM3(PP,dim):
    
    parA=[]
    parB=[]
    parC=[]
    for i in range(PP):
        parA.append(Parameter(f'a{i:010b}'))
        parB.append(Parameter(f'b{i:010b}'))
        parC.append(Parameter(f'c{i:010b}'))
    
    ci = QuantumCircuit(dim)
    for i in range(dim):
        ci.h(i)
    #ci.barrier()
    for p in range(PP):
        for i in range(dim//2):
            ci.rzz(2*parA[p],2*i,2*i+1)
        for i in range(dim//2):
            ci.rzz(2*parA[p],2*i+1,(2*i+2)%dim)
        if (dim%2==1):
            ci.rzz(2*parA[p],dim-1,0)
        #circ.barrier()
        for ii in range(dim):
            ci.rz(-2*parB[p],ii)     
        for ii in range(dim):
            ci.rx(-2*parC[p],ii) 
            
    return ci     


def makecirc_LTFIM(PP,dim,sign,h):
    parA=[]
    parB=[]
    #parC=[]
    for i in range(PP):
        parA.append(Parameter(f'a{i:010b}'))
        parB.append(Parameter(f'b{i:010b}'))
        #parC.append(Parameter(f'c{i:010b}'))
    
    ci = QuantumCircuit(dim)
    for i in range(dim):
        ci.h(i)
    #ci.barrier()
    for p in range(PP):
        for i in range(dim//2):
            ci.rzz(2*sign*parA[p],2*i,2*i+1)
        for i in range(dim//2):
            ci.rzz(2*sign*parA[p],2*i+1,(2*i+2)%dim)
        if (dim%2==1):
            ci.rzz(2*sign*parA[p],dim-1,0)
        #circ.barrier()
        for ii in range(dim):
            ci.rz(-2*h*parA[p],ii)     
        for ii in range(dim):
            ci.rx(-2*parB[p],ii) 
            
    return ci        
   


