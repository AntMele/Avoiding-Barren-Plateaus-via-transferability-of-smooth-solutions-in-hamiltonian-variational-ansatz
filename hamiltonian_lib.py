import numpy as np
from qiskit.opflow import *

"""
Build an interaction term as XX, YY, ZZ
""" 

def termXX(i,j,n):
    i=n-i-1
    j=n-j-1
    for pos in range(n):
        if pos==0:
            if(pos==i or pos==j):
                ter=X
            else:
                ter=I
        else:        
            if(pos==i or pos==j):
                ter=ter^X
            else:
                ter=ter^I
    return ter

def termYY(i,j,n):
    i=n-i-1
    j=n-j-1
    for pos in range(n):
        if pos==0:
            if(pos==i or pos==j):
                ter=Y
            else:
                ter=I
        else:        
            if(pos==i or pos==j):
                ter=ter^Y
            else:
                ter=ter^I
    return ter

def termZZ(i,j,n):
    i=n-i-1
    j=n-j-1
    for pos in range(n):
        if pos==0:
            if(pos==i or pos==j):
                ter=Z
            else:
                ter=I
        else:        
            if(pos==i or pos==j):
                ter=ter^Z
            else:
                ter=ter^I
    return ter


"""
Build the interaction Hamiltonian, given a weighted graph
"""

def makeXX(edges_list,coeff_list,n):
    Hnow=0
    for (edge,w) in zip(edges_list,coeff_list):
        i=edge[0]
        j=edge[1]
        Hnow = Hnow + w*(termXX(i,j,n))
    return Hnow

def makeYY(edges_list,coeff_list,n):
    Hnow=0
    for (edge,w) in zip(edges_list,coeff_list):
        i=edge[0]
        j=edge[1]
        Hnow = Hnow + w*(termYY(i,j,n))
    return Hnow

def makeZZ(edges_list,coeff_list,n):
    Hnow=0
    for (edge,w) in zip(edges_list,coeff_list):
        i=edge[0]
        j=edge[1]
        Hnow = Hnow + w*(termZZ(i,j,n))
    return Hnow


"""
Build the one-body Hx Hamiltonian
"""
def termZ(i,n):
    i=n-i-1
    te=1
    for pos in range(n):
        if pos==0:
            if(pos==i):
                te=Z
            else:
                te=I
        else:        
            if(pos==i):
                te=te^Z
            else:
                te=te^I
    return te

def termX(i,n):
    i=n-i-1
    te=1
    for pos in range(n):
        if pos==0:
            if(pos==i):
                te=X
            else:
                te=I
        else:        
            if(pos==i):
                te=te^X
            else:
                te=te^I
    return te

def makeX(n): # = sum_i X_i
    Hxnow=1.0*(termX(0,n))
    for i in range(1,n):
        Hxnow = Hxnow + 1.0*(termX(i,n))
    return Hxnow

def makeZ(n): # = sum_i X_i
    Hxnow=1.0*(termZ(0,n))
    for i in range(1,n):
        Hxnow = Hxnow + 1.0*(termZ(i,n))
    return Hxnow


"""
Create only interaction terms on the first even AND odd links (we shall use translational invariance)
"""

def makeZZfirstEO(n): 
    Hnow = 1*(termZZ(0,1,n)) + 1*(termZZ(1,2,n))
    return Hnow

def makeYYfirstEO(n):
    Hnow = 1*(termYY(0,1,n)) + 1*(termYY(1,2,n))
    return Hnow

def makeXXfirstEO(n):
    Hnow = 1*(termXX(0,1,n)) + 1*(termXX(1,2,n))
    return Hnow


# TFIM

def makeZZfirst_TFIM(n): # only pair (0,1)
    Hnow = 1*(termZZ(0,1,n))
    return Hnow

def makeXfirst(n): # only qubit #0
    Hnow = 1*(termX(0,n))
    return Hnow

