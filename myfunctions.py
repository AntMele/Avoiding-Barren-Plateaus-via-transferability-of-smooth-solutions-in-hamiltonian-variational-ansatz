import numpy as np

def fidel(psifin,psi0):
    return np.linalg.norm(np.vdot(psifin,psi0))

def residual(now,emin,emax):
    return (now-emin)/(emax-emin)

def makeinit(par_old,P,prep): #Zhou INTERP formula, prep=2 in the case you have alpha and beta
    par_new=[]
    for j in range(prep): #cycle on alpha and beta 
        for i in range(P+1):
            if (i==0):
                par_new.append( ((P-i)/P)*par_old[i + j*P] )
            elif (i==P):
                par_new.append( ((i)/P)*par_old[i + j*P - 1 ] )
            else:
                par_new.append( ((i)/P)*par_old[i + j*P -1] + ((P-i)/P)*par_old[i+j*P] )
    par_new=np.array(par_new)
    if(len(par_new)!= ((P+1)*prep)):
        print("ERROR")
    return par_new

