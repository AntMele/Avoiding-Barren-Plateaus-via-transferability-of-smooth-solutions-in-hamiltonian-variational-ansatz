#!/bin/bash


#SBATCH --job-name=EdGuess_LTFIM       # Job name, will show up in squeue outputi
#SBATCH --ntasks=1                    # Number of cores
#SBATCH --nodes=1                      # Ensure that all cores are on one machine
#SBATCH --time=13-18:00:00              # Runtime in DAYS-HH:MM:SS format
#SBATCH --mem-per-cpu=100000             # Memory per cpu in MB (see also --mem) 
#SBATCH --output=job1_%j.out           # File to which standard out will be written
#SBATCH --error=job1_%j.err            # File to which standard err will be written
#SBATCH --mail-type=END                # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=antoniomele.p@gmail.com # Email to which notifications will be sent 
 

##############################################
# INITIALISATION STEPS
##############################################


g=1
h=1
maxiter=100
Linp=8
Pmax=100
ans="ans2"
sign=1



python main_VQE_LTFIM.py $g $h $maxiter $Linp $Pmax $ans $sign

