#!/bin/bash


#SBATCH --job-name=SlBP       # Job name, will show up in squeue outputi
#SBATCH --ntasks=4                    # Number of cores
#SBATCH --nodes=1                      # Ensure that all cores are on one machine
#SBATCH --time=2-18:00:00              # Runtime in DAYS-HH:MM:SS format
#SBATCH --mem-per-cpu=100000             # Memory per cpu in MB (see also --mem) 
#SBATCH --output=job1_%j.out           # File to which standard out will be written
#SBATCH --error=job1_%j.err            # File to which standard err will be written
#SBATCH --mail-type=END                # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=antoniomele.p@gmail.com # Email to which notifications will be sent 
 

##############################################
# INITIALISATION STEPS
##############################################


deltay=1.25
deltaz=0.75
maxiter=100
Linp=8
Pmax=40




python main_VQE_XYZ.py $deltay $deltaz $maxiter $Linp $Pmax

