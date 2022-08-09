#!/bin/bash


#SBATCH --job-name=LTFIM_transf       # Job name, will show up in squeue outputi
#SBATCH --ntasks=1                    # Number of cores
#SBATCH --nodes=1                      # Ensure that all cores are on one machine
#SBATCH --time=1-01:00:00              # Runtime in DAYS-HH:MM:SS format
#SBATCH --mem-per-cpu=100000             # Memory per cpu in MB (see also --mem) 
#SBATCH --output=job1_%j.out           # File to which standard out will be written
#SBATCH --error=job1_%j.err            # File to which standard err will be written
#SBATCH --mail-type=END                # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=antoniomele.p@gmail.com # Email to which notifications will be sent 
 

##############################################
# INITIALISATION STEPS
##############################################


g=0.2
h=1.5
Lmax=20
Lguess=6
Pmaxguess=10
iterguess=100
P=10
ans="ans2"
sign=1

python main_VQE_LTFIM.py $g $h $iterguess $Lguess $Pmaxguess $ans $sign
time sleep 10
python main_transferability_LTFIM.py $g $h $Lmax $Lguess $Pmaxguess $iterguess $P $ans $sign

