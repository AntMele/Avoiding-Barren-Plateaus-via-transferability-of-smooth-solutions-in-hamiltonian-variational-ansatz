#!/bin/bash


#SBATCH --job-name=randLTFIM_transf       # Job name, will show up in squeue outputi
#SBATCH --ntasks=1                    # Number of cores
#SBATCH --nodes=1                      # Ensure that all cores are on one machine
#SBATCH --time=2-01:00:00              # Runtime in DAYS-HH:MM:SS format
#SBATCH --mem-per-cpu=100000             # Memory per cpu in MB (see also --mem) 
#SBATCH --output=job1_%j.out           # File to which standard out will be written
#SBATCH --error=job1_%j.err            # File to which standard err will be written
#SBATCH --mail-type=END                # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=antoniomele.p@gmail.com # Email to which notifications will be sent 
 

##############################################
# INITIALISATION STEPS
##############################################







g=1
h=0.0
iterguess=100000 #1000000
ans="ans2"
sign=-1
Lguess=8
P=10
Nrandominits=20


python main_random_init_LTFIM.py $g $h $iterguess $ans $sign $Lguess $P $Nrandominits 

