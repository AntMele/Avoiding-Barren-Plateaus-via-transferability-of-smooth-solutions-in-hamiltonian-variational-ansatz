#!/bin/bash


#SBATCH --job-name=RandEdGuess_LTFIM       # Job name, will show up in squeue outputi
#SBATCH --ntasks=1                    # Number of cores
#SBATCH --nodes=1                      # Ensure that all cores are on one machine
#SBATCH --time=0-18:00:00              # Runtime in DAYS-HH:MM:SS format
#SBATCH --mem-per-cpu=100000             # Memory per cpu in MB (see also --mem) 
#SBATCH --array=0-20
#SBATCH --output=job1_%j.out           # File to which standard out will be written
#SBATCH --error=job1_%j.err            # File to which standard err will be written
#SBATCH --mail-type=END                # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=antoniomele.p@gmail.com # Email to which notifications will be sent 
 

##############################################
# INITIALISATION STEPS
##############################################

declare -a ArrR

g=1
h=1
maxiter=1000000
Linp=8
Pmax=10
ans="ans2"
sign=1

for nrep in {1..20..1} #20
do	
    ArrR+=($nrep)
done

python main_VQE_LTFIM_NewStartRand.py $g $h $maxiter $Linp $Pmax $ans $sign ${ArrR[$SLURM_ARRAY_TASK_ID]}

Lmax=24

python main_transferability_LTFIM_NewStartRand.py $g $h $Lmax $Linp $Pmax $maxiter $Pmax $ans $sign ${ArrR[$SLURM_ARRAY_TASK_ID]}