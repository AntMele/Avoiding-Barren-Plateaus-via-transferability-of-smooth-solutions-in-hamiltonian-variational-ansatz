#!/bin/bash


#SBATCH --job-name=LTFIM_VQE       # Job name, will show up in squeue outputi
#SBATCH --ntasks=1                   # Number of cores
#SBATCH --nodes=1                      # Ensure that all cores are on one machine
#SBATCH --time=2-18:00:00              # Runtime in DAYS-HH:MM:SS format
#SBATCH --mem-per-cpu=100000             # Memory per cpu in MB (see also --mem) 
#SBATCH --array=0-7
#SBATCH --output=job1_%j.out           # File to which standard out will be written
#SBATCH --error=job1_%j.err            # File to which standard err will be written
#SBATCH --mail-type=END                # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=antoniomele.p@gmail.com # Email to which notifications will be sent 
 

##############################################
# INITIALISATION STEPS
##############################################

declare -a Linp
g=1.0
h=0
maxiter=1000
Pmax=16 
ans="ans2"
sign=-1

for L in {4..16..2}
do
	Linp+=($L)
done

python main_VQE_LTFIM.py $g $h $maxiter ${Linp[$SLURM_ARRAY_TASK_ID]} $Pmax $ans $sign

time sleep 20

Lmax=24
Lguess=8
P=10

python main_transferability_LTFIM.py $g $h $Lmax $Lguess $Pmax $maxiter $P $ans $sign
