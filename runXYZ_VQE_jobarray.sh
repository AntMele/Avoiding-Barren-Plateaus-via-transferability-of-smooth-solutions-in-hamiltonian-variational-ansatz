#!/bin/bash


#SBATCH --job-name=SlBP       # Job name, will show up in squeue outputi
#SBATCH --ntasks=2                   # Number of cores
#SBATCH --nodes=1                      # Ensure that all cores are on one machine
#SBATCH --time=3-18:00:00              # Runtime in DAYS-HH:MM:SS format
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
deltay=0.75
deltaz=1.25
maxiter=100
Pmax=16 

for L in {4..16..2}
do
	Linp+=($L)
done

python main_VQE_XYZ.py $deltay $deltaz $maxiter ${Linp[$SLURM_ARRAY_TASK_ID]} $Pmax

time sleep 20

Lmax=24
Lguess=6
P=10

python main_transferability_XYZ.py $deltay $deltaz $Lmax $Lguess $Pmax $maxiter $P
