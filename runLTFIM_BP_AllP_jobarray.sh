#!/bin/bash


#SBATCH --job-name=LTFIM_BPallP       # Job name, will show up in squeue outputi
#SBATCH --ntasks=1                    # Number of cores
#SBATCH --nodes=1                      # Ensure that all cores are on one machine
#SBATCH --time=3-18:00:00              # Runtime in DAYS-HH:MM:SS format
#SBATCH --mem-per-cpu=100000             # Memory per cpu in MB (see also --mem) 
#SBATCH --array=0-220
#SBATCH --output=job1_%j.out           # File to which standard out will be written
#SBATCH --error=job1_%j.err            # File to which standard err will be written
#SBATCH --mail-type=END                # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=antoniomele.p@gmail.com # Email to which notifications will be sent 
 

##############################################
# INITIALISATION STEPS
##############################################

declare -a Linp
declare -a Arrp
randed=1 #0 if rand, 1 educated
g=1.0
h=1.0
nrands=1000
Lguess=8
eps_opt=0.05
Pmaxguess=100
iterguess=100
ans="ans2"
onlyZZ=0
sign=1

for L in {4..20..2} #11
do
	for nrep in {5..100..5} #20
	do	
		Linp+=($L)
		Arrp+=($nrep)
	done
done

python main_BP_LTFIM.py $randed $g $h $nrands ${Linp[$SLURM_ARRAY_TASK_ID]} $Lguess $eps_opt ${Arrp[$SLURM_ARRAY_TASK_ID]} $Pmaxguess $iterguess $ans $onlyZZ $sign

