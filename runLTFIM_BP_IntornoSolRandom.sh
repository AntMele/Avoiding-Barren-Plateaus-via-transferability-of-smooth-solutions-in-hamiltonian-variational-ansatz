#!/bin/bash


#SBATCH --job-name=BP_LTFIM       # Job name, will show up in squeue outputi
#SBATCH --ntasks=1                    # Number of cores
#SBATCH --nodes=1                      # Ensure that all cores are on one machine
#SBATCH --time=2-18:00:00              # Runtime in DAYS-HH:MM:SS format
#SBATCH --mem-per-cpu=100000             # Memory per cpu in MB (see also --mem) 
#SBATCH --array=0-180
#SBATCH --output=job1_%j.out           # File to which standard out will be written
#SBATCH --error=job1_%j.err            # File to which standard err will be written
#SBATCH --mail-type=END                # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=antoniomele.p@gmail.com # Email to which notifications will be sent 
 

##############################################
# INITIALISATION STEPS
##############################################

declare -a Linp
declare -a ArrR

g=1
h=1
nrands=1000
eps_opt=0.05
P=100
ans="ans2"
onlyZZ=0
sign=1
Lg=8
Pmaxguess=100

for L in {4..20..2} #9
do
	for nrep in {1..20..1} #20
	do	
		Linp+=($L)
		ArrR+=($nrep)
	done
done

python main_BP_LTFIM_IntornoSolRandom.py $g $h $nrands ${Linp[$SLURM_ARRAY_TASK_ID]} $eps_opt $P ${ArrR[$SLURM_ARRAY_TASK_ID]} $ans $onlyZZ $sign $Lg $Pmaxguess
