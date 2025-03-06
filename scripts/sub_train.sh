#!/bin/sh                                                                                                                                                                                               
#SBATCH -C gpu                                                                                                                                                                                          
#SBATCH -q regular                                                                                                                                                                                      
#SBATCH -n 64                                                                                                                                                                                          
#SBATCH --ntasks-per-node 4                                                                       
#SBATCH --gpus-per-task 1
#SBATCH --gpu-bind=none                                                                                                                                                                                 
#SBATCH -t 15:00:00                                                                                                                                                                                        
#SBATCH -A m3246                                                                                                                                                                                 
export MPICH_ALLGATHERV_PIPELINE_MSG_SIZE=0
export MPICH_MAX_THREAD_SAFETY=multiple
export MPIR_CVAR_GPU_EAGER_DEVICE_MEM=0

cd $PSCRATCH/par/parnassus/scripts/

module load tensorflow

echo python train.py
srun python train.py
