#!/bin/bash
#
#-----------------------------------------------------------------------------
# This Maverick job script is designed to create an Rstudio session on 
# visualization nodes through the SLURM batch system. Once the job
# is scheduled, check the output of your job (which by default is
# stored in your home directory in a file named Rstudio.out)
# and it will tell you the port number that has been setup for you so
# that you can attach via a separate web browser to any Maverick login 
# node (e.g., login1.maverick.tacc.utexas.edu).
#
# Note: you can fine tune the SLURM submission variables below as
# needed.  Typical items to change are the runtime limit, location of
# the job output, and the allocation project to submit against (it is
# commented out for now, but is required if you have multiple
# allocations).  
#
# To submit the job, issue: "sbatch /share/doc/slurm/job.Rstudio" 
#
# For more information, please consult the User Guide at: 
#
# http://www.tacc.utexas.edu/user-services/user-guides/maverick-user-guide
#-----------------------------------------------------------------------------
#
#SBATCH -J default                           # Job name
#SBATCH -o job_outputs/controller/default-%j.out                     # Name of stdout output file (%j expands to jobId)
#SBATCH -p gpu                                      # Queue name
#SBATCH -N 1                                        # Total number of nodes requested (20 cores/node)
#SBATCH -n 1                                        # Total number of mpi tasks requested
#SBATCH --mail-user=kk28695@tacc.utexas.edu
#SBATCH --mail-type=end
#SBATCH -t 4:00:00                                  # Run time (hh:mm:ss) - 4 hours
#SBATCH -A CS381V-Visual-Recogn

module load cuda/8.0 cudnn/5.1
module load tensorflow-gpu
# TODO: REMEMBER TO CHANGE JOB NAME
python3 one_shot_learning.py --dataset_type=kinetics_dynamic --controller_type=alex --batch_size=16 --image_width=64  --image_height=64 --summary_writer=True --model_saver=False --debug=True --memory_size=128 --memory_vector_dim=40 --seq_length=100 --n_classes=25 --class_difficulty=all --use_pretrained=False --num_epoches=250 --rnn_size=200 --batches_validation=5
