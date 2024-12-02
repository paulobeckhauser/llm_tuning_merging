#!/bin/sh

### –- specify queue --
#BSUB -q gpuv100

### -- set the job Name --
#BSUB -J testjob

### -- ask for number of cores (default: 1) --
#BSUB -n 4

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 8:00

# request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"

### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address

### -- send notification at start --
#BSUB -B

### -- send notification at completion--
#BSUB -N

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err

nvidia-smi
module load cuda/11.6

/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery

source ../.venv/bin/activate
python "LoRA_Fine_tuning.py"
