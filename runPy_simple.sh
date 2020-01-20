#!/bin/bash
#SBATCH -J gen-net-job
#SBATCH --time=6:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alvinmeltsov@gmail.com
#SBATCH --mem-per-cpu=8000
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=15

module load python/3.6.3/CUDA-9.0

# This is a runner script for running some python scripts in SLURM
# Use for bigger jobs.
#
# TODO: Activate the gqn enviroment first!
# TODO: EMAIL and TIME fields!

srun --ntasks=1 --exclusive python view_pointer.py "2d_64block128_25.pt" &
srun --ntasks=1 --exclusive python view_pointer.py "2d_64block128_26.pt" &
srun --ntasks=1 --exclusive python view_pointer.py "2d_64block128_27.pt" &
srun --ntasks=1 --exclusive python view_pointer.py "2d_64block128_28.pt" &
srun --ntasks=1 --exclusive python view_pointer.py "2d_64block128_29.pt" &
srun --ntasks=1 --exclusive python view_pointer.py "2d_64block128_30.pt" &
srun --ntasks=1 --exclusive python view_pointer.py "2d_64block128_31.pt" &
srun --ntasks=1 --exclusive python view_pointer.py "2d_64block128_32.pt" &
srun --ntasks=1 --exclusive python view_pointer.py "2d_64block128_33.pt" &
srun --ntasks=1 --exclusive python view_pointer.py "2d_64block128_34.pt" &
srun --ntasks=1 --exclusive python view_pointer.py "2d_64block128_35.pt" &
srun --ntasks=1 --exclusive python view_pointer.py "2d_64block128_36.pt" &
srun --ntasks=1 --exclusive python view_pointer.py "2d_64block128_37.pt" &
srun --ntasks=1 --exclusive python view_pointer.py "2d_64block128_38.pt" &
srun --ntasks=1 --exclusive python view_pointer.py "2d_64block128_39.pt" &
srun --ntasks=1 --exclusive python view_pointer.py "2d_64block128_40.pt" &

wait

echo "done"
