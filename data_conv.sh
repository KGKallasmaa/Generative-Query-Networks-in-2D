#!/usr/bin/env bash
#SBATCH -J gen-net-job
#SBATCH --time=6:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alvinmeltsov@gmail.com
#SBATCH --mem=64000
#SBATCH --cpus-per-task=16

module load python-3.6.3
conda activate gqn

# It is suggested that you run this script from a command line
# with a python enviroment with deps installed activated.

LOCATION="./"   # example: /tmp/data
BATCH_SIZE=64 # example: 64

echo "Downloading data"
gsutil -m cp -R "gs://gqn-dataset/shepard_metzler_5_parts" $LOCATION

echo "Deleting small records less than 10MB"
DATA_PATH="$LOCATION/shepard_metzler_5_parts/**/*.tfrecord"
find $DATA_PATH -type f -size -10M | xargs rm

# # if something gets messed up, use the following
# echo "Removing previous converted files"
# rm -rf "$LOCATION/shepard_metzler_5_parts/**/*.pt.gz"

echo "Converting data"
python tfrecord-converter.py $LOCATION shepard_metzler_5_parts -b $BATCH_SIZE -m "train"
echo "Training data: done"
python tfrecord-converter.py $LOCATION shepard_metzler_5_parts -b $BATCH_SIZE -m "test"
echo "Testing data: done"

echo "Removing original records"
rm -rf "$LOCATION/shepard_metzler_5_parts/**/*.tfrecord"

