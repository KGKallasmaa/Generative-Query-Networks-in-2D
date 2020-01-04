# Generative-Query-Networks-in-2D
Work in progress

## Medium article:
https://medium.com/computational-neuroscience-students/room-layout-prediction-using-neural-networks-4ae2223daecf

## Comments
Alvin 

**13.11:**

I took the test data from
[google
cloud](https://console.cloud.google.com/storage/browser/gqn-dataset/rooms_ring_camera/)
and edited the `data_reader.py` from [our
project](https://github.com/shivamsaboo17/Neural-Scene-Representation-and-Rendering)
and scripts [from
here](https://github.com/wohlert/generative-query-network-pytorch) to convert
downloaded `.tfrecord`s to torch `.pt` format and supplied some examples of this
result in _sodi/_. Also wrote the `driver.py` for some
interaction with the `data_reader.py` class.

Further exploration will ensue to try out the aforementionted git projects
scripts to run visualisations and representations of the available data.

To get stuff running install `tensorflow==1.15`(!), `pytorch`.

**03.01:**

**Creating the SM model on '5-part' dataset**

The work was moved to UT's Rocket server at
`/gpfs/hpc/home/vootorav/proj/gen-query-net`. Modified the enviroment from
@wohlert. If running things in server, create enviroment from *enviroment.yml*
file (from @wohlert again) using `conda env create -f enviroment.yml` and take a
coffee break, as it goes on for a looooong time.

The current pipeline of scripts is:

1. `data_conv.sh` Download data and convert it to `pytorch` format, clean up
   afterwards. This script loads right now the *shepardmetzler_part5* dataset by
   default, and uses `tfrecord-converter.py` for conversion. 
   
   Note that:
   
   + The `rooms_ring_data` set is 213 TB big, so please don't try to download
     it. For that, mayby use the `cloadload.sh` script in master directory and
     modify it for only fraction of data... but don't know if it works...

   + If converting other datasets (than SM-5part), see the `SEQ_INF` parameter on the
     `tfrecord-converter.py` script and modify it as needed.

2. `run-gqn.py` for training the dataset. This script uses the
   `shepardmetzler.py` class, which creates the model from the data. This script
   should also create a log using *Tensorboard* for logging and hopefully
   creates a model still, that can be used in the `mental-rotation.ipynb`
   notebook. This notebook should will be converted to a script tho...
   
   Note that:

   + Train on the SM-5part set only using 10% of the data. This will hasten
     stuff :) Apply it using flag `--fraction [int]`. Results are poor, but at
     least it will show if things are functioning.

3. When you wake up, take another coffee and look at models fresh from the oven.
   For this run `mental_rotation.py` in IPython or just run and modify the
   jupiter notebook. For usage either do a X11 tunnel or just open the folder
   using _sshfs_ in your computer (and prepare your local pyton enviroment too).

**TODO:**

+ devise a way to feed new data, ie the pictures of shapes.

+ modifiy the gqn to take in 1D data, hopefully with some simple modifications
  to some layers and replacing some functions.
