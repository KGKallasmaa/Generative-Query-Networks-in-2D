# Generative-Query-Networks-in-2D
Work in progress

## Medium article:
https://medium.com/@karl.gustav1789/hi-2576005cc748

## Comments
Alvin 13.11:

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
