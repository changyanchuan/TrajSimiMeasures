## Heuristic Trajectory kNN queries

Code implementation of heuristic measure-based trajectory kNN queries. The folder mainly include the implementation of the trajectory indices (DFT and DITA) and heuristic trajectory measures (DTW, ERP, Frechet and Hausdorff). `./knnquery_exp.cpp` is the main file.


### install:
Install the following libraries and update the environment variables. Please modify the variables in makefile, such as INCLUDES and LIBS that I previously used fixed values.

* CRoaring -- see https://github.com/RoaringBitmap/CRoaring
* hdf5 -- conda install -c anaconda hdf5
* boost -- conda install -c statiskit libboost-dev
* HighFive -- see https://github.com/BlueBrain/HighFive
* python3.7
* numpy c-api


### build:
* (make clean && make -j8 all) &> result


### run:
./knnquery_exp dbset_sizes DTW xian_7_20inf.h5 0 1000000 10 20 200 20 200 50 10  &> result
