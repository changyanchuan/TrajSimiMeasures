# Trajectory Similarity Measures

Trajectory similarity computation benchmark from an efficiency perspective, including 10 heuristic measures and 6 learned measures. All measures can be run on CPU and GPU.

Code implementation of the VLDB 2024 paper: **Trajectory Similarity Measurement: An Efficiency Perspective**. [[paper]](https://www.vldb.org/pvldb/vol17/p2293-qi.pdf) [[full report]](https://github.com/changyanchuan/TrajSimiMeasures/blob/master/paper_technical_report.pdf)

```
@article{trajsimi_survey,
  title={{Trajectory Similarity Measurement: An Efficiency Perspective}},
  author={Chang, Yanchuan and Tanin, Egemen and Cong, Gao and Jensen, Christian S. and Qi, Jianzhong},
  journal={PVLDB},
  volume={17},
  number={9},
  pages={2293--2306},
  year={2024}
}
```


## Requirements
- An Nvidia GPU with CUDA 10.2
- Ubuntu 20.04 LTS with Python 3.7.7
- `pip install -r requirements377.txt` (See section FAQ below first.)
- Datasets and the snaphots of TrajGAT and RSTS can be downloaded from [here](https://drive.google.com/drive/folders/1wgT09SLHQLKIY1bnjwflp2ExRbf3iqJv), `tar -zxvf trajsimi_dataset.tar.gz -C ./data/` and and `tar -zxvf trajsimi_snapshot.tar.gz -C ./exp/snapshot`
- Compile pyx files `cd core_cpu && python setup.py build_ext --inplace --force && cd ..`


## Quick Start

We provide python scripts that are originally for the empirical studies in our paper, to help get stared. Try the following prompts.

\- Compute the similarity values of trajectory pairs without embedding resue:
```bash
python test_trajsimi_time.py --dataset porto --effi_exp numtrajs --gpu
```
```bash
python test_trajsimi_time.py --dataset porto --effi_exp numtrajs 
```


\- Compute the similarity values of trajectories in two data sets:
```bash
python test_trajsimi_time_DQ.py --dataset porto --effi_exp numtrajs --gpu
```
```bash
python test_trajsimi_time.py --dataset porto --effi_exp numtrajs 
```


\- Query kNN trajectories (learned measures):
```bash
python test_knn_time.py --dataset porto --knn_exp dbset_sizes
```


\- Query kNN trajectories (heuristic measures):
```markdown
See `./query/cpp./readme.md`
```


\- Cluster trajectories:
```bash
python test_clustering_time.py --dataset geolife --clustering_exp numtrajs --gpu
```
```bash
python test_clustering_time.py --dataset geolife --clustering_exp numtrajs 
```


## FAQ
### Datasets
We have provided the pre-processed Porto, Germany and Geolife datasets (see Requirements Section). Chengdu and Xian datasets are not public datasets, and thus we do not provide them, while they can be downloaded at [here](https://outreach.didichuxing.com/) under the agreement of the data provider. 
We also provide the dataset preprocessing scripts in `./preprocessing/`. Alternatively, you can pre-process original datasets from the scrath, such as [Porto](https://www.kaggle.com/competitions/pkdd-15-predict-taxi-service-trajectory-i/data?select=train.csv.zip) by using `./preprocessing/porto.py`.

To use your own datasets, you may need to create your own pre-processing scripts which should be similar to `./preprocessing/porto.py`. The meta data of the dataset (see function post_value_updates() in `./config.py`) is required as well.


### Fail to install packages
If you fail to install the kmedoids==0.4.3 package from pip repositories, see the [guide](https://github.com/kno10/python-kmedoids?tab=readme-ov-file#compilation-from-source) to compile from the source and install.

We specified the cuda version of dgl package in `requirements377.py`. You may need to modify the version to the one your environment's need or install it separately.



## Contact
Email changyanchuan@gmail.com if you have any queries.