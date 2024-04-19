## RecSys2024_CTR_Challenge

The RecSys 2024 Challenge: https://www.recsyschallenge.com/2024/

The Ekstra Bladet RecSys Challenge aims to predict which article a user will click on from a list of articles that were seen during a specific impression. Utilizing the user's click history, session details (like time and device used), and personal metadata (including gender and age), along with a list of candidate news articles listed in an impression log, the challenge's objective is to rank the candidate articles based on the user's personal preferences. 

This baseline is built on top of [FuxiCTR](https://github.com/reczoo/FuxiCTR), a configurable, tunable, and reproducible library for CTR prediction. The library has been selected among [the list of recommended evaluation frameworks](https://github.com/ACMRecSys/recsys-evaluation-frameworks) by the ACM RecSys Conference. By using FuxiCTR, we develop a simple yet strong baseline (AUC: 0.7154) without heavy tuning. We open source the code to help beginers get familar with FuxiCTR and quickly get started on this task.

🔥 If you find our code helpful in your competition, please cite the following paper:

+ Jieming Zhu, Jinyang Liu, Shuai Yang, Qi Zhang, Xiuqiang He. [Open Benchmarking for Click-Through Rate Prediction](https://arxiv.org/abs/2009.05794). *The 30th ACM International Conference on Information and Knowledge Management (CIKM)*, 2021.


### Data Preparation

Note that the dataset is quite large. Preparing the full dataset needs about 1T disk space. Although some optimizations can be made to save space (e.g., store sequence features sperately), we leave it for future exploration.

1. Download the datasets at: https://recsys.eb.dk/#dataset

2. Unzip the data files to the following

    ```bash
    cd ~/RecSys2024_CTR_Challenge/data/Ebnerd/
    find -L .

    .
    ./train
    ./train/history.parquet
    ./train/articles.parquet
    ./train/behaviors.parquet
    ./validation
    ./validation/history.parquet
    ./validation/behaviors.parquet
    ./test
    ./test/history.parquet
    ./test/articles.parquet
    ./test/behaviors.parquet
    ./image_embeddings.parquet
    ./contrastive_vector.parquet
    ./prepare_data_v1.py
    ```

3. Convert the data to csv format

    ```bash
    cd ~/RecSys2024_CTR_Challenge/data/Ebnerd/
    python prepare_data_v1.py
    ```

### Environment

Please set up the environment as follows. We run the experiments on a P100 GPU server with 16G GPU memory and 750G RAM.

+ torch==1.10.2+cu113
+ fuxictr==2.2.3

```
conda create -n fuxictr python==3.9
pip install -r requirements.txt
source activate fuxictr
```

### Version 1

1. Train the model on train and validation sets:

    ```
    python run_param_tuner.py --config config/DIN_ebnerd_large_x1_tuner_config_01.yaml --gpu 0
    ```

    We get validation avgAUC: 0.7113. Note that in FuxiCTR, AUC is the global AUC, while avgAUC is averaged over impression ID groups.

2. Make predictions on the test set:

    Get the experiment_id from running logs or the result csv file, and then you can run prediction on the test.

    ```
    python submit.py --config config/DIN_ebnerd_large_x1_tuner_config_01 --expid DIN_ebnerd_large_x1_001_1860e41e --gpu 1
    ```

3. Make a submission. We get test AUC: 0.7154.

    <div align="left">
        <img width="99%" src="./img/submit_v1.png">
    </div>

### Potential Improvements

+ To build the baseline, we simply reuse the DIN model, which is popular for sequential user interest modeling. We encourage to explore some other alternatives for user behavior sequence modeling.
+ We currently only consider the click behaviors, but leave out other important singnals of reading times and percentiles. It is desired to consider them with multi-objective modeling.
+ We use contrast vectors and image embeddings in a straightforward way. It is interesting to explore other embedding features.
+ How to bridge the user sequence modeling with large pretrained models (e.g., Bert, LLMs) is a promising direction to explore.

### Discussion
We also welcome contributors to help improve the space and time efficiency of FuxiCTR for handling large-scale sequence datasets. If you have any question, please feel free to open an issue.
