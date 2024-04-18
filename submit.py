# =========================================================================
# Copyright (C) 2024. FuxiCTR Authors. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
import logging
import fuxictr_version
from fuxictr import datasets
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import RankDataLoader
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.preprocess import FeatureProcessor, build_dataset
import src
import gc
import argparse
import os
from pathlib import Path
import polars as pl
import shutil
import multiprocessing as mp
import pandas as pd


def grank(x):
    scores = x["score"].tolist()
    tmp = [(i, s) for i, s in enumerate(scores)]
    tmp = sorted(tmp, key=lambda y: y[-1], reverse=True)
    rank = [(i+1, t[0]) for i, t in enumerate(tmp)]
    rank = [str(r[0]) for r in sorted(rank, key=lambda y: y[-1])]
    rank = "[" + ",".join(rank) + "]"
    return str(x["impression_id"].iloc[0]) + " " + rank

if __name__ == '__main__':
    ''' Usage: python run_expid.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='DeepFM_test', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    args = vars(parser.parse_args())
    
    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    params['gpu'] = args['gpu']
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])

    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    if params["data_format"] == "csv":
        # Build feature_map and transform data
        feature_encoder = FeatureProcessor(**params)
        params["train_data"], params["valid_data"], params["test_data"] = \
            build_dataset(feature_encoder, **params)
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))
    
    model_class = getattr(src, params['model'])
    model = model_class(feature_map, **params)
    model.count_parameters() # print number of parameters used in model
    model.to(device=model.device)
    model.load_weights(model.checkpoint)
    
    params["batch_size"] = 16000
    test_gen = RankDataLoader(feature_map, stage='test', **params).make_iterator()
    ans = pl.scan_csv("./data/Ebnerd_large_x1/test.csv")
    ans = ans.select(['impression_id', 'user_id']).collect().to_pandas()
    logging.info("Predicting scores...")
    ans["score"] = model.predict(test_gen)
    logging.info("Ranking samples...")
    ans = ans.groupby(['impression_id', 'user_id'], sort=False).apply(grank).reset_index(drop=True)
    logging.info("Writing results...")
    os.makedirs("submit", exist_ok=True)
    with open('submit/predictions.txt', "w") as fout:
        fout.write("\n".join(ans.to_list()))
    shutil.make_archive(f'submit/{experiment_id}', 'zip', 'submit/', 'predictions.txt')
    logging.info("All done.")