# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""MFKD2 Algorithm"""
import os
import copy
import json
import random
import torch.nn as nn
import vega
from vega.core.common.utils import update_dict
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common import UserConfig, TaskOps, FileOps
from vega.search_space.networks import NetTypes, NetworkFactory, NetworkDesc
from vega.search_space.search_algs import SearchAlgorithm
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF
import itertools
from sklearn import preprocessing
import numpy as np
import logging

@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class MFKD2(SearchAlgorithm):
    def __init__(self, search_space):
        super(MFKD2, self).__init__(search_space)
        self.search_space = copy.deepcopy(search_space.search_space)

        self._get_all_arcs()

        self.sample_count = 0
        self.model_idx = None
        self.best_model_idx = None

        self.accuracy = {}
        self.points = []

        self.init_points = self.cfg.get("init_samples", 1)
        self.max_points = self.cfg["max_samples"]

    def _run_GPR_bayesopt(self, targets, checked_points):

        X_train = []
        y_train = []

        for idx in checked_points:
            X_train.append(self.X[idx])
            y = targets[idx]

            y_train.append(y / 100)

        gpr = GPR(kernel = RBF(1.0), n_restarts_optimizer = 3, normalize_y = True)
        gpr.fit(X_train, y_train)

        mean, std = gpr.predict(self.X, return_std = True)

        max_score = 0
        best_new_idx = 0

        for idx in range(len(self.X)):
            if idx not in checked_points:
                score = mean[idx] + 0.5 * std[idx]

                if score > max_score:
                    max_score = score
                    best_new_idx = idx

        checked_points.append(best_new_idx)

    def _search(self):

        if len(self.points) < self.max_points:

            if len(self.points) == 0:
                if self.best_model_idx is None:
                    self.points = [np.random.choice(range(len(self.X)))]
                else:
                    self.points = [self.best_model_idx]
            elif len(self.points) < self.init_points:
                self.points.append(np.random.choice(range(len(self.X))))
            else:
                self._run_GPR_bayesopt(self.accuracy, self.points)

            return self.points[-1]

    def search(self):

        idx = self._search()
        self.model_idx = idx

        choices = self.choices[idx]
        desc = self._desc_from_choices(self.choices[idx])
        logging.info('Checking architecture %d, %s' % (idx, str(desc)))

        self.sample_count += 1
        self._save_model_desc_file(self.sample_count, desc)

        return self.sample_count, NetworkDesc(desc)

    def _sub_config_choice(self, config, choices, pos):
        """Apply choices to config"""

        for key, value in sorted(config.items()):
            if isinstance(value, dict):
                _, pos = self._sub_config_choice(value, choices, pos)
            elif isinstance(value, list):
                choice = value[choices[pos]]
                config[key] = choice
                pos += 1

        return config, pos

    def _desc_from_choices(self, choices):
        """Create description object from choices"""

        desc = {}
        pos = 0

        for key in self.search_space.modules:
            config_space = copy.deepcopy(self.search_space[key])
            module_cfg, pos = self._sub_config_choice(config_space, choices, pos)
            desc[key] = module_cfg

        desc = update_dict(desc, copy.deepcopy(self.search_space))

        return desc

    def _sub_config_all(self, config, vectors, choices):
        """Get all possible choices and their values"""

        for key, value in sorted(config.items()):
            if isinstance(value, dict):
                self._sub_config_all(value, vectors, choices)
            elif isinstance(value, list):
                vectors.append([float(x) for x in value])
                choices.append(list(range(len(value))))

    def _get_all_arcs(self):
        """Get all the architectures from the search space"""

        vectors = []
        choices = []

        for key in self.search_space.modules:
            config_space = copy.deepcopy(self.search_space[key])
            self._sub_config_all(config_space, vectors, choices)

        self.X = list(itertools.product(*vectors))
        self.X = preprocessing.scale(self.X, axis = 0)
        self.choices = list(itertools.product(*choices))

        logging.info('Number of architectures in the search space %d' % len(self.X))

    def update(self, worker_path):

        with open(os.path.join(worker_path, 'performance.txt')) as infile:
            perf = infile.read()

        acc = eval(perf)[0]
        self.accuracy[self.model_idx] = acc

        self.best_model_idx = max(self.accuracy.items(), key = lambda x : x[1])[0]
        self.best_model_desc = self._desc_from_choices(self.choices[self.best_model_idx])

        if self.is_completed:
            logging.info('The best architecture %d, description %s' % (self.best_model_idx, str(self.best_model_desc)))

    @property
    def is_completed(self):
        """Check if the search is finished."""
        return len(self.points) >= self.max_points

    def _save_model_desc_file(self, id, desc):
        output_path = TaskOps(UserConfig().data.general).local_output_path
        desc_file = os.path.join(output_path, "nas", "model_desc_{}.json".format(id))
        FileOps.make_base_dir(desc_file)
        output = {}
        for key in desc:
            if key in ["type", "modules", "custom"]:
                output[key] = desc[key]
        with open(desc_file, "w") as f:
            json.dump(output, f)
