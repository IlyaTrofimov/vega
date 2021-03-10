# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Nas Pipe Step defined in Pipeline."""
import logging
import os
import traceback
import torch
import json
import numpy as np
from .nas_pipe_step import NasPipeStep
from .generator import Generator
from ..scheduler.master import Master
from ..common.class_factory import ClassFactory, ClassType

logger = logging.getLogger(__name__)

@ClassFactory.register(ClassType.PIPE_STEP)
class MFKDNasPipeStep(NasPipeStep):
    """PipeStep is the base components class that can be added in Pipeline."""

    def __init__(self):
        super(MFKDNasPipeStep, self).__init__()
        torch.manual_seed(7)
        torch.cuda.manual_seed(7)

    def _save_model_desc_file(self, desc, desc_file):
        output = {}
        for key in desc:
            if key in ["type", "modules", "custom"]:
                output[key] = desc[key]

        with open(desc_file, "w") as f:
            json.dump(output, f)

    def do(self):
        """Do the main task in this pipe step."""
        logger.info("NasPipeStep started...")

        best_model_file = os.path.join(self.task.local_base_path, 'best_model')

        if os.path.isfile(best_model_file):
            with open(best_model_file) as infile:
                best_model_idx = int(infile.read())
                self.generator.search_alg.best_model_idx = best_model_idx

            logger.info('Reading best model %d from %s' % (best_model_idx, best_model_file))

        while not self.generator.is_completed:
            id, model = self.generator.sample()
            if isinstance(id, list) and isinstance(model, list):
                for id_ele, model_ele in zip(id, model):
                    cls_trainer = ClassFactory.get_cls('trainer')
                    trainer = cls_trainer(model_ele, id_ele)
                    logger.info("submit trainer(id={})!".format(id_ele))
                    self.master.run(trainer)
                self.master.join()
            elif id is not None and model is not None:
                cls_trainer = ClassFactory.get_cls('trainer')
                trainer = cls_trainer(model, id)
                logger.info("submit trainer(id={})!".format(id))
                self.master.run(trainer)
            finished_trainer_info = self.master.pop_finished_worker()
            self.update_generator(self.generator, finished_trainer_info)

        logger.info('Writing best model %d to %s' % (self.generator.search_alg.best_model_idx, best_model_file))

        with open(best_model_file, 'w') as outfile:
            outfile.write(str(self.generator.search_alg.best_model_idx))

        self._save_model_desc_file(self.generator.search_alg.best_model_desc, best_model_file + '_desc.json')

        self.master.join()
        finished_trainer_info = self.master.pop_all_finished_train_worker()
        self.update_generator(self.generator, finished_trainer_info)
        self._backup_output_path()
        self.master.close_client()
