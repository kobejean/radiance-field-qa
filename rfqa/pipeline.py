# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""
Abstracts for the Pipeline class.
"""
from __future__ import annotations

import typing
import math
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union, cast

import torch
import torch.distributed as dist
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from torch import nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn import Parameter
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig, VanillaDataManager
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import profiler, writer
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation


def module_wrapper(ddp_or_model: Union[DDP, Model]) -> Model:
    """
    If DDP, then return the .module. Otherwise, return the model.
    """
    if isinstance(ddp_or_model, DDP):
        return cast(Model, ddp_or_model.module)
    return ddp_or_model

@dataclass

class RFQAPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: RFQAPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = field(default_factory=DataManagerConfig)
    """specifies the datamanager config"""
    model: ModelConfig = field(default_factory=ModelConfig)
    """specifies the model config"""


class RFQAPipeline(VanillaPipeline):
    """The pipeline class for the vanilla nerf setup of multiple cameras for one or a few scenes.

    Args:
        config: configuration to instantiate pipeline
        device: location to place model and data
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        world_size: total number of machines available
        local_rank: rank of current machine
        grad_scaler: gradient scaler used in the trainer

    Attributes:
        datamanager: The data manager that will be used
        model: The model that will be used
    """

    @profiler.time_function
    def get_average_eval_image_metrics(
        self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    ):
        """Iterate over all the images in the eval dataset and get the average.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        metrics_dict = super().get_average_eval_image_metrics(step, output_path, get_std)
        model_size = self.get_model_size()
        combined_score = float(metrics_dict["psnr"] / (math.log10(float(model_size))*1.67 + 9.2))
        metrics_dict["model_size"] = model_size
        metrics_dict["combined_score"] = combined_score
        return metrics_dict
    
    def get_model_size(self):
        data_size = 0
        for name, param in self.model.named_parameters():
            is_blacklisted = "lpips" in name
            if param.requires_grad and param.numel() > 0 and not is_blacklisted:
                size_in_bytes = param.numel() * param.element_size()
                data_size += size_in_bytes
        return data_size


    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""
        callbacks = super().get_training_callbacks(training_callback_attributes)

        def write_model_stats(step):
            # import io
            # buffer = io.BytesIO()
            # torch.save(self.model.state_dict(), buffer)
            # data_size = buffer.tell()
            # del buffer
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            non_trainable_params = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)

            scalar_dict = {
                "total_params": total_params,
                "trainable_params": trainable_params,
                "non_trainable_params": non_trainable_params,
            }
            data_size = 0
            for name, param in self.model.named_parameters():
                is_blacklisted = "lpips" in name
                if param.requires_grad and param.numel() > 0 and not is_blacklisted:
                    size_in_bytes = param.numel() * param.element_size()
                    scalar_dict["param_size."+name] = size_in_bytes
                    data_size += size_in_bytes
            scalar_dict["data_size"] = data_size


            writer.put_dict(name="Model Statistics", scalar_dict=scalar_dict, step=step+1)
            writer.write_out_storage()

        callbacks += [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN],
                func=write_model_stats,
            )
        ]
        return callbacks

