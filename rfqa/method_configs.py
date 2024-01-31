"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations
from collections import OrderedDict
from typing import Dict, Union

import tyro

from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManagerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
    ExponentialDecaySchedulerConfig,
    MultiStepSchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from rfqa.nerfacto_model import NerfactoModelConfig
from rfqa.instant_ngp_model import InstantNGPModelConfig
from rfqa.pipeline import RFQAPipelineConfig
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipelineConfig


from nerfstudio.plugins.types import MethodSpecification
from rfqa.blender_dataparser import BlenderDataParserConfig


rfqa_nerfacto = MethodSpecification(
    config=TrainerConfig(
        method_name="rfqa-nerfacto",
        steps_per_eval_batch=100,
        steps_per_eval_image=1000,
        steps_per_save=25000,
        steps_per_eval_all_images=5000,
        max_num_iterations=25000+1,
        mixed_precision=True,
        pipeline=RFQAPipelineConfig(
            datamanager=ParallelDataManagerConfig(
                dataparser=BlenderDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=NerfactoModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                background_color='random',
                num_proposal_iterations=2,
                proposal_net_args_list=[
                    {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "features_per_level": 2, "use_linear": False},
                    {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "features_per_level": 2, "use_linear": False},
                ],
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Nerfstudio method template.",
)


rfqa_nerfacto_big = MethodSpecification(
    config=TrainerConfig(
        method_name="rfqa-nerfacto-big",
        steps_per_eval_batch=500,
        steps_per_eval_image=5000,
        steps_per_save=5000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=RFQAPipelineConfig(
            datamanager=ParallelDataManagerConfig(
                dataparser=BlenderDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=NerfactoModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                num_nerf_samples_per_ray=128,
                num_proposal_samples_per_ray=(512, 256),
                proposal_net_args_list=[
                    {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 512, "use_linear": False},
                    {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 7, "max_res": 2048, "use_linear": False},
                ],
                hidden_dim=128,
                hidden_dim_color=128,
                max_res=4096,
                proposal_weights_anneal_max_num_iters=5000,
                log2_hashmap_size=21,
                background_color='random'
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000),
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Nerfstudio method template big.",
)




rfqa_instant_ngp = MethodSpecification(
    config=TrainerConfig(
        method_name="rfqa-instant-ngp",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=RFQAPipelineConfig(
            datamanager=ParallelDataManagerConfig(
                dataparser=BlenderDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=InstantNGPModelConfig(
                eval_num_rays_per_chunk=4096,
                near_plane=1.0,
                far_plane=4.0,
                num_nerf_samples_per_ray=128,
                num_proposal_samples_per_ray=(512, 256),
                proposal_net_args_list=[
                    {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "features_per_level": 2, "use_linear": False},
                    {"hidden_dim": 16, "log2_hashmap_size": 18, "num_levels": 8, "max_res": 256, "features_per_level": 2, "use_linear": False},
                ],
                background_color='random'
            ),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000),
            },
            "proposal_networks": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 11),
        vis="viewer",
    ),
    description="Nerfstudio method template instant-ngp.",
)