#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from lerobot.configs.types import FeatureType, NormalizationMode, PipelineFeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .converters import from_tensor_to_numpy, to_tensor
from .core import EnvTransition, PolicyAction, TransitionKey
from .pipeline import PolicyProcessorPipeline, ProcessorStep, ProcessorStepRegistry


from scipy.spatial.transform import Rotation as R


import pdb


@dataclass
@ProcessorStepRegistry.register(name="aa_to_rot6d_processor")
class AxisAngleToRot6d(ProcessorStep):
    """
    A processor step that applies normalization to observations and actions in a transition.

    This class uses the logic from `_NormalizationMixin` to perform forward normalization
    (e.g., scaling data to have zero mean and unit variance, or to the range [-1, 1]).
    It is typically used in the pre-processing pipeline before feeding data to a policy.
    """

    def _convert(self, inp):
        dev = inp.device
        inp = inp.cpu()
        assert inp.shape[-1] == 7, f"Invalid shape: {inp.shape}"
        mx = R.from_rotvec(inp[..., 3:6].flatten(0, -2), degrees=False).as_matrix()
        mx = torch.from_numpy(mx).reshape(*inp.shape[:-1], 3, 3)
        out = torch.concat([
            inp[..., :3],       # translations
            mx[..., :3, 0],       # first col of rotation matrix
            mx[..., :3, 1],       # second col of rotation matrix
            inp[..., 6:]        # gripper
        ], dim=-1)        
        return out.to(dev, non_blocking=True)


    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Convert observation/actions of a transition from axis angle format to rot6d. 
        Assumes that action/observation presentation is axis angle (3 translation, 3 rotation) followed by a single gripper separation float
        """
        transition[TransitionKey.ACTION] = self._convert(transition[TransitionKey.ACTION])
        transition[TransitionKey.OBSERVATION]["observation.state"] = self._convert(transition[TransitionKey.OBSERVATION]["observation.state"])
        return transition


    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Defines how this step modifies the description of pipeline features.

        This method is used to track changes in data shapes, dtypes, or modalities
        as data flows through the pipeline, without needing to process actual data.

        Args:
            features: A dictionary describing the input features for observations, actions, etc.

        Returns:
            A dictionary describing the output features after this step's transformation.
        """
        import pdb
        pdb.set_trace()


        return features

@dataclass
@ProcessorStepRegistry.register(name="rot6d_to_aa_processor")
class Rot6dToAxisAngle(ProcessorStep):
    """
    A processor step that applies normalization to observations and actions in a transition.

    This class uses the logic from `_NormalizationMixin` to perform forward normalization
    (e.g., scaling data to have zero mean and unit variance, or to the range [-1, 1]).
    It is typically used in the pre-processing pipeline before feeding data to a policy.
    """

    def _convert(self, inp):

        
        # TODO: implement this. normalize, gram-schidt, cross product -> convert to axis angle
        
        pdb.set_trace()

        return None


    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Convert observation/actions of a transition from rot6d to axis angle format
        Assumes that action/observation presentation is rot6d (3 translation, 6 rotation, single gripper)
        """
        transition[TransitionKey.ACTION] = self._convert(transition[TransitionKey.ACTION])
        transition[TransitionKey.OBSERVATION] = self._convert(transition[TransitionKey.OBSERVATION])
        return transition


    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Defines how this step modifies the description of pipeline features.

        This method is used to track changes in data shapes, dtypes, or modalities
        as data flows through the pipeline, without needing to process actual data.

        Args:
            features: A dictionary describing the input features for observations, actions, etc.

        Returns:
            A dictionary describing the output features after this step's transformation.
        """
        import pdb
        pdb.set_trace()


        return features



