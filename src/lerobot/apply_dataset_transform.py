
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

from lerobot.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME
from lerobot.processor import AxisAngleToRot6d

import tyro
import pdb
from tqdm import tqdm
import copy
import torch

def convert(input_repo_id: str, output_repo_id: str, aa_to_rot6d: bool = False, push_to_hub: bool = True):
    assert input_repo_id != output_repo_id and output_repo_id != "", f"Invalid output repo ID"
    existing_dataset = LeRobotDataset(input_repo_id)
    new_features = copy.deepcopy(existing_dataset.features)

    if aa_to_rot6d:
        assert new_features['action']['shape'][0] == 7
        assert new_features['observation.state']['shape'][0] == 7
        new_features['action']['shape'] = (10,)
        new_features['action']['names'] = [
            'pose.x', 'pose.x', 'pose.x',
            'pose.rot6d_1x', 'pose.rot6d_1y', 'pose.rot6d_1z',
            'pose.rot6d_2x', 'pose.rot6d_2y', 'pose.rot6d_2z', 
            'gripper.pos'
        ]
        new_features['observation.state']['shape'] = (10,)
        new_features['observation.state']['names'] = [
            'pose.x', 'pose.x', 'pose.x',
            'pose.rot6d_1x', 'pose.rot6d_1y', 'pose.rot6d_1z',
            'pose.rot6d_2x', 'pose.rot6d_2y', 'pose.rot6d_2z', 
            'gripper.pos'
        ]

    # create a new dataset with same metadata settings
    new_dataset = LeRobotDataset.create(
        output_repo_id,
        existing_dataset.fps,
        root=HF_LEROBOT_HOME / output_repo_id,
        robot_type=existing_dataset.meta.robot_type,
        features=new_features,
        use_videos=len(existing_dataset.meta.video_keys) > 0,
        image_writer_processes=0,
        image_writer_threads=8,
        batch_encoding_size=existing_dataset.batch_encoding_size,
        tolerance_s=existing_dataset.tolerance_s
    )

    rot6d_transform = AxisAngleToRot6d()

    # apply transform to each step in dataset, write result to new dataset
    current_ep_idx = 0
    assert existing_dataset[0]['episode_index'].item() == current_ep_idx, f"{existing_dataset[0]['episode_index'].item()} vs. {current_ep_idx}"
    for idx in tqdm(range(len(existing_dataset))):
        frame = existing_dataset[idx]
        if frame['episode_index'] != current_ep_idx:
            new_dataset.save_episode()
            assert frame['episode_index'] == current_ep_idx + 1, "We assume dataset frames are sorted by episode index and that all frames for an episode are contiguous"
            current_ep_idx = frame['episode_index']
        
        if aa_to_rot6d:
            frame['action'] = rot6d_transform._convert(frame['action'].unsqueeze(0)).squeeze().to(torch.float32)
            frame['observation.state'] = rot6d_transform._convert(frame['observation.state'].unsqueeze(0)).squeeze().to(torch.float32)
        

        for k in frame:
            if "image" in k and frame[k].shape[0] == 3:
                frame[k] = frame[k].permute((1, 2, 0))
        
        del frame['timestamp']
        del frame['episode_index']
        del frame['frame_index']
        del frame['index']
        del frame['task_index']

        new_dataset.add_frame(frame)

    if push_to_hub:
        new_dataset.push_to_hub(tags=None, private=False)


if __name__ == "__main__":
    tyro.cli(convert)
































