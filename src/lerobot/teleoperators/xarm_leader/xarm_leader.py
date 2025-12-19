#!/usr/bin/env python

import math
import logging
import os
import sys
import time
import numpy as np
from typing import Any
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from collections import defaultdict
from threading import Thread, Event, Lock

from lerobot.teleoperators import Teleoperator
from .config_xarm_leader import xArmLeaderTeleopConfig

from lerobot.robots.ufactory_robot import UFRobot


# if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
#     logging.info("No DISPLAY set. Skipping pynput import.")
#     raise ImportError("pynput blocked intentionally due to no display.")
from pynput import keyboard


class xArmLeaderTeleop(Teleoperator, UFRobot):
    
    config_class = xArmLeaderTeleopConfig
    name = "xarm_leader"

    def __init__(self, config: xArmLeaderTeleopConfig):
        super().__init__(config)
        Thread.__init__(self) # Do NOT REMOVE!
        UFRobot.__init__(self, config.robot_config)

        self.callback_lock = Lock()
        self.desired_gripper_pos = 0        # default open
        self.listener = None

    def on_space(self, key):
        if key == keyboard.Key.space:
            with self.callback_lock:
                # swap hold open/closed when gripper is pressed
                if self.desired_gripper_pos == 244:
                    self.desired_gripper_pos = 0
                else:
                    self.desired_gripper_pos = 244

    def configure(self) -> None:
        # put leader arm into free drive
        self.real_arm.motion_enable()
        self.real_arm.set_mode(0)
        time.sleep(.01)

        self.real_arm.set_teach_sensitivity(5)       # least stiff setting
        self.real_arm.set_mode(2, 1)
        self.real_arm.set_state(0)

        if not self._get_arm_err() == 0:
            raise RuntimeError(f"Failed to set correct state to UF robot! Controller Error code: {self._get_arm_err()} !")

        
        # start tracking thread
        self.start()
        time.sleep(0.2)

        self.listener = keyboard.Listener(on_press=self.on_space)
        self.listener.start()

    # NOTE: this is an absolute action.... probably want to change this to delta?
    def get_action(self) -> dict[str, Any]:
        obs_dict =  UFRobot.get_observation(self)
        with self.callback_lock:
            obs_dict.update({"gripper.pos": self.desired_gripper_pos})
        return obs_dict
        
    @property
    def action_features(self) -> dict:
        return UFRobot.action_features(self)

    @property
    def feedback_features(self) -> dict:
        return UFRobot.feedback_features(self)

    @property
    def is_connected(self) -> bool:
        return UFRobot.is_connected(self)

    @property
    def is_calibrated(self) -> bool:
        return UFRobot.is_calibrated(self)


    def calibrate(self):
        return UFRobot.calibrate(self)

    def connect(self, calibrate: bool = False) -> None:
        return UFRobot.connect(self)

    def disconnect(self):
        return UFRobot.disconnect(self)
    
    def run(self):
        UFRobot.run(self)

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError        
