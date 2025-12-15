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
from threading import Thread

from lerobot.teleoperators import Teleoperator
from .config_xarm_leader import xArmLeaderTeleopConfig

from lerobot.robots.ufactory_robot import UFRobot

class xArmLeaderTeleop(Teleoperator, UFRobot):
    
    config_class = xArmLeaderTeleopConfig
    name = "xarm_leader"

    def __init__(self, config: xArmLeaderTeleopConfig):
        super().__init__(config)
        Thread.__init__(self) # Do NOT REMOVE!
        UFRobot.__init__(self, config.robot_config)

    def configure(self) -> None:
        # put leader arm into free drive
        
        self.real_arm.motion_enable()
        self.real_arm.set_mode(0)
        time.sleep(.01)
        self.real_arm.set_mode(2, 1)
        self.real_arm.set_state(0)

        if not self._get_arm_err() == 0:
            raise RuntimeError(f"Failed to set correct state to UF robot! Controller Error code: {self._get_arm_err()} !")
        if self.config.gripper_control:
            self.real_arm.robotiq_open()
            self.real_arm.robotiq_reset()
            self.real_arm.robotiq_set_activate()
            if not self._get_arm_err() == 0:
                raise RuntimeError(f"Failed to set correct state to Gripper! Controller Error code: {self._get_arm_err()} !")

        if self._use_rt_report:
            self.start()
        time.sleep(0.2)


    # NOTE: this is an absolute action.... probably want to change this to delta?
    def get_action(self) -> dict[str, Any]:
        return UFRobot.get_observation(self)
        
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
