#!/usr/bin/env python
import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import h5py
import click
import rospy
import ros_numpy
import numpy as np
from pathlib import Path
from typing import Dict
from pynput.keyboard import Listener
from sensor_msgs.msg import JointState, Image
from message_filters import Subscriber, ApproximateTimeSynchronizer


class Collector:

    def __init__(
        self,
        save_dir: Path,
        topics: Dict[str, str]
    ) -> None:
        self.save_dir = save_dir
        self.images = list()
        self.joint_poses = list()
        self.ee_poses = list()
        self.collecting = False
        self.cnt = 0
        # ROS initialization
        rospy.init_node("collect_data", anonymous=True)
        self.image_sub = Subscriber(topics["image_topic"], Image)
        self.joint_sub = Subscriber(topics["joint_topic"], JointState)
        sync = ApproximateTimeSynchronizer(
            [self.image_sub, self.joint_sub], 20, 0.5, True
        )
        sync.registerCallback(self.callback)
        # start keyboard listener
        self.listener = Listener(on_press=self.on_press)
        self.listener.start()

    def callback(
        self,
        image: Image,
        joint_state: JointState
    ) -> None:
        if self.collecting:
            if image is None or joint_state is None:
                rospy.logwarn("No data received!")
            else:
                image = ros_numpy.numpify(image)
                joint_pos = np.array(joint_state.position[0: 7])
                joint_pos[6] = joint_pos[6] > 0.5  # gripper closure
                self.images.append(image)
                self.joint_poses.append(joint_pos)

    def on_press(self, key) -> None:
        try:
            if hasattr(key, "char") and key.char:
                if key.char == "r":
                    self.resume()
                elif key.char == "p":
                    self.pause()
                elif key.char == "s":
                    self.save()
                elif key.char == "e":
                    self.exit()
                else:
                    rospy.logwarn(f"Invalid command {key.char}.")
        except AttributeError:
            pass
    
    def resume(self):
        if not self.collecting:
            rospy.loginfo("Start collecting data")
            self.images = list()
            self.joint_poses = list()
            self.ee_poses = list()
            self.collecting = True
        else:
            rospy.logwarn("Data is being collected!")    
    
    def pause(self):
        if self.collecting:
            self.collecting = False
            rospy.loginfo("Pause")
        else:
            rospy.logwarn("The process is already paused.")
            
    def save(self):
        if not self.collecting:
            save_path = self.save_dir.joinpath(f"episode_{self.cnt}.hdf5")
            images = np.array(self.images)
            joint_poses = np.array(self.joint_poses)
            with h5py.File(str(save_path), "w") as f:
                f["/obs/head_camera"] = images
                f["/proprios"] = joint_poses
                f.close()
            rospy.loginfo("Data saved")
        else:
            rospy.logwarn("Please pause before saving data.")
            
    def exit(self):
        self.collecting = False
        if self.image_sub:
            self.image_sub.unregister()
        if self.joint_sub:
            self.joint_sub.unregister()
        rospy.loginfo("Successfully exit.")
        
                
@click.command("Collect full episodes using Kinova Gen3 Lite.")
@click.option("-s", "--save_dir", type=str, default="data", help="Directory used for saving collected data.")

def main(save_dir):
    save_dir = Path(os.path.expanduser(save_dir)).absolute()
    if not save_dir.is_dir():
        save_dir.mkdir(parents=True)
    topics = {
        "image_topic": "/camera/color/image_raw",
        "joint_topic": "/my_gen3_lite/joint_states"
    }
    collector = Collector(save_dir, topics)
    rospy.spin()


    
