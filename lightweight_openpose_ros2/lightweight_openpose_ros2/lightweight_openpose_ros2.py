#!/usr/bin/env python3
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
import rclpy

import message_filters

from std_srvs.srv import SetBool
from sensor_msgs.msg import Image
from lor_interfaces.msg import *

from cv_bridge import CvBridge, CvBridgeError

from .models.with_mobilenet import PoseEstimationWithMobileNet
from .modules.keypoints import extract_keypoints, group_keypoints
from .modules.load_state import load_state
from .modules.pose import Pose, track_poses
from .val import normalize, pad_width

from ament_index_python.packages import get_package_share_directory
import numpy as np
import torch
import cv2
import traceback
import time
import os


class LightweightOpenPoseRos2(Node):
    def __init__(self):
        super().__init__('lightweight_openpose_ros2')

        # parameters
        self.declare_parameter('checkpoint_path', os.path.join(get_package_share_directory('lightweight_openpose_ros2'), 'datas', 'checkpoint_iter_370000.pth'))
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('height_size', 256)
        self.declare_parameter('debug', True)
        self.declare_parameter('qos.reliability', 'RELIABLE')
        self.declare_parameter('qos.durability', 'VOLATILE')
        self.declare_parameter('qos.depth', 10)
        
        # Ensure use_sim_time parameter is available
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', False)
        
        # min confidence
        self.declare_parameter('min_confidence', 0.7)

        param_checkpoint_path = self.get_parameter('checkpoint_path').value
        self.param_device = self.get_parameter('device').value
        self.debug = self.get_parameter('debug').value
        self.min_confidence = self.get_parameter('min_confidence').value

        self.get_logger().info('''
        LIGHTWEIGHT OPEN POSE ROS2 START.
                               
        CHECKPOINT  : %s                               
        DEVICE      : %s
        '''%(param_checkpoint_path, self.param_device))

        # model setup
        self.net = PoseEstimationWithMobileNet()
        self.checkpoint = torch.load(param_checkpoint_path, map_location='cpu')
        load_state(self.net, self.checkpoint)
        self.net = self.net.eval() if self.param_device == 'cpu' else self.net.cuda()
        self.stride = 8
        self.upsample_ratio = 4
        self.num_keypoints = Pose.num_kpts
        self.previous_poses = []
        self.track = 1
        self.smooth = 1
        # when rise, execute openpose
        self.exec_flag = False

        # subscriber
        qos_reliability_str = self.get_parameter('qos.reliability').value.upper()
        qos_durability_str = self.get_parameter('qos.durability').value.upper()
        qos_depth = self.get_parameter('qos.depth').value

        reliability_policy = {
            'RELIABLE': QoSReliabilityPolicy.RELIABLE,
            'BEST_EFFORT': QoSReliabilityPolicy.BEST_EFFORT
        }.get(qos_reliability_str, QoSReliabilityPolicy.RELIABLE)

        durability_policy = {
            'VOLATILE': QoSDurabilityPolicy.VOLATILE,
            'TRANSIENT_LOCAL': QoSDurabilityPolicy.TRANSIENT_LOCAL
        }.get(qos_durability_str, QoSDurabilityPolicy.VOLATILE)

        qos_profile = QoSProfile(
            reliability=reliability_policy,
            durability=durability_policy,
            depth=qos_depth
        )
        self.image_sub = self.create_subscription(Image, 'image_raw', self.image_cb, qos_profile)

        # service
        self.execute_cli = self.create_service(SetBool, 'execute', self.execute_cb)

        # publisher
        self.poses_pub = self.create_publisher(Persons, 'human_2d_poses', 10)

        self.bridge = CvBridge()
        
        self.get_logger().info('''
        LIGHTWEIGHT OPEN POSE ROS2 SETUP IS DONE !
        When call service, OpenPose will run.
        ''')
    

    def infer_fast(self, net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
        height, width, _ = img.shape
        scale = net_input_height_size / height

        scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        scaled_img = normalize(scaled_img, img_mean, img_scale)
        min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
        padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
        if not cpu == 'cpu':
            tensor_img = tensor_img.cuda()

        stages_output = net(tensor_img)

        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        return heatmaps, pafs, scale, pad
    

    def image_cb(self, msg:Image):
        if self.exec_flag:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
                orig_cv_image = cv_image.copy()
                heatmaps, pafs, scale, pad = self.infer_fast(
                    self.net, cv_image,
                    self.get_parameter('height_size').value,
                    self.stride, self.upsample_ratio, self.param_device)

                total_keypoints_num = 0
                all_keypoints_by_type = []
                for kpt_idx in range(self.num_keypoints):  # 19th for bg
                    total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

                pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
                for kpt_id in range(all_keypoints.shape[0]):
                    all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * self.stride / self.upsample_ratio - pad[1]) / scale
                    all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * self.stride / self.upsample_ratio - pad[0]) / scale
                current_poses = []
                for n in range(len(pose_entries)):
                    if len(pose_entries[n]) == 0:
                        continue
                    pose_keypoints = np.ones((self.num_keypoints, 2), dtype=np.int32) * -1
                    for kpt_id in range(self.num_keypoints):
                        if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                            pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                            pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
                    pose = Pose(pose_keypoints, pose_entries[n][18] / (2 * pose_entries[n][19] - 1))
                    current_poses.append(pose)

                if self.track:
                    track_poses(self.previous_poses, current_poses, smooth=self.smooth)
                    self.previous_poses = current_poses
                
                # Filter by confidence
                current_poses = [pose for pose in current_poses if pose.confidence >= self.min_confidence]

                for pose in current_poses:
                    pose.draw(cv_image)
                cv_image = cv2.addWeighted(orig_cv_image, 0.6, cv_image, 0.4, 0)
                for pose in current_poses:
                    cv2.rectangle(cv_image, (pose.bbox[0], pose.bbox[1]),
                                (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
                    if self.track:
                        cv2.putText(cv_image, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
                    cv2.putText(cv_image, 'conf: {:.2f}'.format(pose.confidence), (pose.bbox[0], pose.bbox[1] - 32 if self.track else pose.bbox[1] - 16),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
                    
                    if self.debug:
                        cv2.imshow('Lightweight Human Pose Estimation ROS2', cv_image)
                        cv2.waitKey(1)

                # Publish poses
                persons_msg = Persons()
                persons_msg.header = msg.header
                persons_msg.persons = []

                for pose in current_poses:
                    person_msg = Person()
                    if pose.id is not None:
                        person_msg.id = pose.id
                    else:
                        person_msg.id = -1
                    person_msg.confidence = float(pose.confidence)
                    
                    # bbox: x, y, w, h
                    person_msg.bbox = [int(pose.bbox[0]), int(pose.bbox[1]), int(pose.bbox[2]), int(pose.bbox[3])]
                    
                    # keypoints
                    for i, kpt in enumerate(pose.keypoints):
                        pt = Point2D()
                        pt.x = int(kpt[0])
                        pt.y = int(kpt[1])
                        person_msg.keypoints[i] = pt
                    
                    persons_msg.persons.append(person_msg)
                
                self.poses_pub.publish(persons_msg)

            except Exception as e:
                self.get_logger().error(f'Error processing image: {str(e)}')
                traceback.print_exc()
    

    def execute_cb(self, req:SetBool.Request, res:SetBool.Response):
        self.exec_flag = req.data
        if self.exec_flag:
            self.get_logger().info('START DETECT')
        else:
            self.get_logger().info('STOP DETECT')
            '''
            if self.debug:
                try:
                    cv2.destroyWindow("Lightweight Human Pose Estimation ROS2")
                    cv2.waitKey(1)
                except cv2.error:
                    pass
            '''
                    
        res.success = True
        return res


def main():
    rclpy.init()
    node = LightweightOpenPoseRos2()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()