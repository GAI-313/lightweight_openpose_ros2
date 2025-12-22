# install
- **Install Python depend packages**
    ```bash
    pip install -r requirements.txt
    ```

- **Donwload model data**
    ```bash
    cd ./lightweight_openpose_ros2/datas/
    ```
    ```bash
    get https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth
    ```

- **Build package**
    ```bash
    colcon build --symlink-install --packages-up-to lightweight_openpose_ros2
    ```

# Execute
## 2D Pose Estimation (lightweight_openpose_ros2)
```bash
ros2 run lightweight_openpose_ros2 lightweight_openpose_ros2 --ros-args -r /image_raw:=<RGB IMAGE TOPIC NAME>
```

- **Parameters**
    |params name|type|default|description|
    |:---|:---:|:---:|:---|
    |**checkpoint_path**|`string`|`.../checkpoint_iter_370000.pth`|Path to the model checkpoint file.|
    |**device**|`string`|`"cpu"`|Computation device (`"cpu"` or `"cuda"`).|
    |**height_size**|`int`|`256`|Input image height for the network.|
    |**qos.reliability**|`string`|`"RELIABLE"`|QoS reliability policy (`"RELIABLE"` or `"BEST_EFFORT"`).|
    |**qos.durability**|`string`|`"VOLATILE"`|QoS durability policy (`"VOLATILE"` or `"TRANSIENT_LOCAL"`).|
    |**qos.depth**|`int`|`10`|QoS history depth.|
    |**debug**|`bool`|`true`|Whether to show the debug window with detection results.|

- **Service**<br>
    This node does not start detection automatically. You need to call the `execute` service to start/stop detection.

    - **Start Detection**
        ```bash
        ros2 service call /execute std_srvs/srv/SetBool "{data: true}"
        ```

    - **Stop Detection**
        ```bash
        ros2 service call /execute std_srvs/srv/SetBool "{data: false}"
        ```

## 3D Pose Estimation (lor_transformer)

This package converts 2D poses to 3D using depth images and camera info.
Run the node with remappings to your camera topics.

```bash
ros2 run lor_transformer lor_transformer --ros-args \
    -r image_raw:=/camera/aligned_depth_to_color/image_raw \
    -r depth_camera_info:=/camera/aligned_depth_to_color/camera_info \
    -r color_camera_info:=/camera/color/camera_info \
    -p target_frame:=base_link
```
> ![NOTE] 
Replace `/camera/...` with your actual RealSense/Depth camera topics.*

- **Parameters**
    |params name|type|default|description|
    |:---|:---:|:---:|:---|
    |**depth_scale**|`double`|`0.001`|Scale factor to convert depth image values to meters (e.g., 0.001 for mm to m).|
    |**depth_sample_size**|`int`|`5`|Window size (NxN) for median depth filtering around the keypoint.|
    |**target_frame**|`string`|`""`|Target TF frame for 3D poses. If empty, uses the depth optical frame.|

- **Visualization**
    The node publishes `MarkerArray` to `human_pose_markers` which can be viewed in RViz2.
The node publishes `MarkerArray` to `human_pose_markers` which can be viewed in RViz2.
