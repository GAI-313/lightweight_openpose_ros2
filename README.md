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
