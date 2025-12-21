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
