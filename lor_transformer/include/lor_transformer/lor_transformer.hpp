#ifndef LOR_TRANSFORMER__LOR_TRANSFORMER_HPP_
#define LOR_TRANSFORMER__LOR_TRANSFORMER_HPP_

#include <rclcpp/rclcpp.hpp>
#include "lor_interfaces/msg/persons.hpp"
#include "lor_interfaces/msg/persons3_d.hpp"
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <opencv2/opencv.hpp>
#include <vector>
#include <array>

namespace lor_transformer
{
    class LorTransformer : public rclcpp::Node
    {
    public:
        LorTransformer();

    private:
        void sync_callback(
            const lor_interfaces::msg::Persons::ConstSharedPtr& poses_msg,
            const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg,
            const sensor_msgs::msg::CameraInfo::ConstSharedPtr& depth_info_msg,
            const sensor_msgs::msg::CameraInfo::ConstSharedPtr& color_info_msg);

        using SyncPolicy = message_filters::sync_policies::ApproximateTime<
            lor_interfaces::msg::Persons,
            sensor_msgs::msg::Image,
            sensor_msgs::msg::CameraInfo,
            sensor_msgs::msg::CameraInfo
        >;
        using Synchronizer = message_filters::Synchronizer<SyncPolicy>;

        message_filters::Subscriber<lor_interfaces::msg::Persons> poses_sub_;
        message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
        message_filters::Subscriber<sensor_msgs::msg::CameraInfo> depth_info_sub_;
        message_filters::Subscriber<sensor_msgs::msg::CameraInfo> color_info_sub_;

        std::shared_ptr<Synchronizer> sync_;

        rclcpp::Publisher<lor_interfaces::msg::Persons3D>::SharedPtr poses_3d_pub_;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_pub_;

        // TF
        std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
        std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

        // Parameters
        double depth_scale_;
        int depth_sample_size_;
        std::string target_frame_;

        // Helpers
        void align_color_to_depth(
            double u_color, double v_color,
            const std::array<double, 9>& color_K,
            const std::array<double, 9>& depth_K,
            double& u_depth, double& v_depth);
            
        double get_median_depth(const cv::Mat& depth_image, int u, int v, int size);

        // Skeleton definition (index pairs)
        // 0:Nose, 1:Neck, 2:RightShoulder, 3:RightElbow, 4:RightWrist, 
        // 5:LeftShoulder, 6:LeftElbow, 7:LeftWrist, 8:RightHip, 9:RightKnee, 
        // 10:RightAnkle, 11:LeftHip, 12:LeftKnee, 13:LeftAnkle, 14:RightEye, 
        // 15:LeftEye, 16:RightEar, 17:LeftEar
        const std::vector<std::pair<int, int>> skeleton_connections_ = {
            {0, 1}, {1, 2}, {2, 3}, {3, 4},       // Nose->Neck->RShoulder->RElbow->RWrist
            {1, 5}, {5, 6}, {6, 7},               // Neck->LShoulder->LElbow->LWrist
            {1, 8}, {8, 9}, {9, 10},              // Neck->RHip->RKnee->RAnkle (Neck connects to hips typically via mid-hip, but linking Neck->Hip is common simple viz)
            {1, 11}, {11, 12}, {12, 13},          // Neck->LHip->LKnee->LAnkle
            {0, 14}, {14, 16},                    // Nose->REye->REar
            {0, 15}, {15, 17}                     // Nose->LEye->LEar
        };
        // Note: The python reference had specific connections, let's try to match standard coco 18 if possible. 
        // The user didn't specify connections, just "skeleton". I'll use a standard set.
    };
} // namespace lor_transformer

#endif // LOR_TRANSFORMER__LOR_TRANSFORMER_HPP_
