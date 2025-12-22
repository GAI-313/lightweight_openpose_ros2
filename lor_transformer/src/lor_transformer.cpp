#include "lor_transformer/lor_transformer.hpp"
#include <cv_bridge/cv_bridge.h>
#include <algorithm>
#include <vector>

namespace lor_transformer
{
    LorTransformer::LorTransformer() : Node("lor_transformer")
    {
        // Declare parameters
        this->declare_parameter("depth_scale", 0.001);
        this->declare_parameter("depth_sample_size", 5);
        this->declare_parameter("target_frame", ""); // Empty means use camera frame (depth frame)
        this->declare_parameter("sync_queue_size", 100);
        
        // Ensure use_sim_time parameter is available
        if (!this->has_parameter("use_sim_time")) {
            this->declare_parameter("use_sim_time", false);
        }

        depth_scale_ = this->get_parameter("depth_scale").as_double();
        depth_sample_size_ = this->get_parameter("depth_sample_size").as_int();
        target_frame_ = this->get_parameter("target_frame").as_string();
        int sync_queue_size = this->get_parameter("sync_queue_size").as_int();

        // Subscribers
        poses_sub_.subscribe(this, "human_2d_poses");
        depth_sub_.subscribe(this, "image_raw");
        depth_info_sub_.subscribe(this, "depth_camera_info");
        color_info_sub_.subscribe(this, "color_camera_info");

        // Publishers
        poses_3d_pub_ = this->create_publisher<lor_interfaces::msg::Persons3D>("human_3d_poses", 10);
        markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("human_pose_markers", 10);

        // TF
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // Sync
        sync_ = std::make_shared<Synchronizer>(SyncPolicy(sync_queue_size), poses_sub_, depth_sub_, depth_info_sub_, color_info_sub_);
        sync_->registerCallback(std::bind(&LorTransformer::sync_callback, this, 
            std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));

        RCLCPP_INFO(this->get_logger(), "LorTransformer node started. Target frame: '%s'", target_frame_.empty() ? "Same as Depth Frame" : target_frame_.c_str());
    }

    void LorTransformer::align_color_to_depth(
        double u_color, double v_color,
        const std::array<double, 9>& color_K,
        const std::array<double, 9>& depth_K,
        double& u_depth, double& v_depth)
    {
        double scale_x = depth_K[0] / color_K[0]; // fx_d / fx_c
        double scale_y = depth_K[4] / color_K[4]; // fy_d / fy_c
        
        u_depth = (u_color - color_K[2]) * scale_x + depth_K[2]; // cx
        v_depth = (v_color - color_K[5]) * scale_y + depth_K[5]; // cy
    }

    double LorTransformer::get_median_depth(const cv::Mat& depth_image, int u, int v, int size)
    {
        int h = depth_image.rows;
        int w = depth_image.cols;
        int half = size / 2;

        int u_min = std::max(0, u - half);
        int u_max = std::min(w, u + half + 1);
        int v_min = std::max(0, v - half);
        int v_max = std::min(h, v + half + 1);

        std::vector<uint16_t> valid_depths;
        valid_depths.reserve(size * size);

        for (int r = v_min; r < v_max; ++r) {
            const uint16_t* ptr = depth_image.ptr<uint16_t>(r);
            for (int c = u_min; c < u_max; ++c) {
                if (ptr[c] > 0) {
                    valid_depths.push_back(ptr[c]);
                }
            }
        }

        if (valid_depths.empty()) {
            return 0.0;
        }

        std::sort(valid_depths.begin(), valid_depths.end());
        
        size_t n = valid_depths.size();
        double q1 = valid_depths[n / 4];
        double q3 = valid_depths[n * 3 / 4];
        double iqr = q3 - q1;
        double lower = q1 - 1.5 * iqr;
        double upper = q3 + 1.5 * iqr;
        
        std::vector<uint16_t> filtered;
        filtered.reserve(n);
        for(auto val : valid_depths) {
            if (val >= lower && val <= upper) {
                filtered.push_back(val);
            }
        }
        
        if (filtered.empty()) {
            if (n % 2 == 0) return (valid_depths[n/2 - 1] + valid_depths[n/2]) / 2.0;
            return valid_depths[n/2];
        }
        
        size_t nf = filtered.size();
        if (nf % 2 == 0) return (filtered[nf/2 - 1] + filtered[nf/2]) / 2.0;
        return filtered[nf/2];
    }

    void LorTransformer::sync_callback(
        const lor_interfaces::msg::Persons::ConstSharedPtr& poses_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg,
        const sensor_msgs::msg::CameraInfo::ConstSharedPtr& depth_info_msg,
        const sensor_msgs::msg::CameraInfo::ConstSharedPtr& color_info_msg)
    {
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        std::string output_frame = depth_msg->header.frame_id;
        geometry_msgs::msg::TransformStamped transform;
        bool do_transform = false;

        if (!target_frame_.empty() && target_frame_ != output_frame) {
            try {
                transform = tf_buffer_->lookupTransform(
                    target_frame_,
                    output_frame,
                    tf2::TimePointZero); // latest
                output_frame = target_frame_;
                do_transform = true;
            } catch (tf2::TransformException &ex) {
                RCLCPP_WARN(this->get_logger(), "Could not transform from %s to %s: %s", 
                    output_frame.c_str(), target_frame_.c_str(), ex.what());
                return; 
            }
        }

        lor_interfaces::msg::Persons3D output_msg;
        output_msg.header = depth_msg->header;
        output_msg.header.frame_id = output_frame;

        visualization_msgs::msg::MarkerArray markers;

        const auto& color_K = color_info_msg->k;
        const auto& depth_K = depth_info_msg->k;

        int marker_id_counter = 0;

        for (const auto& person : poses_msg->persons) {
            lor_interfaces::msg::Person3D person_3d;
            person_3d.id = person.id;
            person_3d.confidence = person.confidence;

            std::vector<geometry_msgs::msg::Point> valid_kpts;
            valid_kpts.resize(18); // store transformed points for visualization

            for (size_t i = 0; i < 18; ++i) {
                const auto& kp_2d = person.keypoints[i];
                geometry_msgs::msg::Point kp_3d;
                bool is_valid = false;

                if (kp_2d.x > 0 && kp_2d.y > 0) {
                    double u_depth, v_depth;
                    align_color_to_depth(kp_2d.x, kp_2d.y, color_K, depth_K, u_depth, v_depth);

                    double depth_raw_val = get_median_depth(cv_ptr->image, (int)u_depth, (int)v_depth, depth_sample_size_);
                    double depth = depth_raw_val * depth_scale_;

                    if (depth > 0) {
                        // Camera frame point
                        double pt_x = (u_depth - depth_K[2]) * depth / depth_K[0];
                        double pt_y = (v_depth - depth_K[5]) * depth / depth_K[4];
                        double pt_z = depth;

                        if (do_transform) {
                            geometry_msgs::msg::PointStamped pt_stamped_in, pt_stamped_out;
                            pt_stamped_in.header = depth_msg->header;
                            pt_stamped_in.point.x = pt_x;
                            pt_stamped_in.point.y = pt_y;
                            pt_stamped_in.point.z = pt_z;
                            
                            tf2::doTransform(pt_stamped_in, pt_stamped_out, transform);
                            
                            kp_3d = pt_stamped_out.point;
                        } else {
                            kp_3d.x = pt_x;
                            kp_3d.y = pt_y;
                            kp_3d.z = pt_z;
                        }
                        is_valid = true;
                    }
                }
                
                if (!is_valid) {
                    kp_3d.x = 0; kp_3d.y = 0; kp_3d.z = 0;
                }
                person_3d.keypoints[i] = kp_3d;
                valid_kpts[i] = kp_3d;
            }
            output_msg.persons.push_back(person_3d);

            // Visualization
            // 1. Text Marker (ID)
            // Use Neck (1) or Nose (0) or average of valid points as position
            geometry_msgs::msg::Point text_pos;
            // Try to find a valid point near the head
            if (person_3d.keypoints[1].z > 0) text_pos = person_3d.keypoints[1];
            else if (person_3d.keypoints[0].z > 0) text_pos = person_3d.keypoints[0];
            else {
                // Find first valid
                 bool found = false;
                 for(auto& p : person_3d.keypoints) { if(p.z > 0) { text_pos = p; found = true; break;} }
                 if(!found) continue; // Skip vis if no valid points
            }

            visualization_msgs::msg::Marker text_marker;
            text_marker.header.frame_id = output_frame;
            text_marker.header.stamp = depth_msg->header.stamp;
            text_marker.ns = "person_id";
            text_marker.id = person.id;
            text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            text_marker.action = visualization_msgs::msg::Marker::ADD;
            text_marker.pose.position = text_pos;
            text_marker.pose.position.z += 0.3; // Show text above person
            text_marker.pose.orientation.w = 1.0;
            text_marker.scale.z = 0.2;
            text_marker.color.r = 1.0; text_marker.color.g = 1.0; text_marker.color.b = 1.0; text_marker.color.a = 1.0;
            text_marker.text = "ID: " + std::to_string(person.id);
            text_marker.lifetime = rclcpp::Duration::from_seconds(0.2); // Short lifetime
            markers.markers.push_back(text_marker);

            // 2. Skeleton Marker
            visualization_msgs::msg::Marker line_marker;
            line_marker.header.frame_id = output_frame;
            line_marker.header.stamp = depth_msg->header.stamp;
            line_marker.ns = "skeleton";
            line_marker.id = person.id;
            line_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
            line_marker.action = visualization_msgs::msg::Marker::ADD;
            line_marker.pose.orientation.w = 1.0;
            line_marker.scale.x = 0.02; // Line width
            // Assign color based on ID (simple hash)
            int c_idx = person.id % 3;
            if(c_idx == 0) { line_marker.color.r = 1.0; line_marker.color.g = 0.0; line_marker.color.b = 0.0; }
            else if(c_idx == 1) { line_marker.color.r = 0.0; line_marker.color.g = 1.0; line_marker.color.b = 0.0; }
            else { line_marker.color.r = 0.0; line_marker.color.g = 0.0; line_marker.color.b = 1.0; }
            line_marker.color.a = 1.0;
            line_marker.lifetime = rclcpp::Duration::from_seconds(0.2);

            for (const auto& conn : skeleton_connections_) {
                int i1 = conn.first;
                int i2 = conn.second;
                
                // Check bounds and validity
                if (i1 < 18 && i2 < 18) {
                    const auto& p1 = valid_kpts[i1];
                    const auto& p2 = valid_kpts[i2];
                    
                    // Simple check: if point is (0,0,0) it's likely invalid (though 0,0,0 is valid in camera frame, it's lens center).
                    // In practice, Z > 0 for valid points in front of camera.
                    // But transformed points might be anywhere.
                    // We rely on the initial check (depth > 0) which we propagated earlier.
                    // Let's use the Z of camera frame... but we don't have it easily here if transformed.
                    // We can check if point is exactly 0,0,0 (initialized invalid). 
                    // This is risky if actual point is at origin. Better to have valid flag.
                    // Re-checking person_3d logic: we set Invalid to 0,0,0.
                    
                    auto is_p_valid = [](const geometry_msgs::msg::Point& p) { return !(p.x == 0 && p.y == 0 && p.z == 0); };
                    
                    if (is_p_valid(p1) && is_p_valid(p2)) {
                        line_marker.points.push_back(p1);
                        line_marker.points.push_back(p2);
                    }
                }
            }
            if (!line_marker.points.empty()) {
                markers.markers.push_back(line_marker);
            }
        }

        poses_3d_pub_->publish(output_msg);
        if (!markers.markers.empty()) {
            markers_pub_->publish(markers);
        }
    }
} // namespace lor_transformer

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<lor_transformer::LorTransformer>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
