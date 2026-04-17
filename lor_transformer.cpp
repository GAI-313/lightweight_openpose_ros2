// Updating lor_transformer.cpp for QoS parameter handling
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>

class LorTransformer : public rclcpp::Node {
public:
    LorTransformer(const std::string & node_name)
    : Node(node_name) {
        // Declare QoS parameters
        int reliability_param;
        int durability_param;
        int depth_param;

        this->declare_parameter("qos.reliability", reliability_param);
        this->declare_parameter("qos.durability", durability_param);
        this->declare_parameter("qos.depth", depth_param);

        // Convert to rclcpp QoS policies
        rclcpp::QoS qos_profile(rclcpp::KeepLast(depth_param));
        qos_profile.reliability(static_cast<rclcpp::ReliabilityPolicy>(reliability_param));
        qos_profile.durability(static_cast<rclcpp::DurabilityPolicy>(durability_param));

        // Create subscribers and publishers with QoSProfile
        subscriber_ = this->create_subscription<YourMsgType>("your_topic", qos_profile,
            std::bind(&LorTransformer::callback, this, std::placeholders::_1));
        publisher_ = this->create_publisher<YourMsgType>("your_topic", qos_profile);
    }

private:
    void callback(const YourMsgType::SharedPtr msg) {
        // Callback implementation
    }
    rclcpp::Publisher<YourMsgType>::SharedPtr publisher_;
    rclcpp::Subscription<YourMsgType>::SharedPtr subscriber_;
};