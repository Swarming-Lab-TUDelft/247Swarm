#include <chrono>
#include <memory>
#include <vector>
#include <cmath>
#include <eigen3/Eigen/Dense>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "geometry_msgs/msg/polygon.hpp"
#include "geometry_msgs/msg/point32.hpp"

using namespace std::chrono_literals;
using std::placeholders::_1;


class WaypointPublisher : public rclcpp::Node
{
public:
    WaypointPublisher();
    ~WaypointPublisher();

private:
    rclcpp::Publisher<geometry_msgs::msg::Polygon>::SharedPtr waypoint_pub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr pattern_switch_sub_;
    rclcpp::TimerBase::SharedPtr timer_;

    void timer_callback();

    void pattern_switch_callback(const std_msgs::msg::String::SharedPtr msg) const;


    //Geometry functions
    static std::vector<Eigen::Vector3d> generate_rotating_diamond();
    std::vector<Eigen::Vector3d> (*current_pattern_function)();

};