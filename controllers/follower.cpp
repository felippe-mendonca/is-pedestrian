#include "bbs.hpp"
#include <armadillo>
#include <boost/circular_buffer.hpp>
#include <boost/program_options.hpp>
#include <is/is.hpp>
#include <is/msgs/camera.hpp>
#include <is/msgs/common.hpp>
#include <is/msgs/ompl.hpp>
#include <is/msgs/robot.hpp>
#include <limits>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>
#include "arma.hpp"
#include "camera.hpp"

#include "/home/felippe/dev/is-robot-controller/msgs/robot-controller.hpp"

namespace po = boost::program_options;
using namespace boost;
using namespace is::msg::camera;
using namespace is::msg::common;
using namespace is::msg::robot;
using namespace is::msg::controller;
using namespace is::msg::ompl;
using namespace arma;
using namespace std::chrono_literals;

namespace std {
std::ostream& operator<<(std::ostream& os, const std::vector<std::string>& vec) {
  for (auto item : vec) {
    os << item << " ";
  }
  return os;
}
}  // ::std

int main(int argc, char* argv[]) {
  std::string uri;
  std::vector<std::string> cameras;
  const std::vector<std::string> default_cameras{"ptgrey.0", "ptgrey.1", "ptgrey.2", "ptgrey.3"};
  double fps;
  std::string path;
  float x_d;
  float y_d;
  float radius;

  po::options_description description("Allowed options");
  auto&& options = description.add_options();
  options("help,", "show available options");
  options("uri,u", po::value<std::string>(&uri)->default_value("amqp://edge.is:30000"), "broker uri");
  options("cameras,c", po::value<std::vector<std::string>>(&cameras)->multitoken()->default_value(default_cameras),
          "cameras");
  options("fps,f", po::value<double>(&fps)->default_value(4.0), "frames rate");
  options("parameters,p", po::value<std::string>(&path)->default_value("../cameras-parameters/"),
          "cameras parameters path");
  options("x_desired,x", po::value<float>(&x_d)->default_value(1000.0), "x desired [mm]");
  options("y_desired,y", po::value<float>(&y_d)->default_value(1000.0), "y desired [mm]");
  options("radius,r", po::value<float>(&radius)->default_value(350.0), "obstacle radius [mm]");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, description), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << description << std::endl;
    return 1;
  }

  auto parameters = camera::load_parameters(path);

  auto is = is::connect(uri);
  auto client = is::make_client(is);

  std::vector<std::string> topics;
  for (auto& camera : cameras) {
    topics.push_back(camera + ".new_bbs");
  }
  topics.push_back("robot-controller.0.pose");

  auto tag = is.subscribe(topics);
  auto period = static_cast<int64_t>(1000.0 / fps);

  circular_buffer<std::map<std::string, arma::mat>> cb_bbs(10);

  is::log::info("Starting");
  while (1) {
    auto msgs = is.consume_sync(tag, topics, period);
    is::log::info("Messages received");

    std::map<std::string, arma::mat> bbs;
    optional<Pose> visual_position;

    std::for_each(std::begin(msgs), std::end(msgs), [&](auto& msg) {

      auto routing_key = msg->RoutingKey();
      auto pos = routing_key.find_last_of('.');
      auto type = routing_key.substr(pos + 1);
      auto camera = routing_key.substr(0, pos);

      if (type == "new_bbs") {
        arma::mat bb = is::msgpack<arma::mat>(msg);
        is::log::info("camera {} -> {} detections", camera, bb.n_rows);
        bbs.emplace(std::make_pair(camera, bb));
      } else if (routing_key == "robot-controller.0.pose") {
        visual_position = is::msgpack<optional<Pose>>(msg);
        if (visual_position) {
          is::log::info("Robot Position: {},{}", (*visual_position).position.x, (*visual_position).position.y);
        } else {
          is::log::warn("Robot not found");
        }
      }
    });

    cb_bbs.push_back(bbs);
    std::map<std::string, arma::mat> grouped_bbs;
    std::transform(cameras.begin(), cameras.end(), std::inserter(grouped_bbs, grouped_bbs.begin()),
                   [](auto camera) { return std::make_pair(camera, arma::mat(0, 0, arma::fill::zeros)); });

    std::for_each(cb_bbs.begin(), cb_bbs.end(), [&](auto& bbs) {
      std::for_each(bbs.begin(), bbs.end(),
                    [&](auto& bb) { grouped_bbs[bb.first] = arma::join_vert(grouped_bbs[bb.first], bb.second); });
    });

    arma::mat positions = bbs::get_position(grouped_bbs, parameters);
    arma::mat clustered_positions = bbs::clustering::distance(positions, 500.0);

    if (!clustered_positions.empty() && visual_position && cb_bbs.full()) {
      arma::vec current_position({(*visual_position).position.x, (*visual_position).position.y});

      arma::vec nearst_pedestrian;
      double distance = std::numeric_limits<double>::max();
      clustered_positions.each_row([&](arma::rowvec& p) {
        current_position.print("current_position");
        p.subvec(0, 1).t().print("current_point");
        auto current_distance = arma::norm(current_position - p.subvec(0, 1).t());
        if (current_distance < distance) {
          distance = current_distance;
          nearst_pedestrian = p.t();
        }
      });

      RobotTask robot_task;
      Point p;
      p.x = nearst_pedestrian(0);
      p.y = nearst_pedestrian(1);
      robot_task.positions.push_back(p);
      robot_task.stop_distance = 800.0;
      auto id = client.request("robot-controller.0.do-task", is::msgpack(robot_task));
      client.receive_for(10ms, id, is::policy::discard_others);
    }

    clustered_positions.print("positions");
  }
  return 0;
}