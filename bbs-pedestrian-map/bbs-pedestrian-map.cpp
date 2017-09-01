#include "bbs.hpp"
#include <armadillo>
#include <boost/program_options.hpp>
#include <chrono>
#include <is/is.hpp>
#include <iterator>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>
#include "arma.hpp"
#include "camera.hpp"
#include "radar.hpp"

namespace po = boost::program_options;
using namespace std;
using namespace arma;

namespace std {
std::ostream& operator<<(std::ostream& os, const std::vector<std::string>& vec) {
  for (auto item : vec) {
    os << item << " ";
  }
  return os;
}
}

int main(int argc, char* argv[]) {
  std::string uri;
  std::string path;
  double fps;
  std::vector<std::string> cameras;
  const std::vector<std::string> default_cameras{"ptgrey.0", "ptgrey.1", "ptgrey.2", "ptgrey.3"};
  std::string prefix;
  std::string bb_topic;

  po::options_description description("Allowed options");
  auto&& options = description.add_options();
  options("help,", "show available options");
  options("uri,u", po::value<std::string>(&uri)->default_value("amqp://localhost"), "broker uri");
  options("cameras,c", po::value<std::vector<std::string>>(&cameras)->multitoken()->default_value(default_cameras),
          "cameras");
  options("parameters,p", po::value<std::string>(&path)->default_value("/data/"), "cameras parameters path");
  options("fps,f", po::value<double>(&fps)->default_value(5.0), "frame rate");
  options("prefix,P", po::value<std::string>(&prefix)->default_value(""), "prefix of topics");
  options("bb_topic,b", po::value<std::string>(&bb_topic)->default_value("bbs"), "boundbox topic");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, description), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << description << std::endl;
    return 1;
  }

  // Load camera parameters
  auto parameters = camera::load_parameters(path);

  is::Connection is(is::connect(uri));

  std::vector<std::string> bbs_topics;
  for (auto& camera : cameras) {
    bbs_topics.push_back(prefix + camera + "." + bb_topic);
  }
  auto bbs_tag = is.subscribe(bbs_topics);
  auto period = static_cast<int64_t>(1000.0 / fps);

  Radar radar(800, 500);
  while (1) {
    auto bbs_msg = is.consume_sync(bbs_tag, bbs_topics, period);
    is::log::info("bss received");

    map<string, arma::mat> bbs;
    transform(begin(bbs_msg), end(bbs_msg), begin(cameras), inserter(bbs, begin(bbs)),
              [](auto& msg, auto& camera) { return make_pair(camera, is::msgpack<arma::mat>(msg)); });

    auto positions = bbs::get_position(bbs, parameters);
    auto clustered_positions = bbs::clustering::distance(positions, 500.0);

    radar.update(positions, cv::Scalar(0, 255, 255), 5, false);
    radar.update(clustered_positions, cv::Scalar(255, 0, 255));

    cv::imshow("Pedestrian position map", radar.get_image());
    cv::waitKey(1);
  }
  return 0;
}