#include <boost/program_options.hpp>
#include <is/is.hpp>
#include <is/msgs/common.hpp>
#include <is/msgs/geometry.hpp>
#include <armadillo>
#include <vector>
#include <string>
#include <chrono>

#include "arma.hpp"
#include "camera.hpp"
#include "bbs.hpp"

namespace po = boost::program_options;
using namespace arma;
using namespace is::msg;
using namespace is::msg::common;
using namespace is::msg::geometry;

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

  po::options_description description("Allowed options");
  auto&& options = description.add_options();
  options("help,", "show available options");
  options("uri,u", po::value<std::string>(&uri)->default_value("amqp://localhost"), "broker uri");
  options("cameras,c", po::value<std::vector<std::string>>(&cameras)->multitoken()->default_value(default_cameras),
          "cameras");
  options("parameters,p", po::value<std::string>(&path)->default_value("/data/"), "cameras parameters path");
  options("fps,f", po::value<double>(&fps)->default_value(5.0), "frame rate");
  options("prefix,P", po::value<std::string>(&prefix)->default_value(""), "prefix of topics");

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
    bbs_topics.push_back(prefix + camera + ".bbs");
  }
  auto bbs_tag = is.subscribe(bbs_topics);
  auto period = static_cast<int64_t>(1000.0 / fps);

  auto validation = [](arma::mat p) {
    auto x = p(0);
    auto y = p(1);
    if (x > 1000.0 || x < -1000.0 || y > 1000.0 || y < -1000.0) {
      // if (x > 2000.0 || x < -3000.0 || y > 1500.0 || y < -1500.0) {
      return false;
    }
    return true;
  };

  while (1) {
    auto bbs_msg = is.consume_sync(bbs_tag, bbs_topics, period);
    
    is::log::info("BBS consumed");
    
    std::map<std::string, arma::mat> bbs;
    std::transform(std::begin(bbs_msg), std::end(bbs_msg), std::begin(cameras), std::inserter(bbs, std::begin(bbs)),
    [](auto& msg, auto& camera) { return std::make_pair(camera, is::msgpack<arma::mat>(msg)); });
       
    auto validated_bbs = bbs::validate_bbs(bbs, parameters);

    for (auto& camera : cameras) {
      is::log::info("Camera: {} | {} BBs received and {} validated.", camera, bbs[camera].n_rows , validated_bbs[camera].n_rows);
      is.publish(prefix + camera + ".new_bbs", is::msgpack(validated_bbs[camera]));
    }
  }

  return 0;
}