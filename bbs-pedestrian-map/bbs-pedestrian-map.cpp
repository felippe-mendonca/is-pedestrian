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

cv::Point pos2pixel(arma::mat const& position, cv::Size const& image_size, cv::Point const& center, double scale = 0.1,
                    int x_dir = 1, int y_dir = 1) {
  cv::Point p(x_dir * scale * position(0), y_dir * scale * position(1));
  return p + center;
}

void draw_axis(cv::Mat& image, cv::Point const& origin, unsigned int const& arrow_length, int const& x_dir = 1,
               int const& y_dir = 1, unsigned int linewidth = 2) {
  // draw x-axis (Red)
  cv::arrowedLine(image, origin, origin + cv::Point(x_dir * arrow_length, 0.0), cv::Scalar(0, 0, 255), linewidth);
  // draw y-axis (Green)
  cv::arrowedLine(image, origin, origin + cv::Point(0.0, y_dir * arrow_length), cv::Scalar(0, 255, 0), linewidth);
}

void draw_grid(cv::Mat& image, cv::Point const& origin, int const& x_tick, int const& y_tick,
               cv::Scalar color = cv::Scalar(255, 255, 255), unsigned int linewidth = 1) {
  auto width = image.cols;
  auto height = image.rows;
  auto x0 = origin.x;
  while (x0 > 0) {
    cv::line(image, cv::Point(x0, 0), cv::Point(x0, height), color, linewidth);
    x0 -= x_tick;
  }
  x0 = origin.x;
  while (x0 < width) {
    cv::line(image, cv::Point(x0, 0), cv::Point(x0, height), color, linewidth);
    x0 += x_tick;
  }
  auto y0 = origin.y;
  while (y0 > 0) {
    cv::line(image, cv::Point(0, y0), cv::Point(width, y0), color, linewidth);
    y0 -= y_tick;
  }
  y0 = origin.y;
  while (y0 < height) {
    cv::line(image, cv::Point(0, y0), cv::Point(width, y0), color, linewidth);
    y0 += y_tick;
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

  const auto width = 800;
  const auto height = 500;
  const auto arrow_length = 50;
  const cv::Point origin(width / 2, height / 2);
  cv::Mat image = cv::Mat::zeros(cv::Size(800, 500), CV_8UC3);

  while (1) {
    auto bbs_msg = is.consume_sync(bbs_tag, bbs_topics, period);
    is::log::info("bss received");

    map<string, arma::mat> bbs;
    transform(begin(bbs_msg), end(bbs_msg), begin(cameras), inserter(bbs, begin(bbs)),
              [](auto& msg, auto& camera) { return make_pair(camera, is::msgpack<arma::mat>(msg)); });

    auto positions = bbs::get_position(bbs, parameters);
    positions.print("positions");

    draw_grid(image, origin, 100, 100);
    positions.each_row([&](arma::rowvec const& p) {
      cv::circle(image, pos2pixel(p, image.size(), origin, 0.1, -1, 1), 5, cv::Scalar(0, 255, 255), -1);
    });
    draw_axis(image, origin, arrow_length, -1, 1, 2);
    cv::imshow("Pedestrian position map", image);
    cv::waitKey(1);
  }
  return 0;
}