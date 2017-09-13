#include "bbs.hpp"
#include <armadillo>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/program_options.hpp>
#include <iomanip>
#include <iostream>
#include <is/is.hpp>
#include <is/msgs/camera.hpp>
#include <is/msgs/common.hpp>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>
#include "arma.hpp"
#include "camera.hpp"
#include "radar.hpp"

namespace po = boost::program_options;
using namespace is::msg::camera;
using namespace is::msg::common;
using namespace std::chrono;
using namespace arma;

namespace std {
std::ostream& operator<<(std::ostream& os, const std::vector<std::string>& vec) {
  for (auto item : vec) {
    os << item << " ";
  }
  return os;
}
}

const std::map<int, cv::Scalar> colors{
    {1, cv::Scalar(0, 255, 0)},    // green
    {2, cv::Scalar(0, 0, 255)},    // red
    {3, cv::Scalar(0, 255, 255)},  // yellow
    {4, cv::Scalar(255, 0, 0)}     // blue
};

namespace is {
void putText(cv::Mat& frame, std::string const& text, cv::Point point, int fontFace, double fontScale, cv::Scalar color,
             cv::Scalar backgroundColor, int tickness = 1, int linetype = 8, bool bottomLeftOrigin = false) {
  int baseline = 0;
  cv::Size text_size = cv::getTextSize(text, fontFace, fontScale, tickness, &baseline);
  cv::rectangle(frame, point + cv::Point(0, baseline), point + cv::Point(text_size.width, -text_size.height),
                backgroundColor, CV_FILLED);
  cv::putText(frame, text, point, fontFace, fontScale, color, tickness, linetype);
}
}  // ::is

int main(int argc, char* argv[]) {
  std::string uri;
  std::vector<std::string> cameras;
  const std::vector<std::string> default_cameras{"ptgrey.0", "ptgrey.1", "ptgrey.2", "ptgrey.3"};
  std::string bb_topic;
  std::string prefix;
  double fps;
  std::string path;
  bool show_radar;
  bool hold_positions;
  bool clustering;
  double clustering_threshold;
  bool original_detections;
  unsigned int width;
  unsigned int height;

  po::options_description description("Allowed options");
  auto&& options = description.add_options();
  options("help,", "show available options");
  options("uri,u", po::value<std::string>(&uri)->default_value("amqp://192.168.1.110:30000"), "broker uri");
  options("cameras,c", po::value<std::vector<std::string>>(&cameras)->multitoken()->default_value(default_cameras),
          "cameras");
  options("bb_topic,b", po::value<std::string>(&bb_topic)->default_value("bbs"), "boundbox topic");
  options("prefix,P", po::value<std::string>(&prefix)->default_value(""), "prefix of topics");
  options("fps,f", po::value<double>(&fps), "fps");
  options("radar,r", po::bool_switch(&show_radar), "enables pedestrian radar");
  options("hold_positions,H", po::bool_switch(&hold_positions), "hold pedestrian positions on radar");
  options("clustering,C", po::bool_switch(&clustering), "enables clustering positions");
  options("original_detections,O", po::bool_switch(&original_detections), "enables original positions");
  options("clustering_distance,d", po::value<double>(&clustering_threshold)->default_value(500.0),
          "clustering threshold");
  options("rwidth,w", po::value<unsigned int>(&width)->default_value(800), "radar width");
  options("rheight,h", po::value<unsigned int>(&height)->default_value(500), "radar height");
  options("parameters,p", po::value<std::string>(&path)->default_value("../cameras-parameters/"),
          "cameras parameters path");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, description), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << description << std::endl;
    return 1;
  }

  auto is = is::connect(uri);
  auto client = is::make_client(is);

  int64_t period;
  if (!vm.count("fps")) {
    std::vector<std::string> ids;
    for (auto& camera : cameras) {
      ids.push_back(client.request(camera + ".get_configuration", is::msgpack(0)));
    }
    auto configuration_msgs = client.receive_until(high_resolution_clock::now() + 1s, ids, is::policy::discard_others);
    if (configuration_msgs.size() != cameras.size())
      exit(0);

    auto configuration = is::msgpack<Configuration>(configuration_msgs.at(ids.front()));
    period = static_cast<int64_t>(1000.0 / *((*(configuration.sampling_rate)).rate));
  } else {
    period = static_cast<int64_t>(1000.0 / fps);
  }

  for (auto& camera : cameras) {
    client.request(prefix + camera + ".reset_iterator", is::msgpack(""));
  }
  client.receive_for(1s);

  // std::vector<std::string> topics;
  // for (auto& camera : cameras) {
  //   topics.push_back(prefix + camera + ".frame");
  // }
  // auto tag = is.subscribe(topics);

  // std::vector<std::string> bbs_topics;
  // for (auto& camera : cameras) {
  //   bbs_topics.push_back(prefix + camera + "." + bb_topic);
  // }
  // auto bbs_tag = is.subscribe(bbs_topics);

  std::vector<std::string> topics;
  for (auto& camera : cameras) {
    topics.push_back(prefix + camera + ".frame");
    topics.push_back(prefix + camera + "." + bb_topic);
  }
  auto tag = is.subscribe(topics);

  is::log::info("Starting capture. Period: {} ms", period);
  Radar radar(width, height);
  auto parameters = camera::load_parameters(path);

  while (1) {
    auto msgs = is.consume_sync(tag, topics, period);
    std::vector<is::Envelope::ptr_t> images_msg;
    std::vector<is::Envelope::ptr_t> bbs_msg;
    std::for_each(std::begin(msgs), std::end(msgs), [&](auto& msg) {
      auto routing_key = msg->RoutingKey();
      auto pos = routing_key.find_last_of('.');
      auto type = routing_key.substr(pos + 1);
      if (type == bb_topic) {
        bbs_msg.push_back(msg);
      } else if (type == "frame") {
        images_msg.push_back(msg);
      }
    });

    // auto images_msg = is.consume_sync(tag, topics, period);
    // auto bbs_msg = is.consume_sync(bbs_tag, bbs_topics, period);

    auto it_begin = boost::make_zip_iterator(boost::make_tuple(images_msg.begin(), bbs_msg.begin()));
    auto it_end = boost::make_zip_iterator(boost::make_tuple(images_msg.end(), bbs_msg.end()));
    std::vector<cv::Mat> up_frames, down_frames;
    int n_frame = 0;
    std::for_each(it_begin, it_end, [&](auto msgs) {
      auto image_msg = boost::get<0>(msgs);
      auto bbs_msg = boost::get<1>(msgs);
      auto image = is::msgpack<CompressedImage>(image_msg);
      arma::mat bbs = is::msgpack<arma::mat>(bbs_msg);

      cv::Mat current_frame = cv::imdecode(image.data, CV_LOAD_IMAGE_COLOR);

      bbs.each_row([&](arma::rowvec const& a) {
        cv::rectangle(current_frame, cv::Rect(a(0), a(1), a(2), a(3)), colors.at(static_cast<int>(a(5))), 3);
        std::stringstream ss_text;
        ss_text << std::setw(4) << a(4);
        std::string text = ss_text.str();
        int fontface = cv::FONT_HERSHEY_SIMPLEX;
        double scale = 1.0;
        int tickness = 2;

        cv::Point point(a(0) + a(2), a(1) + a(3));
        is::putText(current_frame, text, point, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(255, 255, 255),
                    cv::Scalar(0, 0, 0), tickness);
      });

      is::putText(current_frame, "ptgrey." + std::to_string(n_frame), cv::Point(0, 22), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                  colors.at(n_frame + 1), cv::Scalar(0, 0, 0), 2);

      cv::resize(current_frame, current_frame, cv::Size(current_frame.cols / 2, current_frame.rows / 2));
      if (n_frame < 2) {
        up_frames.push_back(current_frame);
      } else {
        down_frames.push_back(current_frame);
      }
      n_frame++;
    });

    cv::Mat output_image;
    cv::Mat up_row, down_row;
    std::vector<cv::Mat> rows_frames;
    cv::hconcat(up_frames, up_row);
    rows_frames.push_back(up_row);
    cv::hconcat(down_frames, down_row);
    rows_frames.push_back(down_row);
    cv::vconcat(rows_frames, output_image);

    cv::imshow("Intelligent Space", output_image);

    if (show_radar) {
      std::map<std::string, arma::mat> bbs;
      transform(std::begin(bbs_msg), std::end(bbs_msg), std::begin(cameras), std::inserter(bbs, std::begin(bbs)),
                [](auto& msg, auto& camera) { return std::make_pair(camera, is::msgpack<arma::mat>(msg)); });
      auto positions = bbs::get_position(bbs, parameters);
      if (original_detections) {
        radar.update(positions, cv::Scalar(0, 255, 255), 5, hold_positions);
      }
      if (clustering) {
        auto clustered_positions = bbs::clustering::distance(positions, clustering_threshold);
        radar.update(clustered_positions, cv::Scalar(255, 0, 255));
      }
      cv::imshow("Radar", radar.get_image());
      cv::waitKey(1);
    }
    cv::waitKey(1);
  }

  is::logger()->info("Exiting");
  return 0;
}