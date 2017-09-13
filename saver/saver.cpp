#include <signal.h>
#include <armadillo>
#include <atomic>
#include <boost/filesystem.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include <boost/program_options.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <is/is.hpp>
#include <is/msgs/camera.hpp>
#include <is/msgs/common.hpp>
#include <is/msgs/robot.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "arma.hpp"
#include "bbs.hpp"

namespace fs = boost::filesystem;
namespace po = boost::program_options;
using namespace boost;
using namespace boost::lockfree;
using namespace is::msg::camera;
using namespace is::msg::common;
using namespace is::msg::robot;
using namespace arma;

template <typename T, size_t Size>
class SyncQueue {
 public:
  SyncQueue() { mtx.lock(); }

  bool push(const T& t) {
    bool ret = queue.push(t);
    mtx.unlock();
    return ret;
  }

  bool pop(T& t) {
    // mtx.lock();
    return queue.pop(t);
  }

  void wait() { mtx.lock(); }

 private:
  bool ready;
  std::mutex mtx;
  boost::lockfree::spsc_queue<T, boost::lockfree::capacity<Size>> queue;
};

namespace std {
std::ostream& operator<<(std::ostream& os, const std::vector<std::string>& vec) {
  for (auto item : vec) {
    os << item << " ";
  }
  return os;
}
}

std::atomic<bool> capturing{true};

void sig_function(int) {
  capturing.store(false);
  is::log::info("SIGINT received");
}

int main(int argc, char* argv[]) {
  std::string uri;
  std::vector<std::string> cameras;
  std::string output;
  std::string prefix;
  std::string parameters_path;

  po::options_description description("Allowed options");
  auto&& options = description.add_options();
  options("help,", "show available options");
  options("uri,u", po::value<std::string>(&uri)->default_value("amqp://edge.is:30000"), "broker uri");
  options("cameras,c", po::value<std::vector<std::string>>(&cameras)->multitoken()->default_value(
                           {"ptgrey.0", "ptgrey.1", "ptgrey.2", "ptgrey.3"}),
          "cameras");
  options("directory,d", po::value<std::string>(&output)->default_value("."), "directory to save data");
  options("prefix,P", po::value<std::string>(&prefix)->default_value(""), "prefix of topics");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, description), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << description << std::endl;
    return 1;
  }

  auto is = is::connect(uri);

  std::vector<std::string> topics;
  for (auto& camera : cameras) {
    topics.push_back(prefix + camera + ".frame");
    topics.push_back(prefix + camera + ".bbs");
    topics.push_back(prefix + camera + ".new_bbs");
  }
  topics.push_back("robot-controller.0.pose");

  auto tag = is.subscribe(topics);

  is::log::info("Starting capture");

  SyncQueue<is::Envelope::ptr_t, 100> messages_to_save;

  auto output_folder = fs::path(output);
  if (!fs::exists(output_folder)) {
    fs::create_directory(output);
    is::log::info("Folder {} doesn't exists. Creating..", output);
  } else if (!fs::is_directory(output_folder)) {
    is::log::warn("{} already exists but isn't a directoty. Exiting.. ", output);
    exit(1);
  }

  std::thread save_images([&] {

    enum SaveType { FRAME, BBS, NEW_BBS, POSE };
    std::map<std::string, SaveType> str2type({{"frame", FRAME}, {"bbs", BBS}, {"new_bbs", NEW_BBS}, {"pose", POSE}});

    std::map<std::string, int64_t> topic_id;
    std::transform(topics.begin(), topics.end(), std::inserter(topic_id, topic_id.begin()),
                   [](auto t) { return std::make_pair(t, 0); });

    std::stringstream topics_log;

    while (1) {
      messages_to_save.wait();
      bool consumed;

      while (1) {
        is::Envelope::ptr_t msg;
        consumed = messages_to_save.pop(msg);
        if (!consumed)
          break;

        auto topic = msg->RoutingKey();
        auto pos = topic.find_last_of('.');
        auto type = topic.substr(pos + 1);
        auto prefix_camera = topic.substr(0, pos);
        auto camera = prefix_camera.substr(prefix.size(), prefix_camera.size());

        topics_log << topic << ';' << topic_id[topic] << ';' << msg->Message()->Timestamp() << '\n';

        switch (str2type[type]) {
        case FRAME: {
          auto image = is::msgpack<CompressedImage>(msg);
          auto filename =
              (output_folder / fs::path(topic + "_" + std::to_string(topic_id[topic]) + image.format)).string();

          std::ofstream ofs;
          ofs.open(filename, std::ofstream::binary);
          std::copy(std::begin(image.data), std::end(image.data), std::ostream_iterator<unsigned char>(ofs));
          ofs.close();
          break;
        }

        case BBS:
        case NEW_BBS: {
          arma::mat bb = is::msgpack<arma::mat>(msg);
          auto filename = (output_folder / fs::path(topic + "_" + std::to_string(topic_id[topic]) + ".mat")).string();
          bb.save(filename, arma::raw_ascii);
          break;
        }

        case POSE: {
          auto pose = is::msgpack<optional<Pose>>(msg);
          auto filename = (output_folder / fs::path(topic + "_" + std::to_string(topic_id[topic]) + ".mat")).string();

          arma::mat arma_pose;
          if (pose) {
            arma_pose = arma::mat({(*pose).position.x, (*pose).position.y, (*pose).heading});
          }
          arma_pose.save(filename, arma::raw_ascii);
          break;
        }
        }

        topic_id[topic]++;
      }

      if (!capturing.load() && !consumed) {
        std::ofstream ofs;
        auto filename = (output_folder / fs::path("topics_log")).string();

        is::log::info("Writing topics_log file");
        ofs.open(filename, std::ios::binary);
        ofs << topics_log.rdbuf();
        ofs.close();
        break;
      }
    }
  });

  signal(SIGINT, sig_function);

  while (capturing.load()) {
    auto envelope = is.consume(tag);
    messages_to_save.push(envelope);
  }

  save_images.join();
  is::log::info("Exiting");
  return 0;
}