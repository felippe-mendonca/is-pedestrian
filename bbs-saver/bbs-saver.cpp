#include "bbs.hpp"
#include <armadillo>
#include <atomic>
#include <boost/lockfree/spsc_queue.hpp>
#include <boost/program_options.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <is/is.hpp>
#include <is/msgs/camera.hpp>
#include <is/msgs/common.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include "arma.hpp"
#include "camera.hpp"

namespace po = boost::program_options;
using namespace boost;
using namespace boost::lockfree;
using namespace is::msg::camera;
using namespace is::msg::common;
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

int main(int argc, char* argv[]) {
  std::string uri;
  std::vector<std::string> cameras;
  const std::vector<std::string> default_cameras{"ptgrey.0", "ptgrey.1", "ptgrey.2", "ptgrey.3"};
  Resolution resolution;
  SamplingRate sample_rate;
  double fps;
  std::string image_type;
  std::string folder;
  int n_images;
  std::string prefix;
  std::string path;

  po::options_description description("Allowed options");
  auto&& options = description.add_options();
  options("help,", "show available options");
  options("uri,u", po::value<std::string>(&uri)->default_value("amqp://localhost"), "broker uri");
  options("cameras,c", po::value<std::vector<std::string>>(&cameras)->multitoken()->default_value(default_cameras),
          "cameras");
  options("height,h", po::value<unsigned int>(&resolution.height)->default_value(728), "image height");
  options("width,w", po::value<unsigned int>(&resolution.width)->default_value(1288), "image width");
  options("fps,f", po::value<double>(&fps)->default_value(5.0), "frames per second");
  options("type,t", po::value<std::string>(&image_type)->default_value("rgb"), "image type");
  options("directory,d", po::value<std::string>(&folder)->default_value("."), "image type");
  options("number_images,n", po::value<int>(&n_images)->default_value(0), "image type");
  options("prefix,P", po::value<std::string>(&prefix)->default_value(""), "prefix of topics");
  options("parameters,p", po::value<std::string>(&path)->default_value("../cameras-parameters/"),
          "cameras parameters path");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, description), vm);
  po::notify(vm);

  if (vm.count("help") || !vm.count("cameras")) {
    std::cout << description << std::endl;
    return 1;
  }

  auto parameters = camera::load_parameters(path);

  auto is = is::connect(uri);
  auto client = is::make_client(is);

  sample_rate.rate = fps;
  for (auto& camera : cameras) {
    client.request(prefix + camera + ".reset_iterator", is::msgpack(""));
    client.request(prefix + camera + ".set_sample_rate", is::msgpack(sample_rate));
    client.request(prefix + camera + ".set_resolution", is::msgpack(resolution));
    client.request(prefix + camera + ".set_image_type", is::msgpack(ImageType{image_type}));
  }
  // client.receive_for(1s);

  std::vector<std::string> topics;
  for (auto& camera : cameras) {
    topics.push_back(prefix + camera + ".frame");
    topics.push_back(prefix + camera + ".bbs");
    topics.push_back(prefix + camera + ".new_bbs");
  }

  auto tag = is.subscribe(topics);
  auto period = static_cast<int64_t>(1000.0 / fps);

  // SyncRequest sr;
  // sr.entities = cameras;
  // sr.sampling_rate = sample_rate;
  // is::logger()->info("Sync request");
  // client.request("is.sync", is::msgpack(sr));

  is::log::info("Starting capture");

  SyncQueue<std::vector<is::Envelope::ptr_t>, 10> images_to_save;
  int n_images_saved = 0;
  std::atomic<bool> capturing{true};

  std::thread save_images([&] {

    enum SaveType { FRAME, BBS, NEW_BBS };
    std::map<std::string, SaveType> str2type({{"frame", FRAME}, {"bbs", BBS}, {"new_bbs", NEW_BBS}});

    std::map<std::string, arma::mat> current_bbs;
    std::map<std::string, arma::mat> current_new_bbs;

    std::map<std::string, arma::mat> bbs;
    std::map<std::string, arma::mat> new_bbs;
    arma::mat xy;
    arma::mat clustered_xy;
    arma::mat new_xy;
    arma::mat new_clustered_xy;

    for (auto& camera : cameras) {
      bbs.emplace(std::make_pair(prefix + camera, arma::mat(0, 0, arma::fill::zeros)));
      new_bbs.emplace(std::make_pair(prefix + camera, arma::mat(0, 0, arma::fill::zeros)));
      current_bbs.emplace(std::make_pair(camera, arma::mat(0, 0, arma::fill::zeros)));
      current_new_bbs.emplace(std::make_pair(camera, arma::mat(0, 0, arma::fill::zeros)));
    }

    while (1) {
      std::vector<is::Envelope::ptr_t> images_message;
      images_to_save.wait();
      bool consumed;
      while (1) {
        consumed = images_to_save.pop(images_message);
        if (!consumed)
          break;

        std::for_each(std::begin(images_message), std::end(images_message), [&](auto& msg) {
          auto routing_key = msg->RoutingKey();
          auto pos = routing_key.find_last_of('.');
          auto type = routing_key.substr(pos + 1);
          auto prefix_camera = routing_key.substr(0, pos);
          auto camera = prefix_camera.substr(prefix.size(), prefix_camera.size());

          switch (str2type[type]) {
          case FRAME: {
            auto image = is::msgpack<CompressedImage>(msg);

            std::ostringstream name;
            name << folder << "/" << prefix_camera << '.' << std::setw(6) << std::setfill('0') << n_images_saved
                 << image.format;

            std::ofstream ofs;
            ofs.open(name.str(), std::ofstream::binary);
            std::copy(std::begin(image.data), std::end(image.data), std::ostream_iterator<unsigned char>(ofs));
            ofs.close();
            break;
          }

          case BBS: {
            arma::mat bb = is::msgpack<arma::mat>(msg);
            current_bbs[camera] = bb;
            if (!bb.empty()) {
              bb = join_horiz(n_images_saved * mat(bb.n_rows, 1, fill::ones), bb.cols(0, bb.n_cols - 2));
              bbs[prefix_camera] = join_vert(bbs[prefix_camera], bb);
            }
            break;
          }

          case NEW_BBS: {
            arma::mat bb = is::msgpack<arma::mat>(msg);
            current_new_bbs[camera] = bb;
            if (!bb.empty()) {
              bb = join_horiz(n_images_saved * mat(bb.n_rows, 1, fill::ones), bb.cols(0, bb.n_cols - 2));
              new_bbs[prefix_camera] = join_vert(new_bbs[prefix_camera], bb);
            }
            break;
          }
          }
        });

        arma::mat bbs_positions = bbs::get_position(current_bbs, parameters);
        if (!bbs_positions.empty()) {
          xy = join_vert(xy, join_horiz(n_images_saved * mat(bbs_positions.n_rows, 1, fill::ones), bbs_positions));

          arma::mat bbs_clustered_positions = bbs::clustering::distance(bbs_positions, 500.0);
          if (!bbs_clustered_positions.empty()) {
            bbs_clustered_positions = join_horiz(n_images_saved * mat(bbs_clustered_positions.n_rows, 1, fill::ones),
                                                 bbs_clustered_positions);
            clustered_xy = join_vert(clustered_xy, bbs_clustered_positions);
          }
        }

        arma::mat new_bbs_positions = bbs::get_position(current_new_bbs, parameters);
        if (!new_bbs_positions.empty()) {
          new_xy = join_vert(
              new_xy, join_horiz(n_images_saved * mat(new_bbs_positions.n_rows, 1, fill::ones), new_bbs_positions));

          arma::mat new_bbs_clustered_positions = bbs::clustering::distance(new_bbs_positions, 500.0);
          if (!new_bbs_clustered_positions.empty()) {
            new_bbs_clustered_positions = join_horiz(
                n_images_saved * mat(new_bbs_clustered_positions.n_rows, 1, fill::ones), new_bbs_clustered_positions);
            new_clustered_xy = join_vert(new_clustered_xy, new_bbs_clustered_positions);
          }
        }
        n_images_saved++;
      }
      if (!capturing.load() && !consumed) {
        for (auto& camera : cameras) {
          bbs[prefix + camera].save(folder + "/" + prefix + camera + ".bbs.mat", arma::raw_ascii);
          new_bbs[prefix + camera].save(folder + "/" + prefix + camera + ".new_bbs.mat", arma::raw_ascii);
        }
        xy.save(folder + "/xy.mat", arma::raw_ascii);
        clustered_xy.save(folder + "/clustered_xy.mat", arma::raw_ascii);
        new_xy.save(folder + "/new_xy.mat", arma::raw_ascii);
        new_clustered_xy.save(folder + "/new_clustered_xy.mat", arma::raw_ascii);
        break;
      }
    }
  });

  for (int n = 0; n < n_images; ++n) {
    images_to_save.push(is.consume_sync(tag, topics, period));
    is::log::info("{}/{}", n + 1, n_images);
  }

  capturing.store(false);

  save_images.join();
  is::log::info("Exiting");
  return 0;
}