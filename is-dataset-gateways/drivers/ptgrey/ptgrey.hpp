#ifndef __IS_DRIVER_PTGREY_HPP__
#define __IS_DRIVER_PTGREY_HPP__

#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/thread/mutex.hpp>
#include <condition_variable>
#include <is/logger.hpp>
#include <is/msgs/camera.hpp>
#include <is/msgs/common.hpp>
#include <mutex>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <thread>

namespace is {
namespace driver {

using namespace std::chrono;
using namespace std::chrono_literals;
using namespace is::msg::camera;
using namespace is::msg::common;
namespace fs = boost::filesystem;

struct PtGrey {
  std::string entity;
  std::string folder;
  std::vector<std::string> images;
  std::vector<std::string>::iterator img_iterator;

  std::mutex mutex;
  std::condition_variable cv;
  std::thread next_image_thread;
  time_point<system_clock> now;
  bool new_image;

  Timestamp last_timestamp;
  cv::Mat last_frame;

  enum State { STOPPED, CAPTURING };
  State state;

  PtGrey(std::string const& entity, std::string const& folder) : entity(entity), folder(folder), new_image(true) {
    state = CAPTURING;

    fs::path images_folder(folder);
    if (images_folder.empty())
      throw std::runtime_error("Empty folder: " + folder);

    fs::recursive_directory_iterator end;
    for (fs::recursive_directory_iterator i(images_folder); i != end; ++i) {
      const fs::path cp = (*i);
      auto filename = cp.string();
      if (is_regular_file(cp) && filename.find(entity) != std::string::npos) {
        this->images.push_back(filename);
      }
    }

    auto file2num = [](std::string const& file) {
      auto last_dot = file.find_last_of('.');
      auto filename = file;
      filename.resize(last_dot);  // remove extension
      last_dot = filename.find_last_of('.');
      return std::stoi(filename.substr(last_dot + 1));
    };

    std::sort(std::begin(this->images), std::end(this->images),
              [&](auto& lhs, auto& rhs) { return file2num(lhs) < file2num(rhs); });

    this->img_iterator = std::begin(this->images);
    this->last_frame = cv::imread(this->next_image());

    this->next_image_thread = std::thread([this]() {
      this->now = high_resolution_clock::now();
      while (1) {
        this->now = this->now + milliseconds(250);
        {
          std::unique_lock<std::mutex> lk(this->mutex);
          auto filename = this->next_image();
          if (this->state == CAPTURING) {
            this->last_frame = cv::imread(filename);
          }
          this->new_image = true;
        }
        this->cv.notify_one();
        is::log::info("new_image");
        std::this_thread::sleep_for(this->now - high_resolution_clock::now());
      }
    });
  }

  ~PtGrey() {}

  void reset_iterator() {
    std::unique_lock<std::mutex> lk(this->mutex);
    this->img_iterator = std::begin(this->images);
  }

  std::string next_image() {
    auto filename = *this->img_iterator;
    if (++this->img_iterator == std::end(this->images))
      this->img_iterator = std::begin(this->images);
    return filename;
  }

  void start_capture() {
    std::unique_lock<std::mutex> lk(this->mutex);
    if (state == STOPPED) {
      is::log::info("Starting capture");
      state = CAPTURING;
    }
  }

  void stop_capture() {
    std::unique_lock<std::mutex> lk(this->mutex);
    if (state == CAPTURING) {
      is::log::info("Stopping capture");
      state = STOPPED;
    }
  }

  void set_sample_rate(SamplingRate) {}

  void set_resolution(Resolution) {}

  void set_image_type(ImageType) {}

  void set_delay(Delay) {}

  void update() {
    if (state == CAPTURING) {
      std::unique_lock<std::mutex> lock(mutex);
      if (this->cv.wait_for(lock, 500ms, [this]() { return this->new_image; })) {
        this->new_image = false;
      } else {
        is::log::warn("Frame timeouted, restarting");
      }
    }
  }

  cv::Mat get_last_frame() {
    std::unique_lock<std::mutex> lk(this->mutex);
    return last_frame;
  }

  Timestamp get_last_timestamp() { return last_timestamp; }
};

}  // ::driver
}  // ::is

#endif  // __IS_DRIVER_PTGREY_HPP__