#ifndef __RADAR_HPP__
#define __RADAR_HPP__

#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <armadillo>

struct Radar {
  unsigned int width;
  unsigned int height;
  unsigned int arrow_length;
  unsigned int x_step;
  unsigned int y_step;
  const cv::Point origin;
  cv::Mat image;

  Radar(unsigned int width, unsigned int height, unsigned int arrow_length = 50, unsigned int x_step = 100,
        unsigned int y_step = 100)
      : width(width),
        height(height),
        arrow_length(arrow_length),
        origin(width / 2, height / 2),
        x_step(x_step),
        y_step(y_step),
        image(cv::Mat::zeros(cv::Size(width, height), CV_8UC3)) {}

  void update(arma::mat const& positions, cv::Scalar const& color, unsigned int radius = 5, bool persistent = true) {
    if (!persistent)
        this->image = cv::Mat::zeros(cv::Size(this->width, this->height), CV_8UC3);
    draw_grid(this->image, this->origin, this->x_step, this->y_step);
    positions.each_row([&](arma::rowvec const& p) {
      cv::circle(this->image, pos2pixel(p, this->image.size(), origin, 0.1, -1, 1), radius, color, -1);
    });
    draw_axis(this->image, this->origin, this->arrow_length, -1, 1, 2);
  }

  cv::Mat get_image() { return this->image; }

 private:
  cv::Point pos2pixel(arma::mat const& position, cv::Size const& image_size, cv::Point const& center,
                      double scale = 0.1, int x_dir = 1, int y_dir = 1) {
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
};

#endif  // __RADAR_HPP__