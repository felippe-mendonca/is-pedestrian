#ifndef __BBS_HPP__
#define __BBS_HPP__

#include <algorithm>
#include <armadillo>
#include <functional>
#include <is/is.hpp>
#include <map>
#include <string>
#include <vector>
#include "camera.hpp"

namespace bbs {

using namespace arma;
using namespace camera;

mat change_frame(mat const& bb, std::map<std::string, CameraParameters> const& parameters, int dst,
                 std::function<bool(mat)> validation = [](mat) { return true; }) {
  if (bb.is_empty())
    return mat(0, 0);
  // return bb;
  auto id = static_cast<int>(bb(0, 5) - 1.0);
  is::log::info("src/dst {}/{}", id, dst);
  if (id == dst)
    return mat(0, 0);
  // return bb;

  auto src_camera = "ptgrey." + std::to_string(id);
  auto dst_camera = "ptgrey." + std::to_string(dst);

  auto x = bb(0, 0);
  auto y = bb(0, 1);
  auto w = bb(0, 2);
  auto h = bb(0, 3);
  mat pc = mat({x + w / 2, y + h, 1.0}).t();
  double z_src, z_dst;
  mat wpc = c2w(pc, src_camera, parameters, 0.0, z_src);
  if (!validation(wpc)) {
    is::log::warn("Out of workspace. {},{}", wpc(0), wpc(1));
    return mat(0, 0);
  }
  mat npc = w2c(wpc, dst_camera, parameters, z_dst);

  double scale = z_src / z_dst;

  double nw = w * scale;
  double nh = h * scale;
  double nx = npc(0) - nw / 2;
  double ny = npc(1) - nh;
  // is::log::info("z_src {} | z_dst {} | <nw, nh, nx, ny> = {}, {}, {}, {}", z_src, z_dst, nw, nh, nx, ny);
  mat nbb = {nx, ny, nw, nh, bb(0, 4), static_cast<double>(dst) + 1.0};
  return nbb;
}

mat change_frame(std::vector<mat> const& cameras_bbs, std::map<std::string, CameraParameters> const& parameters,
                 int dst, std::function<bool(mat)> validation = [](mat) { return true; }) {
  mat output;
  for (auto& bbs : cameras_bbs) {
    if (bbs.is_empty())
      continue;
    auto src = static_cast<int>(bbs(0, 5) - 1.0);
    if (src == dst)
      continue;

    auto src_camera = "ptgrey." + std::to_string(src);
    auto dst_camera = "ptgrey." + std::to_string(dst);
    // is::log::info("src/dst {}/{}", src, dst);

    bbs.each_row([&](rowvec const& bb) {
      auto x = bb(0, 0);
      auto y = bb(0, 1);
      auto w = bb(0, 2);
      auto h = bb(0, 3);
      mat pc = mat({x + w / 2, y + h, 1.0}).t();
      double z_src, z_dst;
      mat wpc = c2w(pc, src_camera, parameters, 0.0, z_src);
      if (!validation(wpc)) {
        is::log::warn("Out of workspace. {},{}", wpc(0), wpc(1));
        return mat(0, 0);
      }
      mat npc = w2c(wpc, dst_camera, parameters, z_dst);

      double scale = z_src / z_dst;
      double nw = w * scale;
      double nh = h * scale;
      double nx = npc(0) - nw / 2;
      double ny = npc(1) - nh;
      // is::log::info("z_src {} | z_dst {} | <nw, nh, nx, ny> = {}, {}, {}, {}", z_src, z_dst, nw, nh, nx, ny);
      mat nbb = {nx, ny, nw, nh, bb(0, 4), static_cast<double>(dst) + 1.0};
      output = join_vert(output, nbb);
    });
  }
  return output;
}

std::map<std::string, mat> change_frame(std::map<std::string, mat> cameras_bbs,
                                        std::map<std::string, CameraParameters> const& parameters,
                                        std::string const& dst_camera, bool discard_dst = true,
                                        std::function<bool(mat)> validation = [](mat) { return true; }) {
  if (discard_dst)
    cameras_bbs.erase(cameras_bbs.find(dst_camera));

  auto dst = stoi(dst_camera.substr(dst_camera.find(".") + 1));
  mat output_bbs;
  std::for_each(std::begin(cameras_bbs), std::end(cameras_bbs), [&](auto camera_bbs) {
    auto src_camera = camera_bbs.first;
    auto bbs = camera_bbs.second;
    bbs.each_row([&](rowvec const& bb) {
      auto x = bb(0, 0);
      auto y = bb(0, 1);
      auto w = bb(0, 2);
      auto h = bb(0, 3);
      mat pc = mat({x + w / 2, y + h, 1.0}).t();
      double z_src, z_dst;
      mat wpc = c2w(pc, src_camera, parameters, 0.0, z_src);
      if (!validation(wpc)) {
        is::log::warn("Out of workspace. {},{}", wpc(0), wpc(1));
        return mat(0, 0);
      }
      mat npc = w2c(wpc, dst_camera, parameters, z_dst);

      double scale = z_src / z_dst;
      double nw = w * scale;
      double nh = h * scale;
      double nx = npc(0) - nw / 2;
      double ny = npc(1) - nh;
      mat nbb = {nx, ny, nw, nh, bb(0, 4), static_cast<double>(dst) + 1.0};
      output_bbs = join_vert(output_bbs, nbb);
    });
  });

  return {{dst_camera, output_bbs}};
}

double iou(mat const& bb_src, mat const& bb) {
  auto w = std::min(bb_src(0) + bb_src(2), bb(0) + bb(2)) -
           std::max(bb_src(0), bb(0));  // min(x_src + w_src, x + w) - max(x_src, x)
  if (w <= 0.0)
    return 0.0;
  auto h = std::min(bb_src(3) + bb_src(1), bb(3) + bb(1)) - std::max(bb_src(1), bb(1));
  if (h <= 0.0)
    return 0.0;

  auto i = w * h;
  auto u = bb_src(2) * bb_src(3) + bb(2) * bb(3) - i;
  // is::log::info("<w,h,i,u,i/u> {},{},{},{},{}", w, h, i, u, i / u);
  return i / u;
}

mat iou_validation(mat const& bbs, mat const& transformed_bbs, unsigned int min_intersections = 1, double th = 0.5) {
  mat output;
  bbs.each_row([&](rowvec const& bb) {
    auto n_intersections = 0;
    transformed_bbs.each_row([&](rowvec const& t_bb) {
      if (iou(bb, t_bb) > th)
        ++n_intersections;
    });
    if (n_intersections >= min_intersections)
      output = join_vert(output, bb);
  });
  return output;
}

mat validate_bbs(mat& bbs, std::vector<mat> const& cameras_bbs,
                 std::map<std::string, CameraParameters> const& parameters) {
  if (bbs.is_empty())
    return mat(0, 0);
  auto dst = static_cast<int>(bbs(0, 5) - 1.0);
  // is::log::info("CAMERA: {}", dst);
  mat transformed_bbs = change_frame(cameras_bbs, parameters, dst);
  // transformed_bbs.print("transformed_bbs " + std::to_string(dst));
  mat validated_bbs = iou_validation(bbs, transformed_bbs);
  // validated_bbs.print("val_bbs " + std::to_string(dst));
  return validated_bbs;
}

mat get_position(std::map<std::string, mat> const& cameras_bbs,
                 std::map<std::string, CameraParameters> const& parameters) {
  mat output;
  std::for_each(std::begin(cameras_bbs), std::end(cameras_bbs), [&](auto& bbs) {
    auto src_camera = bbs.first;
    bbs.second.each_row([&](rowvec const& bb) {
      auto x = bb(0, 0);
      auto y = bb(0, 1);
      auto w = bb(0, 2);
      auto h = bb(0, 3);
      mat pc = mat({x + w / 2, y + h, 1.0}).t();
      mat wpc = c2w(pc, src_camera, parameters);
      output = join_vert(output, wpc.rows(0, 1).t());
    });
  });
  return output;
}

namespace clustering {
arma::mat distance(arma::mat positions, double const& threshold) {
  std::vector<std::pair<arma::mat, double>> position_dist;
  positions.each_row([&](arma::rowvec const& position) { position_dist.push_back(std::make_pair(position, 0.0)); });

  arma::mat clustered_points;
  for (auto it = std::begin(position_dist); it != std::end(position_dist);) {
    auto ref_point = it->first;
    std::for_each(it, std::end(position_dist), [&](auto& pos_dist) {
      auto point = pos_dist.first;
      pos_dist.second = arma::norm(ref_point - point, 2);
    });
    std::sort(it, std::end(position_dist), [](auto const& lhs, auto const& rhs) { return lhs.second < rhs.second; });
    auto lb_it = std::lower_bound(it, std::end(position_dist), threshold,
                                  [](auto const& a, auto const& b) { return a.second < b; });
    arma::mat sum = std::accumulate(it, lb_it, arma::mat(1, 2, arma::fill::zeros),
                                    [](auto const& a, auto const& b) { return a + b.first; });
    clustered_points = arma::join_vert(clustered_points, sum / std::distance(it, lb_it));
    it = lb_it;
  }
  return clustered_points;
}

arma::mat distance(std::map<std::string, mat> cameras_bbs, std::map<std::string, CameraParameters> const& parameters,
                   double const& threshold) {
  auto positions = get_position(cameras_bbs, parameters);
  return distance(positions, threshold);
}
}  // ::bbs::clustering

}  // ::bbs

#endif  // __BBS_HPP__