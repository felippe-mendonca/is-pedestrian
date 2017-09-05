#ifndef __CAMERA_HPP__
#define __CAMERA_HPP__

#include <armadillo>
#include <map>
#include <string>
#include <vector>
#include "boost/filesystem.hpp"

namespace camera {

using namespace std;
using namespace arma;
namespace fs = boost::filesystem;

struct CameraParameters {
  mat K;  // Intrinsic Parameters
  mat R;  // Rotation matrix (extrinsic parameters)
  mat T;  // Translation vector (extrinsic parameters)
};

std::map<std::string, CameraParameters> load_parameters(std::string const& dir) {
  
  fs::path parameters_dir(dir);
  if (parameters_dir.empty())
    throw std::runtime_error("Empty dir: " + dir);

  std::vector<std::string> cameras;
  fs::recursive_directory_iterator end;
  for (fs::recursive_directory_iterator i(parameters_dir); i != end; ++i) {
      const fs::path cp = (*i);
      if (is_directory(cp)) {
        cameras.push_back(cp.string());
      }
  }

  // Load camera parameters
  std::map<std::string, CameraParameters> parameters;
  for (auto& camera : cameras) {
    mat E, I;
    E.load(camera + "/extrinsic.dat");
    I.load(camera + "/intrinsic.dat");
    auto pos = camera.find_last_of('/');
    auto camera_key = camera.substr(pos+1);
    parameters.insert({camera_key, {I, E(span(0, 2), span(0, 2)), E(span(0, 2), 3)}});
    is::log::info("{} parameters loaded", camera_key);
  }
  return parameters;
}

mat c2w(mat const& p, std::string camera, std::map<std::string, CameraParameters> const& parameters,
    double const& z, double& zc) {
mat E = join_vert(join_horiz(parameters.at(camera).R, parameters.at(camera).T), mat({0.0, 0.0, 0.0, 1.0}));
mat B = parameters.at(camera).K * eye(3, 4) * E;
mat A = join_horiz(B.cols(0, 1), -p);
mat b = -B.col(3) - z * B.col(2);
mat P = inv(A) * b;
zc = P.at(2);
P.at(2) = z;
return join_vert(P, mat({1.0}));
}

mat c2w(mat const& p, std::string camera, std::map<std::string, CameraParameters> const& parameters,
    double z = 0.0) {
double zc;
return c2w(p, camera, parameters, z, zc);
}

mat w2c(mat const& P, std::string camera, std::map<std::string, CameraParameters> const& parameters,
    double& zc) {
mat E = join_vert(join_horiz(parameters.at(camera).R, parameters.at(camera).T), mat({0.0, 0.0, 0.0, 1.0}));
mat p = parameters.at(camera).K * eye(3, 4) * E * P;
zc = p.at(2);
p.at(0) = p.at(0) / p.at(2);
p.at(1) = p.at(1) / p.at(2);
p.at(2) = 1.0;
return p;
}

mat w2c(mat const& P, std::string camera, std::map<std::string, CameraParameters> const& parameters) {
double zc;
return w2c(P, camera, parameters, zc);
}

// mat c2c(mat const& p, std::string src, std::string dst,
//               std::map<std::string, CameraParameters> const& parameters, double z = 0.0) {
//   return w2c(c2w(p, src, parameters, z), dst, parameters);
// };

mat im2world(CameraParameters const& parameters, mat pt, double const& z) {
  if (pt.n_elem != 2)
    throw std::runtime_error("In function 'im2world': Invalid matrix size. Must be 2x1 or 1x2.");
  if(pt.n_cols == 2)
    pt = pt.t();

  pt = join_vert(pt, mat(1, 1, fill::ones));
  mat E = join_vert(join_horiz(parameters.R, parameters.T),
                    mat({0.0, 0.0, 0.0, 1.0}));
  mat B = parameters.K * eye(3, 4) * E;
  mat A = join_horiz(B.cols(0, 1), -pt);
  mat b = -B.col(3) - z * B.col(2);
  mat P = inv(A) * b;
  P(2,0) = z;
  return P;
}

} // camera::

#endif // __CAMERA_HPP__