#ifndef __IS_MSG_ARMA_HPP__
#define __IS_MSG_ARMA_HPP__

#include <armadillo>
#include <is/packer.hpp>

namespace msgpack {

MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS) {
  namespace adaptor {

  template <>
  struct convert<arma::mat> {
    msgpack::object const& operator()(msgpack::object const& o, arma::mat& mat) const {
      if (o.type != msgpack::type::ARRAY)
        throw msgpack::type_error();

      if (o.via.array.size != 3)
        throw msgpack::type_error();

      auto rows = o.via.array.ptr[0].as<int>();
      auto cols = o.via.array.ptr[1].as<int>();
      auto data = o.via.array.ptr[2].as<std::vector<double>>();

      mat = arma::mat(&data[0], rows, cols, true);
      return o;
    }
  };

  template <>
  struct pack<arma::mat> {
    template <typename Stream>
    packer<Stream>& operator()(msgpack::packer<Stream>& o, arma::mat const& mat) const {
      o.pack_array(3);
      o.pack(mat.n_rows);
      o.pack(mat.n_cols);
      
      o.pack_array(mat.n_elem);
      const double* data = mat.memptr();
      for (int n = mat.n_elem; n != 0; --n) {
        o.pack(*data);
        ++data;
      }
      return o;
    }
  };

  }  // ::adaptor

}  // MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)

}  // ::msgpack

#endif  // __IS_MSG_ARMA_HPP__