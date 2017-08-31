#include <iostream>
#include <boost/program_options.hpp>
#include "../../drivers/ptgrey/ptgrey.hpp"
#include "../camera/camera.hpp"

namespace po = boost::program_options;

int main(int argc, char* argv[]) {
  std::string uri;
  std::string entity;
  std::string folder;
  std::string format;

  po::options_description description("Allowed options");
  auto&& options = description.add_options();
  options("help,", "show available options");
  options("uri,u", po::value<std::string>(&uri)->default_value("amqp://localhost"), "broker uri");
  options("entity,e", po::value<std::string>(&entity), "entity name");
  options("images,i", po::value<std::string>(&folder), "images folder");
  options("format,f", po::value<std::string>(&format)->default_value("jpeg"), "image format [png/jpeg]");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, description), vm);
  po::notify(vm);

  if (vm.count("help") || !vm.count("entity") || !vm.count("images")) {
    std::cout << description << std::endl;
    return 1;
  }

  is::driver::PtGrey camera(entity, folder);
  is::gw::Camera<is::driver::PtGrey> gw(entity, uri, format, camera);
  return 0;
}