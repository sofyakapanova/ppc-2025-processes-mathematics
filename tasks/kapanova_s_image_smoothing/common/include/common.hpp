#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace kapanova_s_image_smoothing {

struct ImageData {
  std::vector<uint8_t> pixels;
  int width = 0;
  int height = 0;
  int kernel_size = 0;

  bool operator==(const ImageData &other) const {
    return pixels == other.pixels && width == other.width && height == other.height && kernel_size == other.kernel_size;
  }
};

using InType = ImageData;
using OutType = ImageData;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kapanova_s_image_smoothing
