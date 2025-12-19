#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace kapanova_s_image_smoothing {

using InType = std::vector<std::vector<uint8_t>>;
using OutType = std::vector<uint8_t>;
using TestType = std::tuple<std::vector<uint8_t>, int, int, std::vector<uint8_t>>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kapanova_s_image_smoothing
