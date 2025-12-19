#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace kapanova_s_image_smoothing {

using InType = std::vector<std::vector<uint8_t>>;
using OutType = std::vector<uint8_t>;
using TestType = std::tuple<std::vector<uint8_t>, int, int,
                            std::vector<uint8_t>>;  // Изображение, ширина, высота, ожидаемый результат
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kapanova_s_image_smoothing
