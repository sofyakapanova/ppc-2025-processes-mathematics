#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace kapanova_s_dijkstra {

struct GraphData {
  std::vector<int> row_ptr;
  std::vector<int> col_idx;
  std::vector<double> weights;
  int vertices_count = 0;
  int start_node = 0;
};

using InType = GraphData;
using OutType = std::vector<double>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kapanova_s_dijkstra