#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace kapanova_s_dijkstra {

struct GraphCSR {
  std::vector<int> distances;
  std::vector<size_t> columns;
  std::vector<size_t> row_ptrs;
  size_t num_vertices;
  int start_vertex;
};

using InType = GraphCSR;
using OutType = std::vector<int>;
using TestType = std::tuple<InType, OutType>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kapanova_s_dijkstra
