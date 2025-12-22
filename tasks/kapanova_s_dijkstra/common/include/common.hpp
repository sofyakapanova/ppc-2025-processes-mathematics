#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace kapanova_s_dijkstra {

struct GraphRepresentation {
  std::vector<int> row_pointers;
  std::vector<int> column_indices;
  std::vector<double> weight_values;
  int total_nodes = 0;
  int source_node = 0;
};

using AlgorithmInput = GraphRepresentation;
using AlgorithmOutput = std::vector<double>;
using TestDescription = std::tuple<int, std::string>;
using BaseSolver = ppc::task::Task<AlgorithmInput, AlgorithmOutput>;

}  // namespace kapanova_s_dijkstra
