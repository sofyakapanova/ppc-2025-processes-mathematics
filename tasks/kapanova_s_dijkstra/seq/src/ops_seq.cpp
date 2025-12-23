#include "kapanova_s_dijkstra/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <queue>
#include <ranges>
#include <utility>
#include <vector>

#include "kapanova_s_dijkstra/common/include/common.hpp"

namespace kapanova_s_dijkstra {

KapanovaSDijkstraSEQ::KapanovaSDijkstraSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>();
}

bool KapanovaSDijkstraSEQ::ValidationImpl() {
  const auto &input = GetInput();

  if (input.vertices_count <= 0) {
    return false;
  }

  if (input.start_node < 0 || input.start_node >= input.vertices_count) {
    return false;
  }

  if (input.row_ptr.size() != static_cast<std::size_t>(input.vertices_count) + 1) {
    return false;
  }

  // Проверка на отрицательные веса с использованием std::ranges::all_of
  return std::ranges::all_of(input.weights, [](double weight) { return weight >= 0.0; });
}

bool KapanovaSDijkstraSEQ::PreProcessingImpl() {
  return true;
}

bool KapanovaSDijkstraSEQ::RunImpl() {
  const auto &graph = GetInput();
  const int node_count = graph.vertices_count;
  const int source = graph.start_node;
  std::vector<double> distances(static_cast<std::size_t>(node_count), std::numeric_limits<double>::infinity());
  distances[static_cast<std::size_t>(source)] = 0.0;

  std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, std::greater<>> min_heap;
  min_heap.emplace(0.0, source);

  std::vector<bool> processed(static_cast<std::size_t>(node_count), false);

  while (!min_heap.empty()) {
    auto [current_dist, u] = min_heap.top();
    min_heap.pop();

    auto u_idx = static_cast<std::size_t>(u);
    if (processed[u_idx]) {
      continue;
    }
    processed[u_idx] = true;

    int begin = graph.row_ptr[u];
    int end = graph.row_ptr[u + 1];

    for (int idx = begin; idx < end; ++idx) {
      int v = graph.col_idx[idx];
      double w = graph.weights[idx];

      auto v_idx = static_cast<std::size_t>(v);
      if (!processed[v_idx]) {
        double new_dist = current_dist + w;
        if (new_dist < distances[v_idx]) {
          distances[v_idx] = new_dist;
          min_heap.emplace(new_dist, v);
        }
      }
    }
  }

  GetOutput() = distances;
  return true;
}

bool KapanovaSDijkstraSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kapanova_s_dijkstra
