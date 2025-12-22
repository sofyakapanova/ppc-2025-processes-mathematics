#include "kapanova_s_dijkstra/seq/include/ops_seq.hpp"

#include <cstddef>
#include <functional>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

#include "kapanova_s_dijkstra/common/include/common.hpp"

namespace kapanova_s_dijkstra {

KapanovaDijkstraCRSSEQ::KapanovaDijkstraCRSSEQ(const AlgorithmInput &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>();
}

bool KapanovaDijkstraCRSSEQ::ValidationImpl() {
  const auto &input = GetInput();
  if (input.total_nodes <= 0) {
    return false;
  }
  if (input.source_node < 0 || input.source_node >= input.total_nodes) {
    return false;
  }
  if (input.row_pointers.size() != static_cast<std::size_t>(input.total_nodes) + 1) {
    return false;
  }
  return true;
}

bool KapanovaDijkstraCRSSEQ::PreProcessingImpl() {
  return true;
}

bool KapanovaDijkstraCRSSEQ::RunImpl() {
  const auto &graph = GetInput();
  const int n = graph.total_nodes;
  const int source = graph.source_node;
  std::vector<double> dist(n, std::numeric_limits<double>::infinity());
  dist[source] = 0.0;

  std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, std::greater<>> pq;
  pq.emplace(0.0, source);

  std::vector<bool> visited(n, false);

  while (!pq.empty()) {
    auto [current_dist, u] = pq.top();
    pq.pop();

    if (visited[u]) {
      continue;
    }
    visited[u] = true;

    int start = graph.row_pointers[u];
    int end = graph.row_pointers[u + 1];

    for (int idx = start; idx < end; ++idx) {
      int v = graph.column_indices[idx];
      double weight = graph.weight_values[idx];

      if (!visited[v]) {
        double new_dist = current_dist + weight;
        if (new_dist < dist[v]) {
          dist[v] = new_dist;
          pq.emplace(new_dist, v);
        }
      }
    }
  }

  GetOutput() = dist;
  return true;
}

bool KapanovaDijkstraCRSSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kapanova_s_dijkstra