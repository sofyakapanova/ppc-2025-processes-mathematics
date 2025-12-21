#include "kapanova_s_dijkstra/seq/include/ops_seq.hpp"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <queue>
#include <utility>
#include <vector>

namespace kapanova_s_dijkstra {

KapanovaSDijkstraSEQ::KapanovaSDijkstraSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>();
}

bool KapanovaSDijkstraSEQ::ValidationImpl() {
  const auto &graph = GetInput();

  if (graph.row_ptrs.empty()) {
    return false;
  }

  if (graph.row_ptrs.size() != graph.num_vertices + 1) {
    return false;
  }

  if (graph.row_ptrs.back() != graph.columns.size()) {
    return false;
  }

  if (graph.columns.size() != graph.distances.size()) {
    return false;
  }

  if (graph.start_vertex < 0 || static_cast<size_t>(graph.start_vertex) >= graph.num_vertices) {
    return false;
  }

  for (int weight : graph.distances) {
    if (weight < 0) {
      return false;
    }
  }

  return true;
}

bool KapanovaSDijkstraSEQ::PreProcessingImpl() {
  const auto &graph = GetInput();
  GetOutput() = std::vector<int>(graph.num_vertices, INT_MAX);
  return true;
}

bool KapanovaSDijkstraSEQ::RunImpl() {
  const auto &graph = GetInput();
  auto &distances = GetOutput();

  const size_t n = graph.num_vertices;
  const int start = graph.start_vertex;

  distances[start] = 0;
  std::vector<bool> visited(n, false);

  using Pair = std::pair<int, size_t>;
  std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> pq;
  pq.emplace(0, start);

  while (!pq.empty()) {
    const auto current_dist = pq.top().first;
    const auto u = pq.top().second;
    pq.pop();

    if (visited[u]) {
      continue;
    }
    visited[u] = true;

    for (size_t i = graph.row_ptrs[u]; i < graph.row_ptrs[u + 1]; ++i) {
      const size_t v = graph.columns[i];
      const int weight = graph.distances[i];

      // Проверка на переполнение
      if (current_dist > INT_MAX - weight) {
        continue;  // Пропускаем ребро, которое может вызвать переполнение
      }

      const int new_dist = current_dist + weight;
      if (new_dist < distances[v]) {
        distances[v] = new_dist;
        pq.emplace(new_dist, v);
      }
    }
  }

  for (int &dist : distances) {
    if (dist == INT_MAX) {
      dist = -1;
    }
  }

  return true;
}

bool KapanovaSDijkstraSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kapanova_s_dijkstra
