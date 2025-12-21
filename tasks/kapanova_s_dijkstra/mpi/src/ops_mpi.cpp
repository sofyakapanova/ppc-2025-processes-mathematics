#include "kapanova_s_dijkstra/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

namespace kapanova_s_dijkstra {

KapanovaSDijkstraMPI::KapanovaSDijkstraMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>();
}

bool KapanovaSDijkstraMPI::ValidationImpl() {
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

bool KapanovaSDijkstraMPI::PreProcessingImpl() {
  const auto &graph = GetInput();
  GetOutput() = std::vector<int>(graph.num_vertices, INT_MAX);
  return true;
}

bool KapanovaSDijkstraMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &graph = GetInput();
  auto &distances = GetOutput();

  const size_t n = graph.num_vertices;
  const int start = graph.start_vertex;

  size_t vertices_per_proc = n / size;
  size_t remainder = n % size;

  size_t local_start = rank * vertices_per_proc + std::min<size_t>(rank, remainder);
  size_t local_end = local_start + vertices_per_proc + (static_cast<size_t>(rank) < remainder ? 1 : 0);
  std::vector<int> local_dist(n, INT_MAX);
  std::vector<bool> local_visited(n, false);

  if (start >= static_cast<int>(local_start) && start < static_cast<int>(local_end)) {
    local_dist[start] = 0;
  }

  MPI_Allreduce(MPI_IN_PLACE, local_dist.data(), n, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  bool changed = true;
  while (changed) {
    changed = false;

    for (size_t u = local_start; u < local_end; ++u) {
      if (local_dist[u] == INT_MAX) {
        continue;
      }

      for (size_t i = graph.row_ptrs[u]; i < graph.row_ptrs[u + 1]; ++i) {
        size_t v = graph.columns[i];
        int weight = graph.distances[i];

        int new_dist = local_dist[u] + weight;
        if (new_dist < local_dist[v]) {
          local_dist[v] = new_dist;
          changed = true;
        }
      }
    }

    int global_changed = changed ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &global_changed, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    changed = (global_changed != 0);

    MPI_Allreduce(MPI_IN_PLACE, local_dist.data(), n, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  }

  distances = local_dist;

  for (int &dist : distances) {
    if (dist == INT_MAX) {
      dist = -1;
    }
  }

  return true;
}

bool KapanovaSDijkstraMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kapanova_s_dijkstra
