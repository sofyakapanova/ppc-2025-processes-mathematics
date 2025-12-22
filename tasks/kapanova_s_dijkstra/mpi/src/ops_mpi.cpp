#include "kapanova_s_dijkstra/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "kapanova_s_dijkstra/common/include/common.hpp"

namespace kapanova_s_dijkstra {

namespace {
void ComputeNodeDistribution(int rank, int size, int total_nodes, int &local_first, int &local_last, int &local_nodes) {
  int base_nodes = total_nodes / size;
  int extra_nodes = total_nodes % size;

  local_first = rank * base_nodes + std::min(rank, extra_nodes);
  local_last = local_first + base_nodes + (rank < extra_nodes ? 1 : 0);
  local_nodes = local_last - local_first;
}

void LoadLocalEdges(const GraphData &graph, int local_first, int local_last, int total_nodes,
                    std::vector<int> &local_cols, std::vector<double> &local_vals, std::vector<int> &local_rows) {
  if (local_last > total_nodes) {
    local_last = total_nodes;
  }

  int first_edge = graph.row_ptr[local_first];
  int last_edge = graph.row_ptr[local_last];
  int edge_count = last_edge - first_edge;

  local_rows.resize(local_last - local_first + 1);

  if (edge_count > 0) {
    local_cols.resize(edge_count);
    local_vals.resize(edge_count);

    for (size_t i = 0; i < local_rows.size(); ++i) {
      int global_idx = local_first + static_cast<int>(i);
      if (global_idx <= total_nodes) {
        local_rows[i] = graph.row_ptr[global_idx] - first_edge;
      } else {
        local_rows[i] = edge_count;
      }
    }

    std::copy(graph.col_idx.begin() + first_edge, graph.col_idx.begin() + last_edge, local_cols.begin());
    std::copy(graph.weights.begin() + first_edge, graph.weights.begin() + last_edge, local_vals.begin());
  } else {
    local_cols.clear();
    local_vals.clear();
    for (size_t i = 0; i < local_rows.size(); ++i) {
      local_rows[i] = 0;
    }
  }
}

// Убрали неиспользуемые параметры
bool RunOptimizedDijkstraMPI(const std::vector<int> &local_rows, const std::vector<int> &local_cols,
                             const std::vector<double> &local_vals, int local_first, int local_nodes, int total_nodes,
                             std::vector<double> &global_dist) {
  std::vector<double> current_dist = global_dist;
  std::vector<double> next_dist = global_dist;

  bool any_changed = true;
  int iterations = 0;

  while (any_changed && iterations < total_nodes) {
    iterations++;
    any_changed = false;

    for (int local_idx = 0; local_idx < local_nodes; ++local_idx) {
      int u = local_first + local_idx;
      if (u >= total_nodes) {
        continue;
      }

      double dist_u = current_dist[u];
      if (dist_u == std::numeric_limits<double>::infinity()) {
        continue;
      }

      if (local_idx >= static_cast<int>(local_rows.size()) - 1) {
        continue;
      }

      int start = local_rows[local_idx];
      int end = local_rows[local_idx + 1];

      for (int edge_idx = start; edge_idx < end; ++edge_idx) {
        if (static_cast<size_t>(edge_idx) >= local_cols.size()) {
          continue;
        }

        int v = local_cols[edge_idx];
        if (v < 0 || v >= total_nodes) {
          continue;
        }

        double new_dist = dist_u + local_vals[edge_idx];
        if (new_dist < next_dist[v]) {
          next_dist[v] = new_dist;
          any_changed = true;
        }
      }
    }

    // КРИТИЧЕСКИЙ МОМЕНТ: Проверяем размеры перед Allreduce
    if (static_cast<int>(next_dist.size()) != total_nodes) {
      next_dist.resize(total_nodes, std::numeric_limits<double>::infinity());
    }

    MPI_Allreduce(MPI_IN_PLACE, next_dist.data(), total_nodes, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    std::swap(current_dist, next_dist);

    int global_changed = any_changed ? 1 : 0;
    int all_changed = 0;
    MPI_Allreduce(&global_changed, &all_changed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    any_changed = (all_changed > 0);
  }

  global_dist = current_dist;
  return true;
}

void FastInitializeDistances(const GraphData &graph, int rank, int size, std::vector<double> &global_dist,
                             int total_nodes) {
  const int source = graph.start_node;

  global_dist.resize(total_nodes, std::numeric_limits<double>::infinity());

  int source_owner = source % size;

  if (rank == source_owner) {
    if (source >= 0 && source < total_nodes) {
      global_dist[source] = 0.0;
    }
  }

  MPI_Bcast(global_dist.data(), total_nodes, MPI_DOUBLE, source_owner, MPI_COMM_WORLD);
}

}  // namespace

KapanovaSDijkstraMPI::KapanovaSDijkstraMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>();
}

bool KapanovaSDijkstraMPI::ValidationImpl() {
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
  return true;
}

bool KapanovaSDijkstraMPI::PreProcessingImpl() {
  MPI_Comm_size(MPI_COMM_WORLD, &proc_count_);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank_);
  return true;
}

void KapanovaSDijkstraMPI::PartitionGraph() {
  const auto &graph = GetInput();
  const int total_nodes = graph.vertices_count;

  int local_first = 0;
  int local_last = 0;
  ComputeNodeDistribution(proc_rank_, proc_count_, total_nodes, local_first, local_last, local_vertices_);

  if (local_last > total_nodes) {
    local_last = total_nodes;
    local_vertices_ = local_last - local_first;
  }

  LoadLocalEdges(graph, local_first, local_last, total_nodes, local_col_idx_, local_weights_, local_row_ptr_);
}

bool KapanovaSDijkstraMPI::RunImpl() {
  const auto &graph = GetInput();
  int total_nodes = graph.vertices_count;

  if (total_nodes <= 0) {
    GetOutput() = std::vector<double>();
    return true;
  }

  MPI_Bcast(&total_nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_nodes < 1000 && proc_count_ > 2) {
    std::vector<double> result(total_nodes, std::numeric_limits<double>::infinity());

    if (proc_rank_ == 0) {
      result[graph.start_node] = 0.0;

      std::vector<bool> visited(total_nodes, false);

      for (int i = 0; i < total_nodes; ++i) {
        int u = -1;
        double min_dist = std::numeric_limits<double>::infinity();

        for (int v = 0; v < total_nodes; ++v) {
          if (!visited[v] && result[v] < min_dist) {
            min_dist = result[v];
            u = v;
          }
        }

        if (u == -1 || min_dist == std::numeric_limits<double>::infinity()) {
          break;
        }

        visited[u] = true;
        int start = graph.row_ptr[u];
        int end = graph.row_ptr[u + 1];

        for (int j = start; j < end; ++j) {
          int v = graph.col_idx[j];
          if (v >= 0 && v < total_nodes) {
            double new_dist = result[u] + graph.weights[j];
            if (new_dist < result[v]) {
              result[v] = new_dist;
            }
          }
        }
      }
    }

    MPI_Bcast(result.data(), total_nodes, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    GetOutput() = result;
    return true;
  }

  PartitionGraph();

  if (local_vertices_ < 0) {
    local_vertices_ = 0;
  }

  std::vector<double> global_dist;
  FastInitializeDistances(graph, proc_rank_, proc_count_, global_dist, total_nodes);

  int local_first = 0;
  int local_last = 0;
  int local_nodes = 0;
  ComputeNodeDistribution(proc_rank_, proc_count_, total_nodes, local_first, local_last, local_nodes);

  if (local_nodes != local_vertices_) {
    local_vertices_ = local_nodes;
  }

  RunOptimizedDijkstraMPI(local_row_ptr_, local_col_idx_, local_weights_, local_first, local_vertices_, total_nodes,
                          global_dist);

  GetOutput() = global_dist;

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool KapanovaSDijkstraMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kapanova_s_dijkstra
