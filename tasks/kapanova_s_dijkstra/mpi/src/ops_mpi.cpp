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

  local_first = (rank * base_nodes) + std::min(rank, extra_nodes);
  local_last = local_first + base_nodes + (rank < extra_nodes ? 1 : 0);
  local_nodes = local_last - local_first;
}

void LoadLocalEdges(const GraphData &graph, int local_first, int local_last, int total_nodes,
                    std::vector<int> &local_cols, std::vector<double> &local_vals, std::vector<int> &local_rows) {
  local_last = std::min(local_last, total_nodes);

  int first_edge = graph.row_ptr[local_first];
  int last_edge = graph.row_ptr[local_last];
  int edge_count = last_edge - first_edge;

  local_rows.resize(static_cast<std::size_t>(local_last - local_first) + 1);

  if (edge_count > 0) {
    local_cols.resize(static_cast<std::size_t>(edge_count));
    local_vals.resize(static_cast<std::size_t>(edge_count));

    for (std::size_t i = 0; i < local_rows.size(); ++i) {
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
    for (int &local_row : local_rows) {
      local_row = 0;
    }
  }
}

bool ProcessVertex(double dist, const std::vector<int> &local_rows, const std::vector<int> &local_cols,
                   const std::vector<double> &local_vals, int local_idx, int total_nodes,
                   std::vector<double> &next_dist) {
  if (local_idx >= static_cast<int>(local_rows.size()) - 1) {
    return false;
  }

  bool changed = false;
  int start = local_rows[local_idx];
  int end = local_rows[local_idx + 1];

  for (int edge_idx = start; edge_idx < end; ++edge_idx) {
    if (static_cast<std::size_t>(edge_idx) >= local_cols.size()) {
      continue;
    }

    int neighbor = local_cols[edge_idx];
    if (neighbor < 0 || neighbor >= total_nodes) {
      continue;
    }

    double new_dist = dist + local_vals[edge_idx];
    if (new_dist < next_dist[static_cast<std::size_t>(neighbor)]) {
      next_dist[static_cast<std::size_t>(neighbor)] = new_dist;
      changed = true;
    }
  }

  return changed;
}

bool RunOptimizedDijkstraMPI(const std::vector<int> &local_rows, const std::vector<int> &local_cols,
                             const std::vector<double> &local_vals, int local_first, int local_nodes, int total_nodes,
                             std::vector<double> &global_dist) {
  std::vector<double> current_dist = global_dist;
  std::vector<double> next_dist = global_dist;

  bool any_changed = true;
  int iterations = 0;

  while (any_changed && iterations < total_nodes) {
    ++iterations;
    any_changed = false;

    for (int local_idx = 0; local_idx < local_nodes; ++local_idx) {
      int vertex = local_first + local_idx;
      if (vertex >= total_nodes) {
        continue;
      }

      double dist_vertex = current_dist[static_cast<std::size_t>(vertex)];
      if (dist_vertex == std::numeric_limits<double>::infinity()) {
        continue;
      }

      if (ProcessVertex(dist_vertex, local_rows, local_cols, local_vals, local_idx, total_nodes, next_dist)) {
        any_changed = true;
      }
    }

    if (static_cast<std::size_t>(total_nodes) != next_dist.size()) {
      next_dist.resize(static_cast<std::size_t>(total_nodes), std::numeric_limits<double>::infinity());
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

  global_dist.resize(static_cast<std::size_t>(total_nodes), std::numeric_limits<double>::infinity());

  int source_owner = source % size;

  if (rank == source_owner) {
    if (source >= 0 && source < total_nodes) {
      global_dist[static_cast<std::size_t>(source)] = 0.0;
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

  local_last = std::min(local_last, total_nodes);
  local_vertices_ = local_last - local_first;

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
    std::vector<double> result(static_cast<std::size_t>(total_nodes), std::numeric_limits<double>::infinity());

    if (proc_rank_ == 0) {
      result[static_cast<std::size_t>(graph.start_node)] = 0.0;

      std::vector<bool> visited(static_cast<std::size_t>(total_nodes), false);

      for (int i = 0; i < total_nodes; ++i) {
        int current_vertex = -1;
        double min_dist = std::numeric_limits<double>::infinity();

        for (int vertex_idx = 0; vertex_idx < total_nodes; ++vertex_idx) {
          if (!visited[static_cast<std::size_t>(vertex_idx)] &&
              result[static_cast<std::size_t>(vertex_idx)] < min_dist) {
            min_dist = result[static_cast<std::size_t>(vertex_idx)];
            current_vertex = vertex_idx;
          }
        }

        if (current_vertex == -1 || min_dist == std::numeric_limits<double>::infinity()) {
          break;
        }

        visited[static_cast<std::size_t>(current_vertex)] = true;
        int start = graph.row_ptr[current_vertex];
        int end = graph.row_ptr[current_vertex + 1];

        for (int j = start; j < end; ++j) {
          int neighbor = graph.col_idx[j];
          if (neighbor >= 0 && neighbor < total_nodes) {
            double new_dist = result[static_cast<std::size_t>(current_vertex)] + graph.weights[j];
            result[static_cast<std::size_t>(neighbor)] = std::min(new_dist, result[static_cast<std::size_t>(neighbor)]);
          }
        }
      }
    }

    MPI_Bcast(result.data(), total_nodes, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    GetOutput() = result;
    return true;
  }

  PartitionGraph();

  local_vertices_ = std::max(local_vertices_, 0);

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
