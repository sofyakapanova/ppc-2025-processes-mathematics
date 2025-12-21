#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

#include "kapanova_s_dijkstra/common/include/common.hpp"
#include "kapanova_s_dijkstra/seq/include/ops_seq.hpp"

using kapanova_s_dijkstra::GraphCSR;

namespace {

GraphCSR GenerateRandomGraph(size_t num_vertices, size_t avg_degree) {
  GraphCSR graph;
  graph.num_vertices = num_vertices;
  graph.start_vertex = 0;

  std::vector<std::vector<size_t>> adj_lists(num_vertices);
  std::vector<std::vector<int>> weight_lists(num_vertices);

  std::mt19937 gen(42);
  std::uniform_int_distribution<size_t> target_dist(0, num_vertices - 1);
  std::uniform_int_distribution<int> weight_dist(1, 100);

  for (size_t i = 0; i < num_vertices; ++i) {
    size_t degree = avg_degree + (gen() % 3) - 1;
    degree = std::min(degree, num_vertices - 1);

    for (size_t j = 0; j < degree; ++j) {
      size_t target = target_dist(gen);
      while (target == i || std::find(adj_lists[i].begin(), adj_lists[i].end(), target) != adj_lists[i].end()) {
        target = target_dist(gen);
      }
      adj_lists[i].push_back(target);
      weight_lists[i].push_back(weight_dist(gen));
    }
  }

  graph.row_ptrs.resize(num_vertices + 1, 0);
  for (size_t i = 0; i < num_vertices; ++i) {
    graph.row_ptrs[i + 1] = graph.row_ptrs[i] + adj_lists[i].size();
  }

  const size_t total_edges = graph.row_ptrs.back();
  graph.columns.resize(total_edges);
  graph.distances.resize(total_edges);

  size_t idx = 0;
  for (size_t i = 0; i < num_vertices; ++i) {
    for (size_t j = 0; j < adj_lists[i].size(); ++j) {
      graph.columns[idx] = adj_lists[i][j];
      graph.distances[idx] = weight_lists[i][j];
      ++idx;
    }
  }

  return graph;
}

GraphCSR GenerateGridGraph(size_t side) {
  GraphCSR graph;
  const size_t num_vertices = side * side;
  graph.num_vertices = num_vertices;
  graph.start_vertex = 0;

  std::vector<std::vector<size_t>> adj_lists(num_vertices);
  std::vector<std::vector<int>> weight_lists(num_vertices);

  for (size_t i = 0; i < side; ++i) {
    for (size_t j = 0; j < side; ++j) {
      const size_t idx = i * side + j;

      if (j + 1 < side) {
        adj_lists[idx].push_back(idx + 1);
        weight_lists[idx].push_back(1);
      }
      if (i + 1 < side) {
        adj_lists[idx].push_back(idx + side);
        weight_lists[idx].push_back(1);
      }
      if (j > 0) {
        adj_lists[idx].push_back(idx - 1);
        weight_lists[idx].push_back(1);
      }
      if (i > 0) {
        adj_lists[idx].push_back(idx - side);
        weight_lists[idx].push_back(1);
      }
    }
  }

  graph.row_ptrs.resize(num_vertices + 1, 0);
  for (size_t i = 0; i < num_vertices; ++i) {
    graph.row_ptrs[i + 1] = graph.row_ptrs[i] + adj_lists[i].size();
  }

  const size_t total_edges = graph.row_ptrs.back();
  graph.columns.resize(total_edges);
  graph.distances.resize(total_edges);

  size_t edge_idx = 0;
  for (size_t i = 0; i < num_vertices; ++i) {
    for (size_t j = 0; j < adj_lists[i].size(); ++j) {
      graph.columns[edge_idx] = adj_lists[i][j];
      graph.distances[edge_idx] = weight_lists[i][j];
      ++edge_idx;
    }
  }

  return graph;
}

TEST(KapanovaSDijkstraPerf, RandomGraphLarge) {
  const size_t num_vertices = 10000;
  const size_t avg_degree = 10;

  auto graph = GenerateRandomGraph(num_vertices, avg_degree);

  kapanova_s_dijkstra::KapanovaSDijkstraSEQ task(graph);

  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());

  EXPECT_TRUE(task.Run());

  EXPECT_TRUE(task.PostProcessing());

  const auto &result = task.GetOutput();
  EXPECT_EQ(num_vertices, result.size());
}

TEST(KapanovaSDijkstraPerf, GridGraphLarge) {
  const size_t side = 100;
  const size_t num_vertices = side * side;

  auto graph = GenerateGridGraph(side);

  kapanova_s_dijkstra::KapanovaSDijkstraSEQ task(graph);

  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());

  EXPECT_TRUE(task.Run());

  EXPECT_TRUE(task.PostProcessing());

  const auto &result = task.GetOutput();
  EXPECT_EQ(num_vertices, result.size());
}

TEST(KapanovaSDijkstraPerf, SparseGraph) {
  const size_t num_vertices = 50000;
  const size_t avg_degree = 4;

  auto graph = GenerateRandomGraph(num_vertices, avg_degree);

  kapanova_s_dijkstra::KapanovaSDijkstraSEQ task(graph);

  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());

  EXPECT_TRUE(task.Run());

  EXPECT_TRUE(task.PostProcessing());
}

}  // namespace
