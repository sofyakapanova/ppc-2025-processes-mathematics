#include <gtest/gtest.h>
#include <mpi.h>

#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

#include "kapanova_s_dijkstra/common/include/common.hpp"
#include "kapanova_s_dijkstra/mpi/include/ops_mpi.hpp"
#include "kapanova_s_dijkstra/seq/include/ops_seq.hpp"

namespace kapanova_s_dijkstra {

namespace {

GraphData GenerateRandomGraph(int vertices, double density, int seed = 42) {
  GraphData graph;
  graph.vertices_count = vertices;
  graph.start_node = 0;

  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> weight_dist(1.0, 100.0);
  std::uniform_real_distribution<double> prob_dist(0.0, 1.0);

  graph.row_ptr.resize(vertices + 1, 0);

  int edge_count = 0;
  for (int i = 0; i < vertices; ++i) {
    for (int j = 0; j < vertices; ++j) {
      if (i != j && prob_dist(gen) < density) {
        graph.col_idx.push_back(j);
        graph.weights.push_back(weight_dist(gen));
        edge_count++;
      }
    }
    graph.row_ptr[i + 1] = edge_count;
  }

  return graph;
}

GraphData CreateChainGraph(int vertices) {
  GraphData graph;
  graph.vertices_count = vertices;
  graph.start_node = 0;

  graph.row_ptr.resize(vertices + 1, 0);

  int edge_count = 0;
  for (int i = 0; i < vertices - 1; ++i) {
    graph.col_idx.push_back(i + 1);
    graph.weights.push_back(1.0 + (i % 3));
    edge_count++;
    graph.row_ptr[i + 1] = edge_count;
  }
  graph.row_ptr[vertices] = edge_count;

  return graph;
}

GraphData CreateGridGraph(int size) {
  GraphData graph;
  graph.vertices_count = size * size;
  graph.start_node = 0;

  graph.row_ptr.resize(size * size + 1, 0);

  int edge_count = 0;
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      int current = i * size + j;

      if (j < size - 1) {
        graph.col_idx.push_back(current + 1);
        graph.weights.push_back(1.0);
        edge_count++;
      }

      if (i < size - 1) {
        graph.col_idx.push_back(current + size);
        graph.weights.push_back(1.0);
        edge_count++;
      }

      graph.row_ptr[current + 1] = edge_count;
    }
  }

  return graph;
}

}  // namespace

class KapanovaSDijkstraPerfTests : public ::testing::Test {
 protected:
  void RunSeqPerformanceTest(const GraphData &graph, const std::string &test_name) {
    auto start = std::chrono::high_resolution_clock::now();
    KapanovaSDijkstraSEQ seq_task(graph);

    EXPECT_TRUE(seq_task.Validation());
    EXPECT_TRUE(seq_task.PreProcessing());
    EXPECT_TRUE(seq_task.Run());
    EXPECT_TRUE(seq_task.PostProcessing());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    auto result = seq_task.GetOutput();

    EXPECT_EQ(result.size(), static_cast<size_t>(graph.vertices_count));

    if (graph.start_node >= 0 && graph.start_node < graph.vertices_count) {
      EXPECT_NEAR(result[graph.start_node], 0.0, 1e-6);
    }

    for (const auto &dist : result) {
      EXPECT_GE(dist, -1e-6);
    }

    std::cout << "\n=== " << test_name << " ===" << std::endl;
    std::cout << "Vertices: " << graph.vertices_count << std::endl;
    std::cout << "Edges: " << graph.col_idx.size() << std::endl;
    std::cout << "SEQ time: " << duration.count() << " ms" << std::endl;
  }

  void RunMPIPerformanceTest(const GraphData &graph, const std::string &test_name) {
    auto start = std::chrono::high_resolution_clock::now();
    KapanovaSDijkstraMPI mpi_task(graph);

    EXPECT_TRUE(mpi_task.Validation());
    EXPECT_TRUE(mpi_task.PreProcessing());
    EXPECT_TRUE(mpi_task.Run());
    EXPECT_TRUE(mpi_task.PostProcessing());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    auto result = mpi_task.GetOutput();

    EXPECT_EQ(result.size(), static_cast<size_t>(graph.vertices_count));

    if (graph.start_node >= 0 && graph.start_node < graph.vertices_count) {
      EXPECT_NEAR(result[graph.start_node], 0.0, 1e-6);
    }

    for (const auto &dist : result) {
      EXPECT_GE(dist, -1e-6);
    }

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
      std::cout << "\n=== " << test_name << " (MPI) ===" << std::endl;
      std::cout << "Vertices: " << graph.vertices_count << std::endl;
      std::cout << "Edges: " << graph.col_idx.size() << std::endl;
      std::cout << "MPI time: " << duration.count() << " ms" << std::endl;
    }
  }
};

TEST_F(KapanovaSDijkstraPerfTests, SmallRandomGraphSEQ) {
  GraphData graph = GenerateRandomGraph(100, 0.1);
  RunSeqPerformanceTest(graph, "Small Random Graph SEQ (100 vertices)");
}

TEST_F(KapanovaSDijkstraPerfTests, MediumRandomGraphSEQ) {
  GraphData graph = GenerateRandomGraph(10000, 0.05);
  RunSeqPerformanceTest(graph, "Medium Random Graph SEQ (500 vertices)");
}

TEST_F(KapanovaSDijkstraPerfTests, ChainGraphSEQ) {
  GraphData graph = CreateChainGraph(1000);
  RunSeqPerformanceTest(graph, "Chain Graph SEQ (1000 vertices)");
}

TEST_F(KapanovaSDijkstraPerfTests, GridGraphSEQ) {
  GraphData graph = CreateGridGraph(20);
  RunSeqPerformanceTest(graph, "Grid Graph SEQ (20x20)");
}

TEST_F(KapanovaSDijkstraPerfTests, SmallRandomGraphMPI) {
  GraphData graph = GenerateRandomGraph(100, 0.1);
  RunMPIPerformanceTest(graph, "Small Random Graph MPI (100 vertices)");
}

TEST_F(KapanovaSDijkstraPerfTests, MediumRandomGraphMPI) {
  GraphData graph = GenerateRandomGraph(10000, 0.05);
  RunMPIPerformanceTest(graph, "Medium Random Graph MPI (500 vertices)");
}

TEST_F(KapanovaSDijkstraPerfTests, ChainGraphMPI) {
  GraphData graph = CreateChainGraph(1000);
  RunMPIPerformanceTest(graph, "Chain Graph MPI (1000 vertices)");
}

TEST_F(KapanovaSDijkstraPerfTests, GridGraphMPI) {
  GraphData graph = CreateGridGraph(20);
  RunMPIPerformanceTest(graph, "Grid Graph MPI (20x20)");
}

}  // namespace kapanova_s_dijkstra
