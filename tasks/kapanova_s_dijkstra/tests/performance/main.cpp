#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "kapanova_s_dijkstra/common/include/common.hpp"
#include "kapanova_s_dijkstra/mpi/include/ops_mpi.hpp"
#include "kapanova_s_dijkstra/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kapanova_s_dijkstra {

class KapanovaDijkstraPerfTests : public ppc::util::BaseRunPerfTests<AlgorithmInput, AlgorithmOutput> {
 protected:
  void SetUp() override {
    int num_vertices = 50000;
    int edges_per_vertex = 100;

    GraphRepresentation graph;
    graph.total_nodes = num_vertices;
    graph.source_node = 0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> vertex_dis(0, num_vertices - 1);
    std::uniform_real_distribution<> weight_dis(0.1, 10.0);

    graph.row_pointers.resize(num_vertices + 1);
    graph.row_pointers[0] = 0;

    for (int i = 0; i < num_vertices; ++i) {
      int num_edges = edges_per_vertex;
      graph.row_pointers[i + 1] = graph.row_pointers[i] + num_edges;

      for (int j = 0; j < num_edges; ++j) {
        int target = vertex_dis(gen);
        double weight = weight_dis(gen);

        graph.column_indices.push_back(target);
        graph.weight_values.push_back(weight);
      }
    }

    input_data_ = graph;
  }

  bool CheckTestOutputData(AlgorithmOutput &output_data) final {
    return !output_data.empty();
  }

  AlgorithmInput GetTestInputData() final {
    return input_data_;
  }

 private:
  AlgorithmInput input_data_;
};

TEST_P(KapanovaDijkstraPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<AlgorithmInput, 
                                                       KapanovaDijkstraCRSMPI, 
                                                       KapanovaDijkstraCRSSEQ>(
    PPC_SETTINGS_kapanova_s_dijkstra);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KapanovaDijkstraPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KapanovaDijkstraPerfTests, kGtestValues, kPerfTestName);

}  // namespace kapanova_s_dijkstra