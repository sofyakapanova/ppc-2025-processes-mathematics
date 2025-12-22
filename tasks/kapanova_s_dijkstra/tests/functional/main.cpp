#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "kapanova_s_dijkstra/common/include/common.hpp"
#include "kapanova_s_dijkstra/mpi/include/ops_mpi.hpp"
#include "kapanova_s_dijkstra/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace kapanova_s_dijkstra {

struct DijkstraTestCase {
  std::string name;
  GraphData graph;
  std::vector<double> expected_distances;
};

class DijkstraTestFactory {
 public:
  static std::map<int, DijkstraTestCase> CreateTestCases() {
    std::map<int, DijkstraTestCase> cases;

    {
      DijkstraTestCase tc;
      tc.name = "simple_graph_4_vertices";
      tc.graph.vertices_count = 4;
      tc.graph.start_node = 0;
      tc.graph.row_ptr = {0, 2, 4, 6, 6};
      tc.graph.col_idx = {1, 2, 2, 3, 1, 3};
      tc.graph.weights = {10.0, 5.0, 2.0, 1.0, 3.0, 9.0};
      tc.expected_distances = {0.0, 8.0, 5.0, 9.0};
      cases[1] = tc;
    }

    {
      DijkstraTestCase tc;
      tc.name = "no_paths_graph";
      tc.graph.vertices_count = 4;
      tc.graph.start_node = 0;
      tc.graph.row_ptr = {0, 0, 0, 0, 0};
      tc.graph.col_idx = {};
      tc.graph.weights = {};
      tc.expected_distances = {0.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::infinity()};
      cases[2] = tc;
    }

    {
      DijkstraTestCase tc;
      tc.name = "long_chain_graph";
      tc.graph.vertices_count = 7;
      tc.graph.start_node = 0;
      tc.graph.row_ptr = {0, 1, 2, 3, 4, 5, 6, 6};
      tc.graph.col_idx = {1, 2, 3, 4, 5, 6};
      tc.graph.weights = {5.0, 3.0, 2.0, 4.0, 6.0, 1.0};
      tc.expected_distances = {0.0, 5.0, 8.0, 10.0, 14.0, 20.0, 21.0};
      cases[3] = tc;
    }

    {
      DijkstraTestCase tc;
      tc.name = "linear_graph";
      tc.graph.vertices_count = 5;
      tc.graph.start_node = 0;
      tc.graph.row_ptr = {0, 1, 2, 3, 4, 4};
      tc.graph.col_idx = {1, 2, 3, 4};
      tc.graph.weights = {1.0, 2.0, 3.0, 4.0};
      tc.expected_distances = {0.0, 1.0, 3.0, 6.0, 10.0};
      cases[4] = tc;
    }

    {
      DijkstraTestCase tc;
      tc.name = "single_node_case";
      tc.graph.vertices_count = 1;
      tc.graph.start_node = 0;
      tc.graph.row_ptr = {0, 0};
      tc.graph.col_idx = {};
      tc.graph.weights = {};
      tc.expected_distances = {0.0};
      cases[5] = tc;
    }

    {
      DijkstraTestCase tc;
      tc.name = "bidirectional_weighted";
      tc.graph.vertices_count = 4;
      tc.graph.start_node = 0;
      tc.graph.row_ptr = {0, 2, 5, 7, 8};
      tc.graph.col_idx = {1, 2, 0, 2, 3, 1, 3, 2};
      tc.graph.weights = {4.0, 2.0, 4.0, 1.0, 5.0, 2.0, 3.0, 5.0};
      tc.expected_distances = {0.0, 4.0, 2.0, 5.0};
      cases[6] = tc;
    }

    {
      DijkstraTestCase tc;
      tc.name = "star_topology_graph";
      tc.graph.vertices_count = 6;
      tc.graph.start_node = 0;
      tc.graph.row_ptr = {0, 5, 5, 5, 5, 5, 5};
      tc.graph.col_idx = {1, 2, 3, 4, 5};
      tc.graph.weights = {7.0, 9.0, 14.0, 15.0, 10.0};
      tc.expected_distances = {0.0, 7.0, 9.0, 14.0, 15.0, 10.0};
      cases[7] = tc;
    }

    {
      DijkstraTestCase tc;
      tc.name = "graph_with_self_loops";
      tc.graph.vertices_count = 3;
      tc.graph.start_node = 0;
      tc.graph.row_ptr = {0, 3, 5, 6};
      tc.graph.col_idx = {0, 1, 2, 1, 2, 2};
      tc.graph.weights = {1.0, 4.0, 3.0, 2.0, 1.0, 5.0};
      tc.expected_distances = {0.0, 4.0, 3.0};
      cases[8] = tc;
    }

    {
      DijkstraTestCase tc;
      tc.name = "complete_graph_3_vertices";
      tc.graph.vertices_count = 3;
      tc.graph.start_node = 0;
      tc.graph.row_ptr = {0, 2, 4, 6};
      tc.graph.col_idx = {1, 2, 0, 2, 0, 1};
      tc.graph.weights = {1.0, 4.0, 1.0, 2.0, 4.0, 2.0};
      tc.expected_distances = {0.0, 1.0, 3.0};
      cases[9] = tc;
    }

    {
      DijkstraTestCase tc;
      tc.name = "bidirectional_graph";
      tc.graph.vertices_count = 3;
      tc.graph.start_node = 0;
      tc.graph.row_ptr = {0, 2, 5, 6};
      tc.graph.col_idx = {1, 2, 0, 2, 0, 1};
      tc.graph.weights = {5.0, 3.0, 5.0, 1.0, 3.0, 1.0};
      tc.expected_distances = {0.0, 4.0, 3.0};
      cases[10] = tc;
    }

    {
      DijkstraTestCase tc;
      tc.name = "disconnected_graph";
      tc.graph.vertices_count = 6;
      tc.graph.start_node = 0;
      tc.graph.row_ptr = {0, 1, 2, 2, 3, 4, 4};
      tc.graph.col_idx = {1, 0, 4, 3};
      tc.graph.weights = {2.0, 2.0, 3.0, 4.0};
      tc.expected_distances = {0.0,
                               2.0,
                               std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::infinity()};
      cases[11] = tc;
    }

    {
      DijkstraTestCase tc;
      tc.name = "zero_weight_edges";
      tc.graph.vertices_count = 4;
      tc.graph.start_node = 0;
      tc.graph.row_ptr = {0, 2, 4, 6, 6};
      tc.graph.col_idx = {1, 2, 0, 3, 1, 3};
      tc.graph.weights = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      tc.expected_distances = {0.0, 0.0, 0.0, 0.0};
      cases[12] = tc;
    }

    {
      DijkstraTestCase tc;
      tc.name = "uniform_weights_graph";
      tc.graph.vertices_count = 5;
      tc.graph.start_node = 0;
      tc.graph.row_ptr = {0, 2, 4, 6, 8, 8};
      tc.graph.col_idx = {1, 2, 0, 3, 0, 4, 1, 4};
      tc.graph.weights = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
      tc.expected_distances = {0.0, 1.0, 1.0, 2.0, 2.0};
      cases[13] = tc;
    }

    {
      DijkstraTestCase tc;
      tc.name = "graph_with_self_loops_2";
      tc.graph.vertices_count = 3;
      tc.graph.start_node = 0;
      tc.graph.row_ptr = {0, 2, 3, 3};
      tc.graph.col_idx = {1, 2, 2};
      tc.graph.weights = {4.0, 2.0, 1.0};
      tc.expected_distances = {0.0, 4.0, 2.0};
      cases[14] = tc;
    }

    return cases;
  }
};

class KapanovaSDijkstraFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    auto param = GetParam();
    int test_case = std::get<0>(std::get<2>(param));

    const auto &test_cases = DijkstraTestFactory::CreateTestCases();
    auto it = test_cases.find(test_case);

    if (it != test_cases.end()) {
      const auto &tc = it->second;
      input_data_ = tc.graph;
      expected_output_ = tc.expected_distances;
    } else {
      GraphData graph;
      graph.vertices_count = 4;
      graph.start_node = 0;
      graph.row_ptr = {0, 2, 4, 6, 6};
      graph.col_idx = {1, 2, 2, 3, 1, 3};
      graph.weights = {10.0, 5.0, 2.0, 1.0, 3.0, 9.0};
      input_data_ = graph;
      expected_output_ = {0.0, 8.0, 5.0, 9.0};
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_output_.size()) {
      return false;
    }

    const double tolerance = 1e-6;

    for (size_t idx = 0; idx < output_data.size(); ++idx) {
      const bool actual_infinite = std::isinf(output_data[idx]);
      const bool expected_infinite = std::isinf(expected_output_[idx]);

      if (actual_infinite && expected_infinite) {
        continue;
      }

      if (actual_infinite != expected_infinite) {
        return false;
      }

      if (std::fabs(expected_output_[idx] - output_data[idx]) > tolerance) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(KapanovaSDijkstraFuncTests, ValidateShortestPaths) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 14> kTestParameters = {
    std::make_tuple(1, "simple_graph_4_vertices"),   std::make_tuple(2, "no_paths_graph"),
    std::make_tuple(3, "long_chain_graph"),          std::make_tuple(4, "linear_graph"),
    std::make_tuple(5, "single_node_case"),          std::make_tuple(6, "bidirectional_weighted"),
    std::make_tuple(7, "star_topology_graph"),       std::make_tuple(8, "graph_with_self_loops"),
    std::make_tuple(9, "complete_graph_3_vertices"), std::make_tuple(10, "bidirectional_graph"),
    std::make_tuple(11, "disconnected_graph"),       std::make_tuple(12, "zero_weight_edges"),
    std::make_tuple(13, "uniform_weights_graph"),    std::make_tuple(14, "graph_with_self_loops_2"),
};

const auto kTestTasksCollection = std::tuple_cat(
    ppc::util::AddFuncTask<KapanovaSDijkstraMPI, InType>(kTestParameters, PPC_SETTINGS_kapanova_s_dijkstra),
    ppc::util::AddFuncTask<KapanovaSDijkstraSEQ, InType>(kTestParameters, PPC_SETTINGS_kapanova_s_dijkstra));

const auto kGTestParameterValues = ppc::util::ExpandToValues(kTestTasksCollection);

const auto kTestNameFormatter = KapanovaSDijkstraFuncTests::PrintFuncTestName<KapanovaSDijkstraFuncTests>;

INSTANTIATE_TEST_SUITE_P(DijkstraImplementationTests, KapanovaSDijkstraFuncTests, kGTestParameterValues,
                         kTestNameFormatter);

}  // namespace

}  // namespace kapanova_s_dijkstra
