#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "kapanova_s_dijkstra/common/include/common.hpp"
#include "kapanova_s_dijkstra/mpi/include/ops_mpi.hpp"
#include "kapanova_s_dijkstra/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace kapanova_s_dijkstra {

class KapanovaSDijkstraFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    auto param = GetParam();
    int test_case = std::get<0>(std::get<2>(param));

    switch (test_case) {
      case 1: {
        GraphData graph;
        graph.vertices_count = 4;
        graph.start_node = 0;
        graph.row_ptr = {0, 2, 4, 6, 6};
        graph.col_idx = {1, 2, 2, 3, 1, 3};
        graph.weights = {10.0, 5.0, 2.0, 1.0, 3.0, 9.0};
        input_data_ = graph;
        expected_output_ = {0.0, 8.0, 5.0, 9.0};
        break;
      }
      case 2: {
        GraphData graph;
        graph.vertices_count = 4;
        graph.start_node = 0;
        graph.row_ptr = {0, 0, 0, 0, 0};
        graph.col_idx = {};
        graph.weights = {};
        input_data_ = graph;
        expected_output_ = {0.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::infinity()};
        break;
      }
      case 3: {
        GraphData graph;
        graph.vertices_count = 7;
        graph.start_node = 0;
        graph.row_ptr = {0, 1, 2, 3, 4, 5, 6, 6};
        graph.col_idx = {1, 2, 3, 4, 5, 6};
        graph.weights = {5.0, 3.0, 2.0, 4.0, 6.0, 1.0};
        input_data_ = graph;
        expected_output_ = {0.0, 5.0, 8.0, 10.0, 14.0, 20.0, 21.0};
        break;
      }
      case 4: {
        GraphData graph;
        graph.vertices_count = 5;
        graph.start_node = 0;
        graph.row_ptr = {0, 1, 2, 3, 4, 4};
        graph.col_idx = {1, 2, 3, 4};
        graph.weights = {1.0, 2.0, 3.0, 4.0};
        input_data_ = graph;
        expected_output_ = {0.0, 1.0, 3.0, 6.0, 10.0};
        break;
      }
      case 5: {
        GraphData graph;
        graph.vertices_count = 1;
        graph.start_node = 0;
        graph.row_ptr = {0, 0};
        graph.col_idx = {};
        graph.weights = {};
        input_data_ = graph;
        expected_output_ = {0.0};
        break;
      }
      case 6: {
        GraphData graph;
        graph.vertices_count = 4;
        graph.start_node = 0;
        graph.row_ptr = {0, 2, 5, 7, 8};
        graph.col_idx = {1, 2, 0, 2, 3, 1, 3, 2};
        graph.weights = {4.0, 2.0, 4.0, 1.0, 5.0, 2.0, 3.0, 5.0};
        input_data_ = graph;
        expected_output_ = {0.0, 4.0, 2.0, 5.0};
        break;
      }
      case 7: {
        GraphData graph;
        graph.vertices_count = 6;
        graph.start_node = 0;
        graph.row_ptr = {0, 5, 5, 5, 5, 5, 5};
        graph.col_idx = {1, 2, 3, 4, 5};
        graph.weights = {7.0, 9.0, 14.0, 15.0, 10.0};
        input_data_ = graph;
        expected_output_ = {0.0, 7.0, 9.0, 14.0, 15.0, 10.0};
        break;
      }
      case 8: {
        GraphData graph;
        graph.vertices_count = 3;
        graph.start_node = 0;
        graph.row_ptr = {0, 3, 5, 6};
        graph.col_idx = {0, 1, 2, 1, 2, 2};
        graph.weights = {1.0, 4.0, 3.0, 2.0, 1.0, 5.0};
        input_data_ = graph;
        expected_output_ = {0.0, 4.0, 3.0};
        break;
      }
      case 9: {
        GraphData graph;
        graph.vertices_count = 3;
        graph.start_node = 0;
        graph.row_ptr = {0, 2, 4, 6};
        graph.col_idx = {1, 2, 0, 2, 0, 1};
        graph.weights = {1.0, 4.0, 1.0, 2.0, 4.0, 2.0};
        input_data_ = graph;
        expected_output_ = {0.0, 1.0, 3.0};
        break;
      }
      case 10: {
        GraphData graph;
        graph.vertices_count = 3;
        graph.start_node = 0;
        graph.row_ptr = {0, 2, 5, 6};
        graph.col_idx = {1, 2, 0, 2, 0, 1};
        graph.weights = {5.0, 3.0, 5.0, 1.0, 3.0, 1.0};
        input_data_ = graph;
        expected_output_ = {0.0, 4.0, 3.0};
        break;
      }
      case 11: {
        GraphData graph;
        graph.vertices_count = 6;
        graph.start_node = 0;
        graph.row_ptr = {0, 1, 2, 2, 3, 4, 4};
        graph.col_idx = {1, 0, 4, 3};
        graph.weights = {2.0, 2.0, 3.0, 4.0};
        input_data_ = graph;
        expected_output_ = {0.0,
                            2.0,
                            std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::infinity()};
        break;
      }
      case 12: {
        GraphData graph;
        graph.vertices_count = 4;
        graph.start_node = 0;
        graph.row_ptr = {0, 2, 4, 6, 6};
        graph.col_idx = {1, 2, 0, 3, 1, 3};
        graph.weights = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        input_data_ = graph;
        expected_output_ = {0.0, 0.0, 0.0, 0.0};
        break;
      }
      case 13: {
        GraphData graph;
        graph.vertices_count = 5;
        graph.start_node = 0;
        graph.row_ptr = {0, 2, 4, 6, 8, 8};
        graph.col_idx = {1, 2, 0, 3, 0, 4, 1, 4};
        graph.weights = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        input_data_ = graph;
        expected_output_ = {0.0, 1.0, 1.0, 2.0, 2.0};
        break;
      }
      case 14: {
        GraphData graph;
        graph.vertices_count = 3;
        graph.start_node = 0;
        graph.row_ptr = {0, 2, 3, 3};
        graph.col_idx = {1, 2, 2};
        graph.weights = {4.0, 2.0, 1.0};
        input_data_ = graph;
        expected_output_ = {0.0, 4.0, 2.0};
        break;
      }
      default: {
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
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_output_.size()) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); ++i) {
      bool expected_inf = std::isinf(expected_output_[i]);
      bool actual_inf = std::isinf(output_data[i]);

      if (expected_inf && actual_inf) {
        continue;
      }

      if (expected_inf != actual_inf) {
        return false;
      }

      if (std::abs(expected_output_[i] - output_data[i]) > 1e-6) {
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

TEST_P(KapanovaSDijkstraFuncTests, FindShortestPaths) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 14> kTestParam = {
    std::make_tuple(1, "simple_graph_4_vertices"),   std::make_tuple(2, "no_paths_graph"),
    std::make_tuple(3, "long_chain_graph"),          std::make_tuple(4, "linear_graph"),
    std::make_tuple(5, "single_node_case"),          std::make_tuple(6, "bidirectional_weighted"),
    std::make_tuple(7, "star_topology_graph"),       std::make_tuple(8, "graph_with_self_loops"),
    std::make_tuple(9, "complete_graph_3_vertices"), std::make_tuple(10, "bidirectional_graph"),
    std::make_tuple(11, "disconnected_graph"),       std::make_tuple(12, "zero_weight_edges"),
    std::make_tuple(13, "uniform_weights_graph"),    std::make_tuple(14, "graph_with_self_loops_2"),
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<KapanovaSDijkstraMPI, InType>(kTestParam, PPC_SETTINGS_kapanova_s_dijkstra),
                   ppc::util::AddFuncTask<KapanovaSDijkstraSEQ, InType>(kTestParam, PPC_SETTINGS_kapanova_s_dijkstra));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KapanovaSDijkstraFuncTests::PrintFuncTestName<KapanovaSDijkstraFuncTests>;

INSTANTIATE_TEST_SUITE_P(ShortestPathTests, KapanovaSDijkstraFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace kapanova_s_dijkstra
