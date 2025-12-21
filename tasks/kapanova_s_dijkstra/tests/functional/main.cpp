#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "kapanova_s_dijkstra/common/include/common.hpp"
#include "kapanova_s_dijkstra/mpi/include/ops_mpi.hpp"
#include "kapanova_s_dijkstra/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kapanova_s_dijkstra {

class KapanovaSDijkstraFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    static std::atomic<int> counter{0};
    int test_id = counter.fetch_add(1);

    const auto &graph = std::get<0>(test_param);

    std::string name = "test" + std::to_string(test_id) + "_v" + std::to_string(graph.num_vertices) + "_e" +
                       std::to_string(graph.distances.size());

    std::ranges::replace(name, '-', 'n');
    return name;
  }

 protected:
  void SetUp() override {
    test_params_ = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_graph_ = std::get<0>(test_params_);
    expected_output_ = std::get<1>(test_params_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_output_;
  }

  InType GetTestInputData() final {
    return input_graph_;
  }

 private:
  TestType test_params_;
  InType input_graph_;
  OutType expected_output_;
};

namespace {

using kapanova_s_dijkstra::GraphCSR;

GraphCSR CreateSimpleGraph() {
  GraphCSR graph;
  graph.num_vertices = 4;
  graph.start_vertex = 0;

  graph.row_ptrs = {0, 2, 4, 5, 6};
  graph.columns = {1, 2, 0, 3, 1, 3};
  graph.distances = {1, 4, 1, 2, 2, 3};

  return graph;
}

std::vector<int> ExpectedSimpleGraph() {
  return {0, 1, 4, 3};
}

GraphCSR CreateDisconnectedGraph() {
  GraphCSR graph;
  graph.num_vertices = 5;
  graph.start_vertex = 0;

  graph.row_ptrs = {0, 2, 3, 3, 4, 4};
  graph.columns = {1, 2, 0, 3};
  graph.distances = {2, 1, 2, 1};

  return graph;
}

std::vector<int> ExpectedDisconnectedGraph() {
  return {0, 2, 1, -1, -1};
}

GraphCSR CreateSingleVertexGraph() {
  GraphCSR graph;
  graph.num_vertices = 1;
  graph.start_vertex = 0;

  graph.row_ptrs = {0, 0};
  graph.columns = {};
  graph.distances = {};

  return graph;
}

std::vector<int> ExpectedSingleVertexGraph() {
  return {0};
}

GraphCSR CreateChainGraph() {
  GraphCSR graph;
  graph.num_vertices = 5;
  graph.start_vertex = 0;

  graph.row_ptrs = {0, 1, 2, 3, 4, 4};
  graph.columns = {1, 2, 3, 4};
  graph.distances = {1, 1, 1, 1};

  return graph;
}

std::vector<int> ExpectedChainGraph() {
  return {0, 1, 2, 3, 4};
}

TEST_P(KapanovaSDijkstraFuncTests, FindShortestPaths) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kTestParam = {
    std::make_tuple(CreateSimpleGraph(), ExpectedSimpleGraph()),
    std::make_tuple(CreateDisconnectedGraph(), ExpectedDisconnectedGraph()),
    std::make_tuple(CreateSingleVertexGraph(), ExpectedSingleVertexGraph()),
    std::make_tuple(CreateChainGraph(), ExpectedChainGraph()),
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<KapanovaSDijkstraMPI, InType>(kTestParam, PPC_SETTINGS_kapanova_s_dijkstra),
                   ppc::util::AddFuncTask<KapanovaSDijkstraSEQ, InType>(kTestParam, PPC_SETTINGS_kapanova_s_dijkstra));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KapanovaSDijkstraFuncTests::PrintFuncTestName<KapanovaSDijkstraFuncTests>;

INSTANTIATE_TEST_SUITE_P(DijkstraTests, KapanovaSDijkstraFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace kapanova_s_dijkstra
