#include <gtest/gtest.h>

#include <cmath>
#include <fstream>  // для std::ifstream, std::getline
#include <random>
#include <sstream>  // для std::istringstream
#include <string>   // для std::string
#include <vector>

#include "kapanova_s_min_of_matrix_elements/common/include/common.hpp"
#include "kapanova_s_min_of_matrix_elements/mpi/include/ops_mpi.hpp"
#include "kapanova_s_min_of_matrix_elements/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kapanova_s_min_of_matrix_elements {

class KapanovaSRunFuncTestsMinMatrixElements : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  InType input_data_;

  static InType GenerateMatrix(int n) {
    InType matrix;
    int seed = 999;
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> distribution(-1000000, 1000000);

    for (int i = 0; i < n; i++) {
      std::vector<int> row;
      row.reserve(n);
      for (int j = 0; j < n; j++) {
        row.push_back(distribution(generator));
      }
      matrix.push_back(row);
    }
    matrix[n / 2][n / 2] = -2000000;  // Минимальный элемент
    return matrix;
  }

  void SetUp() override {
    input_data_ = GenerateMatrix(5000);  // 4096x4096 = 16.7 млн элементов
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == -2000000;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(KapanovaSRunFuncTestsMinMatrixElements, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KapanovaSMinOfMatrixElementsMPI, KapanovaSMinOfMatrixElementsSEQ>(
        PPC_SETTINGS_kapanova_s_min_of_matrix_elements);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KapanovaSRunFuncTestsMinMatrixElements::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KapanovaSRunFuncTestsMinMatrixElements, kGtestValues, kPerfTestName);

}  // namespace kapanova_s_min_of_matrix_elements

// Оставляем старые тесты для совместимости
namespace {

std::vector<std::vector<int>> LoadMatrixFromFile(const std::string &filename) {
  std::vector<std::vector<int>> matrix;
  std::ifstream file(filename);

  if (file.is_open()) {
    std::string line;
    while (std::getline(file, line)) {
      std::vector<int> row;
      std::istringstream iss(line);
      int value = 0;
      while (iss >> value) {
        row.push_back(value);
      }
      if (!row.empty()) {
        matrix.push_back(row);
      }
    }
  }
  return matrix;
}

TEST(KapanovaSMinOfMatrixElementsPerf, TestPerformance) {
  const int rows = 2000;
  const int cols = 2000;
  std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix[i][j] = (i * cols) + j;
    }
  }

  matrix[50][50] = -100;

  kapanova_s_min_of_matrix_elements::KapanovaSMinOfMatrixElementsSEQ task(matrix);

  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());

  EXPECT_EQ(-100, task.GetOutput());
}

TEST(KapanovaSMinOfMatrixElementsPerf, TestLoadFromFile) {
  auto matrix = LoadMatrixFromFile("tasks/kapanova_s_min_of_matrix_elements/data/matrix_3x3.txt");

  if (matrix.empty()) {
    matrix = {{1, 2, 3}, {4, 0, 6}, {7, 8, 9}};
  }

  kapanova_s_min_of_matrix_elements::KapanovaSMinOfMatrixElementsSEQ task(matrix);

  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());

  EXPECT_EQ(0, task.GetOutput());
}

}  // namespace
