#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "kapanova_s_image_smoothing/common/include/common.hpp"
#include "kapanova_s_image_smoothing/mpi/include/ops_mpi.hpp"
#include "kapanova_s_image_smoothing/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kapanova_s_image_smoothing {

class KapanovaSImageSmoothingPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  const int kWidth = 2048;
  const int kHeight = 2048;
  const int kKernelSize = 5;

  InType input_data_;

  void SetUp() override {
    input_data_.width = kWidth;
    input_data_.height = kHeight;
    input_data_.kernel_size = kKernelSize;

    const size_t total_pixels = static_cast<size_t>(kWidth) * kHeight;
    input_data_.pixels.resize(total_pixels);

    // Заполняем тестовыми данными
    for (size_t i = 0; i < total_pixels; ++i) {
      input_data_.pixels[i] = static_cast<uint8_t>((i * 13) % 256);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Для MPI: на процессе 0 проверяем результат, на других процессах
    // только проверяем, что метаданные корректны
    if (rank == 0) {
      // Проверяем размер
      if (output_data.pixels.size() != static_cast<size_t>(kWidth) * kHeight) {
        return false;
      }

      // Проверяем, что все значения в диапазоне [0, 255]
      return std::ranges::all_of(output_data.pixels, [](uint8_t val) { return val >= 0 && val <= 255; });
    } else {
      // На других процессах проверяем только метаданные
      return output_data.width == kWidth && output_data.height == kHeight && output_data.kernel_size == kKernelSize;
    }
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(KapanovaSImageSmoothingPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, KapanovaSImageSmoothingMPI, KapanovaSImageSmoothingSEQ>(
    PPC_SETTINGS_kapanova_s_image_smoothing);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KapanovaSImageSmoothingPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KapanovaSImageSmoothingPerfTests, kGtestValues, kPerfTestName);

}  // namespace kapanova_s_image_smoothing
