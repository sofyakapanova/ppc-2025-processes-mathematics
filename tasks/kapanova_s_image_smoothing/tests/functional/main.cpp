#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "kapanova_s_image_smoothing/common/include/common.hpp"
#include "kapanova_s_image_smoothing/mpi/include/ops_mpi.hpp"
#include "kapanova_s_image_smoothing/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kapanova_s_image_smoothing {

class KapanovaSImageSmoothingFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    auto test_params = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int kernel_size = std::get<0>(test_params);
    std::string test_name = std::get<1>(test_params);

    // Создаем тестовые данные
    if (test_name == "kernel3") {
      CreateSmallImage(kernel_size);
    } else if (test_name == "kernel5") {
      CreateUniformImage(kernel_size);
    } else if (test_name == "kernel7") {
      CreateGradientImage(kernel_size);
    } else if (test_name == "small_image") {
      CreateSmallTestImage(kernel_size);
    } else if (test_name == "kernel1") {
      CreateKernelSizeOneImage();
    } else {
      CreateDefaultImage(kernel_size);
    }

    // Создаем эталонный результат с помощью SEQ версии
    KapanovaSImageSmoothingSEQ seq_task(input_data_);
    if (seq_task.Validation()) {
      seq_task.PreProcessing();
      seq_task.Run();
      seq_task.PostProcessing();
      expected_output_ = seq_task.GetOutput();
    }
  }

  void CreateSmallImage(int kernel_size) {
    input_data_.width = 4;
    input_data_.height = 4;
    input_data_.kernel_size = kernel_size;
    input_data_.pixels.resize(16);

    // Простой паттерн
    for (int i = 0; i < 16; ++i) {
      input_data_.pixels[i] = static_cast<uint8_t>(i * 16);
    }
  }

  void CreateUniformImage(int kernel_size) {
    input_data_.width = 8;
    input_data_.height = 8;
    input_data_.kernel_size = kernel_size;
    input_data_.pixels.assign(64, 100);
  }

  void CreateGradientImage(int kernel_size) {
    input_data_.width = 5;
    input_data_.height = 5;
    input_data_.kernel_size = kernel_size;
    input_data_.pixels.resize(25);

    for (int y = 0; y < 5; ++y) {
      for (int x = 0; x < 5; ++x) {
        int index = y * 5 + x;
        input_data_.pixels[index] = static_cast<uint8_t>((x + y) * 20);
      }
    }
  }

  void CreateSmallTestImage(int kernel_size) {
    input_data_.width = 3;
    input_data_.height = 3;
    input_data_.kernel_size = kernel_size;
    input_data_.pixels = {10, 20, 30, 40, 50, 60, 70, 80, 90};
  }

  void CreateKernelSizeOneImage() {
    input_data_.width = 4;
    input_data_.height = 3;
    input_data_.kernel_size = 1;
    input_data_.pixels.resize(12);

    for (size_t i = 0; i < input_data_.pixels.size(); ++i) {
      input_data_.pixels[i] = static_cast<uint8_t>(i * 20);
    }
  }

  void CreateDefaultImage(int kernel_size) {
    input_data_.width = 10;
    input_data_.height = 10;
    input_data_.kernel_size = kernel_size;
    input_data_.pixels.resize(100);

    for (size_t i = 0; i < input_data_.pixels.size(); ++i) {
      input_data_.pixels[i] = static_cast<uint8_t>((i * 7) % 256);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Сравниваем с эталонным результатом
    return output_data.pixels == expected_output_.pixels;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(KapanovaSImageSmoothingFuncTests, SmoothImage) {
  ExecuteTest(GetParam());
}

// Тестовые параметры: (размер_ядра, имя_теста)
const std::array<TestType, 6> kTestParam = {std::make_tuple(3, "kernel3"), std::make_tuple(5, "kernel5"),
                                            std::make_tuple(7, "kernel7"), std::make_tuple(3, "small_image"),
                                            std::make_tuple(1, "kernel1"), std::make_tuple(3, "default_image")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<KapanovaSImageSmoothingMPI, InType>(kTestParam, PPC_SETTINGS_kapanova_s_image_smoothing),
    ppc::util::AddFuncTask<KapanovaSImageSmoothingSEQ, InType>(kTestParam, PPC_SETTINGS_kapanova_s_image_smoothing));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KapanovaSImageSmoothingFuncTests::PrintFuncTestName<KapanovaSImageSmoothingFuncTests>;

INSTANTIATE_TEST_SUITE_P(ImageSmoothingTests, KapanovaSImageSmoothingFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace kapanova_s_image_smoothing
