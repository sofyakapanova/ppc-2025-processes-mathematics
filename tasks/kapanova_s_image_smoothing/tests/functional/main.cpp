#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
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
    int width = std::get<1>(test_param);
    int height = std::get<2>(test_param);

    std::string name = "image_" + std::to_string(width) + "x" + std::to_string(height);

    std::ranges::replace(name, '-', 'n');
    return name;
  }

 protected:
  void SetUp() override {
    test_params_ = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    image_data_ = std::get<0>(test_params_);
    width_ = std::get<1>(test_params_);
    height_ = std::get<2>(test_params_);
    expected_output_ = std::get<3>(test_params_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (expected_output_.empty()) {
      size_t expected_size = static_cast<size_t>(width_) * static_cast<size_t>(height_) * 3;
      return !output_data.empty() && output_data.size() == expected_size;
    }
    return output_data == expected_output_;
  }

  InType GetTestInputData() final {
    InType formatted_input;
    std::vector<uint8_t> data;

    data.push_back(static_cast<uint8_t>(width_ & 0xFF));
    data.push_back(static_cast<uint8_t>((width_ >> 8) & 0xFF));
    data.push_back(static_cast<uint8_t>(height_ & 0xFF));
    data.push_back(static_cast<uint8_t>((height_ >> 8) & 0xFF));

    data.insert(data.end(), image_data_.begin(), image_data_.end());

    formatted_input.push_back(data);
    return formatted_input;
  }

 private:
  TestType test_params_;
  std::vector<uint8_t> image_data_;
  int width_ = 0;
  int height_ = 0;
  std::vector<uint8_t> expected_output_;
};

namespace {

const std::vector<uint8_t> kImage3x3 = {255, 0,   0, 0,   255, 0,   0,   0,  255, 255, 255, 0,   0,  255,
                                        255, 255, 0, 255, 128, 128, 128, 64, 64,  64,  192, 192, 192};
const int kWidth3x3 = 3;
const int kHeight3x3 = 3;

const std::vector<uint8_t> kImage2x2 = {255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255};
const int kWidth2x2 = 2;
const int kHeight2x2 = 2;

const std::vector<uint8_t> kImage4x4(static_cast<size_t>(4 * 4 * 3), 128);
const int kWidth4x4 = 4;
const int kHeight4x4 = 4;

TEST_P(KapanovaSImageSmoothingFuncTests, SmoothImage) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(kImage3x3, kWidth3x3, kHeight3x3, std::vector<uint8_t>()),
                                            std::make_tuple(kImage2x2, kWidth2x2, kHeight2x2, std::vector<uint8_t>()),
                                            std::make_tuple(kImage4x4, kWidth4x4, kHeight4x4, std::vector<uint8_t>())};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<KapanovaSImageSmoothingMPI, InType>(kTestParam, PPC_SETTINGS_kapanova_s_image_smoothing),
    ppc::util::AddFuncTask<KapanovaSImageSmoothingSEQ, InType>(kTestParam, PPC_SETTINGS_kapanova_s_image_smoothing));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kPerfTestName = KapanovaSImageSmoothingFuncTests::PrintFuncTestName<KapanovaSImageSmoothingFuncTests>;

INSTANTIATE_TEST_SUITE_P(ImageSmoothingTests, KapanovaSImageSmoothingFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace kapanova_s_image_smoothing
