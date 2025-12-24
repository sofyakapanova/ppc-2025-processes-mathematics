#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
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
    const int width = std::get<1>(test_param);
    const int height = std::get<2>(test_param);

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
    const size_t expected_size = static_cast<size_t>(width_) * static_cast<size_t>(height_) * 3;
    if (output_data.empty() || output_data.size() != expected_size) {
      return false;
    }

    if (!expected_output_.empty()) {
      return output_data == expected_output_;
    }

    if (IsConstantImage(image_data_)) {
      return CheckConstantImage(output_data);
    }

    if (width_ > 2 && height_ > 2) {
      return CheckSmoothingEffect(image_data_, output_data);
    }

    return true;
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

  [[nodiscard]] static bool IsConstantImage(const std::vector<uint8_t> &image) {
    if (image.size() < 3) {
      return true;
    }

    const uint8_t first_pixel_r = image[0];
    const uint8_t first_pixel_g = image[1];
    const uint8_t first_pixel_b = image[2];

    for (size_t i = 3; i < image.size(); i += 3) {
      if (image[i] != first_pixel_r || image[i + 1] != first_pixel_g || image[i + 2] != first_pixel_b) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] static bool CheckConstantImage(const std::vector<uint8_t> &output) {
    if (output.size() < 3) {
      return true;
    }

    const uint8_t first_pixel_r = output[0];
    const uint8_t first_pixel_g = output[1];
    const uint8_t first_pixel_b = output[2];

    constexpr int kTolerance = 5;

    for (size_t i = 3; i < output.size(); i += 3) {
      if (std::abs(static_cast<int>(output[i]) - static_cast<int>(first_pixel_r)) > kTolerance ||
          std::abs(static_cast<int>(output[i + 1]) - static_cast<int>(first_pixel_g)) > kTolerance ||
          std::abs(static_cast<int>(output[i + 2]) - static_cast<int>(first_pixel_b)) > kTolerance) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool CheckSmoothingEffect(const std::vector<uint8_t> &input, const std::vector<uint8_t> &output) const {
    if (input.size() != output.size()) {
      return false;
    }

    double input_gradient = 0.0;
    double output_gradient = 0.0;

    size_t checked_pairs = 0;

    for (int y_coord = 0; y_coord < height_; ++y_coord) {
      for (int x_coord = 0; x_coord < width_ - 1; ++x_coord) {
        // Исправлено: явное приведение каждого int к size_t перед умножением
        const auto idx1 =
            (static_cast<size_t>(y_coord) * static_cast<size_t>(width_) * 3) + (static_cast<size_t>(x_coord) * 3);
        const auto idx2 =
            (static_cast<size_t>(y_coord) * static_cast<size_t>(width_) * 3) + (static_cast<size_t>(x_coord + 1) * 3);

        const double brightness1_input = (0.299 * input[idx1]) + (0.587 * input[idx1 + 1]) + (0.114 * input[idx1 + 2]);
        const double brightness2_input = (0.299 * input[idx2]) + (0.587 * input[idx2 + 1]) + (0.114 * input[idx2 + 2]);

        const double brightness1_output =
            (0.299 * output[idx1]) + (0.587 * output[idx1 + 1]) + (0.114 * output[idx1 + 2]);
        const double brightness2_output =
            (0.299 * output[idx2]) + (0.587 * output[idx2 + 1]) + (0.114 * output[idx2 + 2]);

        input_gradient += std::abs(brightness1_input - brightness2_input);
        output_gradient += std::abs(brightness1_output - brightness2_output);
        checked_pairs++;
      }
    }

    if (checked_pairs == 0) {
      return true;
    }

    input_gradient /= static_cast<double>(checked_pairs);
    output_gradient /= static_cast<double>(checked_pairs);

    return output_gradient <= input_gradient * 1.1;
  }
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
