#pragma once

#include "kapanova_s_image_smoothing/common/include/common.hpp"

namespace kapanova_s_image_smoothing {

class KapanovaSImageSmoothingSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit KapanovaSImageSmoothingSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<float> create_gaussian_kernel(int radius, float sigma);
  void convolve_rows(const std::vector<uint8_t> &input, int height, int width, const std::vector<float> &kernel,
                     std::vector<float> &temp);
  void convolve_columns(const std::vector<float> &temp, int height, int width, const std::vector<float> &kernel,
                        std::vector<uint8_t> &output);

  int width_{0};
  int height_{0};
  std::vector<uint8_t> input_;
  std::vector<uint8_t> output_;
};

}  // namespace kapanova_s_image_smoothing
