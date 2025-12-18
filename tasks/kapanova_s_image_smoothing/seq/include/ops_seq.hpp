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

  // Методы для обработки изображения
  static std::vector<float> CreateGaussianKernel(int kernel_radius, float sigma_val);
  static void ProcessRows(const std::vector<uint8_t> &input_img, int img_h, int img_w, const std::vector<float> &kernel,
                          std::vector<float> &temp_buf);
  static void ProcessColumns(const std::vector<float> &temp_buf, int img_h, int img_w, const std::vector<float> &kernel,
                             std::vector<uint8_t> &output_img);

  int img_width_{0};
  int img_height_{0};
  std::vector<uint8_t> input_data_;
  std::vector<uint8_t> output_data_;
};

}  // namespace kapanova_s_image_smoothing
