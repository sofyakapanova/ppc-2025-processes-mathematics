#pragma once

#include "kapanova_s_image_smoothing/common/include/common.hpp"

namespace kapanova_s_image_smoothing {

class KapanovaSImageSmoothingMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit KapanovaSImageSmoothingMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  
  static std::vector<float> create_gaussian_kernel(int radius, float sigma);
  static void convolve_rows(const std::vector<uint8_t>& input, int height, int width, 
                           const std::vector<float>& kernel, std::vector<float>& temp);
  static void convolve_columns(const std::vector<float>& temp, int height, int width,
                              const std::vector<float>& kernel, std::vector<uint8_t>& output);
};

}  // namespace kapanova_s_image_smoothing