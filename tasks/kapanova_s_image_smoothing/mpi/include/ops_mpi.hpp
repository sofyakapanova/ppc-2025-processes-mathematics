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
  
  // Вспомогательные функции для Гауссова ядра
  std::vector<float> CreateGaussianKernel();
  uint8_t ApplyGaussianFilter(int x, int y, const std::vector<uint8_t>& local_data, 
                              int local_width, int local_height, 
                              int offset_row, const std::vector<float>& kernel);
};

}  // namespace kapanova_s_image_smoothing