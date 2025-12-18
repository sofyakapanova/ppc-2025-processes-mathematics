#pragma once

#include <memory>
#include <vector>

#include "task/include/task.hpp"

namespace kapanova_s_image_smoothing {

class KapanovaSImageSmoothingSEQ
    : public ppc::task::Task<std::vector<std::vector<int>>, std::vector<std::vector<int>>> {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KapanovaSImageSmoothingSEQ(const std::vector<std::vector<int>> &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // Вспомогательные методы
  void generateGaussianKernel();
  void smoothPixel(int x, int y);
  int limitToRange(int value, int lower, int upper);

  // Данные
  int img_height_;
  int img_width_;
  std::vector<std::vector<int>> input_image_;
  std::vector<std::vector<int>> result_image_;
  std::vector<float> gaussian_filter_;
};

}  // namespace kapanova_s_image_smoothing
