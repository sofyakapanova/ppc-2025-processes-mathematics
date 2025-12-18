#pragma once

#include "kapanova_s_image_smoothing/common/include/common.hpp"
#include "task/include/task.hpp"

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

  // Вспомогательные методы
  int Clamp(int value, int min_val, int max_val) const;
  uint8_t CalculatePixelAverage(int x, int y) const;
};

}  // namespace kapanova_s_image_smoothing
