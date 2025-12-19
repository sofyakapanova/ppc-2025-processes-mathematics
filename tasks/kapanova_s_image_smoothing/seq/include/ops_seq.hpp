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

  void CreateKernel();
  void SmoothPixel(int x, int y);

  int width = 0;
  int height = 0;
  std::vector<uint8_t> input;
  std::vector<uint8_t> result;
  int radius = 1;
  float *kernel = nullptr;
};

}  // namespace kapanova_s_image_smoothing
