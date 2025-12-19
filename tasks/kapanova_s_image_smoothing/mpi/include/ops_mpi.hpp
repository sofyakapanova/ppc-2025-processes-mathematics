#pragma once

#include "kapanova_s_image_smoothing/common/include/common.hpp"
#include "task/include/task.hpp"

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

  void SmoothPixel(uint8_t *out, int x, int y);
  float *CreateKernel();

  int width = 0;
  int height = 0;
  std::vector<uint8_t> input;
  std::vector<uint8_t> result;
  int radius = 1;
  float *kernel = nullptr;
};

}  // namespace kapanova_s_image_smoothing
