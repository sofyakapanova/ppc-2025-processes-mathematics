#pragma once

#include <cstdint>
#include <vector>

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

  int width_ = 0;
  int height_ = 0;
  std::vector<uint8_t> input_;
  std::vector<uint8_t> result_;
  int radius_ = 1;
  std::vector<float> kernel_;
};

}  // namespace kapanova_s_image_smoothing
