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

  std::vector<float> CreateGaussianKernel();
};

}  // namespace kapanova_s_image_smoothing
