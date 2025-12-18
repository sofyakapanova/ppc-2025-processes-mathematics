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
};

}  // namespace kapanova_s_image_smoothing
