#pragma once

#include <vector>

#include "/opt/homebrew/opt/boost/include/boost/mpi/communicator.hpp"
#include "task/include/task.hpp"

namespace kapanova_s_image_smoothing {

class KapanovaSImageSmoothingMPI
    : public ppc::task::Task<std::vector<std::vector<int>>, std::vector<std::vector<int>>> {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KapanovaSImageSmoothingMPI(const std::vector<std::vector<int>> &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // Вспомогательные методы
  void createGaussianKernel();
  void processPixel(int x, int y);
  int clampValue(int value, int min_val, int max_val);

  // Данные
  int image_height_;
  int image_width_;
  std::vector<std::vector<int>> input_matrix_;
  std::vector<std::vector<int>> output_matrix_;
  std::vector<float> gaussian_kernel_;
  boost::mpi::communicator mpi_communicator_;
};

}  // namespace kapanova_s_image_smoothing
