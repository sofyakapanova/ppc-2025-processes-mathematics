#pragma once

#include <cstdint>
#include <vector>

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

  void SmoothPixel(uint8_t *out, int x_coord, int y_coord, bool use_local = false,
                   const std::vector<uint8_t> *local_input = nullptr, int local_width = 0, int local_height = 0);

  [[nodiscard]] static std::vector<float> CreateKernel();

  void ProcessBorderRows();
  void ProcessRowRange(int start_row, int num_rows);
  void SendImageData(int worker_rank, int row);
  [[nodiscard]] int CalculateDataSize(int row) const;
  [[nodiscard]] int CalculateStartPosition(int row) const;

  void MasterProcess();
  void DistributeRowsToWorkers(int num_workers);
  void AssignRowsToWorkers(int start_row, int num_workers);
  void ReceiveResultsFromWorkers(int start_row, int num_workers);
  static void SendExitSignalToWorkers(int num_workers);

  void WorkerProcess();
  static int ReceiveImageData(std::vector<uint8_t> &buffer);
  void ProcessAndSendResult(int local_width, const std::vector<uint8_t> &input, std::vector<uint8_t> &result,
                            int rows_received, int row_to_process);
  [[nodiscard]] static int GetCommRank();
  [[nodiscard]] static int GetCommSize();

  int width_ = 0;
  int height_ = 0;
  std::vector<uint8_t> input_;
  std::vector<uint8_t> result_;
  int radius_ = 1;
  std::vector<float> kernel_;
};

}  // namespace kapanova_s_image_smoothing
