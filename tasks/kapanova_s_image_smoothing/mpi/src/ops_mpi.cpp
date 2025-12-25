#include "kapanova_s_image_smoothing/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "kapanova_s_image_smoothing/common/include/common.hpp"

namespace kapanova_s_image_smoothing {

namespace {
constexpr int kTagExit = 0;
constexpr int kTagData = 2;
constexpr int kTagResult = 3;
constexpr int kRadius = 1;
constexpr int kKernelSize = (2 * kRadius) + 1;
constexpr float kSigma = 1.5F;
constexpr float kSigmaSquared = kSigma * kSigma;
constexpr int kEscapeSignal = 0;
constexpr int kNoEscapeSignal = 1;
}  // namespace

KapanovaSImageSmoothingMPI::KapanovaSImageSmoothingMPI(const InType &in) : radius_(kRadius) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in.empty() ? InType() : in;
}

bool KapanovaSImageSmoothingMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if (rank == 0) {
    const auto &input_data = GetInput();
    return !input_data.empty() && !input_data[0].empty();
  }
  
  return true;
}

bool KapanovaSImageSmoothingMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if (rank == 0) {
    const auto &input_data = GetInput();
    if (input_data.empty() || input_data[0].size() < 4) {
      return false;
    }

    const auto &data = input_data[0];
    width_ = (data[1] << 8) | data[0];
    height_ = (data[3] << 8) | data[2];

    const auto width_u = static_cast<size_t>(width_);
    const auto height_u = static_cast<size_t>(height_);
    const size_t required_pixels = width_u * height_u * 3U;
    const size_t total_required_size = 4U + required_pixels;

    if (data.size() < total_required_size) {
      return false;
    }

    input_.assign(data.begin() + 4, data.end());
    
    int dimensions[2] = {width_, height_};
    MPI_Bcast(dimensions, 2, MPI_INT, 0, MPI_COMM_WORLD);
    
    int image_size = static_cast<int>(input_.size());
    MPI_Bcast(&image_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (image_size > 0) {
      MPI_Bcast(input_.data(), image_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    }
    
  } else {
    int dimensions[2];
    MPI_Bcast(dimensions, 2, MPI_INT, 0, MPI_COMM_WORLD);
    width_ = dimensions[0];
    height_ = dimensions[1];
    
    int image_size;
    MPI_Bcast(&image_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (image_size > 0) {
      input_.resize(static_cast<size_t>(image_size));
      MPI_Bcast(input_.data(), image_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    }
  }
  
  const auto width_u = static_cast<size_t>(width_);
  const auto height_u = static_cast<size_t>(height_);
  const size_t required_pixels = width_u * height_u * 3U;
  result_.resize(required_pixels);
  kernel_ = CreateKernel();
  
  return true;
}

std::vector<float> KapanovaSImageSmoothingMPI::CreateKernel() {
  const auto kernel_size = static_cast<size_t>(kKernelSize) * static_cast<size_t>(kKernelSize);
  std::vector<float> kernel(kernel_size, 0.0F);
  float norm = 0.0F;

  for (int i = -kRadius; i <= kRadius; ++i) {
    for (int j = -kRadius; j <= kRadius; ++j) {
      const int index = ((i + kRadius) * kKernelSize) + (j + kRadius);
      kernel[static_cast<size_t>(index)] = std::exp(-static_cast<float>((i * i) + (j * j)) / (2.0F * kSigmaSquared));
      norm += kernel[static_cast<size_t>(index)];
    }
  }

  if (norm > 0.0F) {
    for (auto &value : kernel) {
      value /= norm;
    }
  }

  return kernel;
}

void KapanovaSImageSmoothingMPI::ProcessBorderRows() {
  for (int x_coord = 0; x_coord < width_; ++x_coord) {
    const auto top_pos = static_cast<size_t>(x_coord) * 3U;
    SmoothPixel(&result_[top_pos], x_coord, 0);

    const auto bottom_pos = static_cast<size_t>((height_ - 1) * width_ * 3) + (static_cast<size_t>(x_coord) * 3U);
    SmoothPixel(&result_[bottom_pos], x_coord, height_ - 1);
  }
}

void KapanovaSImageSmoothingMPI::ProcessRowRange(int start_row, int num_rows) {
  const int end_row = std::min(start_row + num_rows, height_ - 1);

  for (int y_coord = start_row; y_coord < end_row; ++y_coord) {
    for (int x_coord = 0; x_coord < width_; ++x_coord) {
      const auto pos = static_cast<size_t>(y_coord * width_ * 3) + (static_cast<size_t>(x_coord) * 3U);
      SmoothPixel(&result_[pos], x_coord, y_coord);
    }
  }
}

void KapanovaSImageSmoothingMPI::SendImageData(int worker_rank, int row) {
  const int data_size = CalculateDataSize(row);
  const int start_pos = CalculateStartPosition(row);

  if (data_size > 0) {
    MPI_Send(&input_[static_cast<size_t>(start_pos)], data_size, MPI_UNSIGNED_CHAR, worker_rank, kTagData,
             MPI_COMM_WORLD);
  }
}

int KapanovaSImageSmoothingMPI::CalculateDataSize(int row) const {
  if (row <= 0 || row >= height_ - 1) {
    return 0;
  }
  
  return 3 * width_ * 3;
}

int KapanovaSImageSmoothingMPI::CalculateStartPosition(int row) const {
  if (row <= 0 || row >= height_ - 1) {
    return 0;
  }
  
  return (row - 1) * width_ * 3;
}

void KapanovaSImageSmoothingMPI::MasterProcess() {
  const int satellites = GetCommSize() - 1;

  ProcessBorderRows();
  
  DistributeRowsToWorkers(satellites);
  
  SendExitSignalToWorkers(satellites);
}

void KapanovaSImageSmoothingMPI::DistributeRowsToWorkers(int num_workers) {
  int current_row = 1;

  while (current_row < height_ - 1) {
    const int processes_to_use = std::min(num_workers, height_ - 1 - current_row);
    
    AssignRowsToWorkers(current_row, processes_to_use);
    ReceiveResultsFromWorkers(current_row, processes_to_use);
    current_row += processes_to_use;
  }
}

void KapanovaSImageSmoothingMPI::AssignRowsToWorkers(int start_row, int num_workers) {
  for (int i = 0; i < num_workers; ++i) {
    const int worker_rank = i + 1;
    const int row_to_process = start_row + i;

    if (row_to_process <= 0 || row_to_process >= height_ - 1) {
      continue;
    }

    int work_info[2] = {kNoEscapeSignal, row_to_process};
    MPI_Send(work_info, 2, MPI_INT, worker_rank, kTagExit, MPI_COMM_WORLD);
    
    SendImageData(worker_rank, row_to_process);
  }
}

void KapanovaSImageSmoothingMPI::ReceiveResultsFromWorkers(int start_row, int num_workers) {
  for (int i = 0; i < num_workers; ++i) {
    const int worker_rank = i + 1;
    const int result_row = start_row + i;

    if (result_row >= 1 && result_row < height_ - 1) {
      const auto result_pos = static_cast<size_t>(result_row) * static_cast<size_t>(width_) * 3;
      MPI_Recv(&result_[result_pos], width_ * 3, MPI_UNSIGNED_CHAR, worker_rank, kTagResult, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }
  }
}

void KapanovaSImageSmoothingMPI::SendExitSignalToWorkers(int num_workers) {
  for (int i = 1; i <= num_workers; ++i) {
    int exit_info[2] = {kEscapeSignal, 0};
    MPI_Send(exit_info, 2, MPI_INT, i, kTagExit, MPI_COMM_WORLD);
  }
}

void KapanovaSImageSmoothingMPI::WorkerProcess() {
  int local_width = width_;
  
  std::vector<uint8_t> local_input;
  std::vector<uint8_t> local_result(static_cast<size_t>(local_width * 3));
  
  int work_info[2];
  
  while (true) {
    MPI_Recv(work_info, 2, MPI_INT, 0, kTagExit, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    int escape_signal = work_info[0];
    int row_to_process = work_info[1];
    
    if (escape_signal == kEscapeSignal) {
      break;
    }
    
    const int received_bytes = ReceiveImageData(local_input);
    const int rows_received = received_bytes / (local_width * 3);
    
    if (rows_received > 0) {
      ProcessAndSendResult(local_width, local_input, local_result, rows_received, row_to_process);
    }
  }
}

int KapanovaSImageSmoothingMPI::ReceiveImageData(std::vector<uint8_t> &buffer) {
  MPI_Status status;
  MPI_Probe(0, kTagData, MPI_COMM_WORLD, &status);

  int count = 0;
  MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &count);

  buffer.resize(static_cast<size_t>(count));
  MPI_Recv(buffer.data(), count, MPI_UNSIGNED_CHAR, 0, kTagData, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  
  return count;
}

void KapanovaSImageSmoothingMPI::ProcessAndSendResult(int local_width, const std::vector<uint8_t> &input,
                                                      std::vector<uint8_t> &result, int rows_received,
                                                      int row_to_process) {
  const int target_row = row_to_process;
  
  for (int x_coord = 0; x_coord < local_width; ++x_coord) {
    const auto pos = static_cast<size_t>(x_coord) * 3U;
    SmoothPixel(&result[pos], x_coord, target_row, true, &input, local_width, rows_received);
  }

  MPI_Send(result.data(), local_width * 3, MPI_UNSIGNED_CHAR, 0, kTagResult, MPI_COMM_WORLD);
}

int KapanovaSImageSmoothingMPI::GetCommRank() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}

int KapanovaSImageSmoothingMPI::GetCommSize() {
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  return size;
}

bool KapanovaSImageSmoothingMPI::RunImpl() {
  int rank = GetCommRank();
  int size = GetCommSize();
  
  if (size == 1) {
    ProcessRowRange(1, height_ - 2);
    ProcessBorderRows();
    return true;
  }

  if (rank == 0) {
    MasterProcess();
  } else {
    WorkerProcess();
  }

  MPI_Barrier(MPI_COMM_WORLD);
  
  return true;
}

bool KapanovaSImageSmoothingMPI::PostProcessingImpl() {
  GetOutput() = result_;
  return true;
}

void KapanovaSImageSmoothingMPI::SmoothPixel(uint8_t *out, int x_coord, int y_coord, bool use_local,
                                             const std::vector<uint8_t> *local_input, int local_width,
                                             int local_height) {
  const int k_size = (2 * radius_) + 1;
  float out_r = 0.0F;
  float out_g = 0.0F;
  float out_b = 0.0F;

  auto clamp = [](int n, int lo, int hi) { return std::min(std::max(n, lo), hi); };

  if (use_local && local_input != nullptr) {
    const int local_stride = local_width * 3;
    const int local_y_center = 1;
    
    for (int ry = -radius_; ry <= radius_; ++ry) {
      int local_y = local_y_center + ry;
      
      if (local_y < 0 || local_y >= local_height) {
        continue;
      }
      
      for (int rx = -radius_; rx <= radius_; ++rx) {
        int local_x = clamp(x_coord + rx, 0, local_width - 1);
        
        const auto pixel_pos = static_cast<size_t>(local_y * local_stride) + (static_cast<size_t>(local_x) * 3U);
        const auto kernel_pos = static_cast<size_t>((ry + radius_) * k_size) + static_cast<size_t>(rx + radius_);
        
        out_r += static_cast<float>((*local_input)[pixel_pos]) * kernel_[kernel_pos];
        out_g += static_cast<float>((*local_input)[pixel_pos + 1U]) * kernel_[kernel_pos];
        out_b += static_cast<float>((*local_input)[pixel_pos + 2U]) * kernel_[kernel_pos];
      }
    }
  } else {
    for (int ry = -radius_; ry <= radius_; ++ry) {
      int global_y = clamp(y_coord + ry, 0, height_ - 1);
      
      for (int rx = -radius_; rx <= radius_; ++rx) {
        int global_x = clamp(x_coord + rx, 0, width_ - 1);
        
        const auto pixel_pos = static_cast<size_t>(global_y * width_ * 3) + (static_cast<size_t>(global_x) * 3U);
        const auto kernel_pos = static_cast<size_t>((ry + radius_) * k_size) + static_cast<size_t>(rx + radius_);
        
        out_r += static_cast<float>(input_[pixel_pos]) * kernel_[kernel_pos];
        out_g += static_cast<float>(input_[pixel_pos + 1U]) * kernel_[kernel_pos];
        out_b += static_cast<float>(input_[pixel_pos + 2U]) * kernel_[kernel_pos];
      }
    }
  }

  out[0] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(out_r)), 0, 255));
  out[1] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(out_g)), 0, 255));
  out[2] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(out_b)), 0, 255));
}

}  // namespace kapanova_s_image_smoothing