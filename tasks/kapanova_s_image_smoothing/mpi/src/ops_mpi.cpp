#include "kapanova_s_image_smoothing/mpi/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <cmath>
#include <vector>

namespace kapanova_s_image_smoothing {

KapanovaSImageSmoothingMPI::KapanovaSImageSmoothingMPI(const std::vector<std::vector<int>> &in) : mpi_communicator_() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = in;
}

bool KapanovaSImageSmoothingMPI::ValidationImpl() {
  const auto &matrix = GetInput();
  if (matrix.empty()) {
    return true;
  }

  const size_t cols = matrix[0].size();
  return std::ranges::all_of(matrix, [cols](const auto &row) { return row.size() == cols; });
}

bool KapanovaSImageSmoothingMPI::PreProcessingImpl() {
  input_matrix_ = GetInput();
  if (input_matrix_.empty()) {
    return true;
  }

  image_height_ = static_cast<int>(input_matrix_.size());
  image_width_ = static_cast<int>(input_matrix_[0].size());

  // Инициализируем выходную матрицу
  output_matrix_.resize(image_height_);
  for (int i = 0; i < image_height_; ++i) {
    output_matrix_[i].resize(image_width_);
  }

  // Создаем гауссово ядро
  createGaussianKernel();

  return true;
}

void KapanovaSImageSmoothingMPI::createGaussianKernel() {
  const int kernel_radius = 1;
  const int kernel_size = 2 * kernel_radius + 1;
  gaussian_kernel_.resize(kernel_size * kernel_size);
  float sigma = 1.5f;
  float total_weight = 0.0f;

  for (int i = -kernel_radius; i <= kernel_radius; ++i) {
    for (int j = -kernel_radius; j <= kernel_radius; ++j) {
      int idx = (i + kernel_radius) * kernel_size + (j + kernel_radius);
      gaussian_kernel_[idx] = std::exp(-(i * i + j * j) / (2 * sigma * sigma));
      total_weight += gaussian_kernel_[idx];
    }
  }

  for (float &weight : gaussian_kernel_) {
    weight /= total_weight;
  }
}

int KapanovaSImageSmoothingMPI::clampValue(int value, int min_val, int max_val) {
  return std::max(min_val, std::min(value, max_val));
}

void KapanovaSImageSmoothingMPI::processPixel(int x, int y) {
  const int kernel_radius = 1;
  const int kernel_size = 2 * kernel_radius + 1;

  float pixel_sum = 0.0f;

  for (int ky = -kernel_radius; ky <= kernel_radius; ++ky) {
    for (int kx = -kernel_radius; kx <= kernel_radius; ++kx) {
      int pixel_x = clampValue(x + kx, 0, image_width_ - 1);
      int pixel_y = clampValue(y + ky, 0, image_height_ - 1);
      int kernel_idx = (ky + kernel_radius) * kernel_size + (kx + kernel_radius);

      pixel_sum += input_matrix_[pixel_y][pixel_x] * gaussian_kernel_[kernel_idx];
    }
  }

  output_matrix_[y][x] = static_cast<int>(pixel_sum);
}

bool KapanovaSImageSmoothingMPI::RunImpl() {
  if (input_matrix_.empty()) {
    return true;
  }

  int process_rank = mpi_communicator_.rank();
  int total_processes = mpi_communicator_.size();

  // Распределяем строки между процессами
  int rows_per_process = image_height_ / total_processes;
  int remainder_rows = image_height_ % total_processes;

  int start_row = process_rank * rows_per_process + std::min(process_rank, remainder_rows);
  int end_row = start_row + rows_per_process + (process_rank < remainder_rows ? 1 : 0);

  // Обрабатываем строки, назначенные текущему процессу
  for (int row = start_row; row < end_row; ++row) {
    for (int col = 0; col < image_width_; ++col) {
      processPixel(col, row);
    }
  }

  // Собираем результаты на процессе 0
  if (process_rank == 0) {
    // Получаем результаты от других процессов
    for (int proc = 1; proc < total_processes; ++proc) {
      int proc_start_row = proc * rows_per_process + std::min(proc, remainder_rows);
      int proc_end_row = proc_start_row + rows_per_process + (proc < remainder_rows ? 1 : 0);

      for (int row = proc_start_row; row < proc_end_row; ++row) {
        std::vector<int> row_data(image_width_);
        mpi_communicator_.recv(proc, row, row_data);
        output_matrix_[row] = row_data;
      }
    }
  } else {
    // Отправляем результаты на процесс 0
    for (int row = start_row; row < end_row; ++row) {
      mpi_communicator_.send(0, row, output_matrix_[row]);
    }
  }

  return true;
}

bool KapanovaSImageSmoothingMPI::PostProcessingImpl() {
  GetOutput() = output_matrix_;
  return true;
}

}  // namespace kapanova_s_image_smoothing
