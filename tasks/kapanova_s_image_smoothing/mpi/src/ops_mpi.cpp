#include "kapanova_s_image_smoothing/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace kapanova_s_image_smoothing {

KapanovaSImageSmoothingMPI::KapanovaSImageSmoothingMPI(const InType &in) : BaseTask() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KapanovaSImageSmoothingMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  auto &input = GetInput();
  
  // Процесс 0 проверяет данные
  if (rank == 0) {
    if (input.width <= 0 || input.height <= 0 || input.kernel_size <= 0) {
      return false;
    }
    
    if (input.kernel_size % 2 == 0) {
      return false;  // Ядро должно быть нечетного размера
    }
    
    if (input.pixels.size() != static_cast<size_t>(input.width) * input.height) {
      return false;
    }
  }
  
  // Рассылаем результат проверки всем процессам
  int valid = 1;
  if (rank == 0) {
    valid = 1;  // Если дошли сюда, данные валидны
  }
  MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  return valid == 1;
}

bool KapanovaSImageSmoothingMPI::PreProcessingImpl() {
  auto &output = GetOutput();
  output = GetInput();  // Копируем метаданные
  
  // На процессе 0 выделяем память для результата
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if (rank == 0) {
    output.pixels.resize(static_cast<size_t>(output.width) * output.height);
  }
  
  return true;
}

std::vector<float> KapanovaSImageSmoothingMPI::CreateGaussianKernel() {
  const auto &input = GetInput();
  const int kernel_radius = input.kernel_size / 2;
  const int kernel_size = input.kernel_size;
  const float sigma = 1.5f;
  
  std::vector<float> kernel(kernel_size * kernel_size);
  float sum = 0.0f;
  
  for (int y = -kernel_radius; y <= kernel_radius; ++y) {
    for (int x = -kernel_radius; x <= kernel_radius; ++x) {
      const float value = std::exp(-(x * x + y * y) / (2 * sigma * sigma));
      kernel[(y + kernel_radius) * kernel_size + (x + kernel_radius)] = value;
      sum += value;
    }
  }
  
  // Нормализуем
  for (float &value : kernel) {
    value /= sum;
  }
  
  return kernel;
}

uint8_t KapanovaSImageSmoothingMPI::ApplyGaussianFilter(int x, int y, 
                                                       const std::vector<uint8_t>& local_data,
                                                       int local_width, int local_height,
                                                       int offset_row, 
                                                       const std::vector<float>& kernel) {
  const auto &input = GetInput();
  const int kernel_radius = input.kernel_size / 2;
  const int kernel_size = input.kernel_size;
  
  float result = 0.0f;
  
  for (int ky = -kernel_radius; ky <= kernel_radius; ++ky) {
    for (int kx = -kernel_radius; kx <= kernel_radius; ++kx) {
      const int neighbor_x = x + kx;
      const int neighbor_y = y + ky;
      
      // Проверяем границы исходного изображения
      if (neighbor_x >= 0 && neighbor_x < input.width &&
          neighbor_y >= 0 && neighbor_y < input.height) {
        
        // Вычисляем индекс в локальных данных
        const int local_y = neighbor_y - offset_row;
        if (local_y >= 0 && local_y < local_height) {
          const int local_idx = local_y * local_width + neighbor_x;
          const float weight = kernel[(ky + kernel_radius) * kernel_size + (kx + kernel_radius)];
          result += local_data[local_idx] * weight;
        }
      }
    }
  }
  
  return static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(result)), 0, 255));
}

bool KapanovaSImageSmoothingMPI::RunImpl() {
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  const auto &input = GetInput();
  auto &output = GetOutput();
  
  const int width = input.width;
  const int height = input.height;
  const int kernel_radius = input.kernel_size / 2;
  
  // Создаем Гауссово ядро
  std::vector<float> kernel;
  if (rank == 0) {
    kernel = CreateGaussianKernel();
  }
  
  // Рассылаем размеры ядра и само ядро
  int kernel_size = input.kernel_size;
  MPI_Bcast(&kernel_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  if (rank != 0) {
    kernel.resize(kernel_size * kernel_size);
  }
  MPI_Bcast(kernel.data(), kernel_size * kernel_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
  
  // Распределяем строки между процессами
  const int rows_per_proc = height / size;
  const int remainder = height % size;
  
  // Вычисляем диапазон строк для каждого процесса
  int start_row = rank * rows_per_proc + std::min(rank, remainder);
  int end_row = start_row + rows_per_proc + (rank < remainder ? 1 : 0);
  int local_rows = end_row - start_row;
  
  // Добавляем halo области для корректной обработки границ
  int halo_start = std::max(0, start_row - kernel_radius);
  int halo_end = std::min(height, end_row + kernel_radius);
  int halo_rows = halo_end - halo_start;
  
  // Подготавливаем данные для Scatterv
  std::vector<int> send_counts(size, 0);
  std::vector<int> displacements(size, 0);
  
  if (rank == 0) {
    for (int proc = 0; proc < size; ++proc) {
      int proc_start = proc * rows_per_proc + std::min(proc, remainder);
      int proc_end = proc_start + rows_per_proc + (proc < remainder ? 1 : 0);
      
      int proc_halo_start = std::max(0, proc_start - kernel_radius);
      int proc_halo_end = std::min(height, proc_end + kernel_radius);
      int proc_halo_rows = proc_halo_end - proc_halo_start;
      
      send_counts[proc] = proc_halo_rows * width;
      displacements[proc] = proc_halo_start * width;
    }
  }
  
  // Распределяем данные
  std::vector<uint8_t> local_data(halo_rows * width);
  MPI_Scatterv(rank == 0 ? input.pixels.data() : nullptr,
               send_counts.data(), displacements.data(), MPI_UNSIGNED_CHAR,
               local_data.data(), halo_rows * width, MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);
  
  // Если у процесса нет строк для обработки
  if (local_rows <= 0) {
    // Отправляем пустые данные для сбора
    std::vector<int> recv_counts(size, 0);
    std::vector<int> recv_displacements(size, 0);
    
    if (rank == 0) {
      for (int proc = 0; proc < size; ++proc) {
        int proc_start = proc * rows_per_proc + std::min(proc, remainder);
        int proc_end = proc_start + rows_per_proc + (proc < remainder ? 1 : 0);
        int proc_rows = proc_end - proc_start;
        
        recv_counts[proc] = proc_rows * width;
        recv_displacements[proc] = proc_start * width;
      }
    }
    
    MPI_Gatherv(nullptr, 0, MPI_UNSIGNED_CHAR,
                rank == 0 ? output.pixels.data() : nullptr,
                recv_counts.data(), recv_displacements.data(), MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);
    
    return true;
  }
  
  // Обрабатываем локальные данные
  std::vector<uint8_t> local_result(local_rows * width);
  
  for (int local_y = 0; local_y < local_rows; ++local_y) {
    const int global_y = start_row + local_y;
    
    for (int x = 0; x < width; ++x) {
      local_result[local_y * width + x] = 
          ApplyGaussianFilter(x, global_y, local_data, width, halo_rows, 
                             halo_start, kernel);
    }
  }
  
  // Собираем результаты
  std::vector<int> recv_counts(size, 0);
  std::vector<int> recv_displacements(size, 0);
  
  if (rank == 0) {
    for (int proc = 0; proc < size; ++proc) {
      int proc_start = proc * rows_per_proc + std::min(proc, remainder);
      int proc_end = proc_start + rows_per_proc + (proc < remainder ? 1 : 0);
      int proc_rows = proc_end - proc_start;
      
      recv_counts[proc] = proc_rows * width;
      recv_displacements[proc] = proc_start * width;
    }
  }
  
  MPI_Gatherv(local_result.data(), local_rows * width, MPI_UNSIGNED_CHAR,
              rank == 0 ? output.pixels.data() : nullptr,
              recv_counts.data(), recv_displacements.data(), MPI_UNSIGNED_CHAR,
              0, MPI_COMM_WORLD);
  
  return true;
}

bool KapanovaSImageSmoothingMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if (rank == 0) {
    auto &output = GetOutput();
    // Проверяем, что результат имеет правильный размер
    return output.pixels.size() == static_cast<size_t>(output.width) * output.height;
  }
  
  return true;
}

}  // namespace kapanova_s_image_smoothing