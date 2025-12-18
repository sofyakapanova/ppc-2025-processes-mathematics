#include "kapanova_s_image_smoothing/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "kapanova_s_image_smoothing/common/include/common.hpp"

namespace kapanova_s_image_smoothing {

KapanovaSImageSmoothingMPI::KapanovaSImageSmoothingMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KapanovaSImageSmoothingMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Собираем параметры на процессе 0 и рассылаем всем
  int params[4] = {0, 0, 0, 0};  // valid, width, height, kernel_size

  if (rank == 0) {
    auto &input = GetInput();

    params[1] = input.width;
    params[2] = input.height;
    params[3] = input.kernel_size;

    // Проверяем валидность
    if (input.width > 0 && input.height > 0 && input.kernel_size > 0 && input.kernel_size % 2 == 1 &&
        input.pixels.size() == static_cast<size_t>(input.width) * static_cast<size_t>(input.height)) {
      params[0] = 1;
    }
  }

  // Рассылаем параметры всем процессам
  MPI_Bcast(params, 4, MPI_INT, 0, MPI_COMM_WORLD);

  // Обновляем параметры на всех процессах
  GetInput().width = params[1];
  GetInput().height = params[2];
  GetInput().kernel_size = params[3];

  return params[0] == 1;
}

bool KapanovaSImageSmoothingMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Инициализируем выходные данные
  GetOutput() = GetInput();

  // Важно: на всех процессах должны быть правильные размеры выходных данных
  const int width = GetInput().width;
  const int height = GetInput().height;
  const size_t total_size = static_cast<size_t>(width) * height;

  if (rank == 0) {
    GetOutput().pixels.resize(total_size, 0);
  } else {
    // На других процессах также нужно выделить память, но она будет пустой
    GetOutput().pixels.clear();
  }

  return true;
}

namespace {

// Вспомогательные функции
struct ProcessRange {
  int start_row;
  int end_row;
  int total_rows;
};

ProcessRange CalculateRowRange(int rank, int size, int total_height) {
  ProcessRange range;

  // Базовое количество строк на процесс
  int base_rows = total_height / size;
  int extra_rows = total_height % size;

  // Определяем строки для текущего процесса
  range.start_row = rank * base_rows + std::min(rank, extra_rows);
  range.end_row = range.start_row + base_rows + (rank < extra_rows ? 1 : 0);
  range.total_rows = range.end_row - range.start_row;

  return range;
}

void DistributeData(int rank, int size, const std::vector<uint8_t> &all_data, std::vector<uint8_t> &local_data,
                    int width, int height, int kernel_radius) {
  // Рассчитываем, какие строки нужны каждому процессу (с учетом перекрытия для ядра)
  std::vector<int> send_counts(size, 0);
  std::vector<int> displacements(size, 0);

  if (rank == 0) {
    for (int proc = 0; proc < size; ++proc) {
      ProcessRange range = CalculateRowRange(proc, size, height);

      // Добавляем перекрытие для ядра сверху и снизу
      int actual_start = std::max(0, range.start_row - kernel_radius);
      int actual_end = std::min(height, range.end_row + kernel_radius);

      send_counts[proc] = (actual_end - actual_start) * width;
      displacements[proc] = actual_start * width;
    }
  }

  // Получаем информацию о своей части данных
  ProcessRange my_range = CalculateRowRange(rank, size, height);
  int my_start = std::max(0, my_range.start_row - kernel_radius);
  int my_end = std::min(height, my_range.end_row + kernel_radius);
  int my_count = (my_end - my_start) * width;

  // Выделяем память для локальных данных
  local_data.resize(my_count);

  // Распределяем данные
  MPI_Scatterv(rank == 0 ? all_data.data() : nullptr, send_counts.data(), displacements.data(), MPI_UNSIGNED_CHAR,
               local_data.data(), my_count, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
}

uint8_t SmoothPixel(int global_x, int global_y, int width, int height, int kernel_radius, int local_start_row,
                    const std::vector<uint8_t> &local_data) {
  int sum = 0;
  int count = 0;

  // Проходим по окрестности пикселя
  for (int dy = -kernel_radius; dy <= kernel_radius; ++dy) {
    for (int dx = -kernel_radius; dx <= kernel_radius; ++dx) {
      int neighbor_x = global_x + dx;
      int neighbor_y = global_y + dy;

      // Проверяем границы изображения
      if (neighbor_x >= 0 && neighbor_x < width && neighbor_y >= 0 && neighbor_y < height) {
        // Преобразуем глобальные координаты в локальные
        int local_y = neighbor_y - local_start_row;
        int index = local_y * width + neighbor_x;

        sum += local_data[index];
        ++count;
      }
    }
  }

  return count > 0 ? static_cast<uint8_t>(sum / count) : 0;
}

void CollectResults(int rank, int size, std::vector<uint8_t> &local_result, std::vector<uint8_t> &all_result, int width,
                    int height) {
  // Рассчитываем, сколько данных каждый процесс должен отправить
  std::vector<int> recv_counts(size, 0);
  std::vector<int> displacements(size, 0);

  if (rank == 0) {
    for (int proc = 0; proc < size; ++proc) {
      ProcessRange range = CalculateRowRange(proc, size, height);
      recv_counts[proc] = range.total_rows * width;
      displacements[proc] = range.start_row * width;
    }
  }

  // Собираем результаты
  MPI_Gatherv(local_result.data(), static_cast<int>(local_result.size()), MPI_UNSIGNED_CHAR,
              rank == 0 ? all_result.data() : nullptr, recv_counts.data(), displacements.data(), MPI_UNSIGNED_CHAR, 0,
              MPI_COMM_WORLD);

  // Синхронизируем процессы
  MPI_Barrier(MPI_COMM_WORLD);
}

}  // namespace

bool KapanovaSImageSmoothingMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto &input = GetInput();
  auto &output = GetOutput();

  const int width = input.width;
  const int height = input.height;
  const int kernel_radius = input.kernel_size / 2;

  // 1. Распределяем данные между процессами
  std::vector<uint8_t> local_data;
  DistributeData(rank, size, input.pixels, local_data, width, height, kernel_radius);

  // 2. Определяем диапазон строк для обработки
  ProcessRange my_range = CalculateRowRange(rank, size, height);

  // Если у процесса нет строк для обработки, просто возвращаемся
  if (my_range.total_rows <= 0) {
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  int local_start_row = std::max(0, my_range.start_row - kernel_radius);

  // 3. Обрабатываем локальные строки
  std::vector<uint8_t> local_result(static_cast<size_t>(my_range.total_rows) * width);

  for (int local_row = 0; local_row < my_range.total_rows; ++local_row) {
    int global_row = my_range.start_row + local_row;

    for (int col = 0; col < width; ++col) {
      int index = local_row * width + col;
      local_result[index] = SmoothPixel(col, global_row, width, height, kernel_radius, local_start_row, local_data);
    }
  }

  // 4. Собираем результаты на процессе 0
  CollectResults(rank, size, local_result, output.pixels, width, height);

  // 5. На всех процессах должны быть правильные размеры выходных данных
  if (rank == 0) {
    // На процессе 0 данные уже собраны
    output.width = width;
    output.height = height;
    output.kernel_size = input.kernel_size;
  } else {
    // На других процессах очищаем пиксели, но сохраняем метаданные
    output.pixels.clear();
    output.width = width;
    output.height = height;
    output.kernel_size = input.kernel_size;
  }

  return true;
}

bool KapanovaSImageSmoothingMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // На процессе 0 должны быть данные, на других - только метаданные
  if (rank == 0) {
    return !GetOutput().pixels.empty();
  } else {
    return GetOutput().width > 0 && GetOutput().height > 0 && GetOutput().kernel_size > 0;
  }
}

}  // namespace kapanova_s_image_smoothing
