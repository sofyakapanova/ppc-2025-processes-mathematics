#include "kapanova_s_image_smoothing/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <vector>

namespace kapanova_s_image_smoothing {

KapanovaSImageSmoothingMPI::KapanovaSImageSmoothingMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KapanovaSImageSmoothingMPI::ValidationImpl() {
  const auto &inputData = GetInput();
  return !inputData.empty() && !inputData[0].empty();
}

bool KapanovaSImageSmoothingMPI::PreProcessingImpl() {
  const auto &inputData = GetInput();
  if (inputData.empty() || inputData[0].size() < 4) {
    return false;
  }

  // Первые 4 элемента - ширина и высота (по 2 байта каждое)
  const auto &data = inputData[0];
  width = (data[1] << 8) | data[0];
  height = (data[3] << 8) | data[2];

  // Проверяем размер данных
  size_t required_pixels = static_cast<size_t>(width) * static_cast<size_t>(height) * 3;
  size_t total_required_size = 4 + required_pixels;

  if (data.size() < total_required_size) {
    return false;
  }

  input.assign(data.begin() + 4, data.end());
  result = std::vector<uint8_t>(required_pixels);
  kernel = CreateKernel();  // Теперь kernel - это std::vector<float>

  return true;
}

std::vector<float> KapanovaSImageSmoothingMPI::CreateKernel() {
  int size = 2 * radius + 1;
  std::vector<float> kernel_local(size * size, 0.0f);
  float sigma = 1.5f;
  float norm = 0;

  for (int i = -radius; i <= radius; i++) {
    for (int j = -radius; j <= radius; j++) {
      kernel_local[static_cast<size_t>((i + radius) * size + j + radius)] =
          std::exp(-(i * i + j * j) / (2 * sigma * sigma));
      norm += kernel_local[static_cast<size_t>((i + radius) * size + j + radius)];
    }
  }

  for (int i = 0; i < size * size; i++) {
    kernel_local[static_cast<size_t>(i)] /= norm;
  }

  return kernel_local;
}

void KapanovaSImageSmoothingMPI::SmoothPixel(uint8_t *out, int x, int y) {
  int stride = width * 3;
  size_t sizek = static_cast<size_t>(2 * radius + 1);
  float outR = 0.0f;
  float outG = 0.0f;
  float outB = 0.0f;

  auto clamp = [](int n, int lo, int hi) { return std::min(std::max(n, lo), hi); };

  for (int ry = -radius; ry <= radius; ry++) {
    for (int rx = -radius; rx <= radius; rx++) {
      int idX = clamp(x + rx, 0, width - 1);
      int idY = clamp(y + ry, 0, height - 1);
      int pos = idY * stride + idX * 3;
      int kernelPos = static_cast<int>((ry + radius) * sizek + rx + radius);

      outR += static_cast<float>(input[static_cast<size_t>(pos)]) * kernel[static_cast<size_t>(kernelPos)];
      outG += static_cast<float>(input[static_cast<size_t>(pos + 1)]) * kernel[static_cast<size_t>(kernelPos)];
      outB += static_cast<float>(input[static_cast<size_t>(pos + 2)]) * kernel[static_cast<size_t>(kernelPos)];
    }
  }

  out[0] = static_cast<uint8_t>(outR);
  out[1] = static_cast<uint8_t>(outG);
  out[2] = static_cast<uint8_t>(outB);
}

bool KapanovaSImageSmoothingMPI::RunImpl() {
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Определяем теги для MPI сообщений
  constexpr int TAG_EXIT = 0;
  constexpr int TAG_INFO = 1;
  constexpr int TAG_DATA = 2;
  constexpr int TAG_RESULT = 3;

  if (size == 1) {
    // Последовательная обработка если только 1 процесс
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        SmoothPixel(&result[static_cast<size_t>(y * width * 3 + x * 3)], x, y);
      }
    }
    return true;
  }

  if (rank == 0) {
    // Процесс 0 - координатор
    int satellites = size - 1;
    int escape = 0;
    int noescape = 1;

    // Отправляем ширину изображения всем процессам
    for (int i = 1; i <= satellites; i++) {
      MPI_Send(&width, 1, MPI_INT, i, TAG_INFO, MPI_COMM_WORLD);
    }

    // Распределяем строки изображения
    int row = 0;
    while (row < height - 2) {
      // Отправляем данные доступным процессам
      int processes_to_use = std::min(satellites, height - 2 - row);

      for (int i = 0; i < processes_to_use; i++) {
        MPI_Send(&noescape, 1, MPI_INT, i + 1, TAG_EXIT, MPI_COMM_WORLD);

        // Отправляем 3 строки: текущая и по одной сверху и снизу
        // Для каждой отправки отправляем 3 строки
        if (row + i == 0) {
          // Первая строка - отправляем 2 строки
          MPI_Send(&input[0], 2 * width * 3, MPI_UNSIGNED_CHAR, i + 1, TAG_DATA, MPI_COMM_WORLD);
        } else if (row + i == height - 2) {
          // Предпоследняя строка - отправляем 2 строки
          int start_pos = (height - 2) * width * 3;
          MPI_Send(&input[static_cast<size_t>(start_pos)], 2 * width * 3, MPI_UNSIGNED_CHAR, i + 1, TAG_DATA,
                   MPI_COMM_WORLD);
        } else {
          // Обычный случай - отправляем 3 строки
          int start_pos = (row + i - 1) * width * 3;
          MPI_Send(&input[static_cast<size_t>(start_pos)], 3 * width * 3, MPI_UNSIGNED_CHAR, i + 1, TAG_DATA,
                   MPI_COMM_WORLD);
        }
      }

      // Получаем результаты
      for (int i = 0; i < processes_to_use; i++) {
        int result_row = row + i + 1;
        if (result_row > 0 && result_row < height - 1) {
          int result_pos = result_row * width * 3;
          MPI_Recv(&result[static_cast<size_t>(result_pos)], width * 3, MPI_UNSIGNED_CHAR, i + 1, TAG_RESULT,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
      }

      row += processes_to_use;
    }

    // Отправляем сигнал завершения
    for (int i = 1; i <= satellites; i++) {
      MPI_Send(&escape, 1, MPI_INT, i, TAG_EXIT, MPI_COMM_WORLD);
    }

    // Обрабатываем первую и последнюю строку
    for (int x = 0; x < width; x++) {
      SmoothPixel(&result[static_cast<size_t>(x * 3)], x, 0);
      SmoothPixel(&result[static_cast<size_t>((height - 1) * width * 3 + x * 3)], x, height - 1);
    }

  } else {
    // Вспомогательные процессы
    int local_width = 0;
    MPI_Recv(&local_width, 1, MPI_INT, 0, TAG_INFO, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    std::vector<uint8_t> local_input;
    std::vector<uint8_t> local_result(static_cast<size_t>(local_width * 3));  // Одна строка результата

    int escape = 0;

    while (true) {
      MPI_Recv(&escape, 1, MPI_INT, 0, TAG_EXIT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (escape == 0) {
        break;
      }

      // Получаем данные - размер зависит от позиции строки
      MPI_Status status;
      MPI_Probe(0, TAG_DATA, MPI_COMM_WORLD, &status);
      int count = 0;
      MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &count);

      local_input.resize(static_cast<size_t>(count));
      MPI_Recv(local_input.data(), count, MPI_UNSIGNED_CHAR, 0, TAG_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      // Определяем, сколько строк получили
      int rows_received = count / (local_width * 3);

      // Обрабатываем среднюю строку (если получили 3 строки)
      if (rows_received == 3) {
        // Обрабатываем среднюю строку (индекс 1)
        for (int x = 0; x < local_width; x++) {
          SmoothPixel(&local_result[static_cast<size_t>(x * 3)], x, 1);
        }

        // Отправляем результат
        MPI_Send(local_result.data(), local_width * 3, MPI_UNSIGNED_CHAR, 0, TAG_RESULT, MPI_COMM_WORLD);
      } else if (rows_received == 2) {
        // Для краевых случаев обрабатываем первую строку
        for (int x = 0; x < local_width; x++) {
          SmoothPixel(&local_result[static_cast<size_t>(x * 3)], x, 0);
        }

        // Отправляем результат
        MPI_Send(local_result.data(), local_width * 3, MPI_UNSIGNED_CHAR, 0, TAG_RESULT, MPI_COMM_WORLD);
      }
    }
  }

  return true;
}

bool KapanovaSImageSmoothingMPI::PostProcessingImpl() {
  // Теперь не нужно удалять kernel, так как это std::vector
  // vector автоматически очистится при разрушении объекта
  GetOutput() = result;
  return true;
}

}  // namespace kapanova_s_image_smoothing
