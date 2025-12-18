#include "kapanova_s_image_smoothing/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "kapanova_s_image_smoothing/common/include/common.hpp"

namespace kapanova_s_image_smoothing {

KapanovaSImageSmoothingSEQ::KapanovaSImageSmoothingSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KapanovaSImageSmoothingSEQ::ValidationImpl() {
  // Получаем ссылку на входные данные (не константную)
  auto &input = GetInput();

  // Проверяем, что все параметры валидны
  if (input.width <= 0 || input.height <= 0 || input.kernel_size <= 0) {
    return false;
  }

  // Размер ядра должен быть нечетным
  if (input.kernel_size % 2 == 0) {
    return false;
  }

  // Проверяем размер данных
  const size_t expected_size = static_cast<size_t>(input.width) * static_cast<size_t>(input.height);
  if (input.pixels.size() != expected_size) {
    return false;
  }

  return true;
}

bool KapanovaSImageSmoothingSEQ::PreProcessingImpl() {
  // Инициализируем выходные данные
  GetOutput() = GetInput();
  GetOutput().pixels.assign(GetInput().pixels.size(), 0);
  return true;
}

// Статическая функция для вычисления сглаженного значения пикселя
static uint8_t ComputePixelAverage(int x, int y, int width, int height, int kernel_radius,
                                   const std::vector<uint8_t> &pixels) {
  int sum = 0;
  int count = 0;

  // Проходим по окрестности пикселя
  for (int dy = -kernel_radius; dy <= kernel_radius; ++dy) {
    for (int dx = -kernel_radius; dx <= kernel_radius; ++dx) {
      int neighbor_x = x + dx;
      int neighbor_y = y + dy;

      // Проверяем границы изображения
      if (neighbor_x >= 0 && neighbor_x < width && neighbor_y >= 0 && neighbor_y < height) {
        int index = neighbor_y * width + neighbor_x;
        sum += pixels[index];
        ++count;
      }
    }
  }

  // Вычисляем среднее значение
  return count > 0 ? static_cast<uint8_t>(sum / count) : 0;
}

bool KapanovaSImageSmoothingSEQ::RunImpl() {
  // Получаем ссылки на данные (не константные)
  auto &input = GetInput();
  auto &output = GetOutput();

  const int width = input.width;
  const int height = input.height;
  const int kernel_radius = input.kernel_size / 2;

  const int total_pixels = width * height;

  // Обрабатываем каждый пиксель
  for (int i = 0; i < total_pixels; ++i) {
    int y = i / width;
    int x = i % width;

    // Используем статическую функцию
    output.pixels[i] = ComputePixelAverage(x, y, width, height, kernel_radius, input.pixels);
  }

  return true;
}

bool KapanovaSImageSmoothingSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kapanova_s_image_smoothing
