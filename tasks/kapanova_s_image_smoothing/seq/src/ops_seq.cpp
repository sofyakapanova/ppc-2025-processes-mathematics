#include "kapanova_s_image_smoothing/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace kapanova_s_image_smoothing {

KapanovaSImageSmoothingSEQ::KapanovaSImageSmoothingSEQ(const InType &in) : BaseTask() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KapanovaSImageSmoothingSEQ::ValidationImpl() {
  auto &input = GetInput();

  if (input.width <= 0 || input.height <= 0 || input.kernel_size <= 0) {
    return false;
  }

  if (input.kernel_size % 2 == 0) {
    return false;
  }

  if (input.pixels.size() != static_cast<size_t>(input.width) * input.height) {
    return false;
  }

  return true;
}

bool KapanovaSImageSmoothingSEQ::PreProcessingImpl() {
  GetOutput() = GetInput();
  GetOutput().pixels.resize(GetInput().pixels.size());
  return true;
}

std::vector<float> KapanovaSImageSmoothingSEQ::CreateGaussianKernel() {
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

bool KapanovaSImageSmoothingSEQ::RunImpl() {
  const auto &input = GetInput();
  auto &output = GetOutput();

  const int width = input.width;
  const int height = input.height;
  const int kernel_radius = input.kernel_size / 2;
  const int kernel_size = input.kernel_size;

  // Создаем Гауссово ядро
  std::vector<float> kernel = CreateGaussianKernel();

  // Применяем фильтр к каждому пикселю
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float result = 0.0f;

      for (int ky = -kernel_radius; ky <= kernel_radius; ++ky) {
        for (int kx = -kernel_radius; kx <= kernel_radius; ++kx) {
          const int neighbor_x = std::clamp(x + kx, 0, width - 1);
          const int neighbor_y = std::clamp(y + ky, 0, height - 1);

          const float weight = kernel[(ky + kernel_radius) * kernel_size + (kx + kernel_radius)];
          result += input.pixels[neighbor_y * width + neighbor_x] * weight;
        }
      }

      output.pixels[y * width + x] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(result)), 0, 255));
    }
  }

  return true;
}

bool KapanovaSImageSmoothingSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kapanova_s_image_smoothing
