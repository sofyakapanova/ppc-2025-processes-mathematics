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
  
  if (input.width <= 0 || input.height <= 0) {
    return false;
  }
  
  if (input.pixels.size() != static_cast<size_t>(input.width) * input.height) {
    return false;
  }
  
  return true;
}

bool KapanovaSImageSmoothingSEQ::PreProcessingImpl() {
  auto &input = GetInput();
  width_ = input.width;
  height_ = input.height;
  
  input_ = input.pixels;
  
  GetOutput() = input;
  output_.resize(input_.size());
  
  return true;
}

std::vector<float> KapanovaSImageSmoothingSEQ::create_gaussian_kernel(int radius, float sigma) {
  int size = 2 * radius + 1;
  std::vector<float> kernel(size * size);
  float norm = 0.0f;
  
  for (int i = -radius; i <= radius; ++i) {
    for (int j = -radius; j <= radius; ++j) {
      kernel[(i + radius) * size + (j + radius)] = std::exp(-(i * i + j * j) / (2 * sigma * sigma));
      norm += kernel[(i + radius) * size + (j + radius)];
    }
  }
  
  for (float& val : kernel) {
    val /= norm;
  }
  
  return kernel;
}

void KapanovaSImageSmoothingSEQ::convolve_rows(const std::vector<uint8_t>& input, int height, int width,
                                              const std::vector<float>& kernel, std::vector<float>& temp) {
  int kernel_radius = 1;
  
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float sum = 0.0f;
      for (int k = -kernel_radius; k <= kernel_radius; ++k) {
        int pixel_x = std::clamp(x + k, 0, width - 1);
        sum += input[y * width + pixel_x] * kernel[k + kernel_radius];
      }
      temp[y * width + x] = sum;
    }
  }
}

void KapanovaSImageSmoothingSEQ::convolve_columns(const std::vector<float>& temp, int height, int width,
                                                 const std::vector<float>& kernel, std::vector<uint8_t>& output) {
  int kernel_radius = 1;
  
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float sum = 0.0f;
      for (int k = -kernel_radius; k <= kernel_radius; ++k) {
        int pixel_y = std::clamp(y + k, 0, height - 1);
        sum += temp[pixel_y * width + x] * kernel[k + kernel_radius];
      }
      output[y * width + x] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(sum)), 0, 255));
    }
  }
}

bool KapanovaSImageSmoothingSEQ::RunImpl() {
  const int radius = 1;
  const float sigma = 1.5f;
  
  std::vector<float> horizontal_kernel = create_gaussian_kernel(radius, sigma);
  const std::vector<float>& vertical_kernel = horizontal_kernel;
  
  std::vector<float> temp(width_ * height_, 0.0f);
  
  convolve_rows(input_, height_, width_, horizontal_kernel, temp);
  convolve_columns(temp, height_, width_, vertical_kernel, output_);
  
  return true;
}

bool KapanovaSImageSmoothingSEQ::PostProcessingImpl() {
  GetOutput().pixels = output_;
  return true;
}

}  // namespace kapanova_s_image_smoothing