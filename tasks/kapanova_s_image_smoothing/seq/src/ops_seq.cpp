#include "kapanova_s_image_smoothing/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace kapanova_s_image_smoothing {

KapanovaSImageSmoothingSEQ::KapanovaSImageSmoothingSEQ(const std::vector<std::vector<int>> &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = in;
}

bool KapanovaSImageSmoothingSEQ::ValidationImpl() {
  const auto &matrix = GetInput();
  if (matrix.empty()) {
    return true;
  }

  const size_t cols = matrix[0].size();
  return std::ranges::all_of(matrix, [cols](const auto &row) { 
    return row.size() == cols; 
  });
}

bool KapanovaSImageSmoothingSEQ::PreProcessingImpl() {
  input_image_ = GetInput();
  if (input_image_.empty()) {
    return true;
  }
  
  img_height_ = static_cast<int>(input_image_.size());
  img_width_ = static_cast<int>(input_image_[0].size());
  
  // Инициализируем выходное изображение
  result_image_.resize(img_height_);
  for (int i = 0; i < img_height_; ++i) {
    result_image_[i].resize(img_width_);
  }
  
  // Создаем гауссово ядро
  generateGaussianKernel();
  
  return true;
}

void KapanovaSImageSmoothingSEQ::generateGaussianKernel() {
  const int filter_radius = 1;
  const int filter_size = 2 * filter_radius + 1;
  gaussian_filter_.resize(filter_size * filter_size);
  float sigma_param = 1.5f;
  float normalization = 0.0f;

  for (int i = -filter_radius; i <= filter_radius; ++i) {
    for (int j = -filter_radius; j <= filter_radius; ++j) {
      int idx = (i + filter_radius) * filter_size + (j + filter_radius);
      gaussian_filter_[idx] = std::exp(-(i * i + j * j) / (2 * sigma_param * sigma_param));
      normalization += gaussian_filter_[idx];
    }
  }

  for (float& val : gaussian_filter_) {
    val /= normalization;
  }
}

int KapanovaSImageSmoothingSEQ::limitToRange(int value, int lower, int upper) {
  return std::max(lower, std::min(value, upper));
}

void KapanovaSImageSmoothingSEQ::smoothPixel(int x, int y) {
  const int filter_radius = 1;
  const int filter_size = 2 * filter_radius + 1;
  
  float pixel_sum = 0.0f;
  
  for (int row_offset = -filter_radius; row_offset <= filter_radius; ++row_offset) {
    for (int col_offset = -filter_radius; col_offset <= filter_radius; ++col_offset) {
      int pixel_x = limitToRange(x + col_offset, 0, img_width_ - 1);
      int pixel_y = limitToRange(y + row_offset, 0, img_height_ - 1);
      int kernel_index = (row_offset + filter_radius) * filter_size + (col_offset + filter_radius);
      
      float weight = gaussian_filter_[kernel_index];
      pixel_sum += input_image_[pixel_y][pixel_x] * weight;
    }
  }
  
  // Записываем результат
  result_image_[y][x] = static_cast<int>(pixel_sum);
}

bool KapanovaSImageSmoothingSEQ::RunImpl() {
  if (input_image_.empty()) {
    return true;
  }

  for (int y = 0; y < img_height_; ++y) {
    for (int x = 0; x < img_width_; ++x) {
      smoothPixel(x, y);
    }
  }
  
  return true;
}

bool KapanovaSImageSmoothingSEQ::PostProcessingImpl() {
  GetOutput() = result_image_;
  return true;
}

}  // namespace kapanova_s_image_smoothing