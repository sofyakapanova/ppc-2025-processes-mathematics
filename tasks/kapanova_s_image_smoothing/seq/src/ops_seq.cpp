#include "kapanova_s_image_smoothing/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace kapanova_s_image_smoothing {

KapanovaSImageSmoothingSEQ::KapanovaSImageSmoothingSEQ(const InType &in) : BaseTask() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KapanovaSImageSmoothingSEQ::ValidationImpl() {
  const auto &input_img = GetInput();

  if (input_img.width <= 0 || input_img.height <= 0) {
    return false;
  }

  size_t expected_size = static_cast<size_t>(input_img.width) * input_img.height;
  if (input_img.pixels.size() != expected_size) {
    return false;
  }

  return true;
}

bool KapanovaSImageSmoothingSEQ::PreProcessingImpl() {
  const auto &input_img = GetInput();
  img_width_ = input_img.width;
  img_height_ = input_img.height;

  input_data_ = input_img.pixels;
  output_data_.resize(input_data_.size());

  return true;
}

std::vector<float> KapanovaSImageSmoothingSEQ::CreateGaussianKernel(int kernel_radius, float sigma_val) {
  int kernel_size = 2 * kernel_radius + 1;
  std::vector<float> kernel_vals(kernel_size);
  float total_sum = 0.0f;

  for (int i = -kernel_radius; i <= kernel_radius; ++i) {
    kernel_vals[i + kernel_radius] = std::exp(-(i * i) / (2 * sigma_val * sigma_val));
    total_sum += kernel_vals[i + kernel_radius];
  }

  for (float &val : kernel_vals) {
    val /= total_sum;
  }

  return kernel_vals;
}

void KapanovaSImageSmoothingSEQ::ProcessRows(const std::vector<uint8_t> &input_img, int img_h, int img_w,
                                             const std::vector<float> &kernel, std::vector<float> &temp_buf) {
  int kernel_half = static_cast<int>(kernel.size()) / 2;

  for (int y = 0; y < img_h; ++y) {
    for (int x = 0; x < img_w; ++x) {
      float sum_val = 0.0f;
      for (int k = -kernel_half; k <= kernel_half; ++k) {
        int sample_x = x + k;
        if (sample_x < 0) {
          sample_x = 0;
        }
        if (sample_x >= img_w) {
          sample_x = img_w - 1;
        }
        sum_val += input_img[y * img_w + sample_x] * kernel[k + kernel_half];
      }
      temp_buf[y * img_w + x] = sum_val;
    }
  }
}

void KapanovaSImageSmoothingSEQ::ProcessColumns(const std::vector<float> &temp_buf, int img_h, int img_w,
                                                const std::vector<float> &kernel, std::vector<uint8_t> &output_img) {
  int kernel_half = static_cast<int>(kernel.size()) / 2;

  for (int y = 0; y < img_h; ++y) {
    for (int x = 0; x < img_w; ++x) {
      float sum_val = 0.0f;
      for (int k = -kernel_half; k <= kernel_half; ++k) {
        int sample_y = y + k;
        if (sample_y < 0) {
          sample_y = 0;
        }
        if (sample_y >= img_h) {
          sample_y = img_h - 1;
        }
        sum_val += temp_buf[sample_y * img_w + x] * kernel[k + kernel_half];
      }
      int result_val = static_cast<int>(std::round(sum_val));
      if (result_val < 0) {
        result_val = 0;
      }
      if (result_val > 255) {
        result_val = 255;
      }
      output_img[y * img_w + x] = static_cast<uint8_t>(result_val);
    }
  }
}

bool KapanovaSImageSmoothingSEQ::RunImpl() {
  const int kernel_radius = 1;
  const float sigma_val = 1.5f;

  std::vector<float> row_kernel = CreateGaussianKernel(kernel_radius, sigma_val);
  const std::vector<float> &col_kernel = row_kernel;

  std::vector<float> temp_buffer(img_width_ * img_height_, 0.0f);

  ProcessRows(input_data_, img_height_, img_width_, row_kernel, temp_buffer);
  ProcessColumns(temp_buffer, img_height_, img_width_, col_kernel, output_data_);

  return true;
}

bool KapanovaSImageSmoothingSEQ::PostProcessingImpl() {
  GetOutput().pixels = output_data_;
  GetOutput().width = img_width_;
  GetOutput().height = img_height_;
  GetOutput().kernel_size = 3;

  return true;
}

}  // namespace kapanova_s_image_smoothing
