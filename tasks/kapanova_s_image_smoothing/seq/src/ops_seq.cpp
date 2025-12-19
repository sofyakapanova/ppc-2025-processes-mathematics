#include "kapanova_s_image_smoothing/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace kapanova_s_image_smoothing {

KapanovaSImageSmoothingSEQ::KapanovaSImageSmoothingSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  if (!in.empty()) {
    GetInput() = in;
  } else {
    GetInput() = InType();
  }
  kernel_ = nullptr;
  width_ = 0;
  height_ = 0;
}

bool KapanovaSImageSmoothingSEQ::ValidationImpl() {
  const auto &input_data = GetInput();
  return !input_data.empty() && !input_data[0].empty();
}

bool KapanovaSImageSmoothingSEQ::PreProcessingImpl() {
  const auto &input_data = GetInput();
  if (input_data.empty() || input_data[0].size() < 4) {
    return false;
  }

  const auto &data = input_data[0];
  width_ = static_cast<int>((data[1] << 8) | data[0]);
  height_ = static_cast<int>((data[3] << 8) | data[2]);

  const auto expected_size = static_cast<size_t>(4 + (width_ * height_ * 3));
  if (data.size() < expected_size) {
    return false;
  }

  input_.assign(data.begin() + 4, data.end());
  result_ = std::vector<uint8_t>(static_cast<size_t>(width_ * height_ * 3));

  CreateKernel();
  return true;
}

void KapanovaSImageSmoothingSEQ::CreateKernel() {
  const int size = (2 * radius_) + 1;
  kernel_ = new float[static_cast<size_t>(size * size)]{0};
  constexpr float kSigma = 1.5F;
  float norm = 0.0F;

  for (int i = -radius_; i <= radius_; ++i) {
    for (int j = -radius_; j <= radius_; ++j) {
      const int kernel_index = ((i + radius_) * size) + j + radius_;
      kernel_[kernel_index] = std::exp(-(static_cast<float>((i * i) + (j * j))) / ((2.0F * kSigma) * kSigma));
      norm += kernel_[kernel_index];
    }
  }

  for (int i = 0; i < (size * size); ++i) {
    kernel_[i] /= norm;
  }
}

void KapanovaSImageSmoothingSEQ::SmoothPixel(int x, int y) {
  const int stride = width_ * 3;
  const auto sizek = static_cast<size_t>((2 * radius_) + 1);
  float out_r = 0.0F;
  float out_g = 0.0F;
  float out_b = 0.0F;

  auto clamp = [](int n, int lo, int hi) { return std::min(std::max(n, lo), hi); };

  for (int ry = -radius_; ry <= radius_; ++ry) {
    for (int rx = -radius_; rx <= radius_; ++rx) {
      const int idx = clamp(x + rx, 0, width_ - 1);
      const int idy = clamp(y + ry, 0, height_ - 1);
      const int pos = (idy * stride) + (idx * 3);
      const int kernel_pos = static_cast<int>(((ry + radius_) * static_cast<int>(sizek)) + rx + radius_);

      out_r += static_cast<float>(input_[static_cast<size_t>(pos)]) * kernel_[kernel_pos];
      out_g += static_cast<float>(input_[static_cast<size_t>(pos + 1)]) * kernel_[kernel_pos];
      out_b += static_cast<float>(input_[static_cast<size_t>(pos + 2)]) * kernel_[kernel_pos];
    }
  }

  const int pos = (y * stride) + (x * 3);
  result_[static_cast<size_t>(pos)] = static_cast<uint8_t>(out_r);
  result_[static_cast<size_t>(pos + 1)] = static_cast<uint8_t>(out_g);
  result_[static_cast<size_t>(pos + 2)] = static_cast<uint8_t>(out_b);
}

bool KapanovaSImageSmoothingSEQ::RunImpl() {
  for (int idx_y = 0; idx_y < height_; ++idx_y) {
    for (int idx_x = 0; idx_x < width_; ++idx_x) {
      SmoothPixel(idx_x, idx_y);
    }
  }
  return true;
}

bool KapanovaSImageSmoothingSEQ::PostProcessingImpl() {
  delete[] kernel_;
  kernel_ = nullptr;

  GetOutput() = result_;
  return true;
}

}  // namespace kapanova_s_image_smoothing
