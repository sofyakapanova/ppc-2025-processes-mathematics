#include "kapanova_s_image_smoothing/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "kapanova_s_image_smoothing/common/include/common.hpp"

namespace kapanova_s_image_smoothing {

KapanovaSImageSmoothingSEQ::KapanovaSImageSmoothingSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  if (!in.empty()) {
    GetInput() = in;
  } else {
    GetInput() = InType();
  }
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

  const size_t expected_size = 4 + static_cast<size_t>(width_ * height_ * 3);
  if (data.size() < expected_size) {
    return false;
  }

  input_.assign(data.begin() + 4, data.end());
  result_.resize(static_cast<size_t>(width_ * height_ * 3));

  CreateKernel();
  return true;
}

void KapanovaSImageSmoothingSEQ::CreateKernel() {
  const int size = (2 * radius_) + 1;
  kernel_.resize(static_cast<size_t>(size * size));
  constexpr float kSigma = 1.5F;
  float norm = 0.0F;

  for (int i = -radius_; i <= radius_; ++i) {
    for (int j = -radius_; j <= radius_; ++j) {
      const auto kernel_index = static_cast<size_t>(((i + radius_) * size) + j + radius_);
      kernel_[kernel_index] = std::exp(-(static_cast<float>((i * i) + (j * j))) / ((2.0F * kSigma) * kSigma));
      norm += kernel_[kernel_index];
    }
  }

  for (auto &value : kernel_) {
    value /= norm;
  }
}

void KapanovaSImageSmoothingSEQ::SmoothPixel(int x_coord, int y_coord) {
  const int stride = width_ * 3;
  const auto size_k = static_cast<size_t>((2 * radius_) + 1);
  float out_r = 0.0F;
  float out_g = 0.0F;
  float out_b = 0.0F;

  auto clamp = [](int n, int lo, int hi) { return std::min(std::max(n, lo), hi); };

  for (int ry = -radius_; ry <= radius_; ++ry) {
    for (int rx = -radius_; rx <= radius_; ++rx) {
      const int idx = clamp(x_coord + rx, 0, width_ - 1);
      const int idy = clamp(y_coord + ry, 0, height_ - 1);
      const int pos = (idy * stride) + (idx * 3);
      const size_t kernel_pos = static_cast<size_t>((ry + radius_) * static_cast<int>(size_k) + rx + radius_);

      out_r += static_cast<float>(input_[static_cast<size_t>(pos)]) * kernel_[kernel_pos];
      out_g += static_cast<float>(input_[static_cast<size_t>(pos + 1)]) * kernel_[kernel_pos];
      out_b += static_cast<float>(input_[static_cast<size_t>(pos + 2)]) * kernel_[kernel_pos];
    }
  }

  const int pos = (y_coord * stride) + (x_coord * 3);
  result_[static_cast<size_t>(pos)] = static_cast<uint8_t>(out_r);
  result_[static_cast<size_t>(pos + 1)] = static_cast<uint8_t>(out_g);
  result_[static_cast<size_t>(pos + 2)] = static_cast<uint8_t>(out_b);
}

bool KapanovaSImageSmoothingSEQ::RunImpl() {
  for (int y_coord = 0; y_coord < height_; ++y_coord) {
    for (int x_coord = 0; x_coord < width_; ++x_coord) {
      SmoothPixel(x_coord, y_coord);
    }
  }
  return true;
}

bool KapanovaSImageSmoothingSEQ::PostProcessingImpl() {
  kernel_.clear();
  kernel_.shrink_to_fit();

  GetOutput() = result_;
  return true;
}

}  // namespace kapanova_s_image_smoothing
