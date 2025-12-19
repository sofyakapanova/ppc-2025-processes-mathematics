#include "kapanova_s_image_smoothing/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
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
  kernel = nullptr;
  width = 0;
  height = 0;
}

bool KapanovaSImageSmoothingSEQ::ValidationImpl() {
  const auto &inputData = GetInput();
  return !inputData.empty() && !inputData[0].empty();
}

bool KapanovaSImageSmoothingSEQ::PreProcessingImpl() {
  const auto &inputData = GetInput();
  if (inputData.empty() || inputData[0].size() < 4) {
    return false;
  }

  const auto &data = inputData[0];
  width = (data[1] << 8) | data[0];
  height = (data[3] << 8) | data[2];

  size_t expected_size = static_cast<size_t>(4 + width * height * 3);
  if (data.size() < expected_size) {
    return false;
  }

  input.assign(data.begin() + 4, data.end());
  result = std::vector<uint8_t>(static_cast<size_t>(width * height * 3));

  CreateKernel();
  return true;
}

void KapanovaSImageSmoothingSEQ::CreateKernel() {
  int size = 2 * radius + 1;
  kernel = new float[size * size]{0};
  float sigma = 1.5f;
  float norm = 0;

  for (int i = -radius; i <= radius; i++) {
    for (int j = -radius; j <= radius; j++) {
      kernel[(i + radius) * size + j + radius] = std::exp(-(i * i + j * j) / (2 * sigma * sigma));
      norm += kernel[(i + radius) * size + j + radius];
    }
  }

  for (int i = 0; i < size * size; i++) {
    kernel[i] /= norm;
  }
}

void KapanovaSImageSmoothingSEQ::SmoothPixel(int x, int y) {
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

      outR += input[static_cast<size_t>(pos)] * kernel[kernelPos];
      outG += input[static_cast<size_t>(pos + 1)] * kernel[kernelPos];
      outB += input[static_cast<size_t>(pos + 2)] * kernel[kernelPos];
    }
  }

  int pos = y * stride + x * 3;
  result[static_cast<size_t>(pos)] = static_cast<uint8_t>(outR);
  result[static_cast<size_t>(pos + 1)] = static_cast<uint8_t>(outG);
  result[static_cast<size_t>(pos + 2)] = static_cast<uint8_t>(outB);
}

bool KapanovaSImageSmoothingSEQ::RunImpl() {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      SmoothPixel(x, y);
    }
  }
  return true;
}

bool KapanovaSImageSmoothingSEQ::PostProcessingImpl() {
  delete[] kernel;
  kernel = nullptr;

  GetOutput() = result;
  return true;
}

}  // namespace kapanova_s_image_smoothing
