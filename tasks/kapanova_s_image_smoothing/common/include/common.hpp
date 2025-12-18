#pragma once

#include <cstdint>
#include <random>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace kapanova_s_image_smoothing {

// RGB/BGR изображение
struct ImageMatrix {
  int columns;                      // ширина
  int rows;                         // высота
  std::vector<uint8_t> pixel_data;  // линейный массив: columns * rows * 3
  bool is_bgr_format;               // true для BGR, false для RGB

  size_t total_size() const {
    return static_cast<size_t>(columns * rows * 3);
  }
  bool is_empty() const {
    return pixel_data.empty();
  }
};

using InputType = ImageMatrix;
using OutputType = ImageMatrix;
using TestDataType = std::tuple<InputType, OutputType>;
using CoreTask = ppc::task::Task<InputType, OutputType>;

// Вспомогательные функции
inline ImageMatrix generate_random_image(int rows, int columns) {
  ImageMatrix image;
  image.rows = rows;
  image.columns = columns;
  image.is_bgr_format = false;  // По умолчанию RGB
  size_t total_elements = static_cast<size_t>(rows) * columns * 3;
  image.pixel_data.resize(total_elements);

  std::random_device device;
  std::mt19937 generator(device());
  std::uniform_int_distribution<> distribution(0, 255);

  for (size_t i = 0; i < total_elements; i++) {
    image.pixel_data[i] = static_cast<uint8_t>(distribution(generator));
  }

  return image;
}

inline ImageMatrix create_uniform_image(int rows, int columns, uint8_t value = 128) {
  ImageMatrix image;
  image.rows = rows;
  image.columns = columns;
  image.is_bgr_format = false;
  size_t total_elements = static_cast<size_t>(rows) * columns * 3;
  image.pixel_data.resize(total_elements, value);

  return image;
}

inline ImageMatrix create_pattern_image(int rows, int columns) {
  ImageMatrix image;
  image.rows = rows;
  image.columns = columns;
  image.is_bgr_format = false;
  image.pixel_data.resize(static_cast<size_t>(rows) * columns * 3);

  for (int y = 0; y < rows; y++) {
    for (int x = 0; x < columns; x++) {
      size_t index = static_cast<size_t>(y * columns * 3 + x * 3);
      bool is_dark = ((x / 10) + (y / 10)) % 2 == 0;

      if (is_dark) {
        image.pixel_data[index] = 0;      // R
        image.pixel_data[index + 1] = 0;  // G
        image.pixel_data[index + 2] = 0;  // B
      } else {
        image.pixel_data[index] = 255;      // R
        image.pixel_data[index + 1] = 255;  // G
        image.pixel_data[index + 2] = 255;  // B
      }
    }
  }

  return image;
}

inline bool compare_images(const ImageMatrix &img1, const ImageMatrix &img2, int tolerance = 2) {
  if (img1.columns != img2.columns || img1.rows != img2.rows) {
    return false;
  }

  if (img1.pixel_data.size() != img2.pixel_data.size()) {
    return false;
  }

  for (size_t i = 0; i < img1.pixel_data.size(); i++) {
    int diff = std::abs(static_cast<int>(img1.pixel_data[i]) - static_cast<int>(img2.pixel_data[i]));
    if (diff > tolerance) {
      return false;
    }
  }

  return true;
}

}  // namespace kapanova_s_image_smoothing
