#include "kapanova_s_image_smoothing/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <vector>

namespace kapanova_s_image_smoothing {

KapanovaSImageSmoothingMPI::KapanovaSImageSmoothingMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  if (!in.empty()) {
    Getinput_() = in;
  } else {
    Getinput_() = InType();
  }
  width_ = 0;
  height_ = 0;
}

bool KapanovaSImageSmoothingMPI::ValidationImpl() {
  const auto &input_Data = Getinput_();
  return !input_Data.empty() && !input_Data[0].empty();
}

bool KapanovaSImageSmoothingMPI::PreProcessingImpl() {
  const auto &input_Data = Getinput_();
  if (input_Data.empty() || input_Data[0].size() < 4) {
    return false;
  }

  const auto &data = input_Data[0];
  width_ = (data[1] << 8) | data[0];
  height_ = (data[3] << 8) | data[2];

  size_t required_pixels = static_cast<size_t>(width_) * static_cast<size_t>(height_) * 3;
  size_t total_required_size = 4 + required_pixels;

  if (data.size() < total_required_size) {
    return false;
  }

  input_.assign(data.begin() + 4, data.end());
  result_ = std::vector<uint8_t>(required_pixels);
  kernel_ = CreateKernel()();

  return true;
}

std::vector<float> KapanovaSImageSmoothingMPI::CreateKernel()() const {
  int size = 2 * radius_ + 1;
  std::vector<float> kernel__local(size * size, 0.0F);
  float sigma = 1.5F;
  float norm = 0;

  for (int i = -radius_; i <= radius_; i++) {
    for (int j = -radius_; j <= radius_; j++) {
      kernel__local[static_cast<size_t>((i + radius_) * size + j + radius_)] =
          std::exp(-(i * i + j * j) / (2 * sigma * sigma));
      norm += kernel__local[static_cast<size_t>((i + radius_) * size + j + radius_)];
    }
  }

  for (int i = 0; i < size * size; i++) {
    kernel__local[static_cast<size_t>(i)] /= norm;
  }

  return kernel__local;
}

void KapanovaSImageSmoothingMPI::SmoothPixel(uint8_t *out, int x, int y) {
  int stride = width_ * 3;
  size_t sizek = static_cast<size_t>(2 * radius_ + 1);
  float outR = 0.0F;
  float outG = 0.0F;
  float outB = 0.0F;

  auto clamp = [](int n, int lo, int hi) { return std::min(std::max(n, lo), hi); };

  for (int ry = -radius_; ry <= radius_; ry++) {
    for (int rx = -radius_; rx <= radius_; rx++) {
      int idX = clamp(x + rx, 0, width_ - 1);
      int idY = clamp(y + ry, 0, height_ - 1);
      int pos = idY * stride + idX * 3;
      int kernel_Pos = static_cast<int>((ry + radius_) * sizek + rx + radius_);

      outR += static_cast<float>(input_[static_cast<size_t>(pos)]) * kernel_[static_cast<size_t>(kernel_Pos)];
      outG += static_cast<float>(input_[static_cast<size_t>(pos + 1)]) * kernel_[static_cast<size_t>(kernel_Pos)];
      outB += static_cast<float>(input_[static_cast<size_t>(pos + 2)]) * kernel_[static_cast<size_t>(kernel_Pos)];
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

  constexpr int kTagExit = 0;
  constexpr int kTagInfo = 1;
  constexpr int kTagData = 2;
  constexpr int kTagResult_ = 3;

  if (size == 1) {
    for (int y = 0; y < height_; y++) {
      for (int x = 0; x < width_; x++) {
        SmoothPixel(&result_[static_cast<size_t>(y * width_ * 3 + x * 3)], x, y);
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
      MPI_Send(&width_, 1, MPI_INT, i, kTagInfo, MPI_COMM_WORLD);
    }

    int row = 0;
    while (row < height_ - 2) {
      int processes_to_use = std::min(satellites, height_ - 2 - row);

      for (int i = 0; i < processes_to_use; i++) {
        MPI_Send(&noescape, 1, MPI_INT, i + 1, kTagExit, MPI_COMM_WORLD);

        if (row + i == 0) {
          MPI_Send(&input_[0], 2 * width_ * 3, MPI_UNSIGNED_CHAR, i + 1, kTagData, MPI_COMM_WORLD);
        } else if (row + i == height_ - 2) {
          int start_pos = (height_ - 2) * width_ * 3;
          MPI_Send(&input_[static_cast<size_t>(start_pos)], 2 * width_ * 3, MPI_UNSIGNED_CHAR, i + 1, kTagData,
                   MPI_COMM_WORLD);
        } else {
          int start_pos = (row + i - 1) * width_ * 3;
          MPI_Send(&input_[static_cast<size_t>(start_pos)], 3 * width_ * 3, MPI_UNSIGNED_CHAR, i + 1, kTagData,
                   MPI_COMM_WORLD);
        }
      }

      // Получаем результаты
      for (int i = 0; i < processes_to_use; i++) {
        int result__row = row + i + 1;
        if (result__row > 0 && result__row < height_ - 1) {
          int result__pos = result__row * width_ * 3;
          MPI_Recv(&result_[static_cast<size_t>(result__pos)], width_ * 3, MPI_UNSIGNED_CHAR, i + 1, kTagResult_,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
      }

      row += processes_to_use;
    }

    for (int i = 1; i <= satellites; i++) {
      MPI_Send(&escape, 1, MPI_INT, i, kTagExit, MPI_COMM_WORLD);
    }

    // Обрабатываем первую и последнюю строку
    for (int x = 0; x < width_; x++) {
      SmoothPixel(&result_[static_cast<size_t>(x * 3)], x, 0);
      SmoothPixel(&result_[static_cast<size_t>((height_ - 1) * width_ * 3 + x * 3)], x, height_ - 1);
    }

  } else {
    int local_width_ = 0;
    MPI_Recv(&local_width_, 1, MPI_INT, 0, kTagInfo, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    std::vector<uint8_t> local_input_;
    std::vector<uint8_t> local_result_(static_cast<size_t>(local_width_ * 3));

    int escape = 0;

    while (true) {
      MPI_Recv(&escape, 1, MPI_INT, 0, kTagExit, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (escape == 0) {
        break;
      }

      MPI_Status status;
      MPI_Probe(0, kTagData, MPI_COMM_WORLD, &status);
      int count = 0;
      MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &count);

      local_input_.resize(static_cast<size_t>(count));
      MPI_Recv(local_input_.data(), count, MPI_UNSIGNED_CHAR, 0, kTagData, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      // Определяем, сколько строк получили
      int rows_received = count / (local_width_ * 3);

      if (rows_received == 3) {
        for (int x = 0; x < local_width_; x++) {
          SmoothPixel(&local_result_[static_cast<size_t>(x * 3)], x, 1);
        }

        MPI_Send(local_result_.data(), local_width_ * 3, MPI_UNSIGNED_CHAR, 0, kTagResult_, MPI_COMM_WORLD);
      } else if (rows_received == 2) {
        for (int x = 0; x < local_width_; x++) {
          SmoothPixel(&local_result_[static_cast<size_t>(x * 3)], x, 0);
        }

        MPI_Send(local_result_.data(), local_width_ * 3, MPI_UNSIGNED_CHAR, 0, kTagResult_, MPI_COMM_WORLD);
      }
    }
  }

  return true;
}

bool KapanovaSImageSmoothingMPI::PostProcessingImpl() {
  GetOutput() = result_;
  return true;
}

}  // namespace kapanova_s_image_smoothing
