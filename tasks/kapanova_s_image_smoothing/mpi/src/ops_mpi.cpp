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
    GetInput() = in;
  } else {
    GetInput() = InType();
  }
  width_ = 0;
  height_ = 0;
}

bool KapanovaSImageSmoothingMPI::ValidationImpl() {
  const auto &input_data = GetInput();
  return !input_data.empty() && !input_data[0].empty();
}

bool KapanovaSImageSmoothingMPI::PreProcessingImpl() {
  const auto &input_data = GetInput();
  if (input_data.empty() || input_data[0].size() < 4) {
    return false;
  }

  const auto &data = input_data[0];
  width_ = (data[1] << 8) | data[0];
  height_ = (data[3] << 8) | data[2];

  size_t required_pixels = static_cast<size_t>(width_) * static_cast<size_t>(height_) * 3;
  size_t total_required_size = 4 + required_pixels;

  if (data.size() < total_required_size) {
    return false;
  }

  input_.assign(data.begin() + 4, data.end());
  result_.resize(required_pixels);
  kernel_ = CreateKernel();

  return true;
}

std::vector<float> KapanovaSImageSmoothingMPI::CreateKernel() const {
  int size = 2 * radius_ + 1;
  std::vector<float> kernel_local(static_cast<size_t>(size * size), 0.0F);
  constexpr float kSigma = 1.5F;
  float norm = 0.0F;

  for (int i = -radius_; i <= radius_; ++i) {
    for (int j = -radius_; j <= radius_; ++j) {
      kernel_local[static_cast<size_t>((i + radius_) * size + j + radius_)] =
          std::exp(-(static_cast<float>(i * i + j * j)) / (2.0F * kSigma * kSigma));
      norm += kernel_local[static_cast<size_t>((i + radius_) * size + j + radius_)];
    }
  }

  for (size_t i = 0; i < kernel_local.size(); ++i) {
    kernel_local[i] /= norm;
  }

  return kernel_local;
}

void KapanovaSImageSmoothingMPI::SmoothPixel(uint8_t *out, int x, int y) {
  int stride = width_ * 3;
  size_t size_k = static_cast<size_t>(2 * radius_ + 1);
  float out_r = 0.0F;
  float out_g = 0.0F;
  float out_b = 0.0F;

  auto clamp = [](int n, int lo, int hi) { return std::min(std::max(n, lo), hi); };

  for (int ry = -radius_; ry <= radius_; ++ry) {
    for (int rx = -radius_; rx <= radius_; ++rx) {
      int idx = clamp(x + rx, 0, width_ - 1);
      int idy = clamp(y + ry, 0, height_ - 1);
      int pos = idy * stride + idx * 3;
      size_t kernel_pos = static_cast<size_t>((ry + radius_) * static_cast<int>(size_k) + rx + radius_);

      out_r += static_cast<float>(input_[static_cast<size_t>(pos)]) * kernel_[kernel_pos];
      out_g += static_cast<float>(input_[static_cast<size_t>(pos + 1)]) * kernel_[kernel_pos];
      out_b += static_cast<float>(input_[static_cast<size_t>(pos + 2)]) * kernel_[kernel_pos];
    }
  }

  out[0] = static_cast<uint8_t>(out_r);
  out[1] = static_cast<uint8_t>(out_g);
  out[2] = static_cast<uint8_t>(out_b);
}

bool KapanovaSImageSmoothingMPI::RunImpl() {
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  constexpr int kTagExit = 0;
  constexpr int kTagInfo = 1;
  constexpr int kTagData = 2;
  constexpr int kTagResult = 3;

  if (size == 1) {
    for (int y = 0; y < height_; ++y) {
      for (int x = 0; x < width_; ++x) {
        SmoothPixel(&result_[static_cast<size_t>(y * width_ * 3 + x * 3)], x, y);
      }
    }
    return true;
  }

  if (rank == 0) {
    int satellites = size - 1;
    constexpr int kEscape = 0;
    constexpr int kNoEscape = 1;

    for (int i = 1; i <= satellites; ++i) {
      MPI_Send(&width_, 1, MPI_INT, i, kTagInfo, MPI_COMM_WORLD);
    }

    int row = 0;
    while (row < height_ - 2) {
      int processes_to_use = std::min(satellites, height_ - 2 - row);

      for (int i = 0; i < processes_to_use; ++i) {
        MPI_Send(&kNoEscape, 1, MPI_INT, i + 1, kTagExit, MPI_COMM_WORLD);

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

      for (int i = 0; i < processes_to_use; ++i) {
        int result_row = row + i + 1;
        if (result_row > 0 && result_row < height_ - 1) {
          int result_pos = result_row * width_ * 3;
          MPI_Recv(&result_[static_cast<size_t>(result_pos)], width_ * 3, MPI_UNSIGNED_CHAR, i + 1, kTagResult,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
      }

      row += processes_to_use;
    }

    for (int i = 1; i <= satellites; ++i) {
      MPI_Send(&kEscape, 1, MPI_INT, i, kTagExit, MPI_COMM_WORLD);
    }

    for (int x = 0; x < width_; ++x) {
      SmoothPixel(&result_[static_cast<size_t>(x * 3)], x, 0);
      SmoothPixel(&result_[static_cast<size_t>((height_ - 1) * width_ * 3 + x * 3)], x, height_ - 1);
    }

  } else {
    int local_width = 0;
    MPI_Recv(&local_width, 1, MPI_INT, 0, kTagInfo, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    std::vector<uint8_t> local_input;
    std::vector<uint8_t> local_result(static_cast<size_t>(local_width * 3));

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

      local_input.resize(static_cast<size_t>(count));
      MPI_Recv(local_input.data(), count, MPI_UNSIGNED_CHAR, 0, kTagData, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      int rows_received = count / (local_width * 3);

      if (rows_received == 3) {
        for (int x = 0; x < local_width; ++x) {
          SmoothPixel(&local_result[static_cast<size_t>(x * 3)], x, 1);
        }

        MPI_Send(local_result.data(), local_width * 3, MPI_UNSIGNED_CHAR, 0, kTagResult, MPI_COMM_WORLD);
      } else if (rows_received == 2) {
        for (int x = 0; x < local_width; ++x) {
          SmoothPixel(&local_result[static_cast<size_t>(x * 3)], x, 0);
        }

        MPI_Send(local_result.data(), local_width * 3, MPI_UNSIGNED_CHAR, 0, kTagResult, MPI_COMM_WORLD);
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
