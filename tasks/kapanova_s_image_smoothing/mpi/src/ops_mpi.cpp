#include "kapanova_s_image_smoothing/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace kapanova_s_image_smoothing {

KapanovaSImageSmoothingMPI::KapanovaSImageSmoothingMPI(const InType &in) : BaseTask() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KapanovaSImageSmoothingMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  auto &input = GetInput();

  if (rank == 0) {
    if (input.width <= 0 || input.height <= 0) {
      return false;
    }

    if (input.pixels.size() != static_cast<size_t>(input.width) * input.height) {
      return false;
    }
  }

  int width = input.width;
  int height = input.height;
  MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetInput().width = width;
  GetInput().height = height;

  return true;
}

bool KapanovaSImageSmoothingMPI::PreProcessingImpl() {
  auto &output = GetOutput();
  output = GetInput();

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    output.pixels.resize(static_cast<size_t>(output.width) * output.height);
  }

  return true;
}

std::vector<float> KapanovaSImageSmoothingMPI::create_gaussian_kernel(int radius, float sigma) {
  int size = 2 * radius + 1;
  std::vector<float> kernel(size);
  float norm = 0.0f;

  for (int i = -radius; i <= radius; ++i) {
    kernel[i + radius] = std::exp(-(i * i) / (2 * sigma * sigma));
    norm += kernel[i + radius];
  }

  for (float &val : kernel) {
    val /= norm;
  }

  return kernel;
}

void KapanovaSImageSmoothingMPI::convolve_rows(const std::vector<uint8_t> &input, int height, int width,
                                               const std::vector<float> &kernel, std::vector<float> &temp) {
  int kernel_radius = static_cast<int>(kernel.size()) / 2;

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

void KapanovaSImageSmoothingMPI::convolve_columns(const std::vector<float> &temp, int height, int width,
                                                  const std::vector<float> &kernel, std::vector<uint8_t> &output) {
  int kernel_radius = static_cast<int>(kernel.size()) / 2;

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

bool KapanovaSImageSmoothingMPI::RunImpl() {
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &input = GetInput();
  auto &output = GetOutput();

  const int width = input.width;
  const int height = input.height;
  const int radius = 1;
  const float sigma = 1.5f;

  std::vector<float> horizontal_kernel = create_gaussian_kernel(radius, sigma);
  const std::vector<float> &vertical_kernel = horizontal_kernel;

  if (size == 1) {
    std::vector<float> temp(width * height, 0.0f);
    convolve_rows(input.pixels, height, width, horizontal_kernel, temp);
    convolve_columns(temp, height, width, vertical_kernel, output.pixels);
    return true;
  }

  int base_height = height / size;
  int remainder = height % size;

  std::vector<int> send_counts(size, 0);
  std::vector<int> displacements(size, 0);

  if (rank == 0) {
    int current_displacement = 0;
    for (int proc = 0; proc < size; ++proc) {
      int proc_height = base_height + (proc < remainder ? 1 : 0);

      int halo_top = (proc == 0) ? 0 : 1;
      int halo_bottom = (proc == size - 1) ? 0 : 1;

      send_counts[proc] = (proc_height + halo_top + halo_bottom) * width;
      displacements[proc] = current_displacement * width;

      if (proc == 0) {
        current_displacement += (proc_height + halo_bottom);
      } else {
        current_displacement += proc_height;
      }
    }
  }

  int local_height_with_halo = 0;
  if (rank == 0) {
    local_height_with_halo = send_counts[rank] / width;
  }
  MPI_Bcast(&local_height_with_halo, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<uint8_t> local_input(local_height_with_halo * width);
  std::vector<int> all_send_counts = send_counts;
  std::vector<int> all_displacements = displacements;

  MPI_Scatterv(rank == 0 ? input.pixels.data() : nullptr, all_send_counts.data(), all_displacements.data(),
               MPI_UNSIGNED_CHAR, local_input.data(), local_height_with_halo * width, MPI_UNSIGNED_CHAR, 0,
               MPI_COMM_WORLD);

  int halo_top = (rank == 0) ? 0 : 1;
  int halo_bottom = (rank == size - 1) ? 0 : 1;
  int local_height = local_height_with_halo - halo_top - halo_bottom;

  std::vector<float> local_temp(local_height_with_halo * width, 0.0f);
  std::vector<uint8_t> local_output(local_height * width);

  convolve_rows(local_input, local_height_with_halo, width, horizontal_kernel, local_temp);
  convolve_columns(local_temp, local_height_with_halo, width, vertical_kernel, local_output);

  std::vector<int> recv_counts(size, 0);
  std::vector<int> recv_displacements(size, 0);

  if (rank == 0) {
    int current_displacement = 0;
    for (int proc = 0; proc < size; ++proc) {
      int proc_height = base_height + (proc < remainder ? 1 : 0);
      recv_counts[proc] = proc_height * width;
      recv_displacements[proc] = current_displacement * width;
      current_displacement += proc_height;
    }
  }

  MPI_Gatherv(local_output.data(), local_height * width, MPI_UNSIGNED_CHAR, rank == 0 ? output.pixels.data() : nullptr,
              recv_counts.data(), recv_displacements.data(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  return true;
}

bool KapanovaSImageSmoothingMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    auto &output = GetOutput();
    return output.pixels.size() == static_cast<size_t>(output.width) * output.height;
  }

  return true;
}

}  // namespace kapanova_s_image_smoothing
