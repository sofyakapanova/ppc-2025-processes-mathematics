#include "kapanova_s_image_smoothing/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace kapanova_s_image_smoothing {

KapanovaSImageSmoothingMPI::KapanovaSImageSmoothingMPI(const InType &in) : BaseTask() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KapanovaSImageSmoothingMPI::ValidationImpl() {
  int my_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  const auto &input_img = GetInput();

  if (my_rank == 0) {
    if (input_img.width <= 0 || input_img.height <= 0) {
      return false;
    }

    size_t expected_size = static_cast<size_t>(input_img.width) * input_img.height;
    if (input_img.pixels.size() != expected_size) {
      return false;
    }
  }

  return true;
}

bool KapanovaSImageSmoothingMPI::PreProcessingImpl() {
  int my_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  const auto &input_img = GetInput();

  if (my_rank == 0) {
    img_width_ = input_img.width;
    img_height_ = input_img.height;
  }

  MPI_Bcast(&img_width_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&img_height_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (my_rank == 0) {
    auto &output_img = GetOutput();
    output_img = input_img;
    output_img.pixels.resize(static_cast<size_t>(img_width_) * img_height_);
  }

  return true;
}

std::vector<float> KapanovaSImageSmoothingMPI::CreateGaussianKernel(int kernel_radius, float sigma_val) {
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

void KapanovaSImageSmoothingMPI::ProcessRows(const std::vector<uint8_t> &input_img, int img_h, int img_w,
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

void KapanovaSImageSmoothingMPI::ProcessColumns(const std::vector<float> &temp_buf, int img_h, int img_w,
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

bool KapanovaSImageSmoothingMPI::RunImpl() {
  int my_rank = 0, proc_count = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_count);

  const int kernel_radius = 1;
  const float sigma_val = 1.5f;

  std::vector<float> row_kernel = CreateGaussianKernel(kernel_radius, sigma_val);
  const std::vector<float> &col_kernel = row_kernel;

  if (proc_count == 1) {
    const auto &input_img = GetInput();
    auto &output_img = GetOutput();

    std::vector<float> temp_buffer(img_width_ * img_height_, 0.0f);
    ProcessRows(input_img.pixels, img_height_, img_width_, row_kernel, temp_buffer);
    ProcessColumns(temp_buffer, img_height_, img_width_, col_kernel, output_img.pixels);
    return true;
  }

  int base_rows = img_height_ / proc_count;
  int extra_rows = img_height_ % proc_count;

  std::vector<int> send_sizes(proc_count, 0);
  std::vector<int> send_offsets(proc_count, 0);

  if (my_rank == 0) {
    int current_offset = 0;

    for (int proc = 0; proc < proc_count; ++proc) {
      int rows_for_proc = base_rows + (proc < extra_rows ? 1 : 0);

      int top_halo = (proc == 0) ? 0 : 1;
      int bottom_halo = (proc == proc_count - 1) ? 0 : 1;

      send_sizes[proc] = (rows_for_proc + top_halo + bottom_halo) * img_width_;
      send_offsets[proc] = current_offset * img_width_;

      if (proc == 0) {
        current_offset += (rows_for_proc + bottom_halo);
      } else {
        current_offset += rows_for_proc;
      }
    }
  }

  MPI_Bcast(send_sizes.data(), proc_count, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(send_offsets.data(), proc_count, MPI_INT, 0, MPI_COMM_WORLD);

  int my_data_size = send_sizes[my_rank];
  std::vector<uint8_t> local_input(my_data_size);

  const uint8_t *global_data = nullptr;
  if (my_rank == 0) {
    const auto &input_img = GetInput();
    global_data = input_img.pixels.data();
  }

  MPI_Scatterv(global_data, send_sizes.data(), send_offsets.data(), MPI_UNSIGNED_CHAR, local_input.data(), my_data_size,
               MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  int top_halo = (my_rank == 0) ? 0 : 1;
  int bottom_halo = (my_rank == proc_count - 1) ? 0 : 1;
  int local_rows_with_halo = my_data_size / img_width_;
  int local_rows = local_rows_with_halo - top_halo - bottom_halo;

  std::vector<float> local_temp(local_rows_with_halo * img_width_, 0.0f);
  std::vector<uint8_t> local_result(local_rows * img_width_);

  ProcessRows(local_input, local_rows_with_halo, img_width_, row_kernel, local_temp);
  ProcessColumns(local_temp, local_rows_with_halo, img_width_, col_kernel, local_result);

  std::vector<int> recv_sizes(proc_count, 0);
  std::vector<int> recv_offsets(proc_count, 0);

  if (my_rank == 0) {
    int current_offset = 0;
    for (int proc = 0; proc < proc_count; ++proc) {
      int rows_for_proc = base_rows + (proc < extra_rows ? 1 : 0);
      recv_sizes[proc] = rows_for_proc * img_width_;
      recv_offsets[proc] = current_offset * img_width_;
      current_offset += rows_for_proc;
    }
  }

  MPI_Bcast(recv_sizes.data(), proc_count, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(recv_offsets.data(), proc_count, MPI_INT, 0, MPI_COMM_WORLD);

  int my_result_size = local_result.size();
  auto &output_img = GetOutput();

  MPI_Gatherv(local_result.data(), my_result_size, MPI_UNSIGNED_CHAR, my_rank == 0 ? output_img.pixels.data() : nullptr,
              recv_sizes.data(), recv_offsets.data(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  return true;
}

bool KapanovaSImageSmoothingMPI::PostProcessingImpl() {
  int my_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank == 0) {
    auto &output_img = GetOutput();
    output_img.width = img_width_;
    output_img.height = img_height_;
    output_img.kernel_size = 3;

    return output_img.pixels.size() == static_cast<size_t>(output_img.width) * output_img.height;
  }

  return true;
}

}  // namespace kapanova_s_image_smoothing
