#include <gtest/gtest.h>
#include <mpi.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "kapanova_s_image_smoothing/common/include/common.hpp"
#include "kapanova_s_image_smoothing/mpi/include/ops_mpi.hpp"
#include "kapanova_s_image_smoothing/seq/include/ops_seq.hpp"

namespace {

std::vector<uint8_t> CreateTestImageData(int height, int width);
kapanova_s_image_smoothing::InType FormatInputData(const std::vector<uint8_t> &image_data, int width, int height);
bool RunMpiTask(kapanova_s_image_smoothing::KapanovaSImageSmoothingMPI &mpi_task);
bool RunSeqTask(kapanova_s_image_smoothing::KapanovaSImageSmoothingSEQ &seq_task);

std::vector<uint8_t> CreateTestImageData(int height, int width) {
  std::vector<uint8_t> image_data(static_cast<size_t>(height) * static_cast<size_t>(width) * 3);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 255);

  for (size_t i = 0; i < image_data.size(); ++i) {
    image_data[i] = static_cast<uint8_t>(dis(gen));
  }
  return image_data;
}

kapanova_s_image_smoothing::InType FormatInputData(const std::vector<uint8_t> &image_data, int width, int height) {
  kapanova_s_image_smoothing::InType formatted_input;
  std::vector<uint8_t> data;

  data.push_back(static_cast<uint8_t>(width & 0xFF));
  data.push_back(static_cast<uint8_t>((width >> 8) & 0xFF));
  data.push_back(static_cast<uint8_t>(height & 0xFF));
  data.push_back(static_cast<uint8_t>((height >> 8) & 0xFF));

  data.insert(data.end(), image_data.begin(), image_data.end());

  formatted_input.push_back(data);
  return formatted_input;
}

bool RunMpiTask(kapanova_s_image_smoothing::KapanovaSImageSmoothingMPI &mpi_task) {
  return mpi_task.Validation() && mpi_task.PreProcessing() && mpi_task.Run() && mpi_task.PostProcessing();
}

bool RunSeqTask(kapanova_s_image_smoothing::KapanovaSImageSmoothingSEQ &seq_task) {
  return seq_task.Validation() && seq_task.PreProcessing() && seq_task.Run() && seq_task.PostProcessing();
}

TEST(KapanovaSImageSmoothingPerformance, SequentialBaseline) {
  const int image_height = 2500;
  const int image_width = 2500;

  auto test_image = CreateTestImageData(image_height, image_width);
  auto formatted_input = FormatInputData(test_image, image_width, image_height);

  kapanova_s_image_smoothing::KapanovaSImageSmoothingSEQ sequential_task(formatted_input);

  auto start_time = std::chrono::high_resolution_clock::now();
  EXPECT_TRUE(RunSeqTask(sequential_task));
  auto end_time = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  std::cout << "SEQ Baseline (2500x2500): " << duration.count() << " ms\n";

  EXPECT_FALSE(sequential_task.GetOutput().empty());
}

TEST(KapanovaSImageSmoothingPerformance, MPISingleProcess) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size != 1) {
    if (rank == 0) {
      std::cout << "Note: MPISingleProcess skipped (requires exactly 1 process, got " << size << ")\n";
    }
    return;
  }

  const int image_height = 2500;
  const int image_width = 2500;

  auto test_image = CreateTestImageData(image_height, image_width);
  auto formatted_input = FormatInputData(test_image, image_width, image_height);

  kapanova_s_image_smoothing::KapanovaSImageSmoothingMPI mpi_task(formatted_input);

  auto start_time = std::chrono::high_resolution_clock::now();
  EXPECT_TRUE(RunMpiTask(mpi_task));
  auto end_time = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  std::cout << "MPI Single Process (2500x2500): " << duration.count() << " ms\n";

  EXPECT_FALSE(mpi_task.GetOutput().empty());
}

TEST(KapanovaSImageSmoothingPerformance, MPIMultiProcess) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size == 1) {
    if (rank == 0) {
      std::cout << "Note: MPIMultiProcess skipped (requires 2+ processes, got " << size << ")\n";
    }
    return;
  }

  const int image_height = 2500;
  const int image_width = 2500;

  auto test_image = CreateTestImageData(image_height, image_width);
  auto formatted_input = FormatInputData(test_image, image_width, image_height);

  kapanova_s_image_smoothing::KapanovaSImageSmoothingMPI mpi_task(formatted_input);

  MPI_Barrier(MPI_COMM_WORLD);
  auto start_time = std::chrono::high_resolution_clock::now();

  EXPECT_TRUE(RunMpiTask(mpi_task));

  MPI_Barrier(MPI_COMM_WORLD);
  auto end_time = std::chrono::high_resolution_clock::now();

  if (rank == 0) {
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "MPI with " << size << " processes (2500x2500): " << duration.count() << " ms\n";
    EXPECT_FALSE(mpi_task.GetOutput().empty());
  }

  EXPECT_FALSE(mpi_task.GetOutput().empty());
}

TEST(KapanovaSImageSmoothingPerformance, MPIScalability) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    std::cout << "\n=== MPI Scalability Test (Running with " << size << " processes) ===\n";
  }

  std::vector<std::pair<std::string, std::pair<int, int>>> test_cases = {
      {"Small (100x100)", {100, 100}},
      {"Medium (1000x1000)", {1000, 1000}},
      {"Large (2500x2500)", {2500, 2500}},
  };

  for (const auto &[name, dimensions] : test_cases) {
    auto [height, width] = dimensions;
    auto test_image = CreateTestImageData(height, width);
    auto formatted_input = FormatInputData(test_image, width, height);

    kapanova_s_image_smoothing::KapanovaSImageSmoothingMPI mpi_task(formatted_input);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start_time = std::chrono::high_resolution_clock::now();

    EXPECT_TRUE(RunMpiTask(mpi_task));

    MPI_Barrier(MPI_COMM_WORLD);
    auto end_time = std::chrono::high_resolution_clock::now();

    if (rank == 0) {
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
      std::cout << name << ": " << duration.count() << " ms\n";
    }

    EXPECT_FALSE(mpi_task.GetOutput().empty());
  }
}

TEST(KapanovaSImageSmoothingPerformance, CompareSEQvsMPI) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size != 1) {
    if (rank == 0) {
      std::cout << "Note: CompareSEQvsMPI skipped (requires exactly 1 process, got " << size << ")\n";
    }
    return;
  }

  const int image_height = 400;
  const int image_width = 400;

  auto test_image = CreateTestImageData(image_height, image_width);
  auto formatted_input = FormatInputData(test_image, image_width, image_height);

  kapanova_s_image_smoothing::KapanovaSImageSmoothingSEQ seq_task(formatted_input);
  auto seq_start = std::chrono::high_resolution_clock::now();
  EXPECT_TRUE(RunSeqTask(seq_task));
  auto seq_end = std::chrono::high_resolution_clock::now();
  auto seq_duration = std::chrono::duration_cast<std::chrono::milliseconds>(seq_end - seq_start);

  kapanova_s_image_smoothing::KapanovaSImageSmoothingMPI mpi_task(formatted_input);
  auto mpi_start = std::chrono::high_resolution_clock::now();
  EXPECT_TRUE(RunMpiTask(mpi_task));
  auto mpi_end = std::chrono::high_resolution_clock::now();
  auto mpi_duration = std::chrono::duration_cast<std::chrono::milliseconds>(mpi_end - mpi_start);

  std::cout << "\n=== SEQ vs MPI Comparison (400x400, 1 process) ===\n";
  std::cout << "SEQ time: " << seq_duration.count() << " ms\n";
  std::cout << "MPI time: " << mpi_duration.count() << " ms\n";
  std::cout << "Overhead: " << std::fixed << std::setprecision(2)
            << ((static_cast<double>(mpi_duration.count()) / static_cast<double>(seq_duration.count())) - 1.0) * 100.0
            << "%\n";

  EXPECT_EQ(seq_task.GetOutput().size(), mpi_task.GetOutput().size());
  EXPECT_EQ(seq_task.GetOutput(), mpi_task.GetOutput());
}

TEST(KapanovaSImageSmoothingPerformance, BoundaryCases) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  bool all_tests_passed = true;

  auto test_image1 = CreateTestImageData(2, 2);
  auto formatted_input1 = FormatInputData(test_image1, 2, 2);
  kapanova_s_image_smoothing::KapanovaSImageSmoothingMPI mpi_task1(formatted_input1);
  all_tests_passed = all_tests_passed && RunMpiTask(mpi_task1);

  auto test_image2 = CreateTestImageData(100, 10);
  auto formatted_input2 = FormatInputData(test_image2, 10, 100);
  kapanova_s_image_smoothing::KapanovaSImageSmoothingMPI mpi_task2(formatted_input2);
  all_tests_passed = all_tests_passed && RunMpiTask(mpi_task2);

  auto test_image3 = CreateTestImageData(10, 100);
  auto formatted_input3 = FormatInputData(test_image3, 100, 10);
  kapanova_s_image_smoothing::KapanovaSImageSmoothingMPI mpi_task3(formatted_input3);
  all_tests_passed = all_tests_passed && RunMpiTask(mpi_task3);

  if (rank == 0) {
    std::cout << "\n=== Boundary Cases Test ===\n";
    std::cout << "2x2 image: " << (mpi_task1.GetOutput().empty() ? "FAIL" : "OK") << "\n";
    std::cout << "10x100 image: " << (mpi_task2.GetOutput().empty() ? "FAIL" : "OK") << "\n";
    std::cout << "100x10 image: " << (mpi_task3.GetOutput().empty() ? "FAIL" : "OK") << "\n";
  }

  EXPECT_TRUE(all_tests_passed);
}

}  // namespace
