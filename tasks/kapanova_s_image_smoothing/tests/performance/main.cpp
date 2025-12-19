#include <gtest/gtest.h>
#include <mpi.h>

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "kapanova_s_image_smoothing/common/include/common.hpp"
#include "kapanova_s_image_smoothing/mpi/include/ops_mpi.hpp"
#include "kapanova_s_image_smoothing/seq/include/ops_seq.hpp"

// Функция для создания тестового изображения
std::vector<uint8_t> createTestImageData(int height, int width) {
  std::vector<uint8_t> image_data(static_cast<size_t>(height) * static_cast<size_t>(width) * 3);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 255);

  for (size_t i = 0; i < image_data.size(); ++i) {
    image_data[i] = static_cast<uint8_t>(dis(gen));
  }
  return image_data;
}

// Функция для формата данных по шаблону
kapanova_s_image_smoothing::InType formatInputData(const std::vector<uint8_t> &image_data, int width, int height) {
  kapanova_s_image_smoothing::InType formatted_input;
  std::vector<uint8_t> data;

  // Первые 4 байта - ширина и высота (по 2 байта каждое)
  data.push_back(static_cast<uint8_t>(width & 0xFF));
  data.push_back(static_cast<uint8_t>((width >> 8) & 0xFF));
  data.push_back(static_cast<uint8_t>(height & 0xFF));
  data.push_back(static_cast<uint8_t>((height >> 8) & 0xFF));

  // Добавляем пиксельные данные
  data.insert(data.end(), image_data.begin(), image_data.end());

  formatted_input.push_back(data);
  return formatted_input;
}

TEST(KapanovaSImageSmoothingPerformance, SequentialBaseline) {
  // Этот тест запускается только для SEQ версии, не зависит от MPI
  const int image_height = 300;
  const int image_width = 300;

  auto test_image = createTestImageData(image_height, image_width);
  auto formatted_input = formatInputData(test_image, image_width, image_height);

  kapanova_s_image_smoothing::KapanovaSImageSmoothingSEQ sequentialTask(formatted_input);

  // Валидация
  EXPECT_TRUE(sequentialTask.Validation());

  // Предобработка
  auto start_time = std::chrono::high_resolution_clock::now();
  EXPECT_TRUE(sequentialTask.PreProcessing());

  // Основная обработка
  EXPECT_TRUE(sequentialTask.Run());

  // Постобработка
  EXPECT_TRUE(sequentialTask.PostProcessing());
  auto end_time = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

  std::cout << "SEQ Baseline (300x300): " << duration.count() << " ms" << std::endl;

  // Проверка, что результат не пустой
  EXPECT_FALSE(sequentialTask.GetOutput().empty());
}

TEST(KapanovaSImageSmoothingPerformance, MPISingleProcess) {
  // Тест MPI с 1 процессом (должен быть аналогичен SEQ)
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Проверяем условие на всех процессах
  if (size != 1) {
    // Просто возвращаемся, не используем GTEST_SKIP()
    if (rank == 0) {
      std::cout << "Note: MPISingleProcess skipped (requires exactly 1 process, got " << size << ")" << std::endl;
    }
    return;
  }

  const int image_height = 300;
  const int image_width = 300;

  auto test_image = createTestImageData(image_height, image_width);
  auto formatted_input = formatInputData(test_image, image_width, image_height);

  kapanova_s_image_smoothing::KapanovaSImageSmoothingMPI mpiTask(formatted_input);

  auto start_time = std::chrono::high_resolution_clock::now();

  // Правильный порядок вызовов
  EXPECT_TRUE(mpiTask.Validation());
  EXPECT_TRUE(mpiTask.PreProcessing());
  EXPECT_TRUE(mpiTask.Run());
  EXPECT_TRUE(mpiTask.PostProcessing());

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

  std::cout << "MPI Single Process (300x300): " << duration.count() << " ms" << std::endl;

  // Проверка, что результат не пустой
  EXPECT_FALSE(mpiTask.GetOutput().empty());
}

TEST(KapanovaSImageSmoothingPerformance, MPIMultiProcess) {
  // Основной тест MPI - работает с 2+ процессами
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Проверяем условие
  if (size == 1) {
    if (rank == 0) {
      std::cout << "Note: MPIMultiProcess skipped (requires 2+ processes, got " << size << ")" << std::endl;
    }
    return;
  }

  const int image_height = 300;
  const int image_width = 300;

  auto test_image = createTestImageData(image_height, image_width);
  auto formatted_input = formatInputData(test_image, image_width, image_height);

  kapanova_s_image_smoothing::KapanovaSImageSmoothingMPI mpiTask(formatted_input);

  // Синхронизация перед началом теста
  MPI_Barrier(MPI_COMM_WORLD);

  auto start_time = std::chrono::high_resolution_clock::now();

  // Правильный порядок вызовов
  EXPECT_TRUE(mpiTask.Validation());
  EXPECT_TRUE(mpiTask.PreProcessing());
  EXPECT_TRUE(mpiTask.Run());
  EXPECT_TRUE(mpiTask.PostProcessing());

  MPI_Barrier(MPI_COMM_WORLD);
  auto end_time = std::chrono::high_resolution_clock::now();

  if (rank == 0) {
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "MPI with " << size << " processes (300x300): " << duration.count() << " ms" << std::endl;

    // Проверка, что результат не пустой
    EXPECT_FALSE(mpiTask.GetOutput().empty());
  }
}

TEST(KapanovaSImageSmoothingPerformance, MPIScalability) {
  // Тест масштабируемости с разными размерами изображений
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Только процесс 0 управляет выводом, но все процессы должны выполнять вычисления
  if (rank == 0) {
    std::cout << "\n=== MPI Scalability Test (Running with " << size << " processes) ===" << std::endl;
  }

  std::vector<std::pair<std::string, std::pair<int, int>>> test_cases = {{"Small (100x100)", {100, 100}},
                                                                         {"Medium (300x300)", {300, 300}},
                                                                         {"Large (500x500)", {500, 500}},
                                                                         {"Very Large (800x800)", {800, 800}}};

  for (const auto &[name, dimensions] : test_cases) {
    auto [height, width] = dimensions;
    auto test_image = createTestImageData(height, width);
    auto formatted_input = formatInputData(test_image, width, height);

    kapanova_s_image_smoothing::KapanovaSImageSmoothingMPI mpiTask(formatted_input);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start_time = std::chrono::high_resolution_clock::now();

    EXPECT_TRUE(mpiTask.Validation());
    EXPECT_TRUE(mpiTask.PreProcessing());
    EXPECT_TRUE(mpiTask.Run());
    EXPECT_TRUE(mpiTask.PostProcessing());

    MPI_Barrier(MPI_COMM_WORLD);
    auto end_time = std::chrono::high_resolution_clock::now();

    if (rank == 0) {
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
      std::cout << name << ": " << duration.count() << " ms" << std::endl;
    }

    // Все процессы проверяют корректность
    EXPECT_FALSE(mpiTask.GetOutput().empty());
  }
}

TEST(KapanovaSImageSmoothingPerformance, CompareSEQvsMPI) {
  // Сравнение SEQ и MPI версий (запускается только на 1 процессе)
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Проверяем условие
  if (size != 1) {
    if (rank == 0) {
      std::cout << "Note: CompareSEQvsMPI skipped (requires exactly 1 process, got " << size << ")" << std::endl;
    }
    return;
  }

  const int image_height = 400;
  const int image_width = 400;

  auto test_image = createTestImageData(image_height, image_width);
  auto formatted_input = formatInputData(test_image, image_width, image_height);

  // SEQ версия
  kapanova_s_image_smoothing::KapanovaSImageSmoothingSEQ seqTask(formatted_input);

  auto seq_start = std::chrono::high_resolution_clock::now();
  EXPECT_TRUE(seqTask.Validation());
  EXPECT_TRUE(seqTask.PreProcessing());
  EXPECT_TRUE(seqTask.Run());
  EXPECT_TRUE(seqTask.PostProcessing());
  auto seq_end = std::chrono::high_resolution_clock::now();
  auto seq_duration = std::chrono::duration_cast<std::chrono::milliseconds>(seq_end - seq_start);

  // MPI версия (1 процесс)
  kapanova_s_image_smoothing::KapanovaSImageSmoothingMPI mpiTask(formatted_input);

  auto mpi_start = std::chrono::high_resolution_clock::now();
  EXPECT_TRUE(mpiTask.Validation());
  EXPECT_TRUE(mpiTask.PreProcessing());
  EXPECT_TRUE(mpiTask.Run());
  EXPECT_TRUE(mpiTask.PostProcessing());
  auto mpi_end = std::chrono::high_resolution_clock::now();
  auto mpi_duration = std::chrono::duration_cast<std::chrono::milliseconds>(mpi_end - mpi_start);

  std::cout << "\n=== SEQ vs MPI Comparison (400x400, 1 process) ===" << std::endl;
  std::cout << "SEQ time: " << seq_duration.count() << " ms" << std::endl;
  std::cout << "MPI time: " << mpi_duration.count() << " ms" << std::endl;
  std::cout << "Overhead: " << std::fixed << std::setprecision(2)
            << (static_cast<double>(mpi_duration.count()) / seq_duration.count() - 1.0) * 100.0 << "%" << std::endl;

  // Проверяем, что результаты одинаковые
  EXPECT_EQ(seqTask.GetOutput().size(), mpiTask.GetOutput().size());

  // Для такого размера можно сравнить результаты
  EXPECT_EQ(seqTask.GetOutput(), mpiTask.GetOutput());
}

TEST(KapanovaSImageSmoothingPerformance, BoundaryCases) {
  // Тестирование граничных случаев
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Все процессы выполняют тест, но вывод только от процесса 0
  if (rank == 0) {
    std::cout << "\n=== Boundary Cases Test ===" << std::endl;
  }

  // Тест 1: Очень маленькое изображение
  {
    auto test_image = createTestImageData(2, 2);
    auto formatted_input = formatInputData(test_image, 2, 2);

    kapanova_s_image_smoothing::KapanovaSImageSmoothingMPI mpiTask(formatted_input);

    EXPECT_TRUE(mpiTask.Validation());
    EXPECT_TRUE(mpiTask.PreProcessing());
    EXPECT_TRUE(mpiTask.Run());
    EXPECT_TRUE(mpiTask.PostProcessing());

    if (rank == 0) {
      std::cout << "2x2 image: OK" << std::endl;
    }
  }

  // Тест 2: Высокое узкое изображение
  {
    auto test_image = createTestImageData(100, 10);
    auto formatted_input = formatInputData(test_image, 10, 100);

    kapanova_s_image_smoothing::KapanovaSImageSmoothingMPI mpiTask(formatted_input);

    EXPECT_TRUE(mpiTask.Validation());
    EXPECT_TRUE(mpiTask.PreProcessing());
    EXPECT_TRUE(mpiTask.Run());
    EXPECT_TRUE(mpiTask.PostProcessing());

    if (rank == 0) {
      std::cout << "10x100 image: OK" << std::endl;
    }
  }

  // Тест 3: Широкое низкое изображение
  {
    auto test_image = createTestImageData(10, 100);
    auto formatted_input = formatInputData(test_image, 100, 10);

    kapanova_s_image_smoothing::KapanovaSImageSmoothingMPI mpiTask(formatted_input);

    EXPECT_TRUE(mpiTask.Validation());
    EXPECT_TRUE(mpiTask.PreProcessing());
    EXPECT_TRUE(mpiTask.Run());
    EXPECT_TRUE(mpiTask.PostProcessing());

    if (rank == 0) {
      std::cout << "100x10 image: OK" << std::endl;
    }
  }
}
