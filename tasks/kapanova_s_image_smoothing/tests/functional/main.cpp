#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "kapanova_s_image_smoothing/mpi/include/ops_mpi.hpp"
#include "kapanova_s_image_smoothing/seq/include/ops_seq.hpp"

std::vector<uint8_t> generateRandomPixels(int height, int width) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> distrib(0, 255);
  std::vector<uint8_t> pixels(height * width * 3);

  for (size_t i = 0; i < pixels.size(); ++i) {
    pixels[i] = static_cast<uint8_t>(distrib(gen));
  }
  return pixels;
}

TEST(kapanova_s_image_smoothing, Test_IMAGE_RANDOM_SQUARE) {
  boost::mpi::communicator world;

  int image_width = 10;
  int image_height = 10;

  std::vector<uint8_t> original_pixels;
  std::vector<uint8_t> parallel_result;

  std::shared_ptr<ppc::core::TaskData> parallelTaskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Создаем тестовые данные
    original_pixels = generateRandomPixels(image_height, image_width);
    parallel_result.resize(image_height * image_width * 3);

    // Настраиваем TaskData для параллельной версии
    parallelTaskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(original_pixels.data()));
    parallelTaskData->inputs_count.emplace_back(image_width);
    parallelTaskData->inputs_count.emplace_back(image_height);
    parallelTaskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(parallel_result.data()));
    parallelTaskData->outputs_count.emplace_back(image_width);
    parallelTaskData->outputs_count.emplace_back(image_height);
  }

  kapanova_s_image_smoothing::KapanovaSImageSmoothingMPI parallelTask(parallelTaskData);
  ASSERT_TRUE(parallelTask.validation());
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    // Создаем данные для последовательной версии
    std::vector<uint8_t> sequential_result(image_height * image_width * 3);

    // Настраиваем TaskData для последовательной версии
    std::shared_ptr<ppc::core::TaskData> sequentialTaskData = std::make_shared<ppc::core::TaskData>();
    sequentialTaskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(original_pixels.data()));
    sequentialTaskData->inputs_count.emplace_back(image_width);
    sequentialTaskData->inputs_count.emplace_back(image_height);
    sequentialTaskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(sequential_result.data()));
    sequentialTaskData->outputs_count.emplace_back(image_width);
    sequentialTaskData->outputs_count.emplace_back(image_height);

    // Создаем и запускаем последовательную задачу
    kapanova_s_image_smoothing::KapanovaSImageSmoothingSEQ sequentialTask(sequentialTaskData);
    ASSERT_TRUE(sequentialTask.validation());
    sequentialTask.pre_processing();
    sequentialTask.run();
    sequentialTask.post_processing();

    // Проверяем, что результаты совпадают
    ASSERT_EQ(parallel_result.size(), sequential_result.size());
    for (size_t i = 0; i < parallel_result.size(); ++i) {
      // Допускаем разницу в 1 из-за округления
      ASSERT_LE(std::abs(static_cast<int>(parallel_result[i]) - static_cast<int>(sequential_result[i])), 1);
    }
  }
}

TEST(kapanova_s_image_smoothing, Test_IMAGE_RANDOM_LANDSCAPE) {
  boost::mpi::communicator world;

  int image_width = 15;
  int image_height = 5;

  std::vector<uint8_t> original_pixels;
  std::vector<uint8_t> parallel_result;

  std::shared_ptr<ppc::core::TaskData> parallelTaskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Создаем тестовые данные (альбомная ориентация)
    original_pixels = generateRandomPixels(image_height, image_width);
    parallel_result.resize(image_height * image_width * 3);

    // Настраиваем TaskData для параллельной версии
    parallelTaskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(original_pixels.data()));
    parallelTaskData->inputs_count.emplace_back(image_width);
    parallelTaskData->inputs_count.emplace_back(image_height);
    parallelTaskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(parallel_result.data()));
    parallelTaskData->outputs_count.emplace_back(image_width);
    parallelTaskData->outputs_count.emplace_back(image_height);
  }

  kapanova_s_image_smoothing::KapanovaSImageSmoothingMPI parallelTask(parallelTaskData);
  ASSERT_TRUE(parallelTask.validation());
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    // Создаем данные для последовательной версии
    std::vector<uint8_t> sequential_result(image_height * image_width * 3);

    // Настраиваем TaskData для последовательной версии
    std::shared_ptr<ppc::core::TaskData> sequentialTaskData = std::make_shared<ppc::core::TaskData>();
    sequentialTaskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(original_pixels.data()));
    sequentialTaskData->inputs_count.emplace_back(image_width);
    sequentialTaskData->inputs_count.emplace_back(image_height);
    sequentialTaskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(sequential_result.data()));
    sequentialTaskData->outputs_count.emplace_back(image_width);
    sequentialTaskData->outputs_count.emplace_back(image_height);

    // Создаем и запускаем последовательную задачу
    kapanova_s_image_smoothing::KapanovaSImageSmoothingSEQ sequentialTask(sequentialTaskData);
    ASSERT_TRUE(sequentialTask.validation());
    sequentialTask.pre_processing();
    sequentialTask.run();
    sequentialTask.post_processing();

    // Проверяем, что результаты совпадают
    ASSERT_EQ(parallel_result.size(), sequential_result.size());
    for (size_t i = 0; i < parallel_result.size(); ++i) {
      // Допускаем разницу в 1 из-за округления
      ASSERT_LE(std::abs(static_cast<int>(parallel_result[i]) - static_cast<int>(sequential_result[i])), 1);
    }
  }
}

TEST(kapanova_s_image_smoothing, Test_IMAGE_RANDOM_PORTRAIT) {
  boost::mpi::communicator world;

  int image_width = 5;
  int image_height = 15;

  std::vector<uint8_t> original_pixels;
  std::vector<uint8_t> parallel_result;

  std::shared_ptr<ppc::core::TaskData> parallelTaskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Создаем тестовые данные (портретная ориентация)
    original_pixels = generateRandomPixels(image_height, image_width);
    parallel_result.resize(image_height * image_width * 3);

    // Настраиваем TaskData для параллельной версии
    parallelTaskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(original_pixels.data()));
    parallelTaskData->inputs_count.emplace_back(image_width);
    parallelTaskData->inputs_count.emplace_back(image_height);
    parallelTaskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(parallel_result.data()));
    parallelTaskData->outputs_count.emplace_back(image_width);
    parallelTaskData->outputs_count.emplace_back(image_height);
  }

  kapanova_s_image_smoothing::KapanovaSImageSmoothingMPI parallelTask(parallelTaskData);
  ASSERT_TRUE(parallelTask.validation());
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    // Создаем данные для последовательной версии
    std::vector<uint8_t> sequential_result(image_height * image_width * 3);

    // Настраиваем TaskData для последовательной версии
    std::shared_ptr<ppc::core::TaskData> sequentialTaskData = std::make_shared<ppc::core::TaskData>();
    sequentialTaskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(original_pixels.data()));
    sequentialTaskData->inputs_count.emplace_back(image_width);
    sequentialTaskData->inputs_count.emplace_back(image_height);
    sequentialTaskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(sequential_result.data()));
    sequentialTaskData->outputs_count.emplace_back(image_width);
    sequentialTaskData->outputs_count.emplace_back(image_height);

    // Создаем и запускаем последовательную задачу
    kapanova_s_image_smoothing::KapanovaSImageSmoothingSEQ sequentialTask(sequentialTaskData);
    ASSERT_TRUE(sequentialTask.validation());
    sequentialTask.pre_processing();
    sequentialTask.run();
    sequentialTask.post_processing();

    // Проверяем, что результаты совпадают
    ASSERT_EQ(parallel_result.size(), sequential_result.size());
    for (size_t i = 0; i < parallel_result.size(); ++i) {
      // Допускаем разницу в 1 из-за округления
      ASSERT_LE(std::abs(static_cast<int>(parallel_result[i]) - static_cast<int>(sequential_result[i])), 1);
    }
  }
}

TEST(kapanova_s_image_smoothing, Test_IMAGE_SOLID_COLOR) {
  boost::mpi::communicator world;

  int image_width = 8;
  int image_height = 8;

  std::vector<uint8_t> original_pixels(image_height * image_width * 3, 100);  // Все пиксели серые
  std::vector<uint8_t> parallel_result(image_height * image_width * 3);

  std::shared_ptr<ppc::core::TaskData> parallelTaskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Настраиваем TaskData для параллельной версии
    parallelTaskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(original_pixels.data()));
    parallelTaskData->inputs_count.emplace_back(image_width);
    parallelTaskData->inputs_count.emplace_back(image_height);
    parallelTaskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(parallel_result.data()));
    parallelTaskData->outputs_count.emplace_back(image_width);
    parallelTaskData->outputs_count.emplace_back(image_height);
  }

  kapanova_s_image_smoothing::KapanovaSImageSmoothingMPI parallelTask(parallelTaskData);
  ASSERT_TRUE(parallelTask.validation());
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    // Проверяем, что однородное изображение осталось однородным
    for (size_t i = 0; i < parallel_result.size(); ++i) {
      ASSERT_EQ(parallel_result[i], 100);
    }
  }
}

TEST(kapanova_s_image_smoothing, Test_IMAGE_GRADIENT) {
  boost::mpi::communicator world;

  int image_width = 6;
  int image_height = 6;

  // Создаем градиентное изображение
  std::vector<uint8_t> original_pixels(image_height * image_width * 3);
  for (int y = 0; y < image_height; ++y) {
    for (int x = 0; x < image_width; ++x) {
      int pos = (y * image_width + x) * 3;
      original_pixels[pos] = static_cast<uint8_t>((x + y) * 10);      // R
      original_pixels[pos + 1] = static_cast<uint8_t>((x + y) * 10);  // G
      original_pixels[pos + 2] = static_cast<uint8_t>((x + y) * 10);  // B
    }
  }

  std::vector<uint8_t> parallel_result(image_height * image_width * 3);
  std::vector<uint8_t> sequential_result(image_height * image_width * 3);

  std::shared_ptr<ppc::core::TaskData> parallelTaskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Настраиваем TaskData для параллельной версии
    parallelTaskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(original_pixels.data()));
    parallelTaskData->inputs_count.emplace_back(image_width);
    parallelTaskData->inputs_count.emplace_back(image_height);
    parallelTaskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(parallel_result.data()));
    parallelTaskData->outputs_count.emplace_back(image_width);
    parallelTaskData->outputs_count.emplace_back(image_height);
  }

  kapanova_s_image_smoothing::KapanovaSImageSmoothingMPI parallelTask(parallelTaskData);
  ASSERT_TRUE(parallelTask.validation());
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    // Настраиваем TaskData для последовательной версии
    std::shared_ptr<ppc::core::TaskData> sequentialTaskData = std::make_shared<ppc::core::TaskData>();
    sequentialTaskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(original_pixels.data()));
    sequentialTaskData->inputs_count.emplace_back(image_width);
    sequentialTaskData->inputs_count.emplace_back(image_height);
    sequentialTaskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(sequential_result.data()));
    sequentialTaskData->outputs_count.emplace_back(image_width);
    sequentialTaskData->outputs_count.emplace_back(image_height);

    // Создаем и запускаем последовательную задачу
    kapanova_s_image_smoothing::KapanovaSImageSmoothingSEQ sequentialTask(sequentialTaskData);
    ASSERT_TRUE(sequentialTask.validation());
    sequentialTask.pre_processing();
    sequentialTask.run();
    sequentialTask.post_processing();

    // Проверяем, что результаты совпадают
    ASSERT_EQ(parallel_result.size(), sequential_result.size());
    for (size_t i = 0; i < parallel_result.size(); ++i) {
      // Допускаем разницу в 1 из-за округления
      ASSERT_LE(std::abs(static_cast<int>(parallel_result[i]) - static_cast<int>(sequential_result[i])), 1);
    }
  }
}

TEST(kapanova_s_image_smoothing, Test_IMAGE_SMALL) {
  boost::mpi::communicator world;

  int image_width = 3;
  int image_height = 3;

  // Маленькое тестовое изображение 3x3
  std::vector<uint8_t> original_pixels = {255, 0,   0,   0, 255, 0,   0,   0,  255, 0,  255, 255, 255, 0,
                                          255, 255, 255, 0, 128, 128, 128, 64, 64,  64, 192, 192, 192};

  std::vector<uint8_t> parallel_result(image_height * image_width * 3);
  std::vector<uint8_t> sequential_result(image_height * image_width * 3);

  std::shared_ptr<ppc::core::TaskData> parallelTaskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Настраиваем TaskData для параллельной версии
    parallelTaskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(original_pixels.data()));
    parallelTaskData->inputs_count.emplace_back(image_width);
    parallelTaskData->inputs_count.emplace_back(image_height);
    parallelTaskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(parallel_result.data()));
    parallelTaskData->outputs_count.emplace_back(image_width);
    parallelTaskData->outputs_count.emplace_back(image_height);
  }

  kapanova_s_image_smoothing::KapanovaSImageSmoothingMPI parallelTask(parallelTaskData);
  ASSERT_TRUE(parallelTask.validation());
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    // Настраиваем TaskData для последовательной версии
    std::shared_ptr<ppc::core::TaskData> sequentialTaskData = std::make_shared<ppc::core::TaskData>();
    sequentialTaskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(original_pixels.data()));
    sequentialTaskData->inputs_count.emplace_back(image_width);
    sequentialTaskData->inputs_count.emplace_back(image_height);
    sequentialTaskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(sequential_result.data()));
    sequentialTaskData->outputs_count.emplace_back(image_width);
    sequentialTaskData->outputs_count.emplace_back(image_height);

    // Создаем и запускаем последовательную задачу
    kapanova_s_image_smoothing::KapanovaSImageSmoothingSEQ sequentialTask(sequentialTaskData);
    ASSERT_TRUE(sequentialTask.validation());
    sequentialTask.pre_processing();
    sequentialTask.run();
    sequentialTask.post_processing();

    // Проверяем, что результаты совпадают
    ASSERT_EQ(parallel_result.size(), sequential_result.size());
    for (size_t i = 0; i < parallel_result.size(); ++i) {
      // Допускаем разницу в 2 для маленького изображения
      ASSERT_LE(std::abs(static_cast<int>(parallel_result[i]) - static_cast<int>(sequential_result[i])), 2);
    }
  }
}
