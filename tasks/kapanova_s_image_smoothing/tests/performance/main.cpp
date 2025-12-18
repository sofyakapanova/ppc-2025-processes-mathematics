#include <gtest/gtest.h>

#include <chrono>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "kapanova_s_image_smoothing/seq/include/ops_seq.hpp"

std::vector<uint8_t> createTestImageData(int height, int width) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(0, 255);

  std::vector<uint8_t> image_data(height * width * 3);
  for (size_t i = 0; i < image_data.size(); ++i) {
    image_data[i] = static_cast<uint8_t>(distrib(gen));
  }
  return image_data;
}

TEST(kapanova_s_image_smoothing_performance, TestPerformancePipeline) {
  const int image_height = 500;
  const int image_width = 500;

  auto test_image = createTestImageData(image_height, image_width);
  std::vector<uint8_t> output_data(image_height * image_width * 3);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(test_image.data()));
  taskData->inputs_count.emplace_back(image_width);
  taskData->inputs_count.emplace_back(image_height);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  taskData->outputs_count.emplace_back(image_width);
  taskData->outputs_count.emplace_back(image_height);

  auto sequentialTask = std::make_shared<kapanova_s_image_smoothing::KapanovaSImageSmoothingSEQ>(taskData);

  ASSERT_TRUE(sequentialTask->validation());
  sequentialTask->pre_processing();
  sequentialTask->run();
  sequentialTask->post_processing();

  auto performanceAttributes = std::make_shared<ppc::core::PerfAttr>();
  performanceAttributes->num_running = 5;
  const auto start_time = std::chrono::high_resolution_clock::now();
  performanceAttributes->current_timer = [&] {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto time_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time - start_time).count();
    return static_cast<double>(time_duration) * 1e-9;
  };

  auto performanceResults = std::make_shared<ppc::core::PerfResults>();
  auto performanceAnalyzer = std::make_shared<ppc::core::Perf>(sequentialTask);

  performanceAnalyzer->pipeline_run(performanceAttributes, performanceResults);

  ppc::core::Perf::print_perf_statistic(performanceResults);
  ASSERT_EQ(image_height, taskData->inputs_count[1]);
}

TEST(kapanova_s_image_smoothing_performance, TestPerformanceTask) {
  const int image_height = 500;
  const int image_width = 500;

  auto test_image = createTestImageData(image_height, image_width);
  std::vector<uint8_t> output_data(image_height * image_width * 3);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(test_image.data()));
  taskData->inputs_count.emplace_back(image_width);
  taskData->inputs_count.emplace_back(image_height);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
  taskData->outputs_count.emplace_back(image_width);
  taskData->outputs_count.emplace_back(image_height);

  auto sequentialTask = std::make_shared<kapanova_s_image_smoothing::KapanovaSImageSmoothingSEQ>(taskData);

  ASSERT_TRUE(sequentialTask->validation());
  sequentialTask->pre_processing();
  sequentialTask->run();
  sequentialTask->post_processing();

  auto performanceAttributes = std::make_shared<ppc::core::PerfAttr>();
  performanceAttributes->num_running = 5;
  const auto start_time = std::chrono::high_resolution_clock::now();
  performanceAttributes->current_timer = [&] {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto time_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time - start_time).count();
    return static_cast<double>(time_duration) * 1e-9;
  };

  auto performanceResults = std::make_shared<ppc::core::PerfResults>();
  auto performanceAnalyzer = std::make_shared<ppc::core::Perf>(sequentialTask);

  performanceAnalyzer->task_run(performanceAttributes, performanceResults);

  ppc::core::Perf::print_perf_statistic(performanceResults);
  ASSERT_EQ(image_height, taskData->inputs_count[1]);
}

TEST(kapanova_s_image_smoothing_performance, TestDifferentImageSizes) {
  std::vector<std::pair<int, int>> test_dimensions = {{100, 100}, {200, 200}, {300, 300}, {500, 500}};

  for (const auto &[height, width] : test_dimensions) {
    auto test_image = createTestImageData(height, width);
    std::vector<uint8_t> output_data(height * width * 3);

    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(test_image.data()));
    taskData->inputs_count.emplace_back(width);
    taskData->inputs_count.emplace_back(height);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
    taskData->outputs_count.emplace_back(width);
    taskData->outputs_count.emplace_back(height);

    kapanova_s_image_smoothing::KapanovaSImageSmoothingSEQ task(taskData);

    auto processing_start = std::chrono::high_resolution_clock::now();

    ASSERT_TRUE(task.validation());
    ASSERT_TRUE(task.pre_processing());
    ASSERT_TRUE(task.run());
    ASSERT_TRUE(task.post_processing());

    auto processing_end = std::chrono::high_resolution_clock::now();
    auto processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(processing_end - processing_start);

    std::cout << "Image size: " << height << "x" << width << " | Processing time: " << processing_time.count() << " ms"
              << std::endl;
  }
}
