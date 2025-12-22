#pragma once

#include <vector>

#include "kapanova_s_dijkstra/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kapanova_s_dijkstra {

class KapanovaDijkstraCRSMPI : public BaseSolver {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KapanovaDijkstraCRSMPI(const AlgorithmInput &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void DistributeGraphData();
  void GatherResults();

  std::vector<int> local_offsets_;
  std::vector<int> local_columns_;
  std::vector<double> local_values_;
  std::vector<int> vertex_ownership_;
  int local_num_vertices_ = 0;
  int world_size_ = 0;
  int world_rank_ = 0;
};

}  // namespace kapanova_s_dijkstra