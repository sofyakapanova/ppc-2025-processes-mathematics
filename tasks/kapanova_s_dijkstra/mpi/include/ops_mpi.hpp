#pragma once

#include <vector>

#include "kapanova_s_dijkstra/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kapanova_s_dijkstra {

class KapanovaSDijkstraMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KapanovaSDijkstraMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void PartitionGraph();
  void CollectDistances();

  std::vector<int> local_row_ptr_;
  std::vector<int> local_col_idx_;
  std::vector<double> local_weights_;
  std::vector<int> node_assignments_;
  int local_vertices_ = 0;
  int proc_count_ = 0;
  int proc_rank_ = 0;
};

}  // namespace kapanova_s_dijkstra