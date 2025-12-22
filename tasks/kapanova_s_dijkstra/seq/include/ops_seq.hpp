#pragma once

#include "kapanova_s_dijkstra/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kapanova_s_dijkstra {

class KapanovaSDijkstraSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KapanovaSDijkstraSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace kapanova_s_dijkstra