#include "kapanova_s_min_of_matrix_elements/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <utility>  // ДОБАВИТЬ для std::pair
#include <vector>

#include "kapanova_s_min_of_matrix_elements/common/include/common.hpp"

namespace kapanova_s_min_of_matrix_elements {

KapanovaSMinOfMatrixElementsMPI::KapanovaSMinOfMatrixElementsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput().resize(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    GetInput()[i] = in[i];
  }
  GetOutput() = 0;
}

bool KapanovaSMinOfMatrixElementsMPI::ValidationImpl() {
  const auto &matrix = GetInput();
  if (matrix.empty()) {
    return true;
  }

  const size_t cols = matrix[0].size();
  return std::ranges::all_of(matrix, [cols](const auto &row) { return row.size() == cols; });
}

bool KapanovaSMinOfMatrixElementsMPI::PreProcessingImpl() {
  GetOutput() = INT_MAX;
  return true;
}

namespace {

// Вспомогательные функции для уменьшения когнитивной сложности

std::pair<int, int> GetMatrixDimensions(int rank, const InType &matrix) {
  int total_rows = 0;
  int total_cols = 0;

  if (rank == 0) {
    if (!matrix.empty()) {
      total_rows = static_cast<int>(matrix.size());
      total_cols = static_cast<int>(matrix[0].size());
    }
  }

  return {total_rows, total_cols};
}

std::vector<int> PrepareAndBroadcastMatrix(int rank, int total_rows, int total_cols, const InType &matrix) {
  std::vector<int> flat_matrix;
  const size_t total_elements = static_cast<size_t>(total_rows) * static_cast<size_t>(total_cols);

  if (rank == 0) {
    flat_matrix.resize(total_elements);
    for (int i = 0; i < total_rows; ++i) {
      for (int j = 0; j < total_cols; ++j) {
        // ИСПРАВЛЕНО: добавлены скобки
        const size_t index = (static_cast<size_t>(i) * static_cast<size_t>(total_cols)) + static_cast<size_t>(j);
        flat_matrix[index] = matrix[i][j];
      }
    }
  } else {
    flat_matrix.resize(total_elements);
  }

  MPI_Bcast(flat_matrix.data(), static_cast<int>(total_elements), MPI_INT, 0, MPI_COMM_WORLD);
  return flat_matrix;
}

std::pair<int, int> CalculateLocalRange(int rank, int size, int total_rows, int total_cols) {
  const int total_elements = total_rows * total_cols;
  const int elements_per_process = total_elements / size;
  const int remainder = total_elements % size;

  int start_element = 0;
  int end_element = 0;

  if (rank < remainder) {
    start_element = rank * (elements_per_process + 1);
    end_element = start_element + elements_per_process + 1;
  } else {
    // ИСПРАВЛЕНО: добавлены скобки
    start_element = (rank * elements_per_process) + remainder;
    end_element = start_element + elements_per_process;
  }

  return {start_element, end_element};
}

int FindLocalMinimum(const std::vector<int> &flat_matrix, int start_element, int end_element, int total_cols) {
  int local_min = INT_MAX;

  for (int elem_idx = start_element; elem_idx < end_element; ++elem_idx) {
    const int row = elem_idx / total_cols;
    const int col = elem_idx % total_cols;
    // ИСПРАВЛЕНО: добавлены скобки
    const int index = (row * total_cols) + col;

    // ИСПРАВЛЕНО: используем std::min вместо сравнения
    local_min = std::min(flat_matrix[index], local_min);
  }

  return local_min;
}

int FindGlobalMinimum(int local_min, int size) {
  int global_min = local_min;

  if (size > 1) {
    MPI_Allreduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  }

  return global_min;
}

}  // namespace

bool KapanovaSMinOfMatrixElementsMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // 1. Получаем размеры матрицы
  auto [total_rows, total_cols] = GetMatrixDimensions(rank, GetInput());
  MPI_Bcast(&total_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&total_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // 2. Проверяем пустую матрицу
  if (total_rows == 0 || total_cols == 0) {
    GetOutput() = INT_MAX;
    return true;
  }

  // 3. Подготавливаем и рассылаем матрицу
  std::vector<int> flat_matrix = PrepareAndBroadcastMatrix(rank, total_rows, total_cols, GetInput());

  // 4. Определяем диапазон для текущего процесса
  auto [start_element, end_element] = CalculateLocalRange(rank, size, total_rows, total_cols);

  // 5. Находим локальный минимум
  int local_min = FindLocalMinimum(flat_matrix, start_element, end_element, total_cols);

  // 6. Находим глобальный минимум
  int global_min = FindGlobalMinimum(local_min, size);

  GetOutput() = global_min;
  return true;
}

bool KapanovaSMinOfMatrixElementsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kapanova_s_min_of_matrix_elements
