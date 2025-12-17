#include "kapanova_s_min_of_matrix_elements/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstddef>
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

bool KapanovaSMinOfMatrixElementsMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Только rank 0 имеет исходную матрицу
  auto &matrix = GetInput();

  // Переменные для размеров матрицы
  int total_rows = 0;
  int total_cols = 0;

  // 1. Rank 0 определяет размеры матрицы и рассылает их всем
  if (rank == 0) {
    if (matrix.empty()) {
      total_rows = 0;
      total_cols = 0;
    } else {
      total_rows = static_cast<int>(matrix.size());
      total_cols = static_cast<int>(matrix[0].size());
    }
  }

  // Рассылаем размеры матрицы всем процессам
  MPI_Bcast(&total_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&total_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Если матрица пустая
  if (total_rows == 0 || total_cols == 0) {
    GetOutput() = INT_MAX;
    return true;
  }

  // 2. Рассылаем саму матрицу
  // Сначала делаем "плоский" массив для рассылки
  std::vector<int> flat_matrix;

  if (rank == 0) {
    flat_matrix.resize(total_rows * total_cols);
    for (int i = 0; i < total_rows; ++i) {
      for (int j = 0; j < total_cols; ++j) {
        flat_matrix[i * total_cols + j] = matrix[i][j];
      }
    }
  } else {
    // Остальные процессы выделяют память
    flat_matrix.resize(total_rows * total_cols);
  }

  // Рассылаем плоскую матрицу всем процессам
  MPI_Bcast(flat_matrix.data(), total_rows * total_cols, MPI_INT, 0, MPI_COMM_WORLD);

  // 3. Теперь каждый процесс вычисляет свою часть
  const int total_elements = total_rows * total_cols;

  int elements_per_process = total_elements / size;
  int remainder = total_elements % size;

  int start_element = 0;
  int end_element = 0;

  if (rank < remainder) {
    start_element = rank * (elements_per_process + 1);
    end_element = start_element + elements_per_process + 1;
  } else {
    start_element = (rank * elements_per_process) + remainder;
    end_element = start_element + elements_per_process;
  }

  // 4. Вычисляем локальный минимум
  int local_min = INT_MAX;
  for (int elem_idx = start_element; elem_idx < end_element; ++elem_idx) {
    const int row = elem_idx / total_cols;
    const int col = elem_idx % total_cols;

    local_min = std::min(flat_matrix[(row * total_cols) + col], local_min);
  }

  // 5. Собираем глобальный минимум
  int global_min = local_min;
  if (size > 1) {
    MPI_Allreduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  }

  GetOutput() = global_min;
  return true;
}

bool KapanovaSMinOfMatrixElementsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kapanova_s_min_of_matrix_elements
