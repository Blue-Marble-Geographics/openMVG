// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include "ceres/partitioned_matrix_view.h"

#include <algorithm>
#include <cstring>
#include <vector>
#include "ceres/block_sparse_matrix.h"
#include "ceres/block_structure.h"
#include "ceres/internal/eigen.h"
#include "ceres/small_blas.h"
#include "glog/logging.h"

namespace ceres {
namespace internal {

template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
PartitionedMatrixView(
    BlockSparseMatrix& matrix,
    int num_col_blocks_e)
    : matrix_(matrix),
      num_col_blocks_e_(num_col_blocks_e) {
  const CompressedRowBlockStructure* bs = matrix_.block_structure();
  CHECK_NOTNULL(bs);

  num_col_blocks_f_ = bs->col_sizes.size() - num_col_blocks_e_;

  // Compute the number of row blocks in E. The number of row blocks
  // in E maybe less than the number of row blocks in the input matrix
  // as some of the row blocks at the bottom may not have any
  // e_blocks. For a definition of what an e_block is, please see
  // explicit_schur_complement_solver.h
  num_row_blocks_e_ = 0;
  for (int r = 0; r < bs->rows.size(); ++r) {
    const Cell* cells = bs->rows[r].cells;
    if (cells[0].block_id < num_col_blocks_e_) {
      ++num_row_blocks_e_;
    }
  }

  // Compute the number of columns in E and F.
  num_cols_e_ = 0;
  num_cols_f_ = 0;

  for (int c = 0; c < bs->col_sizes.size(); ++c) {
    const int block_size = bs->col_sizes[c];
    if (c < num_col_blocks_e_) {
      num_cols_e_ += block_size;
    } else {
      num_cols_f_ += block_size;
    }
  }

  CHECK_EQ(num_cols_e_ + num_cols_f_, matrix_.num_cols());
}

template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
~PartitionedMatrixView() {
}

// The next four methods don't seem to be particularly cache
// friendly. This is an artifact of how the BlockStructure of the
// input matrix is constructed. These methods will benefit from
// multithreading as well as improved data layout.

template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void
PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
RightMultiplyE(const double* x, double* y) const {
  const CompressedRowBlockStructure* bs = matrix_.block_structure();

  // Iterate over the first num_row_blocks_e_ row blocks, and multiply
  // by the first cell in each row block.
  const double* values = matrix_.values();
  for (int r = 0; r < num_row_blocks_e_; ++r) {
    const Cell& cell = bs->rows[r].cells[0];
    const int row_block_pos = bs->rows[r].block.position;
    const int row_block_size = bs->rows[r].block.size;
    const int col_block_id = cell.block_id;
    const int col_block_pos = bs->col_positions[col_block_id];
    const int col_block_size = bs->col_sizes[col_block_id];
    MatrixVectorMultiply<kRowBlockSize, kEBlockSize, 1>(
        values + cell.position, row_block_size, col_block_size,
        x + col_block_pos,
        y + row_block_pos);
  }
}

template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void
PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
RightMultiplyF(const double* x, double* y) const {
  const CompressedRowBlockStructure* bs = matrix_.block_structure();

  // Iterate over row blocks, and if the row block is in E, then
  // multiply by all the cells except the first one which is of type
  // E. If the row block is not in E (i.e its in the bottom
  // num_row_blocks - num_row_blocks_e row blocks), then all the cells
  // are of type F and multiply by them all.
  const double* values = matrix_.values();
  for (int r = 0; r < num_row_blocks_e_; ++r) {
    const int row_block_pos = bs->rows[r].block.position;
    const int row_block_size = bs->rows[r].block.size;
    const Cell* cells = bs->rows[r].cells;
    const size_t num_cells = bs->rows[r].num_cells;
    for (int c = 1; c < num_cells; ++c) {
      const int col_block_id = cells[c].block_id;
      const int col_block_pos = bs->col_positions[col_block_id];
      const int col_block_size = bs->col_sizes[col_block_id];
      MatrixVectorMultiply<kRowBlockSize, kFBlockSize, 1>(
          values + cells[c].position, row_block_size, col_block_size,
          x + col_block_pos - num_cols_e_,
          y + row_block_pos);
    }
  }

  for (int r = num_row_blocks_e_; r < bs->rows.size(); ++r) {
    const int row_block_pos = bs->rows[r].block.position;
    const int row_block_size = bs->rows[r].block.size;
    const Cell* cells = bs->rows[r].cells;
    const int num_cells = bs->rows[r].block.size;
    for (int c = 0; c < num_cells; ++c) {
      const int col_block_id = cells[c].block_id;
      const int col_block_pos = bs->col_positions[col_block_id];
      const int col_block_size = bs->col_sizes[col_block_id];
      MatrixVectorMultiply<Eigen::Dynamic, Eigen::Dynamic, 1>(
          values + cells[c].position, row_block_size, col_block_size,
          x + col_block_pos - num_cols_e_,
          y + row_block_pos);
    }
  }
}

template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void
PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
LeftMultiplyE(const double* x, double* y) const {
  const CompressedRowBlockStructure* bs = matrix_.block_structure();

  // Iterate over the first num_row_blocks_e_ row blocks, and multiply
  // by the first cell in each row block.
  const double* values = matrix_.values();
  for (int r = 0; r < num_row_blocks_e_; ++r) {
    const Cell& cell = bs->rows[r].cells[0];
    const int row_block_pos = bs->rows[r].block.position;
    const int row_block_size = bs->rows[r].block.size;
    const int col_block_id = cell.block_id;
    const int col_block_pos = bs->col_positions[col_block_id];
    const int col_block_size = bs->col_sizes[col_block_id];
    MatrixTransposeVectorMultiply<kRowBlockSize, kEBlockSize, 1>(
        values + cell.position, row_block_size, col_block_size,
        x + row_block_pos,
        y + col_block_pos);
  }
}

template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void
PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
LeftMultiplyF(const double* x, double* y) const {
  const CompressedRowBlockStructure* bs = matrix_.block_structure();

  // Iterate over row blocks, and if the row block is in E, then
  // multiply by all the cells except the first one which is of type
  // E. If the row block is not in E (i.e its in the bottom
  // num_row_blocks - num_row_blocks_e row blocks), then all the cells
  // are of type F and multiply by them all.
  const double* values = matrix_.values();
  for (int r = 0; r < num_row_blocks_e_; ++r) {
    const int row_block_pos = bs->rows[r].block.position;
    const int row_block_size = bs->rows[r].block.size;
    const Cell* cells = bs->rows[r].cells;
    const size_t num_cells = bs->rows[r].num_cells;
    for (int c = 1; c < num_cells; ++c) {
      const int col_block_id = cells[c].block_id;
      const int col_block_pos = bs->col_positions[col_block_id];
      const int col_block_size = bs->col_sizes[col_block_id];
      MatrixTransposeVectorMultiply<kRowBlockSize, kFBlockSize, 1>(
        values + cells[c].position, row_block_size, col_block_size,
        x + row_block_pos,
        y + col_block_pos - num_cols_e_);
    }
  }

  for (int r = num_row_blocks_e_; r < bs->rows.size(); ++r) {
    const int row_block_pos = bs->rows[r].block.position;
    const int row_block_size = bs->rows[r].block.size;
    const Cell* cells = bs->rows[r].cells;
    const size_t num_cells = bs->rows[r].num_cells;
    for (int c = 0; c < num_cells; ++c) {
      const int col_block_id = cells[c].block_id;
      const int col_block_pos = bs->col_positions[col_block_id];
      const int col_block_size = bs->col_sizes[col_block_id];
      MatrixTransposeVectorMultiply<Eigen::Dynamic, Eigen::Dynamic, 1>(
        values + cells[c].position, row_block_size, col_block_size,
        x + row_block_pos,
        y + col_block_pos - num_cols_e_);
    }
  }
}

// Given a range of columns blocks of a matrix m, compute the block
// structure of the block diagonal of the matrix m(:,
// start_col_block:end_col_block)'m(:, start_col_block:end_col_block)
// and return a BlockSparseMatrix with the this block structure. The
// caller owns the result.
template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
BlockSparseMatrix*
PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
CreateBlockDiagonalMatrixLayout(int start_col_block, int end_col_block) {
  CompressedRowBlockStructure* bs = matrix_.block_structure();
  CompressedRowBlockStructure* block_diagonal_structure =
      new CompressedRowBlockStructure;

  int block_position = 0;
  int diagonal_cell_position = 0;

  bs->all_cells.resize(end_col_block - start_col_block);

  // Iterate over the column blocks, creating a new diagonal block for
  // each column block.
  for (int c = start_col_block; c < end_col_block; ++c) {
    block_diagonal_structure->col_sizes.push_back(-1);
    block_diagonal_structure->col_positions.push_back(-1);

    auto& diagonal_block_size = block_diagonal_structure->col_sizes.back();
    auto& diagonal_block_position2 = block_diagonal_structure->col_positions.back();

    diagonal_block_size = bs->col_sizes[c];
    diagonal_block_position2 = bs->col_positions[c];

    block_diagonal_structure->rows.push_back(CompressedRow());
    CompressedRow& row = block_diagonal_structure->rows.back();
    row.block.position = diagonal_block_position2;
    row.block.size = diagonal_block_size;

    const int block_id = c - start_col_block;
    row.cells = &bs->all_cells[block_id];
    row.num_cells = 1;
    Cell& cell = *row.cells;
    cell.block_id = block_id;
    cell.position = diagonal_cell_position;

    block_position += diagonal_block_size;
    diagonal_cell_position += diagonal_block_size * diagonal_block_size;
  }

  // Build a BlockSparseMatrix with the just computed block
  // structure.
  return new BlockSparseMatrix(block_diagonal_structure);
}

template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
BlockSparseMatrix*
PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
CreateBlockDiagonalEtE() {
  BlockSparseMatrix* block_diagonal =
      CreateBlockDiagonalMatrixLayout(0, num_col_blocks_e_);
  UpdateBlockDiagonalEtE(block_diagonal);
  return block_diagonal;
}

template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
BlockSparseMatrix*
PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
CreateBlockDiagonalFtF() {
  BlockSparseMatrix* block_diagonal =
      CreateBlockDiagonalMatrixLayout(
          num_col_blocks_e_, num_col_blocks_e_ + num_col_blocks_f_);
  UpdateBlockDiagonalFtF(block_diagonal);
  return block_diagonal;
}

// Similar to the code in RightMultiplyE, except instead of the matrix
// vector multiply its an outer product.
//
//    block_diagonal = block_diagonal(E'E)
//
template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void
PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
UpdateBlockDiagonalEtE(
    BlockSparseMatrix* block_diagonal) const {
  const CompressedRowBlockStructure* bs = matrix_.block_structure();
  const CompressedRowBlockStructure* block_diagonal_structure =
      block_diagonal->block_structure();

  block_diagonal->SetZero();
  const double* values = matrix_.values();
  for (int r = 0; r < num_row_blocks_e_ ; ++r) {
    const Cell& cell = bs->rows[r].cells[0];
    const int row_block_size = bs->rows[r].block.size;
    const int block_id = cell.block_id;
    const int col_block_size = bs->col_sizes[block_id];
    const int cell_position =
        block_diagonal_structure->rows[block_id].cells[0].position;

    MatrixTransposeMatrixMultiply
        <kRowBlockSize, kEBlockSize, kRowBlockSize, kEBlockSize, 1>(
            values + cell.position, row_block_size, col_block_size,
            values + cell.position, row_block_size, col_block_size,
            block_diagonal->mutable_values() + cell_position,
            0, 0, col_block_size, col_block_size);
  }
}

// Similar to the code in RightMultiplyF, except instead of the matrix
// vector multiply its an outer product.
//
//   block_diagonal = block_diagonal(F'F)
//
template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void
PartitionedMatrixView<kRowBlockSize, kEBlockSize, kFBlockSize>::
UpdateBlockDiagonalFtF(BlockSparseMatrix* block_diagonal) const {
  const CompressedRowBlockStructure* bs = matrix_.block_structure();
  const CompressedRowBlockStructure* block_diagonal_structure =
      block_diagonal->block_structure();

  block_diagonal->SetZero();
  const double* values = matrix_.values();
  for (int r = 0; r < num_row_blocks_e_; ++r) {
    const int row_block_size = bs->rows[r].block.size;
    const Cell* cells = bs->rows[r].cells;
    const size_t num_cells = bs->rows[r].num_cells;
    for (int c = 1; c < num_cells; ++c) {
      const int col_block_id = cells[c].block_id;
      const int col_block_size = bs->col_sizes[col_block_id];
      const int diagonal_block_id = col_block_id - num_col_blocks_e_;
      const int cell_position =
          block_diagonal_structure->rows[diagonal_block_id].cells[0].position;

      MatrixTransposeMatrixMultiply
          <kRowBlockSize, kFBlockSize, kRowBlockSize, kFBlockSize, 1>(
              values + cells[c].position, row_block_size, col_block_size,
              values + cells[c].position, row_block_size, col_block_size,
              block_diagonal->mutable_values() + cell_position,
              0, 0, col_block_size, col_block_size);
    }
  }

  for (int r = num_row_blocks_e_; r < bs->rows.size(); ++r) {
    const int row_block_size = bs->rows[r].block.size;
    const Cell* cells = bs->rows[r].cells;
    const size_t num_cells = bs->rows[r].num_cells;
    for (int c = 0; c < num_cells; ++c) {
      const int col_block_id = cells[c].block_id;
      const int col_block_size = bs->col_sizes[col_block_id];
      const int diagonal_block_id = col_block_id - num_col_blocks_e_;
      const int cell_position =
          block_diagonal_structure->rows[diagonal_block_id].cells[0].position;

      MatrixTransposeMatrixMultiply
          <Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic, 1>(
              values + cells[c].position, row_block_size, col_block_size,
              values + cells[c].position, row_block_size, col_block_size,
              block_diagonal->mutable_values() + cell_position,
              0, 0, col_block_size, col_block_size);
    }
  }
}

}  // namespace internal
}  // namespace ceres
