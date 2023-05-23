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
//
// TODO(sameeragarwal): row_block_counter can perhaps be replaced by
// Chunk::start ?

#ifndef CERES_INTERNAL_SCHUR_ELIMINATOR_IMPL_H_
#define CERES_INTERNAL_SCHUR_ELIMINATOR_IMPL_H_

// Eigen has an internal threshold switching between different matrix
// multiplication algorithms. In particular for matrices larger than
// EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD it uses a cache friendly
// matrix matrix product algorithm that has a higher setup cost. For
// matrix sizes close to this threshold, especially when the matrices
// are thin and long, the default choice may not be optimal. This is
// the case for us, as the default choice causes a 30% performance
// regression when we moved from Eigen2 to Eigen3.

#define EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD 10

// This include must come before any #ifndef check on Ceres compile options.
#include "ceres/internal/port.h"

#ifdef CERES_USE_OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <map>
#include "ceres/block_random_access_matrix.h"
#include "ceres/block_sparse_matrix.h"
#include "ceres/block_structure.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/fixed_array.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/invert_psd_matrix.h"
#include "ceres/map_util.h"
#include "ceres/schur_eliminator.h"
#include "ceres/small_blas.h"
#include "ceres/stl_util.h"
#include "Eigen/Dense"
#include "glog/logging.h"

namespace ceres {
namespace internal {

template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
SchurEliminator<kRowBlockSize, kEBlockSize, kFBlockSize>::~SchurEliminator() {
  STLDeleteElements(&rhs_locks_);
}

template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void SchurEliminator<kRowBlockSize, kEBlockSize, kFBlockSize>::Init(
    int num_eliminate_blocks,
    bool assume_full_rank_ete,
    const CompressedRowBlockStructure* bs) {
  CHECK_GT(num_eliminate_blocks, 0)
      << "SchurComplementSolver cannot be initialized with "
      << "num_eliminate_blocks = 0.";

  num_eliminate_blocks_ = num_eliminate_blocks;
  assume_full_rank_ete_ = assume_full_rank_ete;

  const int num_col_blocks = bs->col_sizes.size();
  const int num_row_blocks = bs->rows.size();

  buffer_size_ = 1;
  chunks_.clear();
  lhs_row_layout_.clear();

  int lhs_num_rows = 0;
  // Add a map object for each block in the reduced linear system
  // and build the row/column block structure of the reduced linear
  // system.
  lhs_row_layout_.resize(num_col_blocks - num_eliminate_blocks_);
  for (int i = num_eliminate_blocks_; i < num_col_blocks; ++i) {
    lhs_row_layout_[i - num_eliminate_blocks_] = lhs_num_rows;
    lhs_num_rows += bs->col_sizes[i];
  }

  int r = 0;
  // Iterate over the row blocks of A, and detect the chunks. The
  // matrix should already have been ordered so that all rows
  // containing the same y block are vertically contiguous. Along
  // the way also compute the amount of space each chunk will need
  // to perform the elimination.
  while (r < num_row_blocks) {
    const int chunk_block_id = bs->rows[r].cells.front().block_id;
    if (chunk_block_id >= num_eliminate_blocks_) {
      break;
    }

    chunks_.push_back(Chunk());
    Chunk& chunk = chunks_.back();
    chunk.size = 0;
    chunk.start = r;
    int buffer_size = 0;
    const int e_block_size = bs->col_sizes[chunk_block_id];

    // Add to the chunk until the first block in the row is
    // different than the one in the first row for the chunk.
    while (r + chunk.size < num_row_blocks) {
      const CompressedRow& row = bs->rows[r + chunk.size];
      if (row.cells.front().block_id != chunk_block_id) {
        break;
      }

      // Iterate over the blocks in the row, ignoring the first
      // block since it is the one to be eliminated.
      for (int c = 1, cnt = row.cells.size(); c < cnt; ++c) {
        const Cell& cell = row.cells[c];      
        for (const auto& i : chunk.buffer_layout) {
          if (i.first == cell.block_id) {
          	goto duplicate;
          }
        }
        chunk.buffer_layout.emplace_back(cell.block_id, buffer_size);
        buffer_size += e_block_size * bs->col_sizes[cell.block_id];
duplicate:
        ;
      }

      buffer_size_ = std::max(buffer_size, buffer_size_);
      ++chunk.size;
    }

    std::sort(std::begin(chunk.buffer_layout), std::end(chunk.buffer_layout),
      []( const auto& a, const auto& b ) {
        return a.first < b.first;
      }
    );    
    CHECK_GT(chunk.size, 0);
    r += chunk.size;
  }
  const Chunk& chunk = chunks_.back();

  uneliminated_row_begins_ = chunk.start + chunk.size;
  if (num_threads_ > 1) {
    random_shuffle(chunks_.begin(), chunks_.end());
  }

  buffer_.reset(new double[buffer_size_ * num_threads_]);

  // chunk_outer_product_buffer_ only needs to store e_block_size *
  // f_block_size, which is always less than buffer_size_, so we just
  // allocate buffer_size_ per thread.
  chunk_outer_product_buffer_.reset(new double[buffer_size_ * num_threads_]);

  STLDeleteElements(&rhs_locks_);
  rhs_locks_.resize(num_col_blocks - num_eliminate_blocks_);
  for (int i = 0; i < num_col_blocks - num_eliminate_blocks_; ++i) {
    rhs_locks_[i] = new Mutex;
  }
}

template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void
SchurEliminator<kRowBlockSize, kEBlockSize, kFBlockSize>::
Eliminate(const BlockSparseMatrix* A,
          const double* b,
          const double* D,
          BlockRandomAccessMatrix* lhs,
          double* rhs) {
  if (lhs->num_rows() > 0) {
    lhs->SetZero();
    VectorRef(rhs, lhs->num_rows()).setZero();
  }

  const CompressedRowBlockStructure* bs = A->block_structure();
  const auto& bs_col_sizes = bs->col_sizes;
  const auto& bs_col_positions = bs->col_positions;
  const int num_col_blocks = bs_col_sizes.size();

  // Add the diagonal to the schur complement.
  if (D != NULL) {
#pragma omp parallel for num_threads(num_threads_) schedule(dynamic)
    for (int i = num_eliminate_blocks_; i < num_col_blocks; ++i) {
      const int block_id = i - num_eliminate_blocks_;
      int r, c, row_stride, col_stride;
      CellInfo* cell_info = lhs->GetCell(block_id, block_id,
                                         &r, &c,
                                         &row_stride, &col_stride);
      if (cell_info) {
        const int block_size = bs_col_sizes[i];
        typename EigenTypes<Eigen::Dynamic>::ConstVectorRef
          diag(D + bs_col_positions[i], block_size);

        MatrixRef m(cell_info->values, row_stride, col_stride);
        // Removed in 2.1+? CeresMutexLock l(&cell_info->m);
        m.block(r, c, block_size, block_size).diagonal()
          += diag.array().square().matrix();
      }
    }
  }

  // Eliminate y blocks one chunk at a time.  For each chunk, compute
  // the entries of the normal equations and the gradient vector block
  // corresponding to the y block and then apply Gaussian elimination
  // to them. The matrix ete stores the normal matrix corresponding to
  // the block being eliminated and array buffer_ contains the
  // non-zero blocks in the row corresponding to this y block in the
  // normal equations. This computation is done in
  // ChunkDiagonalBlockAndGradient. UpdateRhs then applies gaussian
  // elimination to the rhs of the normal equations, updating the rhs
  // of the reduced linear system by modifying rhs blocks for all the
  // z blocks that share a row block/residual term with the y
  // block. EliminateRowOuterProduct does the corresponding operation
  // for the lhs of the reduced linear system.

  const int cnt = chunks_.size();

  MatrixType matrix_type;
  if (typeid(*lhs) == typeid(BlockRandomAccessDenseMatrix)) {
    matrix_type = eDense;
  } else if (typeid(*lhs) == typeid(BlockRandomAccessDiagonalMatrix)) {
    matrix_type = eDiagonal;
  } else if (typeid(*lhs) == typeid(BlockRandomAccessSparseMatrix)) {
    matrix_type = eSparse;
  } else {
    throw "Unknown matrix lhs.";
  }

#pragma omp parallel for num_threads(num_threads_) schedule(dynamic)
  for (int i = 0; i < cnt; ++i) {
#ifdef CERES_USE_OPENMP
    int thread_id = omp_get_thread_num();
#else
    int thread_id = 0;
#endif
    double* buffer = buffer_.get() + thread_id * buffer_size_;
    const Chunk& chunk = chunks_[i];
    const auto chunk_start = chunk.start;
    const int e_block_id = bs->rows[chunk_start].cells.front().block_id;
    const int e_block_size = bs_col_sizes[e_block_id];

    VectorRef(buffer, buffer_size_).setZero();

    typename EigenTypes<kEBlockSize, kEBlockSize>::Matrix
      ete(e_block_size, e_block_size);

    if (D != NULL) {
      const typename EigenTypes<kEBlockSize>::ConstVectorRef
        diag(D + bs_col_positions[e_block_id], e_block_size);
      ete = diag.array().square().matrix().asDiagonal();
    } else {
      ete.setZero();
    }

    FixedArray<double, 8> g(e_block_size);
    typename EigenTypes<kEBlockSize>::VectorRef gref(g.get(), e_block_size);
    gref.setZero();

    // We are going to be computing
    //
    //   S += F'F - F'E(E'E)^{-1}E'F
    //
    // for each Chunk. The computation is broken down into a number of
    // function calls as below.

    // Compute the outer product of the e_blocks with themselves (ete
    // = E'E). Compute the product of the e_blocks with the
    // corresponding f_blocks (buffer = E'F), the gradient of the terms
    // in this chunk (g) and add the outer product of the f_blocks to
    // Schur complement (S += F'F).
    ChunkDiagonalBlockAndGradient(
        chunk, A, b, chunk_start, &ete, g.get(), buffer, lhs, matrix_type);

    // Normally one wouldn't compute the inverse explicitly, but
    // e_block_size will typically be a small number like 3, in
    // which case its much faster to compute the inverse once and
    // use it to multiply other matrices/vectors instead of doing a
    // Solve call over and over again.
    typename EigenTypes<kEBlockSize, kEBlockSize>::Matrix inverse_ete =
      InvertPSDMatrix<kEBlockSize>(assume_full_rank_ete_, ete);

    // For the current chunk compute and update the rhs of the reduced
    // linear system.
    //
    //   rhs = F'b - F'E(E'E)^(-1) E'b

    FixedArray<double, 8> inverse_ete_g(e_block_size);
    MatrixVectorMultiply<kEBlockSize, kEBlockSize, 0>(
      inverse_ete.data(),
      e_block_size,
      e_block_size,
      g.get(),
      inverse_ete_g.get());

    UpdateRhs(chunk, A, b, chunk_start, inverse_ete_g.get(), rhs);

    // S -= F'E(E'E)^{-1}E'F
    if (eDense == matrix_type) {
      ChunkOuterProduct<BlockRandomAccessDenseMatrix>(bs, inverse_ete, buffer, chunk.buffer_layout, (BlockRandomAccessDenseMatrix*) lhs, thread_id);
    } else if (eDiagonal == matrix_type) {
      ChunkOuterProduct<BlockRandomAccessDiagonalMatrix>(bs, inverse_ete, buffer, chunk.buffer_layout, (BlockRandomAccessDiagonalMatrix*) lhs, thread_id);
    } else {
      ChunkOuterProduct<BlockRandomAccessSparseMatrix>(bs, inverse_ete, buffer, chunk.buffer_layout, (BlockRandomAccessSparseMatrix*) lhs, thread_id);
    }
  }

  // For rows with no e_blocks, the schur complement update reduces to
  // S += F'F.
  NoEBlockRowsUpdate(A, b,  uneliminated_row_begins_, lhs, rhs);
}

template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void
SchurEliminator<kRowBlockSize, kEBlockSize, kFBlockSize>::
BackSubstitute(const BlockSparseMatrix* A,
               const double* b,
               const double* D,
               const double* z,
               double* y) {
  const CompressedRowBlockStructure* __restrict bs = A->block_structure();
  const double* values = A->values();
  const int cnt = (int) chunks_.size();
  const auto& bs_rows = bs->rows;
  const auto& bs_col_sizes = bs->col_sizes;
  const auto& bs_col_positions = bs->col_positions;

#pragma omp parallel for num_threads(num_threads_) schedule(dynamic)
  for (int i = 0; i < cnt; ++i) {
    const Chunk& chunk = chunks_[i];
    const auto chunk_start = chunk.start;
    const int e_block_id = bs_rows[chunk_start].cells.front().block_id;
    const auto e_block_size = bs_col_sizes[e_block_id];
    const auto e_block_position = bs_col_positions[e_block_id];

    double* y_ptr = y + e_block_position;
    typename EigenTypes<kEBlockSize>::VectorRef y_block(y_ptr, e_block_size);

    typename EigenTypes<kEBlockSize, kEBlockSize>::Matrix
        ete(e_block_size, e_block_size);
    if (D != NULL) {
      const typename EigenTypes<kEBlockSize>::ConstVectorRef
          diag(D + e_block_position, e_block_size);
      ete = diag.array().square().matrix().asDiagonal();
    } else {
      ete.setZero();
    }

    SmallBiasHelper sbhOuter1;
    sbhOuter1.num_col_a_ = e_block_size;
    sbhOuter1.C_ = y_ptr;

    SmallBiasHelper sbhOuter2;
    sbhOuter2.num_col_a_ = e_block_size;
    sbhOuter2.num_col_b_ = e_block_size;
    sbhOuter2.cih_.r_ = 0;
    sbhOuter2.cih_.c_ = 0;
    sbhOuter2.cih_.row_stride_ = e_block_size;
    sbhOuter2.cih_.col_stride_ = e_block_size;
    sbhOuter2.C_ = ete.data();

    SmallBiasHelper sbhInner;

    for (size_t j = 0, cnt2 = chunk.size; j < cnt2; ++j) {
      const CompressedRow& row = bs_rows[chunk_start + j];
      const auto& row_cells = row.cells;
      const auto row_block_size = row.block.size;
      const Cell& e_cell = row_cells.front();
      const auto* e_cell_values = values + e_cell.position;
      DCHECK_EQ(e_block_id, e_cell.block_id);

      FixedArray<double, 8> sj(row_block_size);

      typename EigenTypes<kRowBlockSize>::VectorRef(sj.get(), row_block_size) =
        typename EigenTypes<kRowBlockSize>::ConstVectorRef
        (b + row.block.position, row_block_size);

      sbhInner.num_row_a_ = row_block_size;
      sbhInner.C_ = sj.get();

      for (size_t c = 1, cnt3 = row_cells.size(); c < cnt3; ++c) {
        const auto& row_cell_c = row_cells[c];
        const int f_block_id = row_cell_c.block_id;
        const int f_block_size = bs_col_sizes[f_block_id];
        const int r_block = f_block_id - num_eliminate_blocks_;

        sbhInner.A_ = values + row_cell_c.position;
        sbhInner.num_col_a_ = f_block_size;
        sbhInner.B_ = z + lhs_row_layout_[r_block];

        MatrixVectorMultiply2<kRowBlockSize, kFBlockSize, -1>(sbhInner);
      }

      sbhOuter1.A_ = e_cell_values;
      sbhOuter1.num_row_a_ = row_block_size;
      sbhOuter1.B_ = sj.get();

      MatrixTransposeVectorMultiply2<kRowBlockSize, kEBlockSize, 1>(sbhOuter1);

      sbhOuter2.A_ = e_cell_values;
      sbhOuter2.num_row_a_ = row_block_size;
      sbhOuter2.B_ = e_cell_values;
      sbhOuter2.num_row_b_ = row_block_size;

      MatrixTransposeMatrixMultiply2
          <kRowBlockSize, kEBlockSize, kRowBlockSize, kEBlockSize, 1>(sbhOuter2);
    }

    y_block = InvertPSDMatrix<kEBlockSize>(assume_full_rank_ete_, ete)
        * y_block;
  }
}

// Update the rhs of the reduced linear system. Compute
//
//   F'b - F'E(E'E)^(-1) E'b
template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void
SchurEliminator<kRowBlockSize, kEBlockSize, kFBlockSize>::
UpdateRhs(const Chunk& chunk,
          const BlockSparseMatrix* A,
          const double* b,
          int row_block_counter,
          const double* inverse_ete_g,
          double* rhs) {
  const CompressedRowBlockStructure* bs = A->block_structure();
  const auto& bs_rows = bs->rows;
  const auto& bs_cols_sizes = bs->col_sizes;
  const double* values = A->values();
  const int e_block_id = bs_rows[chunk.start].cells.front().block_id;
  const int e_block_size = bs_cols_sizes[e_block_id];

  SmallBiasHelper sbhOuter;
  sbhOuter.num_col_a_ = e_block_size;
  sbhOuter.B_ = inverse_ete_g;

  SmallBiasHelper sbhInner;

  int b_pos = bs_rows[row_block_counter].block.position;
  for (size_t j = 0, cnt = chunk.size; j < cnt; ++j) {
    const CompressedRow& row = bs_rows[row_block_counter + j];
    const auto row_block_size = row.block.size;
    const Cell& e_cell = row.cells.front();

    typename EigenTypes<kRowBlockSize>::Vector sj =
      typename EigenTypes<kRowBlockSize>::ConstVectorRef
        (b + b_pos, row_block_size);

    sbhOuter.A_ = values + e_cell.position;
    sbhOuter.num_row_a_ = row_block_size;
    sbhOuter.C_ = sj.data();

    sbhInner.num_row_a_ = row_block_size;
    sbhInner.B_ = sj.data();

    MatrixVectorMultiply2<kRowBlockSize, kEBlockSize, -1>(sbhOuter);

    for (size_t c = 1, cnt2 = row.cells.size(); c < cnt2; ++c) {
      const auto& row_cell_c = row.cells[c];
      const int block_id = row_cell_c.block_id;
      sbhInner.num_col_a_ = bs_cols_sizes[block_id];
      const int block = block_id - num_eliminate_blocks_;
      sbhInner.A_ = values + row_cell_c.position;
      sbhInner.C_ = rhs + lhs_row_layout_[block];

      const auto& lock = rhs_locks_[block];
      if (num_threads_ > 1) {
        lock->Lock();
      }

      MatrixTransposeVectorMultiply2<kRowBlockSize, kFBlockSize, 1>(sbhInner);

      if (num_threads_ > 1) {
        lock->Unlock();
      }
    }
    b_pos += row_block_size;
  }
}

// Given a Chunk - set of rows with the same e_block, e.g. in the
// following Chunk with two rows.
//
//                E                   F
//      [ y11   0   0   0 |  z11     0    0   0    z51]
//      [ y12   0   0   0 |  z12   z22    0   0      0]
//
// this function computes twp matrices. The diagonal block matrix
//
//   ete = y11 * y11' + y12 * y12'
//
// and the off diagonal blocks in the Guass Newton Hessian.
//
//   buffer = [y11'(z11 + z12), y12' * z22, y11' * z51]
//
// which are zero compressed versions of the block sparse matrices E'E
// and E'F.
//
// and the gradient of the e_block, E'b.
template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void
SchurEliminator<kRowBlockSize, kEBlockSize, kFBlockSize>::
ChunkDiagonalBlockAndGradient(
    const Chunk& chunk,
    const BlockSparseMatrix* A,
    const double* b,
    int row_block_counter,
    typename EigenTypes<kEBlockSize, kEBlockSize>::Matrix* ete,
    double* g,
    double* buffer,
    BlockRandomAccessMatrix* lhs,
    MatrixType lhsType) {
  const CompressedRowBlockStructure* bs = A->block_structure();
  const double* values = A->values();
  const auto& bs_rows = bs->rows;
  const auto& bs_cols_sizes = bs->col_sizes;
  int b_pos = bs_rows[row_block_counter].block.position;
  const int e_block_size = ete->rows();
  auto chunk_buffer_layout_start = std::begin(chunk.buffer_layout);
  auto chunk_buffer_layout_end = std::end(chunk.buffer_layout);
  std::pair<int, int> tmp;
  SmallBiasHelper sbhOuter;
  sbhOuter.num_col_a_ = e_block_size;
  sbhOuter.num_col_b_ = e_block_size;
  sbhOuter.C_ = ete->data();
  sbhOuter.cih_.r_ = 0;
  sbhOuter.cih_.c_ = 0;
  sbhOuter.cih_.row_stride_ = e_block_size;
  sbhOuter.cih_.col_stride_ = e_block_size;

  SmallBiasHelper sbhOuterVector;
  sbhOuterVector.num_col_a_ = e_block_size;
  sbhOuterVector.C_ = g;

  SmallBiasHelper sbhInner;
  sbhInner.num_col_a_ = e_block_size;
  sbhInner.cih_.r_ = 0;
  sbhInner.cih_.c_ = 0;
  sbhInner.cih_.row_stride_ = e_block_size;

  // Iterate over the rows in this chunk, for each row, compute the
  // contribution of its F blocks to the Schur complement, the
  // contribution of its E block to the matrix EE' (ete), and the
  // corresponding block in the gradient vector.
  for (int j = 0, cnt = chunk.size; j < cnt; ++j) {
    const CompressedRow& row = bs_rows[row_block_counter + j];
    const auto& row_cells = row.cells;
    const auto cnt2 = row_cells.size();
    const auto row_block_size = row.block.size;
    sbhOuter.num_row_a_ = row_block_size;
    sbhOuter.num_row_b_ = row_block_size;
    sbhOuterVector.num_row_a_ = row_block_size;

    sbhInner.num_row_a_ = row_block_size;
    sbhInner.num_row_b_ = row_block_size;
    const Cell& e_cell = row_cells.front();

    if (cnt2 > 1) {
      if (eDense == lhsType) {
        EBlockRowOuterProduct<BlockRandomAccessDenseMatrix>(bs, values, row_block_counter + j, (BlockRandomAccessDenseMatrix*) lhs);
      } else if (eDiagonal == lhsType) {
        EBlockRowOuterProduct<BlockRandomAccessDiagonalMatrix>(bs, values, row_block_counter + j, (BlockRandomAccessDiagonalMatrix*) lhs);
      } else {
        EBlockRowOuterProduct<BlockRandomAccessSparseMatrix>(bs, values, row_block_counter + j, (BlockRandomAccessSparseMatrix*) lhs);
      }
    }

    const auto* e_cell_values = values + e_cell.position;

    sbhOuter.A_ = e_cell_values;
    sbhOuter.B_ = e_cell_values;
    sbhOuterVector.A_ = e_cell_values;
    sbhOuterVector.B_ = b + b_pos;
    sbhInner.A_ = e_cell_values;

    // Extract the e_block, ETE += E_i' E_i
    MatrixTransposeMatrixMultiply2
      <kRowBlockSize, kEBlockSize, kRowBlockSize, kEBlockSize, 1>(sbhOuter);

    // g += E_i' b_i
    MatrixTransposeVectorMultiply2<kRowBlockSize, kEBlockSize, 1>(sbhOuterVector);

    // buffer = E'F. This computation is done by iterating over the
    // f_blocks for each row in the chunk.
    for (size_t c = 1; c < cnt2; ++c) {
      const auto& row_cell_c = row_cells[c];
      const int f_block_id = row_cell_c.block_id;
      const int f_block_size = bs_cols_sizes[f_block_id];
      tmp.first = f_block_id;
      const auto it = std::lower_bound(
        chunk_buffer_layout_start,
        chunk_buffer_layout_end,
        tmp,
        [](const auto& lhs, const auto& rhs)
        {
          return lhs.first < rhs.first;
        }
      );

      sbhInner.B_ = values + row_cell_c.position;
      sbhInner.num_col_b_ = f_block_size;
      sbhInner.C_ = buffer + it->second;
      sbhInner.cih_.col_stride_ = f_block_size;

      MatrixTransposeMatrixMultiply2
        <kRowBlockSize, kEBlockSize, kRowBlockSize, kFBlockSize, 1>(sbhInner);
    }
    b_pos += row_block_size;
  }
}

// For rows with no e_blocks, the schur complement update reduces to S
// += F'F. This function iterates over the rows of A with no e_block,
// and calls NoEBlockRowOuterProduct on each row.
template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void
SchurEliminator<kRowBlockSize, kEBlockSize, kFBlockSize>::
NoEBlockRowsUpdate(const BlockSparseMatrix* A,
                   const double* b,
                   int row_block_counter,
                   BlockRandomAccessMatrix* lhs,
                   double* rhs) {
  const CompressedRowBlockStructure* bs = A->block_structure();
  const double* values = A->values();
  const auto& bs_rows = bs->rows;
  const auto& bs_cols_sizes = bs->col_sizes;

  SmallBiasHelper sbh;

  for (int cnt = (int) bs_rows.size(); row_block_counter < cnt; ++row_block_counter) {
    NoEBlockRowOuterProduct(A, row_block_counter, lhs);

    const CompressedRow& row = bs_rows[row_block_counter];
    const auto& row_cells = row.cells;
    sbh.num_row_a_ = row.block.size;
    sbh.B_ = b + row.block.position;

    for (size_t c = 0, cnt2 = row_cells.size(); c < cnt2; ++c) {
      const auto& row_cell_c = row_cells[c];
      const int block_id = row_cell_c.block_id;
      const auto block_size = bs_cols_sizes[block_id];

      sbh.A_ = values + row_cell_c.position;
      sbh.num_col_a_ = block_size;

      const int block = block_id - num_eliminate_blocks_;

      sbh.C_ = rhs + lhs_row_layout_[block];

      MatrixTransposeVectorMultiply2<Eigen::Dynamic, Eigen::Dynamic, 1>(sbh);
    }
  }
}


// A row r of A, which has no e_blocks gets added to the Schur
// Complement as S += r r'. This function is responsible for computing
// the contribution of a single row r to the Schur complement. It is
// very similar in structure to EBlockRowOuterProduct except for
// one difference. It does not use any of the template
// parameters. This is because the algorithm used for detecting the
// static structure of the matrix A only pays attention to rows with
// e_blocks. This is because rows without e_blocks are rare and
// typically arise from regularization terms in the original
// optimization problem, and have a very different structure than the
// rows with e_blocks. Including them in the static structure
// detection will lead to most template parameters being set to
// dynamic. Since the number of rows without e_blocks is small, the
// lack of templating is not an issue.
template <int kRowBlockSize, int kEBlockSize, int kFBlockSize>
void
SchurEliminator<kRowBlockSize, kEBlockSize, kFBlockSize>::
NoEBlockRowOuterProduct(const BlockSparseMatrix* A,
                        int row_block_index,
                        BlockRandomAccessMatrix* lhs) {
  const CompressedRowBlockStructure* bs = A->block_structure();
  const CompressedRow& row = bs->rows[row_block_index];
  const auto& bs_col_sizes = bs->col_sizes;
  const auto row_block_size = row.block.size;
  const double* values = A->values();
  const auto& row_cells = row.cells;

  SmallBiasHelper sbhOuter;
  sbhOuter.num_row_a_ = row_block_size;
  sbhOuter.num_row_b_ = row_block_size;

  SmallBiasHelper sbhInner;
  sbhInner.num_row_a_ = row_block_size;
  sbhInner.num_row_b_ = row_block_size;

  for (size_t i = 0, cnt = row_cells.size(); i < cnt; ++i) {
    const auto& row_cell_i = row_cells[i];
    const double* row_cell_i_values_position = values + row_cell_i.position;
    const int block1 = row_cell_i.block_id - num_eliminate_blocks_;
    DCHECK_GE(block1, 0);

    const int block1_size = bs_col_sizes[row_cell_i.block_id];
    sbhInner.num_col_a_ = block1_size;
    sbhInner.A_ = row_cell_i_values_position;

    CellInfo* __restrict cell_info = lhs->GetCell(block1, block1,
                                                  &sbhOuter.cih_.r_, &sbhOuter.cih_.c_,
                                                  &sbhOuter.cih_.row_stride_, &sbhOuter.cih_.col_stride_);
    if (cell_info) {
      sbhOuter.A_ = row_cell_i_values_position;
      sbhOuter.num_col_a_ = block1_size;
      sbhOuter.B_ = row_cell_i_values_position;
      sbhOuter.num_col_b_ = block1_size;
      sbhOuter.C_ = cell_info->values;

      auto& lock = cell_info->m;
      if (num_threads_ > 1) {
        lock.Lock();
      }

      // This multiply currently ignores the fact that this is a
      // symmetric outer product.
      MatrixTransposeMatrixMultiply2
          <Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic, 1>(sbhOuter);

      if (num_threads_ > 1) {
        lock.Unlock();
      }
    }

    lhs->PrepareGetCellHelper(sbhInner.cih_, block1);

    for (size_t j = i + 1; j < cnt; ++j) {
      const auto& row_cell_j = row_cells[j];

      const int block2 = row_cell_j.block_id - num_eliminate_blocks_;
      DCHECK_GE(block2, 0);
      DCHECK_LT(block1, block2);
      CellInfo* __restrict cell_info = lhs->GetCellHelped(sbhInner.cih_, block2);

      if (cell_info) {
        sbhInner.B_ = values + row_cell_j.position;
        sbhInner.num_col_b_ = bs_col_sizes[row_cell_j.block_id];

        auto& lock = cell_info->m;
        if (num_threads_ > 1) {
          lock.Lock();
        }

        MatrixTransposeMatrixMultiply2
            <Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic, 1>(sbhInner);

        if (num_threads_ > 1) {
          lock.Unlock();
        }
      }
    }
  }
}

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_SCHUR_ELIMINATOR_IMPL_H_
