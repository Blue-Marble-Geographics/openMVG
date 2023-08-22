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

#ifndef CERES_INTERNAL_SCHUR_ELIMINATOR_H_
#define CERES_INTERNAL_SCHUR_ELIMINATOR_H_

#include <map>
#include <vector>
#include "ceres/mutex.h"
#include "ceres/block_random_access_matrix.h"
#include "ceres/block_random_access_dense_matrix.h"
#include "ceres/block_random_access_diagonal_matrix.h"
#include "ceres/block_sparse_matrix.h"
#include "ceres/block_random_access_sparse_matrix.h"
#include "ceres/block_structure.h"
#include "ceres/linear_solver.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/scoped_ptr.h"

namespace ceres {
namespace internal {

enum MatrixType { eDense, eDiagonal, eSparse };

// Classes implementing the SchurEliminatorBase interface implement
// variable elimination for linear least squares problems. Assuming
// that the input linear system Ax = b can be partitioned into
//
//  E y + F z = b
//
// Where x = [y;z] is a partition of the variables.  The paritioning
// of the variables is such that, E'E is a block diagonal matrix. Or
// in other words, the parameter blocks in E form an independent set
// of the of the graph implied by the block matrix A'A. Then, this
// class provides the functionality to compute the Schur complement
// system
//
//   S z = r
//
// where
//
//   S = F'F - F'E (E'E)^{-1} E'F and r = F'b - F'E(E'E)^(-1) E'b
//
// This is the Eliminate operation, i.e., construct the linear system
// obtained by eliminating the variables in E.
//
// The eliminator also provides the reverse functionality, i.e. given
// values for z it can back substitute for the values of y, by solving the
// linear system
//
//  Ey = b - F z
//
// which is done by observing that
//
//  y = (E'E)^(-1) [E'b - E'F z]
//
// The eliminator has a number of requirements.
//
// The rows of A are ordered so that for every variable block in y,
// all the rows containing that variable block occur as a vertically
// contiguous block. i.e the matrix A looks like
//
//              E                 F                   chunk
//  A = [ y1   0   0   0 |  z1    0    0   0    z5]     1
//      [ y1   0   0   0 |  z1   z2    0   0     0]     1
//      [  0  y2   0   0 |   0    0   z3   0     0]     2
//      [  0   0  y3   0 |  z1   z2   z3  z4    z5]     3
//      [  0   0  y3   0 |  z1    0    0   0    z5]     3
//      [  0   0   0  y4 |   0    0    0   0    z5]     4
//      [  0   0   0  y4 |   0   z2    0   0     0]     4
//      [  0   0   0  y4 |   0    0    0   0     0]     4
//      [  0   0   0   0 |  z1    0    0   0     0] non chunk blocks
//      [  0   0   0   0 |   0    0   z3  z4    z5] non chunk blocks
//
// This structure should be reflected in the corresponding
// CompressedRowBlockStructure object associated with A. The linear
// system Ax = b should either be well posed or the array D below
// should be non-null and the diagonal matrix corresponding to it
// should be non-singular. For simplicity of exposition only the case
// with a null D is described.
//
// The usual way to do the elimination is as follows. Starting with
//
//  E y + F z = b
//
// we can form the normal equations,
//
//  E'E y + E'F z = E'b
//  F'E y + F'F z = F'b
//
// multiplying both sides of the first equation by (E'E)^(-1) and then
// by F'E we get
//
//  F'E y + F'E (E'E)^(-1) E'F z =  F'E (E'E)^(-1) E'b
//  F'E y +                F'F z =  F'b
//
// now subtracting the two equations we get
//
// [FF' - F'E (E'E)^(-1) E'F] z = F'b - F'E(E'E)^(-1) E'b
//
// Instead of forming the normal equations and operating on them as
// general sparse matrices, the algorithm here deals with one
// parameter block in y at a time. The rows corresponding to a single
// parameter block yi are known as a chunk, and the algorithm operates
// on one chunk at a time. The mathematics remains the same since the
// reduced linear system can be shown to be the sum of the reduced
// linear systems for each chunk. This can be seen by observing two
// things.
//
//  1. E'E is a block diagonal matrix.
//
//  2. When E'F is computed, only the terms within a single chunk
//  interact, i.e for y1 column blocks when transposed and multiplied
//  with F, the only non-zero contribution comes from the blocks in
//  chunk1.
//
// Thus, the reduced linear system
//
//  FF' - F'E (E'E)^(-1) E'F
//
// can be re-written as
//
//  sum_k F_k F_k' - F_k'E_k (E_k'E_k)^(-1) E_k' F_k
//
// Where the sum is over chunks and E_k'E_k is dense matrix of size y1
// x y1.
//
// Advanced usage. Uptil now it has been assumed that the user would
// be interested in all of the Schur Complement S. However, it is also
// possible to use this eliminator to obtain an arbitrary submatrix of
// the full Schur complement. When the eliminator is generating the
// blocks of S, it asks the RandomAccessBlockMatrix instance passed to
// it if it has storage for that block. If it does, the eliminator
// computes/updates it, if not it is skipped. This is useful when one
// is interested in constructing a preconditioner based on the Schur
// Complement, e.g., computing the block diagonal of S so that it can
// be used as a preconditioner for an Iterative Substructuring based
// solver [See Agarwal et al, Bundle Adjustment in the Large, ECCV
// 2008 for an example of such use].
//
// Example usage: Please see schur_complement_solver.cc
class SchurEliminatorBase {
 public:
  virtual ~SchurEliminatorBase() {}

  // Initialize the eliminator. It is the user's responsibilty to call
  // this function before calling Eliminate or BackSubstitute. It is
  // also the caller's responsibilty to ensure that the
  // CompressedRowBlockStructure object passed to this method is the
  // same one (or is equivalent to) the one associated with the
  // BlockSparseMatrix objects below.
  //
  // assume_full_rank_ete controls how the eliminator inverts with the
  // diagonal blocks corresponding to e blocks in A'A. If
  // assume_full_rank_ete is true, then a Cholesky factorization is
  // used to compute the inverse, otherwise a singular value
  // decomposition is used to compute the pseudo inverse.
  virtual void Init(int num_eliminate_blocks,
                    bool assume_full_rank_ete,
                    const CompressedRowBlockStructure* bs) = 0;

  // Compute the Schur complement system from the augmented linear
  // least squares problem [A;D] x = [b;0]. The left hand side and the
  // right hand side of the reduced linear system are returned in lhs
  // and rhs respectively.
  //
  // It is the caller's responsibility to construct and initialize
  // lhs. Depending upon the structure of the lhs object passed here,
  // the full or a submatrix of the Schur complement will be computed.
  //
  // Since the Schur complement is a symmetric matrix, only the upper
  // triangular part of the Schur complement is computed.
  virtual void Eliminate(const BlockSparseMatrix* A,
                         const double* b,
                         const double* D,
                         BlockRandomAccessMatrix* lhs,
                         double* rhs) = 0;

  // Given values for the variables z in the F block of A, solve for
  // the optimal values of the variables y corresponding to the E
  // block in A.
  virtual void BackSubstitute(const BlockSparseMatrix* A,
                              const double* b,
                              const double* D,
                              const double* z,
                              double* y) = 0;
  // Factory
  static SchurEliminatorBase* Create(const LinearSolver::Options& options);
};

// Templated implementation of the SchurEliminatorBase interface. The
// templating is on the sizes of the row, e and f blocks sizes in the
// input matrix. In many problems, the sizes of one or more of these
// blocks are constant, in that case, its worth passing these
// parameters as template arguments so that they are visible to the
// compiler and can be used for compile time optimization of the low
// level linear algebra routines.
//
// This implementation is mulithreaded using OpenMP. The level of
// parallelism is controlled by LinearSolver::Options::num_threads.
template <int kRowBlockSize = Eigen::Dynamic,
          int kEBlockSize = Eigen::Dynamic,
          int kFBlockSize = Eigen::Dynamic >
class SchurEliminator : public SchurEliminatorBase {
 public:
  explicit SchurEliminator(const LinearSolver::Options& options)
      : num_threads_(options.num_threads) {
  }

  // SchurEliminatorBase Interface
  virtual ~SchurEliminator();
  virtual void Init(int num_eliminate_blocks,
                    bool assume_full_rank_ete,
                    const CompressedRowBlockStructure* bs);
  virtual void Eliminate(const BlockSparseMatrix* A,
                         const double* b,
                         const double* D,
                         BlockRandomAccessMatrix* lhs,
                         double* rhs);
  virtual void BackSubstitute(const BlockSparseMatrix* A,
                              const double* b,
                              const double* D,
                              const double* z,
                              double* y);

 private:
  // Chunk objects store combinatorial information needed to
  // efficiently eliminate a whole chunk out of the least squares
  // problem. Consider the first chunk in the example matrix above.
  //
  //      [ y1   0   0   0 |  z1    0    0   0    z5]
  //      [ y1   0   0   0 |  z1   z2    0   0     0]
  //
  // One of the intermediate quantities that needs to be calculated is
  // for each row the product of the y block transposed with the
  // non-zero z block, and the sum of these blocks across rows. A
  // temporary array "buffer_" is used for computing and storing them
  // and the buffer_layout maps the indices of the z-blocks to
  // position in the buffer_ array.  The size of the chunk is the
  // number of row blocks/residual blocks for the particular y block
  // being considered.
  //
  // For the example chunk shown above,
  //
  // size = 2
  //
  // The entries of buffer_layout will be filled in the following order.
  //
  // buffer_layout[z1] = 0
  // buffer_layout[z5] = y1 * z1
  // buffer_layout[z2] = y1 * z1 + y1 * z5
  typedef std::vector<std::pair<int, int>> BufferLayoutType;
  struct Chunk {
    Chunk() : size(0) {}
    int size;
    int start;
    BufferLayoutType buffer_layout;
  };

  void ChunkDiagonalBlockAndGradient(
      const Chunk& chunk,
      const BlockSparseMatrix* A,
      const double* b,
      int row_block_counter,
      typename EigenTypes<kEBlockSize, kEBlockSize>::Matrix* eet,
      double* g,
      double* buffer,
      BlockRandomAccessMatrix* lhs,
      MatrixType lhsType);

  void UpdateRhs(const Chunk& chunk,
                 const BlockSparseMatrix* A,
                 const double* b,
                 int row_block_counter,
                 const double* inverse_ete_g,
                 double* rhs);

  // Compute the outer product F'E(E'E)^{-1}E'F and subtract it from the
  // Schur complement matrix, i.e
  //
  //  S -= F'E(E'E)^{-1}E'F.
  template<class T>
  void ChunkOuterProduct(const CompressedRowBlockStructure* bs,
                         const typename EigenTypes<kEBlockSize, kEBlockSize>::Matrix& inverse_ete,
                         const double* buffer,
                         const BufferLayoutType& buffer_layout,
                         T* lhs,
                         int thread_id) {
    // This is the most computationally expensive part of this
    // code. Profiling experiments reveal that the bottleneck is not the
    // computation of the right-hand matrix product, but memory
    // references to the left hand side.
    const int e_block_size = inverse_ete.rows();

    double* __restrict b1_transpose_inverse_ete =
      chunk_outer_product_buffer_.get() + thread_id * buffer_size_;
    const auto& bs_cols_sizes = bs->col_sizes;

    SmallBiasHelper sbhOuter;
    sbhOuter.num_row_a_ = e_block_size;
    sbhOuter.B_ = inverse_ete.data();
    sbhOuter.num_row_b_ = e_block_size;
    sbhOuter.num_col_b_ = e_block_size;
    sbhOuter.C_ = b1_transpose_inverse_ete;
    sbhOuter.cih_.r_ = 0;
    sbhOuter.cih_.c_ = 0;
    sbhOuter.cih_.col_stride_ = e_block_size;

    SmallBiasHelper sbhInner;
    sbhInner.A_ = b1_transpose_inverse_ete;
    sbhInner.num_col_a_ = e_block_size;
    sbhInner.num_row_b_ = e_block_size;

    // S(i,j) -= bi' * ete^{-1} b_j

    const int cnt = buffer_layout.size();
    for (auto i = 0; i < cnt; ++i) {
      const auto& layout_i = buffer_layout[i];
      const int block1 = layout_i.first - num_eliminate_blocks_;

      sbhOuter.A_ = buffer + layout_i.second;

      lhs->T::PrepareGetCellHelper(sbhInner.cih_, block1);

      const int block1_size = bs_cols_sizes[layout_i.first];
      sbhOuter.num_col_a_ = block1_size;
      sbhInner.num_row_a_ = block1_size;
      sbhOuter.cih_.row_stride_ = block1_size;

      MatrixTransposeMatrixMultiply2
        <kEBlockSize, kFBlockSize, kEBlockSize, kEBlockSize, 0>(sbhOuter);

      for (auto j = i; j < cnt; ++j) {
        const auto& layout_j = buffer_layout[j];
        const int block2 = layout_j.first - num_eliminate_blocks_;

        CellInfo* __restrict cell_info = lhs->T::GetCellHelped(sbhInner.cih_, block2);
        if (cell_info) {
          const int block2_size = bs_cols_sizes[layout_j.first];
          sbhInner.B_ = buffer + layout_j.second;
          sbhInner.num_col_b_ = block2_size;
          sbhInner.C_ = cell_info->values;

          auto& lock = cell_info->m;
          if (num_threads_ > 1) {
            lock.Lock();
          }

          MatrixMatrixMultiply2
            <kFBlockSize, kEBlockSize, kEBlockSize, kFBlockSize, -1>(sbhInner);

          if (num_threads_ > 1) {
            lock.Unlock();
          }
        }
      }
    }
  }



  // For a row with an e_block, compute the contribution S += F'F. This
  // function has the same structure as NoEBlockRowOuterProduct, except
  // that this function uses the template parameters.
  template<class T>
  void EBlockRowOuterProduct(const CompressedRowBlockStructure* bs,
                            const double* values,
                             int row_block_index,
                            T* lhs) {
    const CompressedRow& row = bs->rows[row_block_index];
    const auto row_block_size = row.block.size;
    const auto& bs_col_sizes = bs->col_sizes;

    SmallBiasHelper sbhOuter;
    sbhOuter.num_row_a_ = row_block_size;
    sbhOuter.num_row_b_ = row_block_size;

    SmallBiasHelper sbhInner;
    sbhInner.num_row_a_ = row_block_size;
    sbhInner.num_row_b_ = row_block_size;

    for (size_t i = 1, cnt = row.cells.size(); i < cnt; ++i) {
      const auto& row_cell_i = row.cells[i];
      const auto* /* restrict? */ row_cell_i_values_position = values + row_cell_i.position;

      sbhInner.A_ = row_cell_i_values_position;

      const int block1 = row_cell_i.block_id - num_eliminate_blocks_;
      DCHECK_GE(block1, 0);

      const int block1_size = bs_col_sizes[row_cell_i.block_id];
      sbhInner.num_col_a_ = block1_size;
      CellInfo* __restrict cell_info = lhs->GetCell(block1, block1,
                                                    &sbhOuter.cih_.r_, &sbhOuter.cih_.c_,
                                                    &sbhOuter.cih_.row_stride_, &sbhOuter.cih_.col_stride_);
      if (cell_info) {
        sbhOuter.A_ = row_cell_i_values_position;
        sbhOuter.B_ = row_cell_i_values_position;
        sbhOuter.num_col_a_ = block1_size;
        sbhOuter.num_col_b_ = block1_size;
        sbhOuter.C_ = cell_info->values;
        auto& lock = cell_info->m;
        if (num_threads_ > 1) {
          lock.Lock();
        }

        // block += b1.transpose() * b1;
        MatrixTransposeMatrixMultiply2
          <kRowBlockSize, kFBlockSize, kRowBlockSize, kFBlockSize, 1>(sbhOuter);

        if (num_threads_ > 1) {
          lock.Unlock();
        }
      }

      lhs->T::PrepareGetCellHelper(sbhInner.cih_, block1);

      for (size_t j = i + 1; j < cnt; ++j) {
        const auto& row_cell_j = row.cells[j];
        const int block2 = row_cell_j.block_id - num_eliminate_blocks_;
        DCHECK_GE(block2, 0);
        DCHECK_LT(block1, block2);
        CellInfo* __restrict cell_info = lhs->T::GetCellHelped(sbhInner.cih_, block2);

        if (cell_info) {
          // block += b1.transpose() * b2;
          const int block2_size = bs_col_sizes[row_cell_j.block_id];
          sbhInner.B_ = values + row_cell_j.position;
          sbhInner.num_col_b_ = block2_size;
          sbhInner.C_ = cell_info->values;

          auto& lock = cell_info->m;
          if (num_threads_ > 1) {
            lock.Lock();
          }

          MatrixTransposeMatrixMultiply2
            <kRowBlockSize, kFBlockSize, kRowBlockSize, kFBlockSize, 1>(sbhInner);

          if (num_threads_ > 1) {
            lock.Unlock();
          }
        }
      }
    }
  }

  void NoEBlockRowsUpdate(const BlockSparseMatrix* A,
                             const double* b,
                             int row_block_counter,
                             BlockRandomAccessMatrix* lhs,
                             double* rhs);

  void NoEBlockRowOuterProduct(const BlockSparseMatrix* A,
                               int row_block_index,
                               BlockRandomAccessMatrix* lhs);

  int num_threads_;
  int num_eliminate_blocks_;
  bool assume_full_rank_ete_;

  // Block layout of the columns of the reduced linear system. Since
  // the f blocks can be of varying size, this vector stores the
  // position of each f block in the row/col of the reduced linear
  // system. Thus lhs_row_layout_[i] is the row/col position of the
  // i^th f block.
  std::vector<int> lhs_row_layout_;

  // Combinatorial structure of the chunks in A. For more information
  // see the documentation of the Chunk object above.
  std::vector<Chunk> chunks_;

  // TODO(sameeragarwal): The following two arrays contain per-thread
  // storage. They should be refactored into a per thread struct.

  // Buffer to store the products of the y and z blocks generated
  // during the elimination phase. buffer_ is of size num_threads *
  // buffer_size_. Each thread accesses the chunk
  //
  //   [thread_id * buffer_size_ , (thread_id + 1) * buffer_size_]
  //
  scoped_array<double> buffer_;

  // Buffer to store per thread matrix matrix products used by
  // ChunkOuterProduct. Like buffer_ it is of size num_threads *
  // buffer_size_. Each thread accesses the chunk
  //
  //   [thread_id * buffer_size_ , (thread_id + 1) * buffer_size_ -1]
  //
  scoped_array<double> chunk_outer_product_buffer_;

  int buffer_size_;
  int uneliminated_row_begins_;

  // Locks for the blocks in the right hand side of the reduced linear
  // system.
  std::vector<Mutex*> rhs_locks_;
};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_SCHUR_ELIMINATOR_H_
