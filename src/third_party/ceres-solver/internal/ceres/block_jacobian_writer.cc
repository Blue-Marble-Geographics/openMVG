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
// Author: keir@google.com (Keir Mierle)

#include "ceres/block_jacobian_writer.h"

#include "ceres/block_evaluate_preparer.h"
#include "ceres/block_sparse_matrix.h"
#include "ceres/parameter_block.h"
#include "ceres/program.h"
#include "ceres/residual_block.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/port.h"
#include "ceres/internal/scoped_ptr.h"

namespace ceres {
namespace internal {

using std::vector;

namespace {

// Given the residual block ordering, build a lookup table to determine which
// per-parameter jacobian goes where in the overall program jacobian.
//
// Since we expect to use a Schur type linear solver to solve the LM step, take
// extra care to place the E blocks and the F blocks contiguously. E blocks are
// the first num_eliminate_blocks parameter blocks as indicated by the parameter
// block ordering. The remaining parameter blocks are the F blocks.
//
// TODO(keir): Consider if we should use a boolean for each parameter block
// instead of num_eliminate_blocks.
void BuildJacobianLayout(const Program& program,
                         int num_eliminate_blocks,
                         vector<int*>* jacobian_layout,
                         vector<int>* jacobian_layout_storage) {
  const vector<ResidualBlock>& residual_blocks = program.residual_blocks();

  // Iterate over all the active residual blocks and determine how many E blocks
  // are there. This will determine where the F blocks start in the jacobian
  // matrix. Also compute the number of jacobian blocks.
  int f_block_pos = 0;
  int num_jacobian_blocks = 0;

  const size_t num_residual_blocks = residual_blocks.size();
  struct BlockInfo_t {
    BlockInfo_t(int num_parameter_blocks, int* n_block_pos_ptr, int jacobian_block_size) :
      num_parameter_blocks_(num_parameter_blocks),
      n_block_pos_ptr_(n_block_pos_ptr),
      jacobian_block_size_(jacobian_block_size)
    {}

    int num_parameter_blocks_;
    int* n_block_pos_ptr_;
    size_t jacobian_block_size_;
  };
  vector<BlockInfo_t> residual_blocks_info;
  vector<char> block_is_constant;

  if (!residual_blocks.empty()) {
    // Guess at the total size
    const size_t representative_parameter_block_size = residual_blocks.front().NumParameterBlocks();
    residual_blocks_info.reserve(num_residual_blocks * representative_parameter_block_size);
    block_is_constant.reserve(num_residual_blocks * representative_parameter_block_size);
  }

  int e_block_pos = 0;

  for (const auto& residual_block : residual_blocks) {
    const int num_residuals = residual_block.NumResiduals();
    const int num_parameter_blocks = residual_block.NumParameterBlocks();
    ParameterBlock* const* parameter_blocks = residual_block.parameter_blocks();

    // Advance f_block_pos over each E block for this residual.
    for (int j = 0; j < num_parameter_blocks; ++j) {
      const ParameterBlock& parameter_block = *parameter_blocks[j];
      const bool is_constant = parameter_block.IsConstant();
      block_is_constant.emplace_back(is_constant);
      if (!is_constant) {
        const int parameter_block_index = parameter_block.index();
        // Only count blocks for active parameters.
        num_jacobian_blocks++;
        const size_t jacobian_block_size = num_residuals * parameter_block.LocalSize();
        int* n_block_pos_ptr = &e_block_pos;

        if (parameter_block_index < num_eliminate_blocks) {
          n_block_pos_ptr = &e_block_pos;
          f_block_pos += jacobian_block_size;
        } else {
          n_block_pos_ptr = &f_block_pos;
        }
        residual_blocks_info.emplace_back(num_parameter_blocks, n_block_pos_ptr, jacobian_block_size);
      }
    }
  }

  // We now know that the E blocks are laid out starting at zero, and the F
  // blocks are laid out starting at f_block_pos. Iterate over the residual
  // blocks again, and this time fill the jacobian_layout array with the
  // position information.
  jacobian_layout->resize(program.NumResidualBlocks());
  jacobian_layout_storage->resize(num_jacobian_blocks);

  int* jacobian_pos = jacobian_layout_storage->data();
  int const_info_idx = 0;

  auto block_info = residual_blocks_info.begin();
  for (int i = 0; i < num_residual_blocks; ++i) {
    (*jacobian_layout)[i] = jacobian_pos;
    int num_parameter_blocks = block_info->num_parameter_blocks_;
    for (int j = 0; j != num_parameter_blocks; ++j) {
      if (!block_is_constant[const_info_idx++]) {
        *jacobian_pos = *block_info->n_block_pos_ptr_; /* Points to e_block_pos or f_block_pos */
        *block_info->n_block_pos_ptr_ += block_info->jacobian_block_size_;
        jacobian_pos++;
        ++block_info;
      }
    }
  }
}

}  // namespace

BlockJacobianWriter::BlockJacobianWriter(const Evaluator::Options& options,
                                         Program* program)
    : program_(program) {
  CHECK_GE(options.num_eliminate_blocks, 0)
      << "num_eliminate_blocks must be greater than 0.";

  BuildJacobianLayout(*program,
                      options.num_eliminate_blocks,
                      &jacobian_layout_,
                      &jacobian_layout_storage_);
}

// Create evaluate prepareres that point directly into the final jacobian. This
// makes the final Write() a nop.
BlockEvaluatePreparer* BlockJacobianWriter::CreateEvaluatePreparers(
    int num_threads) {
  int max_derivatives_per_residual_block =
      program_->MaxDerivativesPerResidualBlock();

  BlockEvaluatePreparer* preparers = new BlockEvaluatePreparer[num_threads];
  for (int i = 0; i < num_threads; i++) {
    preparers[i].Init(&jacobian_layout_[0], max_derivatives_per_residual_block);
  }
  return preparers;
}

SparseMatrix* BlockJacobianWriter::CreateJacobian() const {
  CompressedRowBlockStructure* bs = new CompressedRowBlockStructure;

  const vector<ParameterBlock*>& parameter_blocks =
      program_->parameter_blocks();

  // Construct the column blocks.
  const int num_parameter_blocks = parameter_blocks.size();
  bs->col_sizes.resize(num_parameter_blocks);
  bs->col_positions.resize(num_parameter_blocks);
  for (int i = 0, cursor = 0; i < num_parameter_blocks; ++i) {
    DCHECK_NE(parameter_blocks[i]->index(), -1);
    DCHECK(!parameter_blocks[i]->IsConstant());
    bs->col_sizes[i] = parameter_blocks[i]->LocalSize();
    bs->col_positions[i] = cursor;
    cursor += bs->col_sizes[i];
  }

  // Construct the cells in each row.
  const vector<ResidualBlock>& residual_blocks = program_->residual_blocks();
  const int num_residual_blocks = residual_blocks.size();
  int row_block_position = 0;
  bs->rows.resize(num_residual_blocks);

  size_t num_total_cells = 0;
  for (int i = 0; i < num_residual_blocks; ++i) {
    const ResidualBlock& residual_block = residual_blocks[i];
    auto* __restrict pbb = residual_block.parameter_blocks();
    // Size the row by the number of active parameters in this residual.
    const int num_parameter_blocks = residual_block.NumParameterBlocks();
    int num_active_parameter_blocks = 0;
    for (int j = 0; j < num_parameter_blocks; ++j) {
      if (pbb[j]->index() != -1) {
        num_active_parameter_blocks++;
      }
    }
    num_total_cells += num_active_parameter_blocks;
  }

  bs->all_cells.resize(num_total_cells);

  int cell_index = 0;

  for (int i = 0; i < num_residual_blocks; ++i) {
    const ResidualBlock& residual_block = residual_blocks[i];
    auto* __restrict pbb = residual_block.parameter_blocks();
    CompressedRow* row = &bs->rows[i];

    row->block.size = residual_block.NumResiduals();
    row->block.position = row_block_position;
    row_block_position += row->block.size;

    // Size the row by the number of active parameters in this residual.
    const int num_parameter_blocks = residual_block.NumParameterBlocks();
    int num_active_parameter_blocks = 0;
    for (int j = 0; j < num_parameter_blocks; ++j) {
      if (pbb[j]->index() != -1) {
        num_active_parameter_blocks++;
      }
    }
    row->cells = &bs->all_cells[cell_index]; // row->cells.resize(num_active_parameter_blocks);
    row->num_cells = num_active_parameter_blocks;
    cell_index += num_active_parameter_blocks;

    // Add layout information for the active parameters in this row.
    for (int j = 0, k = 0; j < num_parameter_blocks; ++j) {
      const ParameterBlock& parameter_block = *pbb[j];
      if (!parameter_block.IsConstant()) {
        Cell& cell = row->cells[k];
        cell.block_id = parameter_block.index();
        cell.position = jacobian_layout_[i][k];

        // Only increment k for active parameters, since there is only layout
        // information for active parameters.
        k++;
      }
    }

    std::sort(row->cells, row->cells+row->num_cells, CellLessThan);
  }

  BlockSparseMatrix* jacobian = new BlockSparseMatrix(bs);
  DCHECK_NOTNULL(jacobian);
  return jacobian;
}

}  // namespace internal
}  // namespace ceres
