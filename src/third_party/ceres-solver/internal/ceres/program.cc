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

#include "ceres/program.h"

#include <map>
#include <vector>
#include "ceres/array_utils.h"
#include "ceres/casts.h"
#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/cost_function.h"
#include "ceres/evaluator.h"
#include "ceres/internal/port.h"
#include "ceres/local_parameterization.h"
#include "ceres/loss_function.h"
#include "ceres/map_util.h"
#include "ceres/parameter_block.h"
#include "ceres/problem.h"
#include "ceres/residual_block.h"
#include "ceres/stl_util.h"
#include "ceres/triplet_sparse_matrix.h"

#include <numeric>

namespace ceres {
namespace internal {

using std::max;
using std::set;
using std::string;
using std::vector;

Program::Program() {}

Program::Program(const Program& program)
    : parameter_blocks_(program.parameter_blocks_),
      residual_blocks_(program.residual_blocks_) {
}

const vector<ParameterBlock*>& Program::parameter_blocks() const {
  return parameter_blocks_;
}

const vector<ResidualBlock*>& Program::residual_blocks() const {
  return residual_blocks_;
}

vector<ParameterBlock*>* Program::mutable_parameter_blocks() {
  return &parameter_blocks_;
}

vector<ResidualBlock*>* Program::mutable_residual_blocks() {
  return &residual_blocks_;
}

bool Program::StateVectorToParameterBlocks(const double *state) {
  for (const auto& pb : parameter_blocks_) {
    if (!pb->IsConstant() &&
        !pb->SetState(state)) {
      return false;
    }
    state += pb->Size();
  }
  return true;
}

void Program::ParameterBlocksToStateVector(double *state) const {
  for (int i = 0; i < parameter_blocks_.size(); ++i) {
    parameter_blocks_[i]->GetState(state);
    state += parameter_blocks_[i]->Size();
  }
}

void Program::CopyParameterBlockStateToUserState() {
  for (int i = 0; i < parameter_blocks_.size(); ++i) {
    parameter_blocks_[i]->GetState(parameter_blocks_[i]->mutable_user_state());
  }
}

bool Program::SetParameterBlockStatePtrsToUserStatePtrs() {
  for (int i = 0; i < parameter_blocks_.size(); ++i) {
    if (!parameter_blocks_[i]->IsConstant() &&
        !parameter_blocks_[i]->SetState(parameter_blocks_[i]->user_state())) {
      return false;
    }
  }
  return true;
}

bool Program::Plus(const double* state,
                   const double* delta,
                   double* state_plus_delta) const {
  for (auto& ppb : parameter_blocks_) {
    auto& pb = *ppb;
    if (!pb.Plus(state, delta, state_plus_delta)) {
      return false;
    }
    const auto cnt = pb.Size();
    state += cnt;
    delta += pb.LocalSize();
    state_plus_delta += cnt;
  }
  return true;
}

void Program::SetParameterOffsetsAndIndex() {
  // Set positions for all parameters appearing as arguments to residuals to one
  // past the end of the parameter block array.
  for (const auto& rb : residual_blocks_) {
    auto pb = rb->parameter_blocks();
    for (int j = 0, cnt = rb->NumParameterBlocks(); j < cnt; ++j) {
      pb[j]->set_index(-1);
    }
  }

  // For parameters that appear in the program, set their position and offset.
  int state_offset = 0;
  int delta_offset = 0;
  int i = 0;
  for (const auto& ppb : parameter_blocks_) {
    auto& pb = *ppb;
    pb.set_index(i++);
    pb.set_state_offset(state_offset);
    pb.set_delta_offset(delta_offset);
    state_offset += pb.Size();
    delta_offset += pb.LocalSize();
  }
}

bool Program::IsValid() const {
  for (int i = 0; i < residual_blocks_.size(); ++i) {
    const ResidualBlock* residual_block = residual_blocks_[i];
    if (residual_block->index() != i) {
      LOG(WARNING) << "Residual block: " << i
                   << " has incorrect index: " << residual_block->index();
      return false;
    }
  }

  int state_offset = 0;
  int delta_offset = 0;
  for (int i = 0; i < parameter_blocks_.size(); ++i) {
    const ParameterBlock* parameter_block = parameter_blocks_[i];
    if (parameter_block->index() != i ||
        parameter_block->state_offset() != state_offset ||
        parameter_block->delta_offset() != delta_offset) {
      LOG(WARNING) << "Parameter block: " << i
                   << "has incorrect indexing information: "
                   << parameter_block->ToString();
      return false;
    }

    state_offset += parameter_blocks_[i]->Size();
    delta_offset += parameter_blocks_[i]->LocalSize();
  }

  return true;
}

bool Program::ParameterBlocksAreFinite(string* message) const {
  CHECK_NOTNULL(message);
  for (int i = 0; i < parameter_blocks_.size(); ++i) {
    const ParameterBlock* parameter_block = parameter_blocks_[i];
    const double* array = parameter_block->user_state();
    const int size = parameter_block->Size();
    const int invalid_index = FindInvalidValue(size, array);
    if (invalid_index != size) {
      *message = StringPrintf(
          "ParameterBlock: %p with size %d has at least one invalid value.\n"
          "First invalid value is at index: %d.\n"
          "Parameter block values: ",
          array, size, invalid_index);
      AppendArrayToString(size, array, message);
      return false;
    }
  }
  return true;
}

bool Program::IsBoundsConstrained() const {
  for (int i = 0; i < parameter_blocks_.size(); ++i) {
    const ParameterBlock* parameter_block = parameter_blocks_[i];
    if (parameter_block->IsConstant()) {
      continue;
    }
    const int size = parameter_block->Size();
    for (int j = 0; j < size; ++j) {
      const double lower_bound = parameter_block->LowerBoundForParameter(j);
      const double upper_bound = parameter_block->UpperBoundForParameter(j);
      if (lower_bound > -std::numeric_limits<double>::max() ||
          upper_bound < std::numeric_limits<double>::max()) {
        return true;
      }
    }
  }
  return false;
}

bool Program::IsFeasible(string* message) const {
  CHECK_NOTNULL(message);
  for (int i = 0; i < parameter_blocks_.size(); ++i) {
    const ParameterBlock* parameter_block = parameter_blocks_[i];
    const double* parameters = parameter_block->user_state();
    const int size = parameter_block->Size();
    if (parameter_block->IsConstant()) {
      // Constant parameter blocks must start in the feasible region
      // to ultimately produce a feasible solution, since Ceres cannot
      // change them.
      for (int j = 0; j < size; ++j) {
        const double lower_bound = parameter_block->LowerBoundForParameter(j);
        const double upper_bound = parameter_block->UpperBoundForParameter(j);
        if (parameters[j] < lower_bound || parameters[j] > upper_bound) {
          *message = StringPrintf(
              "ParameterBlock: %p with size %d has at least one infeasible "
              "value."
              "\nFirst infeasible value is at index: %d."
              "\nLower bound: %e, value: %e, upper bound: %e"
              "\nParameter block values: ",
              parameters, size, j, lower_bound, parameters[j], upper_bound);
          AppendArrayToString(size, parameters, message);
          return false;
        }
      }
    } else {
      // Variable parameter blocks must have non-empty feasible
      // regions, otherwise there is no way to produce a feasible
      // solution.
      for (int j = 0; j < size; ++j) {
        const double lower_bound = parameter_block->LowerBoundForParameter(j);
        const double upper_bound = parameter_block->UpperBoundForParameter(j);
        if (lower_bound >= upper_bound) {
          *message = StringPrintf(
              "ParameterBlock: %p with size %d has at least one infeasible "
              "bound."
              "\nFirst infeasible bound is at index: %d."
              "\nLower bound: %e, upper bound: %e"
              "\nParameter block values: ",
              parameters, size, j, lower_bound, upper_bound);
          AppendArrayToString(size, parameters, message);
          return false;
        }
      }
    }
  }

  return true;
}

Program* Program::CreateReducedProgram(
    vector<double*>* removed_parameter_blocks,
    double* fixed_cost,
    string* error) const {
  CHECK_NOTNULL(removed_parameter_blocks);
  CHECK_NOTNULL(fixed_cost);
  CHECK_NOTNULL(error);

  scoped_ptr<Program> reduced_program(new Program(*this));
  if (!reduced_program->RemoveFixedBlocks(removed_parameter_blocks,
                                          fixed_cost,
                                          error)) {
    return NULL;
  }

  reduced_program->SetParameterOffsetsAndIndex();
  return reduced_program.release();
}

bool Program::RemoveFixedBlocks(vector<double*>* removed_parameter_blocks,
                                double* fixed_cost,
                                string* error) {
  DCHECK_NOTNULL(removed_parameter_blocks);
  DCHECK_NOTNULL(fixed_cost);
  DCHECK_NOTNULL(error);

  FixedArray<double, 10, 0 /* no init */> residual_block_evaluate_scratch(MaxScratchDoublesNeededForEvaluate());
  *fixed_cost = 0.0;

  // Mark all the parameters as unused. Abuse the index member of the
  // parameter blocks for the marking.
  const int num_parameter_blocks = parameter_blocks_.size();
  for (int i = 0; i < num_parameter_blocks; ++i) {
    parameter_blocks_[i]->set_index(-1);
  }

  // Filter out residual that have all-constant parameters, and mark
  // all the parameter blocks that appear in residuals.
  int num_active_residual_blocks = 0;
  for (const auto& prb : residual_blocks_) {
    ResidualBlock& residual_block = *prb;
    int num_parameter_blocks = residual_block.NumParameterBlocks();
    ParameterBlock* const * ppb = residual_block.parameter_blocks();
    // Determine if the residual block is fixed, and also mark varying
    // parameters that appear in the residual block.
    bool all_constant = true;
    for (int k = 0; k < num_parameter_blocks; k++) {
      ParameterBlock& pb = *ppb[k];
      if (!pb.IsConstant()) {
        all_constant = false;
        pb.set_index(1);
      }
    }

    if (!all_constant) {
      residual_blocks_[num_active_residual_blocks++] = &residual_block;
      continue;
    }

    // The residual is constant and will be removed, so its cost is
    // added to the variable fixed_cost.
    double cost = 0.0;
    if (!residual_block.Evaluate(true,
                                  &cost,
                                  NULL,
                                  NULL,
                                  residual_block_evaluate_scratch.get())) {
      *error = StringPrintf("Evaluation of the residual failed during "
                            "removal of fixed residual blocks.");
      return false;
    }
    *fixed_cost += cost;
  }
  residual_blocks_.resize(num_active_residual_blocks);

  // Filter out unused or fixed parameter blocks.
  int num_active_parameter_blocks = 0;
  removed_parameter_blocks->clear();
  for (int i = 0; i < num_parameter_blocks; ++i) {
    ParameterBlock& parameter_block = *parameter_blocks_[i];
    if (parameter_block.index() == -1) {
      removed_parameter_blocks->push_back(
          parameter_block.mutable_user_state());
    } else {
      parameter_blocks_[num_active_parameter_blocks++] = &parameter_block;
    }
  }
  parameter_blocks_.resize(num_active_parameter_blocks);

  if (!(((NumResidualBlocks() == 0) &&
         (NumParameterBlocks() == 0)) ||
        ((NumResidualBlocks() != 0) &&
         (NumParameterBlocks() != 0)))) {
    *error =  "Congratulations, you found a bug in Ceres. Please report it.";
    return false;
  }

  return true;
}

bool Program::IsParameterBlockSetIndependent(
    const std::set<double*, std::less<double*>, Mallocator<double*>>& independent_set) const {
  // Loop over each residual block and ensure that no two parameter
  // blocks in the same residual block are part of
  // parameter_block_ptrs as that would violate the assumption that it
  // is an independent set in the Hessian matrix.
  for (vector<ResidualBlock*>::const_iterator it = residual_blocks_.begin();
       it != residual_blocks_.end();
       ++it) {
    ParameterBlock* const* parameter_blocks = (*it)->parameter_blocks();
    const int num_parameter_blocks = (*it)->NumParameterBlocks();
    int count = 0;
    for (int i = 0; i < num_parameter_blocks; ++i) {
      count += independent_set.count(
          parameter_blocks[i]->mutable_user_state());
    }
    if (count > 1) {
      return false;
    }
  }
  return true;
}

TripletSparseMatrix* Program::CreateJacobianBlockSparsityTranspose() const {
  // Matrix to store the block sparsity structure of the Jacobian.
  TripletSparseMatrix* tsm =
      new TripletSparseMatrix(NumParameterBlocks(),
                              NumResidualBlocks(),
                              10 * NumResidualBlocks());
  int num_nonzeros = 0;
  int* rows = tsm->mutable_rows();
  int* cols = tsm->mutable_cols();
  double* values = tsm->mutable_values();

  const int num_residual_blocks = residual_blocks_.size();
  for (int c = 0; c < num_residual_blocks; ++c) {
    const ResidualBlock& residual_block = *residual_blocks_[c];
    const int num_parameter_blocks = residual_block.NumParameterBlocks();
    ParameterBlock* const* parameter_blocks =
        residual_block.parameter_blocks();

    for (int j = 0; j < num_parameter_blocks; ++j) {
      if (parameter_blocks[j]->IsConstant()) {
        continue;
      }

      // Re-size the matrix if needed.
      if (num_nonzeros >= tsm->max_num_nonzeros()) {
        tsm->set_num_nonzeros(num_nonzeros);
        tsm->Reserve(2 * num_nonzeros);
        rows = tsm->mutable_rows();
        cols = tsm->mutable_cols();
        values = tsm->mutable_values();
      }

      const int r = parameter_blocks[j]->index();
      rows[num_nonzeros] = r;
      cols[num_nonzeros] = c;
      values[num_nonzeros] = 1.0;
      ++num_nonzeros;
    }
  }

  tsm->set_num_nonzeros(num_nonzeros);
  return tsm;
}

int Program::NumResidualBlocks() const {
  return residual_blocks_.size();
}

int Program::NumParameterBlocks() const {
  return parameter_blocks_.size();
}

int Program::NumResiduals() const {
  return std::accumulate(
    std::begin(residual_blocks_),
    std::end(residual_blocks_), 0,
    [](const auto& a, const auto& b)
    {
      return a + b->NumResiduals();
  }
  );
}

int Program::NumParameters() const {
  return std::accumulate(
    std::begin(parameter_blocks_),
    std::end(parameter_blocks_), 0,
    [](const auto& a, const auto& b)
    {
      return a + b->Size();
  }
  );
}

int Program::NumEffectiveParameters() const {
  return std::accumulate(
    std::begin(parameter_blocks_),
    std::end(parameter_blocks_), 0,
    [](const auto& a, const auto& b)
    {
      return a + b->LocalSize();
  }
  );
}

int Program::MaxScratchDoublesNeededForEvaluate() const {
  // Compute the scratch space needed for evaluate.
  int max_scratch_bytes_for_evaluate = 0;
  for (const auto& prb :  residual_blocks_) {
    max_scratch_bytes_for_evaluate =
        max(max_scratch_bytes_for_evaluate,
            prb->NumScratchDoublesForEvaluate());
  }
  return max_scratch_bytes_for_evaluate;
}

int Program::MaxDerivativesPerResidualBlock() const {
  int max_derivatives = 0;
  for (const auto& prb :  residual_blocks_) {
    int derivatives = 0;
    ResidualBlock& residual_block = *prb;
    int num_parameters = residual_block.NumParameterBlocks();
    auto pb = residual_block.parameter_blocks();
    for (int j = 0; j < num_parameters; ++j) {
      derivatives += residual_block.NumResiduals() *
                     pb[j]->LocalSize();
    }
    max_derivatives = max(max_derivatives, derivatives);
  }
  return max_derivatives;
}

int Program::MaxParametersPerResidualBlock() const {
  int max_parameters = 0;
  for (const auto& prb : residual_blocks_) {
    max_parameters = max(max_parameters,
                         prb->NumParameterBlocks());
  }
  return max_parameters;
}

int Program::MaxResidualsPerResidualBlock() const {
  int max_residuals = 0;
  for (const auto& prb : residual_blocks_) {
    max_residuals = max(max_residuals, prb->NumResiduals());
  }
  return max_residuals;
}

string Program::ToString() const {
  string ret = "Program dump\n";
  ret += StringPrintf("Number of parameter blocks: %d\n", NumParameterBlocks());
  ret += StringPrintf("Number of parameters: %d\n", NumParameters());
  ret += "Parameters:\n";
  for (int i = 0; i < parameter_blocks_.size(); ++i) {
    ret += StringPrintf("%d: %s\n",
                        i, parameter_blocks_[i]->ToString().c_str());
  }
  return ret;
}

}  // namespace internal
}  // namespace ceres
