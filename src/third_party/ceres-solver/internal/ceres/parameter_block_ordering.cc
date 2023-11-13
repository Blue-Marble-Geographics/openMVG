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

#include "ceres/parameter_block_ordering.h"

#include "ceres/graph.h"
#include "ceres/graph_algorithms.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/map_util.h"
#include "ceres/parameter_block.h"
#include "ceres/program.h"
#include "ceres/residual_block.h"
#include "ceres/wall_time.h"
#include "glog/logging.h"

#include "ceres/internal/fixed_array.h" // Borrow this

namespace ceres {
namespace internal {

using std::map;
using std::set;
using std::vector;

int ComputeStableSchurOrdering(const Program& program,
                         vector<ParameterBlock*>* ordering) {
  CHECK_NOTNULL(ordering)->clear();
  EventLogger event_logger("ComputeStableSchurOrdering");
  Graph< ParameterBlock*> graph(CreateHessianGraph(program));
  event_logger.AddEvent("CreateHessianGraph");

  const vector<ParameterBlock*>& parameter_blocks = program.parameter_blocks();
  const auto cnt = parameter_blocks.size();
  ordering->reserve(cnt); // const + non-const parameter blocks.

  for (const auto& parameter_block : parameter_blocks) {
    if (auto vertex = graph.FindVertex(parameter_block)) {
      ordering->push_back(parameter_block);
    }
  }
  event_logger.AddEvent("Preordering");

  int independent_set_size = StableIndependentSetOrderingFaster(graph, ordering);
  event_logger.AddEvent("StableIndependentSet");

  // Add the excluded blocks to back of the ordering vector.
  for (const auto& parameter_block : parameter_blocks) {
    if (parameter_block->IsConstant()) {
      ordering->push_back(parameter_block);
    }
  }
  event_logger.AddEvent("ConstantParameterBlocks");

  return independent_set_size;
}

int ComputeSchurOrdering(const Program& program,
                         vector<ParameterBlock*>* ordering) {
  CHECK_NOTNULL(ordering)->clear();

  Graph< ParameterBlock*> graph(CreateHessianGraph(program));
  int independent_set_size = IndependentSetOrdering(graph, ordering);
  const vector<ParameterBlock*>& parameter_blocks = program.parameter_blocks();

  // Add the excluded blocks to back of the ordering vector.
  for (int i = 0; i < parameter_blocks.size(); ++i) {
    ParameterBlock* parameter_block = parameter_blocks[i];
    if (parameter_block->IsConstant()) {
      ordering->push_back(parameter_block);
    }
  }

  return independent_set_size;
}

void ComputeRecursiveIndependentSetOrdering(const Program& program,
                                            ParameterBlockOrdering* ordering) {
  CHECK_NOTNULL(ordering)->Clear();
  const vector<ParameterBlock*> parameter_blocks = program.parameter_blocks();
  Graph< ParameterBlock*> graph(CreateHessianGraph(program));

  int num_covered = 0;
  int round = 0;
  while (num_covered < parameter_blocks.size()) {
    vector<ParameterBlock*> independent_set_ordering;
    const int independent_set_size =
        IndependentSetOrdering(graph, &independent_set_ordering);
    for (int i = 0; i < independent_set_size; ++i) {
      ParameterBlock* parameter_block = independent_set_ordering[i];
      ordering->AddElementToGroup(parameter_block->mutable_user_state(), round);
      throw std::runtime_error( "Unsupported" );
      // JPB WIP graph.RemoveVertex(parameter_block);
    }
    num_covered += independent_set_size;
    ++round;
  }
}

Graph<ParameterBlock*> CreateHessianGraph(const Program& program) {
  Graph<ParameterBlock*> graph;

  const vector<ParameterBlock*>& parameter_blocks = program.parameter_blocks();
  const auto cnt = parameter_blocks.size();

  graph.AddVertices(
    std::begin(parameter_blocks),
    std::end(parameter_blocks),
    [](const auto& pb) -> bool {
      return !pb->IsConstant();
    }
  );

  for (const auto& residual_block : program.residual_blocks()) {
    const int num_parameter_blocks = residual_block->NumParameterBlocks();
    FixedArray<std::pair<ParameterBlock*, FlatSet<ParameterBlock*>*>, 10, 0 /* No init */> usedBlocks(num_parameter_blocks);
    size_t numUsedBlocks = 0;

    ParameterBlock* const* parameter_blocks =
        residual_block->parameter_blocks();

    for (int i = 0; i < num_parameter_blocks; ++i) {
      const auto& parameter_block = parameter_blocks[i];
      if (!parameter_block->IsConstant() ) {
        usedBlocks[numUsedBlocks++] = { parameter_block, nullptr };
      }
    }

    for (int j = 0; j < numUsedBlocks; ++j) {
      for (int k = j + 1; k < numUsedBlocks; ++k) {
        graph.AddEdgeExplicitly(usedBlocks[j], usedBlocks[k]);
      }
    }
  }

  return graph;
}

void OrderingToGroupSizes(const ParameterBlockOrdering* ordering,
                          vector<int>* group_sizes) {
  CHECK_NOTNULL(group_sizes)->clear();
  if (ordering == NULL) {
    return;
  }

  for (const auto& i : ordering->group_to_elements()) {
    group_sizes->push_back(i.size());
  }
}

}  // namespace internal
}  // namespace ceres
