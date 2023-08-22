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
  const size_t cnt = parameter_blocks.size();
  ordering->reserve(cnt*2);

  const auto& vertices = graph.vertices();
  const auto last = vertices.end();
  for (const auto& __restrict parameter_block : parameter_blocks) {
    if (vertices.find(parameter_block) != last) {
      ordering->push_back(parameter_block);
    }
  }
  event_logger.AddEvent("Preordering");

  int independent_set_size = StableIndependentSetOrderingFaster(graph, ordering);
  event_logger.AddEvent("StableIndependentSet");

  // Add the excluded blocks to back of the ordering vector.
  for (const auto& __restrict parameter_block : parameter_blocks) {
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
      graph.RemoveVertex(parameter_block);
    }
    num_covered += independent_set_size;
    ++round;
  }
}

Graph<ParameterBlock*> CreateHessianGraph(const Program& program) {
  Graph<ParameterBlock*> graph;

  const vector<ParameterBlock*>& parameter_blocks = program.parameter_blocks();
  const size_t cnt = parameter_blocks.size();
  graph.Reserve(cnt);

  for (const auto& __restrict parameter_block : parameter_blocks) {
    if (!parameter_block->IsConstant()) {
      graph.AddVertex(parameter_block);
    }
  }

  for (const auto& __restrict residual_block : program.residual_blocks()) {
    const int num_parameter_blocks = residual_block->NumParameterBlocks();
    ParameterBlock* const* parameter_blocks =
        residual_block->parameter_blocks();
    for (int j = 0; j < num_parameter_blocks; ++j) {
      const auto& __restrict parameter_block_j = parameter_blocks[j];
      if (parameter_block_j->IsConstant()) {
        continue;
      }

      for (int k = j + 1; k < num_parameter_blocks; ++k) {
        const auto& __restrict parameter_block_k = parameter_blocks[k];
        if (parameter_block_k->IsConstant()) {
          continue;
        }

        graph.AddEdge(parameter_block_j, parameter_block_k);
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

  const map<int, set<double*> >& group_to_elements =
      ordering->group_to_elements();
  for (map<int, set<double*> >::const_iterator it = group_to_elements.begin();
       it != group_to_elements.end();
       ++it) {
    group_sizes->push_back(it->second.size());
  }
}

}  // namespace internal
}  // namespace ceres
