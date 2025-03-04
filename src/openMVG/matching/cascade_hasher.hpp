// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2015 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef OPENMVG_MATCHING_CASCADE_HASHER_HPP
#define OPENMVG_MATCHING_CASCADE_HASHER_HPP

//------------------
//-- Bibliography --
//------------------
//- [1] "Fast and Accurate Image Matching with Cascade Hashing for 3D Reconstruction"
//- Authors: Jian Cheng, Cong Leng, Jiaxiang Wu, Hainan Cui, Hanqing Lu.
//- Date: 2014.
//- Conference: CVPR.
//
// This implementation is based on the Theia library implementation.
//
// Update compare to the initial paper [1] and initial author code:
// - hashing projection is made by using Eigen to use vectorization (Theia)
// - replace the BoxMuller random number generation by C++ 11 random number generation (OpenMVG)
// - this implementation can support various descriptor length and internal type (OpenMVG)
// -  SIFT, SURF, ... all scalar based descriptor
//

// Copyright (C) 2014 The Regents of the University of California (Regents).
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of The Regents or University of California nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Chris Sweeney (cmsweeney@cs.ucsb.edu)


#include <cmath>
#include <random>
#include <utility>
#include <vector>

#include "openMVG/matching/indMatch.hpp"
#include "openMVG/matching/metric.hpp"
#include "openMVG/numeric/eigen_alias_definition.hpp"
#include "openMVG/stl/dynamic_bitset.hpp"

#include "P2PUtils.h"

namespace openMVG {
namespace matching {

static void __forceinline BitSet( _DataI& d, int b /* [0, 128) */, int v)
{
  int dword = b / 32;
  int bit = (v << (b % 32));

  d.m128i_i32[dword] &= ~bit;
  d.m128i_i32[dword] |= bit;
}

struct HashedDescriptions{
  // The hash information.
  std::vector<_DataI> hashed_desc_codes;

  std::vector<std::vector<uint16_t>> hashed_desc_buckets;

  using Bucket = std::vector<int>;
  // buckets[bucket_group][bucket_id] = bucket (container of description ids).
  std::vector<std::vector<Bucket>> buckets;
};

// This hasher will hash descriptors with a two-step hashing system:
// 1. it generates a hash code,
// 2. it determines which buckets the descriptors belong to.
// Retrieval step is fast since:
// - only descriptors in the same bucket are likely to be good matches.
//   - 1. a hamming distance is used for fast neighbor candidate retrieval
//   - 2. the L2 distance is computed only a reduced selection of approximate neighbor
//
// Implementation is based on the paper [1].
// If you use this matcher, please cite the paper.
class CascadeHasher {
private:

  // The number of bucket bits.
  int nb_bits_per_bucket_;
  // The number of dimensions of the Hash code.
  int nb_hash_code_;
  // The number of bucket groups.
  int nb_bucket_groups_;
  // The number of buckets in each group.
  int nb_buckets_per_group_;

public:
  CascadeHasher() = default;

  // Creates the hashing projections (cascade of two level of hash codes)
  bool Init
  (
    const uint8_t nb_hash_code = 128,
    const uint8_t nb_bucket_groups = 6,
    const uint8_t nb_bits_per_bucket = 10,
    const unsigned random_seed = std::mt19937::default_seed)
  {
    nb_bucket_groups_= nb_bucket_groups;
    nb_hash_code_ = nb_hash_code;
    nb_bits_per_bucket_ = nb_bits_per_bucket;
    nb_buckets_per_group_= 1 << nb_bits_per_bucket;

    //
    // Box Muller transform is used in the original paper to get fast random number
    // from a normal distribution with <mean = 0> and <variance = 1>.
    // Here we use C++11 normal distribution random number generator
    std::mt19937 gen(random_seed);
    std::normal_distribution<> d(0,1);

    primary_hash_projection_.resize(nb_hash_code, nb_hash_code);

    // Initialize primary hash projection.
    for (int i = 0; i < nb_hash_code; ++i)
    {
      for (int j = 0; j < nb_hash_code; ++j)
        primary_hash_projection_(i, j) = d(gen);
    }

    // Initialize secondary hash projection.
    secondary_hash_projection_.resize(nb_bucket_groups);
    for (int i = 0; i < nb_bucket_groups; ++i)
    {
      secondary_hash_projection_[i].resize(nb_bits_per_bucket_,
        nb_hash_code);
      for (int j = 0; j < nb_bits_per_bucket_; ++j)
      {
        for (int k = 0; k < nb_hash_code; ++k)
          secondary_hash_projection_[i](j, k) = d(gen);
      }
    }
    return true;
  }

  template <typename MatrixT>
  static Eigen::VectorXf GetZeroMeanDescriptor
  (
    const MatrixT & descriptions
  )
  {
    if (descriptions.rows() == 0) {
      return {};
    }
    // Compute the ZeroMean descriptor
    return descriptions.template cast<float>().colwise().mean();
  }


  template <typename MatrixT>
  std::unique_ptr<HashedDescriptions> CreateHashedDescriptions
  (
    const MatrixT & descriptions,
    const Eigen::VectorXf & zero_mean_descriptor
  ) const
  {
    // Steps:
    //   1) Compute hash code and hash buckets (based on the zero_mean_descriptor).
    //   2) Construct buckets.

    std::unique_ptr<HashedDescriptions> pHashed_descriptions = std::make_unique<HashedDescriptions>();
    if (descriptions.rows() == 0) {
      return pHashed_descriptions;
    }

    auto& hashed_descriptions = *pHashed_descriptions;

    // Create hash codes for each description.
    {
      // Allocate space for hash codes.
      const typename MatrixT::Index nbDescriptions = descriptions.rows();
      hashed_descriptions.hashed_desc_codes.resize(nbDescriptions);
      hashed_descriptions.hashed_desc_buckets.resize(nbDescriptions);
      Eigen::VectorXf descriptor(descriptions.cols());
      for (int i = 0; i < nbDescriptions; ++i)
      {
        // Allocate space for each bucket id.
        hashed_descriptions.hashed_desc_buckets[i].resize(nb_bucket_groups_);

        // Compute hash code.
        auto& hash_code = hashed_descriptions.hashed_desc_codes[i];
        assert(descriptions.cols() == 128);
        descriptor = descriptions.row(i).template cast<float>();
        descriptor -= zero_mean_descriptor;
        const Eigen::VectorXf primary_projection = primary_hash_projection_ * descriptor;
        for (int j = 0; j < nb_hash_code_; ++j)
        {
          BitSet(hash_code, j, primary_projection(j) > 0);
        }

        // Determine the bucket index for each group.
        Eigen::VectorXf secondary_projection;
        for (int j = 0; j < nb_bucket_groups_; ++j)
        {
          uint16_t bucket_id = 0;
          secondary_projection = secondary_hash_projection_[j] * descriptor;

          for (int k = 0; k < nb_bits_per_bucket_; ++k)
          {
            bucket_id = (bucket_id << 1) + (secondary_projection(k) > 0 ? 1 : 0);
          }
          hashed_descriptions.hashed_desc_buckets[i][j] = bucket_id;
        }
      }
    }
    // Build the Buckets
    {
      hashed_descriptions.buckets.resize(nb_bucket_groups_);
      for (int i = 0; i < nb_bucket_groups_; ++i)
      {
        hashed_descriptions.buckets[i].resize(nb_buckets_per_group_);

        // Add the descriptor ID to the proper bucket group and id.
        for (int j = 0; j < hashed_descriptions.hashed_desc_buckets.size(); ++j)
        {
          const uint16_t bucket_id = hashed_descriptions.hashed_desc_buckets[j][i];
          hashed_descriptions.buckets[i][bucket_id].push_back(j);
        }
      }
    }
    return pHashed_descriptions;
  }

  // Matches two collection of hashed descriptions with a fast matching scheme
  // based on the hash codes previously generated.
  template <typename MatrixT, typename DistanceType>
  void Match_HashedDescriptions
  (
    const HashedDescriptions& hashed_descriptions1,
    const MatrixT & descriptions1,
    const HashedDescriptions& hashed_descriptions2,
    const MatrixT & descriptions2,
    IndMatches * pvec_indices,
    std::vector<DistanceType> * pvec_distances,
    const int NN = 2
  ) const
  {
    using MetricT = L2<typename MatrixT::Scalar>;
    MetricT metric;

    static const int kNumTopCandidates = 10;

    // Preallocated hamming distances. Each column indicates the hamming distance
    // and the rows collect the descriptor ids with that
    // distance. num_descriptors_with_hamming_distance keeps track of how many
    // descriptors have that distance.
    Eigen::MatrixXi candidate_hamming_distances(
      hashed_descriptions2.hashed_desc_codes.size(), nb_hash_code_ + 1);
    Eigen::VectorXi num_descriptors_with_hamming_distance(nb_hash_code_ + 1);

    // Preallocate the container for keeping euclidean distances.
    std::vector<std::pair<DistanceType, int>> candidate_euclidean_distances;
    candidate_euclidean_distances.reserve(kNumTopCandidates);

    using HammingMetricType = matching::Hamming<stl::dynamic_bitset::BlockType>;
    static const HammingMetricType metricH = {};
    for (int i = 0; i < hashed_descriptions1.hashed_desc_codes.size(); ++i)
    {
      num_descriptors_with_hamming_distance.setZero();
      candidate_euclidean_distances.clear();

      const auto& hashed_desc_buckets = hashed_descriptions1.hashed_desc_buckets[i];

      // Compute the hamming distance of all candidates based on the comp hash
      // code. Put the descriptors into buckets corresponding to their hamming
      // distance.

      const auto hashCode = hashed_descriptions1.hashed_desc_codes[i];
      for (int j = 0; j < nb_bucket_groups_; ++j)
      {
        const uint16_t bucket_id = hashed_desc_buckets[j];
        for (const auto candidate_id : hashed_descriptions2.buckets[j][bucket_id])
        {
          _DataI xor = _XorI(hashCode, hashed_descriptions2.hashed_desc_codes[candidate_id]);
          const HammingMetricType::ResultType hamming_distance = 
            PopCnt64(_Low64I(xor))
            + PopCnt64(_Low64I(_UnpackHighI(xor, xor)));
          candidate_hamming_distances(
              num_descriptors_with_hamming_distance(hamming_distance)++,
              hamming_distance) = candidate_id;
        }
      }

      // Compute the euclidean distance of the k descriptors with the best hamming
      // distance.
      candidate_euclidean_distances.reserve(kNumTopCandidates);
      for (int j = 0, numCols = candidate_hamming_distances.cols(); j < numCols; ++j)
      {
        for (int k = 0, cnt = num_descriptors_with_hamming_distance(j); k < cnt; ++k)
        {
          const int candidate_id = candidate_hamming_distances(k, j);
          const DistanceType distance = metric(
            descriptions2.row(candidate_id).data(),
            descriptions1.row(i).data(),
            descriptions1.cols());

          for (const auto& l : candidate_euclidean_distances)
          {
            if (l.second == candidate_id) {
               goto skip;
            }
          }
          candidate_euclidean_distances.emplace_back(distance, candidate_id);
          if (candidate_euclidean_distances.size() >= kNumTopCandidates)
            goto quit;
          
          skip:
            ;
        }
      }
    quit:
      ;

      // Assert that each query is having at least NN retrieved neighbors
      if (candidate_euclidean_distances.size() >= NN)
      {
        // Find the top NN candidates based on euclidean distance.
        std::partial_sort(candidate_euclidean_distances.begin(),
          candidate_euclidean_distances.begin() + NN,
          candidate_euclidean_distances.end());
        // save resulting neighbors
        for (int l = 0; l < NN; ++l)
        {
          pvec_distances->emplace_back(candidate_euclidean_distances[l].first);
          pvec_indices->emplace_back(IndMatch(i,candidate_euclidean_distances[l].second));
        }
      }
      //else -> too few candidates... (save no one)
    }
  }

  private:
  // Primary hashing function.
  Eigen::MatrixXf primary_hash_projection_;

  // Secondary hashing function.
  std::vector<Eigen::MatrixXf> secondary_hash_projection_;
};

}  // namespace matching
}  // namespace openMVG

#endif // OPENMVG_MATCHING_CASCADE_HASHER_HPP
