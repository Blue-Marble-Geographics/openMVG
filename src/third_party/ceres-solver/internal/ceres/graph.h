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

#ifndef CERES_INTERNAL_GRAPH_H_
#define CERES_INTERNAL_GRAPH_H_

#include <limits>
#include <utility>
#include "ceres/integral_types.h"
#include "ceres/map_util.h"
#include "ceres/collections_port.h"
#include "ceres/internal/macros.h"
#include "ceres/types.h"
#include "glog/logging.h"

#include <unordered_map>

//https://stackoverflow.com/questions/1271367/radix-sort-implemented-in-c/40457313#40457313
static inline void RadixSort(uint64_t* a, size_t count)
{
size_t mIndex[8][256] = {0};            // count / index matrix
uint64_t * b = new uint64_t[count];    // allocate temp array
size_t i,j,m,n;
uint64_t u;
    for(i = 0; i < count; i++){         // generate histograms
        u = a[i];
        for(j = 0; j < 8; j++){
            mIndex[j][(size_t)(u & 0xff)]++;
            u >>= 8;
        }       
    }
    for(j = 0; j < 8; j++){             // convert to indices
        m = 0;
        for(i = 0; i < 256; i++){
            n = mIndex[j][i];
            mIndex[j][i] = m;
            m += n;
        }       
    }
    for(j = 0; j < 8; j++){             // radix sort
        for(i = 0; i < count; i++){     //  sort by current lsb
            u = a[i];
            m = (size_t)(u>>(j<<3))&0xff;
            b[mIndex[j][m]++] = u;
        }
        std::swap(a, b);                //  swap ptrs
    }
    delete[] b;
}

enum IsFixedSize_t { eNotFixedSize = false, eFixedSize = true };
template<class T, IsFixedSize_t U = eNotFixedSize>
struct FlatSet
{
  enum { kDefaultSize = 8 };

  size_t Hash(const T& vertex) const noexcept {
    return ((size_t) vertex) % mData.size();
  }

  size_t capacity() const noexcept { return mData.size(); }
  size_t size() const noexcept { return mCnt; }

  T* find(const T& v) noexcept
  {
    size_t start = Hash(v);
    size_t i = start;
    const size_t end = capacity();
    do {
      if (mData[i] == v) {
        return &mData[i];
      } else if (!mData[i]) {
        return nullptr;
      }
      if (++i >= end) {
        i = 0;
      }
    } while (i != start);

    return nullptr;
  }

  void rehash(size_t defaultSize = kDefaultSize)
  {
    if (mData.empty()) {
      // Fast case for empty set
      mData.resize(defaultSize);
    } else {
      FlatSet<T> newSet;
      newSet.mData.resize(std::max((size_t) defaultSize, mData.size()*2));
      newSet.mCnt = mCnt;
      for (const auto& from : mData ) {
        if (from) {
          newSet.insert(from);
        }
      }

      mData = newSet.mData;
      mCnt = newSet.mCnt;
    }
  }

  void reserve(size_t cnt)
  {
    rehash(cnt);
  }

  std::pair<T*, bool> insert(const T& e) noexcept
  {
    if constexpr (U == eNotFixedSize) {
      if (size()+1 > capacity()) {
        // Assure room is available in the set even if the
        // item is not inserted.
        rehash();
      }
    }

    auto at = std::begin(mData) + Hash(e);
    auto last = std::begin(mData) + capacity();
    while (*at != e) {
      if (!*at) {
        // Found an empty slot, use it for the item.
        *at = e;
        ++mCnt;
        return std::make_pair<T*, bool>(&*at, true);
      }
      if (++at == last) {
        // Move back to the first slot if we reach the last.
        at = std::begin(mData);
      }
      // Repeat until we find a slot.
      // Room will always be available.
    }

    return std::make_pair<T*, bool>(&*at, false);
  }

  const std::vector<T>& Neighbors() const
  {
    return mData;
  }

  size_t NumNeighbors() const { return mCnt; }

  size_t         mCnt = 0;
  std::vector<T> mData;
};

template<class T>
struct FixedHashSet
{
  FixedHashSet()
  {}

  explicit FixedHashSet(int maxBuckets) :
    mBuckets(maxBuckets)
  {}

  auto insert(const T& vertex)
  {
    return mBuckets[Hash(vertex)].insert(vertex);
  }

  auto& operator[](const T& v) 
  {
    //tttries = tttries + 1;
    // Guaranteed present.
    auto it = std::begin(mBuckets) + Hash(v);
    do {
      //travel = travel + 1;
      if (it->first == v) {
        return it->second;
      } else {
        if (!it->first) {
          it->first = v;
          return it->second;
        }
      }
      if (++it == std::end(mBuckets)) {
        it = std::begin(mBuckets);
      }
    }
    while (1);
  }

  const auto& operator[](const T& v) const noexcept
  {
    return const_cast<FixedHashSet<T>&>(*this)[v];
  }

  size_t Hash(const T& vertex) const noexcept {
    return ((size_t) vertex) % mBuckets.size();
  }

  void resize(size_t maxBuckets)
  {
    mBuckets.resize(maxBuckets);
  }

  std::vector<std::pair<T, FlatSet<T>>> mBuckets;
};

template<class S, class T>
struct FixedHashMap
{
  FixedHashMap()
  {}

  explicit FixedHashMap(int maxBuckets) :
    mBuckets(maxBuckets)
  {}

  auto insert(const S& key, const T& value)
  {
    return mBuckets[Hash(key)].insert(vertex);
  }

  auto& operator[](const S& key)
  {
    //tttries = tttries + 1;
    // Guaranteed present.
    auto it = std::begin(mBuckets) + Hash(key);
    do {
      //travel = travel + 1;
      if (it->first == key) {
        return it->second;
      }
      else {
        if (!it->first) {
          it->first = key;
          return it->second;
        }
      }
      if (++it == std::end( mBuckets )) {
        it = std::begin( mBuckets );
      }
    } while (1);
  }

  const auto& operator[](const S& key) const noexcept
  {
    return const_cast<FixedHashMap<S>&>(*this)[key];
  }

  size_t Hash(const S& key) const noexcept {
    return ((size_t)key) % mBuckets.size();
  }

  void resize(size_t maxBuckets)
  {
    mBuckets.resize(maxBuckets);
  }

  std::vector<std::pair<S, T>> mBuckets;
};

namespace ceres {
namespace internal {

// A unweighted undirected graph templated over the vertex ids. Vertex
// should be hashable.
template<typename T>
struct Graph
{
  template<typename T, typename Functor>
  void AddVertices(
    T first,
    T last,
    Functor functor
  )
  {
    constexpr size_t kFactor = 4;

    if (const size_t cnt = last - first) {
      size_t numUniqueVertices = 0;
      vertices_.reserve(vertices_.size()+kFactor*cnt);
      while (first != last) {
        if (functor(*first)) {
          numUniqueVertices += !!vertices_.insert(*first).second;
    }
        ++first;
      }
      edges_.resize( kFactor*numUniqueVertices);
    }
  }

  T* FindVertex(const T& vertex) {
    return vertices_.find(vertex);
  }

  void AddEdge(const T& vertex1, const T& vertex2) {
    if (edges_[vertex1].insert(vertex2).second) {
      edges_[vertex2].insert(vertex1);
    }
  }

  void AddEdgeExplicitly( std::pair<T, FlatSet<T>*>& edgeVertex1, std::pair<T, FlatSet<T>*>& edgeVertex2 ) {
    if (!edgeVertex1.second) {
      edgeVertex1.second = &edges_[edgeVertex1.first];
    }
    if ( edgeVertex1.second->insert( edgeVertex2.first ).second ) {
      if ( !edgeVertex2.second ) {
        edgeVertex2.second = &edges_[edgeVertex2.first];
      }
      edgeVertex2.second->insert( edgeVertex1.first );
    }
  }

  const std::vector<T>& Neighbors(const T& vertex) const
  {
    return edges_[vertex].Neighbors();
  }

  size_t NumNeighbors(const T& vertex) const { return edges_[vertex].NumNeighbors(); }

  const std::vector<T>& vertices() const // Raw hashset.
  {
    return vertices_.mData;
  }

  FlatSet<T, eFixedSize> vertices_;
  FixedHashSet<T> edges_;
};

// A weighted undirected graph templated over the vertex ids. Vertex
// should be hashable and comparable.
template <typename Vertex>
class WeightedGraph {
 public:
  WeightedGraph() {}

  // Add a weighted vertex. If the vertex already exists in the graph,
  // its weight is set to the new weight.
  void AddVertex(const Vertex& vertex, double weight) {
    if (vertices_.find(vertex) == vertices_.end()) {
      vertices_.insert(vertex);
      edges_[vertex] = HashSet<Vertex>();
    }
    vertex_weights_[vertex] = weight;
  }

  // Uses weight = 1.0. If vertex already exists, its weight is set to
  // 1.0.
  void AddVertex(const Vertex& vertex) {
    AddVertex(vertex, 1.0);
  }

  bool RemoveVertex(const Vertex& vertex) {
    if (vertices_.find(vertex) == vertices_.end()) {
      return false;
    }

    vertices_.erase(vertex);
    vertex_weights_.erase(vertex);
    const HashSet<Vertex>& sinks = edges_[vertex];
    for (typename HashSet<Vertex>::const_iterator it = sinks.begin();
         it != sinks.end(); ++it) {
      if (vertex < *it) {
        edge_weights_.erase(std::make_pair(vertex, *it));
      } else {
        edge_weights_.erase(std::make_pair(*it, vertex));
      }
      edges_[*it].erase(vertex);
    }

    edges_.erase(vertex);
    return true;
  }

  // Add a weighted edge between the vertex1 and vertex2. Calling
  // AddEdge on a pair of vertices which do not exist in the graph yet
  // will result in undefined behavior.
  //
  // It is legal to call this method repeatedly for the same set of
  // vertices.
  void AddEdge(const Vertex& vertex1, const Vertex& vertex2, double weight) {
    DCHECK(vertices_.find(vertex1) != vertices_.end());
    DCHECK(vertices_.find(vertex2) != vertices_.end());

    if (edges_[vertex1].insert(vertex2).second) {
      edges_[vertex2].insert(vertex1);
    }

    if (vertex1 < vertex2) {
      edge_weights_[std::make_pair(vertex1, vertex2)] = weight;
    } else {
      edge_weights_[std::make_pair(vertex2, vertex1)] = weight;
    }
  }

  // Uses weight = 1.0.
  void AddEdge(const Vertex& vertex1, const Vertex& vertex2) {
    AddEdge(vertex1, vertex2, 1.0);
  }

  // Calling VertexWeight on a vertex not in the graph will result in
  // undefined behavior.
  double VertexWeight(const Vertex& vertex) const {
    return FindOrDie(vertex_weights_, vertex);
  }

  // Calling EdgeWeight on a pair of vertices where either one of the
  // vertices is not present in the graph will result in undefined
  // behaviour. If there is no edge connecting vertex1 and vertex2,
  // the edge weight is zero.
  double EdgeWeight(const Vertex& vertex1, const Vertex& vertex2) const {
    if (vertex1 < vertex2) {
      return FindWithDefault(edge_weights_,
                             std::make_pair(vertex1, vertex2), 0.0);
    } else {
      return FindWithDefault(edge_weights_,
                             std::make_pair(vertex2, vertex1), 0.0);
    }
  }

  // Calling Neighbors on a vertex not in the graph will result in
  // undefined behaviour.
  const HashSet<Vertex>& Neighbors(const Vertex& vertex) const {
    return FindOrDie(edges_, vertex);
  }

  const HashSet<Vertex>& vertices() const {
    return vertices_;
  }

  static double InvalidWeight() {
    return std::numeric_limits<double>::quiet_NaN();
  }

 private:
  HashSet<Vertex> vertices_;
  HashMap<Vertex, double> vertex_weights_;
  HashMap<Vertex, HashSet<Vertex> > edges_;
  HashMap<std::pair<Vertex, Vertex>, double> edge_weights_;

  CERES_DISALLOW_COPY_AND_ASSIGN(WeightedGraph);
};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_GRAPH_H_
