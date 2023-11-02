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

#ifndef CERES_PUBLIC_ORDERED_GROUPS_H_
#define CERES_PUBLIC_ORDERED_GROUPS_H_

#include <list>
#include <map>
#include <set>
#include <vector>
#include "ceres/internal/port.h"
#include "glog/logging.h"

template<class T>
struct Mallocator
{
  Mallocator* copyAllocator = nullptr;
  Mallocator<T>* rebindAllocator = nullptr;

  typedef T value_type;

  template <typename U>
  struct rebind
  {
    using other = Mallocator<U>;
  };
  Mallocator() = default;

  Mallocator(Mallocator& allocator) :
    copyAllocator(&Mallocator)
  {
  }

  template <class U>
  Mallocator(const Mallocator<U>& other)
  {
    if (!std::is_same<T, U>::value)
      rebindAllocator = new Mallocator<T>();
  }

  bool operator==(const Mallocator&) const noexcept
  {
    return true;
  }
  bool operator!=(const Mallocator&) const noexcept
  {
    return false;
  }

  template <typename T>
  std::unique_ptr<T> make_unique_uninitialized(const std::size_t size) {
    return std::unique_ptr<T>(new typename std::remove_extent<T>::type[size]);
  }

  enum { kChunkSize = 4096 };
  T* allocate(std::size_t n)
  {
    if (copyAllocator)
      return copyAllocator->allocate(n);

    if (rebindAllocator)
      return rebindAllocator->allocate(n);

    size_t num_bytes = sizeof(T)* n;

    if (chunks.empty() || ( space_used + num_bytes ) > kChunkSize) {
      //cerr << "alloc " << kChunkSize << " calls\n";
      chunks.push_back(make_unique_uninitialized<uint8_t[]>(kChunkSize));
      space_used = 0;
    }

    T* addr = (T*)&chunks.back()[space_used];
    space_used += num_bytes;

    return reinterpret_cast<typename std::allocator<T>::pointer>( addr );
  }

  void deallocate(T* p, std::size_t n) noexcept
  {
    if (copyAllocator) {
      copyAllocator->deallocate(p, n);
      return;
    }

    if (rebindAllocator) {
      rebindAllocator->deallocate(p, n);
      return;
    }
  }

private:
  // Can't use deque... it invalidates.
  size_t space_used = 0;
  std::list<std::unique_ptr<uint8_t[]>> chunks;
};

namespace ceres {

// A class for storing and manipulating an ordered collection of
// groups/sets with the following semantics:
//
// Group ids are non-negative integer values. Elements are any type
// that can serve as a key in a map or an element of a set.
//
// An element can only belong to one group at a time. A group may
// contain an arbitrary number of elements.
//
// Groups are ordered by their group id.
template <typename T>
class OrderedGroups {
 public:
  // Add an element to a group. If a group with this id does not
  // exist, one is created. This method can be called any number of
  // times for the same element. Group ids should be non-negative
  // numbers.
  //
  // Return value indicates if adding the element was a success.
  bool AddElementToGroup(const T element, const int group) {
    if (group < 0) {
      return false;
    }
#if 1 // JPB WIP
    const auto result = element_to_group_.try_emplace(element, group);
    // result.first has an iterator pair to where it is inserted.
    // result.second is false if insertion occurred, true if element already present.
    if (!result.second) { // Element was already there (try_emplace did nothing but locate the element).
      // Is it in the same group?
      auto stored_group = result.first->second;
      if (stored_group == group) {
        // current_group matches nothing to do.
        return true;
      }

      // Replace the stored group.
      result.first->second = group;

      // Remove the element from the stored group's reference.
      auto it = group_to_elements_.find(stored_group);

      auto& stored_groups_elements_set = it->second;

      // Find the element in the group (set) and remove it.
      auto it2 = stored_groups_elements_set.find(element);
      stored_groups_elements_set.erase(it2);

      // If it makes the set empty erase the set.
      if (stored_groups_elements_set.empty()) {
        group_to_elements_.erase(it);
      }
    }
    else {
      group_to_elements_[group].insert(element);
    }
#else
    auto it =
        element_to_group_.find(element);
    if (it != element_to_group_.end()) {
      if (it->second == group) {
        // Element is already in the right group, nothing to do.
        return true;
      }

      group_to_elements_[it->second].erase(element);
      if (group_to_elements_[it->second].size() == 0) {
        group_to_elements_.erase(it->second);
      }
    }

    element_to_group_[element] = group;
    group_to_elements_[group].insert(element);
#endif
    return true;
  }

  void Clear() {
    group_to_elements_.clear();
    element_to_group_.clear();
  }

  // Remove the element, no matter what group it is in. Return value
  // indicates if the element was actually removed.
  bool Remove(const T element) {
    const int current_group = GroupId(element);
    if (current_group < 0) {
      return false;
    }

    group_to_elements_[current_group].erase(element);

    if (group_to_elements_[current_group].size() == 0) {
      // If the group is empty, then get rid of it.
      group_to_elements_.erase(current_group);
    }

    element_to_group_.erase(element);
    return true;
  }

  // Bulk remove elements. The return value indicates the number of
  // elements successfully removed.
  int Remove(const std::vector<T>& elements) {
    if (NumElements() == 0 || elements.size() == 0) {
      return 0;
    }

    int num_removed = 0;
    for (int i = 0; i < elements.size(); ++i) {
      num_removed += Remove(elements[i]);
    }
    return num_removed;
  }

  // Reverse the order of the groups in place.
  void Reverse() {
    if (NumGroups() == 0) {
      return;
    }

    typename std::map<int, std::set<T, std::less<T>, Mallocator<T>>, std::less<int>, Mallocator<T> >::reverse_iterator it =
        group_to_elements_.rbegin();
    std::map<int, std::set<T, std::less<T>, Mallocator<T>>, std::less<int>, Mallocator<T> > new_group_to_elements;
    new_group_to_elements[it->first] = it->second;

    int new_group_id = it->first + 1;
    for (++it; it != group_to_elements_.rend(); ++it) {
      for (auto element_it = it->second.begin();
           element_it != it->second.end();
           ++element_it) {
        element_to_group_[*element_it] = new_group_id;
      }
      new_group_to_elements[new_group_id] = it->second;
      new_group_id++;
    }

    group_to_elements_.swap(new_group_to_elements);
  }

  // Return the group id for the element. If the element is not a
  // member of any group, return -1.
  int GroupId(const T element) const {
    auto it =
        element_to_group_.find(element);
    if (it == element_to_group_.end()) {
      return -1;
    }
    return it->second;
  }

  bool IsMember(const T element) const {
    auto it =
        element_to_group_.find(element);
    return (it != element_to_group_.end());
  }

  // This function always succeeds, i.e., implicitly there exists a
  // group for every integer.
  int GroupSize(const int group) const {
    auto it =
        group_to_elements_.find(group);
    return (it ==  group_to_elements_.end()) ? 0 : it->second.size();
  }

  int NumElements() const {
    return element_to_group_.size();
  }

  // Number of groups with one or more elements.
  int NumGroups() const {
    return group_to_elements_.size();
  }

  // The first group with one or more elements. Calling this when
  // there are no groups with non-zero elements will result in a
  // crash.
  int MinNonZeroGroup() const {
    CHECK_NE(NumGroups(), 0);
    return group_to_elements_.begin()->first;
  }

  const auto& group_to_elements() const {
    return group_to_elements_;
  }

  const auto& element_to_group() const {
    return element_to_group_;
  }

 private:
  std::map<int, std::set<T, std::less<T>, Mallocator<T>>, std::less<int>, Mallocator<T> > group_to_elements_;
  std::map<T, int, std::less<T>, Mallocator<int>> element_to_group_;
};

// Typedef for the most commonly used version of OrderedGroups.
typedef OrderedGroups<double*> ParameterBlockOrdering;

}  // namespace ceres

#endif  // CERES_PUBLIC_ORDERED_GROUP_H_
