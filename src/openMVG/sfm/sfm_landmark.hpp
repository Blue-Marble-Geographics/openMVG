// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2015 Pierre Moulon.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef OPENMVG_SFM_SFM_LANDMARK_HPP
#define OPENMVG_SFM_SFM_LANDMARK_HPP

#include "openMVG/types.hpp"

#include <stdexcept>
#include <utility>

namespace openMVG {
namespace sfm {

/// Define 3D-2D tracking data: 3D landmark with its 2D observations
struct Observation
{
  Observation():id_feat(UndefinedIndexT) {  }
  Observation(const Vec2 & p, IndexT idFeat): x(p), id_feat(idFeat) {}

  Vec2 x;
  IndexT id_feat;

  // Serialization
  template <class Archive>
  void save( Archive & ar) const;

  // Serialization
  template <class Archive>
  void load( Archive & ar);
};

struct ObservationKeyValue_t
{
  ObservationKeyValue_t() {}

  ObservationKeyValue_t(IndexT key_, const Observation& value_) :
    first(key_),
    second(value_)
  {}

  IndexT first;
  Observation second;

  // Serialization
  template <class Archive>
  void save(Archive& ar) const;

  // Serialization
  template <class Archive>
  void load(Archive& ar);
};

/// Observations are indexed by their View_id
struct Observations
{
  using iterator = std::vector<ObservationKeyValue_t>::iterator;
  using const_iterator = std::vector<ObservationKeyValue_t>::const_iterator;

  const_iterator cbegin() const noexcept { return obs.begin(); }
  const_iterator cend() const noexcept { return obs.end(); }
  const_iterator begin() const noexcept { return cbegin(); }
  const_iterator end() const noexcept { return cend(); }
  iterator begin() noexcept { return obs.begin(); }
  iterator end() noexcept { return obs.end(); }

  void clear() noexcept { obs.clear(); }
  bool empty() const noexcept { return obs.empty(); }
  auto size() const noexcept { return obs.size(); }

  const Observation& at(IndexT i) const
  {
    for (auto& ob : obs) {
      if (i == ob.first) {
        return ob.second;
      }
    }

    throw std::out_of_range("Observations");
  }

#if 0 // JPB WIP
  template<class... Args>
  void emplace(Args&&... args)
  {
    return insert(std::forward<Args>(args)...);
  }
#endif

  const_iterator find(IndexT ob) const noexcept {
    int cnt = 0;
    for (auto it = begin(); it != end(); ++it, ++cnt)
    {
      if (ob == it->first)
      {
        return it;
      }
    }

    return end();
  }

#if 0
  auto erase(iterator ob) noexcept {
    int cnt = 0;
    for (auto it = begin(); it != end(); ++it, ++cnt)
    {
      if (ob == it)
      {
        obs.erase(std::begin(obs) + cnt);

        return ob;
      }
    }

    return end();
  }
#endif

  std::pair<iterator, bool> insert(const ObservationKeyValue_t& value)
  {
    for (auto it = begin(); it != end(); ++it) {
      if (value.first == it->first) {
        it->second = value.second;
        return { it, false };
      }
    }

    obs.emplace_back(value.first, value.second);

    sorted = false;

    return { std::prev(obs.end()), true};
  }

  std::pair<iterator, bool> insert(const_iterator hint, const ObservationKeyValue_t& value)
  {
    for (auto it = begin(); it != end(); ++it) {
      if (value.first == it->first) {
        it->second = value.second;
        return { it, false };
      }
    }

    obs.emplace_back(value.first, value.second);

    sorted = false;

    return { std::prev(obs.end()), true };
  }

  Observation& operator[](IndexT i)
  {
    for (auto& ob : obs) {
      if (i == ob.first) {
        return ob.second;
      }
    }

    obs.emplace_back(i, Observation());

    sorted = false;

    return obs.back().second;
  }

  void reserve(size_t cnt)
  {
    obs.reserve(cnt);
  }

  void swap(Observations& other)
  {
    std::swap(sorted, other.sorted);
    std::swap(obs, other.obs);
  }

  // Serialization
  template <class Archive>
  void save( Archive & ar) const;

  template <class Archive>
  void load( Archive & ar);

  bool sorted = false;
  std::vector<ObservationKeyValue_t> obs;
};

/// Define a landmark (a 3D point, with its 2d observations)
struct Landmark
{
  Vec3 X;
  Observations obs;

  // Serialization
  template <class Archive>
  void save( Archive & ar) const;

  template <class Archive>
  void load( Archive & ar);
};

/// Define a collection of landmarks are indexed by their TrackId
using Landmarks = Hash_Map<IndexT, Landmark>;

} // namespace sfm
} // namespace openMVG

#endif // OPENMVG_SFM_SFM_LANDMARK_HPP
