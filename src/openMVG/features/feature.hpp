// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2012, 2013 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef OPENMVG_FEATURES_FEATURE_HPP
#define OPENMVG_FEATURES_FEATURE_HPP

#include <algorithm>
#include <iterator>
#include <fstream>
#include <string>
#include <vector>

#include "P2PUtils.h"

#include "openMVG/numeric/eigen_alias_definition.hpp"
#include "openMVG/system/logger.hpp"

namespace openMVG {
namespace features {

/**
 * Base class for Point features.
 * Store position of a feature point.
 */
class PointFeature {

  friend std::ostream& operator<<(std::ostream& out, const PointFeature& obj);
  friend std::istream& operator>>(std::istream& in, PointFeature& obj);

public:
  PointFeature(float x=0.0f, float y=0.0f);

  float x() const;
  float y() const;
  inline const Vec2f & coords() const{ return coords_;}

  float& x();
  float& y();
  Vec2f& coords();

  template<class Archive>
  void serialize(Archive & ar)
  {
    ar (coords_(0), coords_(1));
  }

protected:
  Vec2f coords_;  // (x, y).
};

/**
 * Base class for ScaleInvariant Oriented Point features.
 * Add scale and orientation description to basis PointFeature.
 */
class SIOPointFeature : public PointFeature {

  friend std::ostream& operator<<(std::ostream& out, const SIOPointFeature& obj);
  friend std::istream& operator>>(std::istream& in, SIOPointFeature& obj);

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ~SIOPointFeature() = default;

  SIOPointFeature(float x=0.0f, float y=0.0f,
                  float scale=0.0f, float orient=0.0f);

  float scale() const;
  float& scale();
  float orientation() const;
  float& orientation();

  bool operator ==(const SIOPointFeature& b) const;

  bool operator !=(const SIOPointFeature& b) const;

  template<class Archive>
  void serialize(Archive & ar)
  {
    ar (
      coords_(0), coords_(1),
      scale_,
      orientation_);
  }

protected:
  float scale_;        // In pixels.
  float orientation_;  // In radians.
};

/// Return the coterminal angle between [0;2*PI].
/// Angle value must be in Radian.
float getCoterminalAngle(float angle);

/**
* Base class for Affine "Point" features.
* Add major & minor ellipse axis & orientation to the basis PointFeature.
*/
class AffinePointFeature : public PointFeature {

  friend std::ostream& operator<<(std::ostream& out, const AffinePointFeature& obj);
  friend std::istream& operator>>(std::istream& in, AffinePointFeature& obj);

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  virtual ~AffinePointFeature() = default;

  AffinePointFeature
  (
    float x = 0.0f,
    float y = 0.0f,
    float a = 0.0f,
    float b = 0.0f,
    float c = 0.0f
  );
  float l1() const;
  float l2() const;
  float orientation() const;

  bool operator ==(const AffinePointFeature& b) const;

  bool operator !=(const AffinePointFeature& rhs) const;

  template<class Archive>
  void serialize(Archive & ar)
  {
    ar (
      coords_(0), coords_(1),
      l1_, l2_, phi_, a_, b_, c_);
  }

  float a() const;
  float b() const;
  float c() const;

protected:
  float l1_, l2_, phi_, a_, b_, c_;
};

/// Read feats from file
#if BINARY_FEATURES
template<typename FeaturesT>
static bool loadFeatsFromBinFile(
  const std::string & sfileNameFeats,
  FeaturesT & vec_feat)
{
  vec_feat.clear();

  std::ifstream fileIn(sfileNameFeats.c_str(), std::ios::in | std::ios::binary);
  if (!fileIn.is_open())
      return false;
  std::size_t numFeats = 0;
  fileIn.read(reinterpret_cast<char*>(&numFeats), sizeof(numFeats));
  vec_feat.resize(numFeats);
  for (auto & it :vec_feat) {
      fileIn.read(reinterpret_cast<char*>(&it),sizeof(it));
  }
  const bool bOk = !fileIn.bad();
  fileIn.close();
  return bOk;
}
#else
template<typename FeaturesT>
static bool loadFeatsFromFile(
  const std::string & sfileNameFeats,
  FeaturesT & vec_feat)
{
  vec_feat.clear();

  std::ifstream fileIn(sfileNameFeats.c_str());
  if (!fileIn.is_open())
  {
    return false;
  }
  std::copy(
    std::istream_iterator<typename FeaturesT::value_type >(fileIn),
    std::istream_iterator<typename FeaturesT::value_type >(),
    std::back_inserter(vec_feat));
  const bool bOk = !fileIn.bad();
  fileIn.close();
  return bOk;
}
#endif

/// Write feats to file
#if BINARY_FEATURES
template<typename FeaturesT >
static bool saveFeatsToBinFile(
  const std::string & sfileNameFeats,
  FeaturesT & vec_feat)
{
  std::ofstream file(sfileNameFeats.c_str(), std::ios::out | std::ios::binary);
  if (!file.is_open())
    return false;
  const std::size_t numFeats = vec_feat.size();
  file.write((const char*) &numFeats,  sizeof(numFeats));
  for (const auto& iter : vec_feat) {
      file.write((const char*) &iter, sizeof(iter));
  }
  const bool bOk = file.good();
  file.close();
  return bOk;
}
#else
template<typename FeaturesT >
static bool saveFeatsToFile(
  const std::string & sfileNameFeats,
  FeaturesT & vec_feat)
{
  std::ofstream file(sfileNameFeats.c_str());
  if (!file.is_open())
    return false;
  std::copy(vec_feat.cbegin(), vec_feat.cend(),
            std::ostream_iterator<typename FeaturesT::value_type >(file,"\n"));
  const bool bOk = file.good();
  file.close();
  return bOk;
}
#endif

/// Export point feature based vector to a matrix [(x,y)'T, (x,y)'T]
template<typename FeaturesT>
void PointsToMat(
  const FeaturesT & vec_feats,
  Mat& m)
{
  m.resize(2, vec_feats.size());

  size_t i = 0;
  for (const auto &feat : vec_feats)
  {
    m.col(i++) << feat.x(), feat.y();
  }
}

} // namespace features
} // namespace openMVG

#endif // OPENMVG_FEATURES_FEATURE_HPP
