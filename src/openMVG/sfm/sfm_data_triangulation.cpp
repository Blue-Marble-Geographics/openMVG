// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2015 Pierre Moulon.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "openMVG/sfm/sfm_data_triangulation.hpp"

#include <deque>
#include <functional>

#include "openMVG/geometry/pose3.hpp"
#include "openMVG/multiview/triangulation_nview.hpp"
#include "openMVG/multiview/triangulation.hpp"
#include "openMVG/robust_estimation/rand_sampling.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_landmark.hpp"
#include "openMVG/system/loggerprogress.hpp"

#include "ceres/internal/fixed_array.h" // Borrow this
using namespace ceres::internal;

namespace openMVG {
namespace sfm {

using namespace openMVG::geometry;
using namespace openMVG::cameras;

SfM_Data_Structure_Computation_Basis::SfM_Data_Structure_Computation_Basis
(
  bool bConsoleVerbose
)
  :bConsole_verbose_(bConsoleVerbose)
{
}

SfM_Data_Structure_Computation_Blind::SfM_Data_Structure_Computation_Blind
(
  bool bConsoleVerbose
)
  :SfM_Data_Structure_Computation_Basis(bConsoleVerbose)
{
}

/// Triangulate a given set of observations
bool track_triangulation
(
  const SfM_Data & sfm_data,
  const Observations & obs,
  Vec3 & X,
  const ETriangulationMethod & etri_method = ETriangulationMethod::DEFAULT
)
{
  const size_t cnt = obs.size();
  if (obs.size() >= 2)
  {
    FixedArray<Vec3, 16, 0 /* No init */> bearing(cnt); // cnt is not the size of the array, but how many elements it can hold before it becomes dynamic.
    FixedArray<Mat34, 16, 0 /* No init */> poses(cnt);
    FixedArray<const Pose3*, 16, 0 /* No init */> poses_(cnt);

    int trueCnt = 0;

    auto observation = std::begin(obs);
    const auto& intrinsics = sfm_data.GetIntrinsics();
    const auto& allPoses = sfm_data.poses;
    for (size_t i = 0; i != cnt; ++i, ++observation)
    {
      const View* view = sfm_data.views.at(observation->first).get();
      if (!view) continue;
      if (view->id_intrinsic == UndefinedIndexT) continue;
      if (view->id_pose == UndefinedIndexT) continue;

      const auto cam = intrinsics.at(view->id_intrinsic).get();
      auto it = intrinsics.find(view->id_intrinsic);
      if (it == intrinsics.end()) continue;
      auto itPose = allPoses.find(view->id_pose);
      if (itPose == allPoses.end()) continue;

      const auto tmp = cam->get_ud_pixel(observation->second.x);
      bearing[trueCnt] = cam->oneBearing(tmp);
      poses[trueCnt] = itPose->second.asMatrix();
      poses_[trueCnt] = &itPose->second;
      ++trueCnt;
    }
    if (trueCnt > 2)
    {
      const Eigen::Map<const Mat3X> bearing_matrix(bearing[0].data(), 3, trueCnt);
      Vec4 Xhomogeneous;
      if (TriangulateNViewAlgebraic2
      (
        bearing_matrix,
        poses.get(),
        &Xhomogeneous))
      {
        X = Xhomogeneous.hnormalized();
        return true;
      }
    }
    else
    {
      return Triangulate2View
      (
        poses_[0]->rotation(),
        poses_[0]->translation(),
        bearing[0],
        poses_[trueCnt-1]->rotation(),
        poses_[trueCnt-1]->translation(),
        bearing[trueCnt-1],
        X,
        etri_method
      );
    }
  }
  return false;
}

// Test if a predicate is true for each observation
// i.e: predicate could be:
// - cheirality test (depth test): cheirality_predicate
// - cheirality and residual error: ResidualAndCheiralityPredicate::predicate
bool track_check_predicate
(
  const Observations & obs,
  const SfM_Data & sfm_data,
  const Vec3 & X,
  std::function<bool(
    const IntrinsicBase&,
    const Pose3&,
    const Vec2&,
    const Vec3&)> predicate
)
{
  bool visibility = false; // assume that no observation has been looked yet
  const auto& intrinsics = sfm_data.GetIntrinsics();
  const auto& allPoses = sfm_data.poses;
  for (const auto & obs_it : obs)
  {
    const View* view = sfm_data.views.at(obs_it.first).get();
    if (!view) continue;
    if (view->id_intrinsic == UndefinedIndexT) continue;
    if (view->id_pose == UndefinedIndexT) continue;

    const auto cam = intrinsics.at(view->id_intrinsic).get();
    auto it = intrinsics.find(view->id_intrinsic);
    if (it == intrinsics.end()) continue;
    auto itPose = allPoses.find(view->id_pose);
    if (itPose == allPoses.end()) continue;

    visibility = true; // at least an observation is evaluated
    if (!predicate(*cam, itPose->second, obs_it.second.x, X))
      return false;
  }
  return visibility;
}

bool cheirality_predicate
(
  const IntrinsicBase& cam,
  const Pose3& pose,
  const Vec2& x,
  const Vec3& X
)
{
  return CheiralityTest(cam.oneBearing(x), pose, X);
}

struct ResidualAndCheiralityPredicate
{
  const double squared_pixel_threshold_;

  ResidualAndCheiralityPredicate(const double squared_pixel_threshold)
    :squared_pixel_threshold_(squared_pixel_threshold){}

  bool predicate
  (
    const IntrinsicBase& cam,
    const Pose3& pose,
    const Vec2& x,
    const Vec3& X
  )
  {
    const Vec2 residual = cam.residual(pose(X), x);
    return CheiralityTest(cam.oneBearing(x), pose, X) &&
           residual.squaredNorm() < squared_pixel_threshold_;
  }
};

void SfM_Data_Structure_Computation_Blind::triangulate
(
  SfM_Data & sfm_data
)
const
{
  std::deque<IndexT> rejectedId;
  std::unique_ptr<system::ProgressInterface> my_progress_bar;
  if (bConsole_verbose_)
    my_progress_bar.reset(
      new system::LoggerProgress(
        sfm_data.structure.size(),
        "Blind triangulation progress" ));
#ifdef OPENMVG_USE_OPENMP
  #pragma omp parallel
#endif
  for (auto& tracks_it :sfm_data.structure)
  {
#ifdef OPENMVG_USE_OPENMP
  #pragma omp single nowait
#endif
    {
      if (bConsole_verbose_)
      {
        ++(*my_progress_bar);
      }

      const Observations & obs = tracks_it.second.obs;
      bool bKeep = false;
      {
        // Generate the track 3D hypothesis
        Vec3 X;
        if (track_triangulation(sfm_data, obs, X))
        {
          // Keep the point only if it has a positive depth for all obs
          if (track_check_predicate(obs, sfm_data, X, cheirality_predicate))
          {
            tracks_it.second.X = X;
            bKeep = true;
          }
        }
      }
      if (!bKeep)
      {
#ifdef OPENMVG_USE_OPENMP
        #pragma omp critical
#endif
        rejectedId.push_front(tracks_it.first);
      }
    }
  }
  // Erase the unsuccessful triangulated tracks
  for (auto& it : rejectedId)
  {
    sfm_data.structure.erase(it);
  }
}

SfM_Data_Structure_Computation_Robust::SfM_Data_Structure_Computation_Robust
(
  const double max_reprojection_error,
  const IndexT min_required_inliers,
  const IndexT min_sample_index,
  const ETriangulationMethod etri_method,
  bool bConsoleVerbose
):
  SfM_Data_Structure_Computation_Basis(bConsoleVerbose),
  max_reprojection_error_(max_reprojection_error),
  min_required_inliers_(min_required_inliers),
  min_sample_index_(min_sample_index),
  etri_method_(etri_method)
{
}

void SfM_Data_Structure_Computation_Robust::triangulate
(
  SfM_Data & sfm_data
)
const
{
  robust_triangulation(sfm_data);
}

/// Robust triangulation of track data contained in the structure
/// All observations must have View with valid Intrinsic and Pose data
/// Invalid landmark are removed.
void SfM_Data_Structure_Computation_Robust::robust_triangulation
(
  SfM_Data& sfm_data
)
const
{
  std::unique_ptr<system::ProgressInterface> my_progress_bar;
  std::vector<Landmarks::iterator> tracks;
  const int cnt = (int)sfm_data.structure.size();
  tracks.reserve( cnt );

  auto last = sfm_data.structure.end();
  for (Landmarks::iterator it = sfm_data.structure.begin(); it != last; ++it)
  {
    tracks.push_back( it );
  }

#ifdef OPENMVG_USE_OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < cnt; i++)
  {
    auto& track = *tracks[ i ];
    if (robust_triangulation( sfm_data, track.second.obs, track.second))
    {
      tracks[ i ] = last;
    }
  }

  for (int i = 0; i != cnt; ++i)
  {
    if (tracks[ i ] != last)
    {
      sfm_data.structure.erase( tracks[ i ] );
    }
  }
}

Observations ObservationsSampler
(
  const Observations & obs,
  const std::vector<std::uint32_t> & samples
)
{
  Observations sampled_obs;
  sampled_obs.reserve(samples.size());
  for (const auto& idx : samples)
  {
    Observations::const_iterator obs_it = obs.cbegin();
    std::advance(obs_it, idx);
    sampled_obs.insert(*obs_it);
  }
  return sampled_obs;
}

void ObservationsSampler
(
  Observations& sampled_obs,
  const Observations & obs,
  const std::uint32_t* samples,
  size_t cnt
)
{
  sampled_obs.clear();
  for (size_t i = 0; i != cnt; ++i)
  {
    const std::uint32_t idx = samples[i];
    Observations::const_iterator obs_it = obs.cbegin();
    std::advance(obs_it, idx);
    sampled_obs.insert(*obs_it);
  }
}

/// Robustly try to estimate the best 3D point using a ransac scheme
/// A point must be seen in at least min_required_inliers views
/// Return true for a successful triangulation
bool SfM_Data_Structure_Computation_Robust::robust_triangulation
(
  const SfM_Data & sfm_data,
  const Observations & obs,
  Landmark & landmark // X & valid observations
)
const
{
  if (obs.size() < min_required_inliers_ || obs.size() < min_sample_index_)
  {
    return false;
  }

  const double dSquared_pixel_threshold = Square(max_reprojection_error_);

  // Predicate to validate a sample (cheirality and residual error)
  ResidualAndCheiralityPredicate predicate(dSquared_pixel_threshold);
  auto predicate_binding = std::bind(&ResidualAndCheiralityPredicate::predicate,
                                     predicate,
                                     std::placeholders::_1,
                                     std::placeholders::_2,
                                     std::placeholders::_3,
                                     std::placeholders::_4);

  // Handle the case where all observations must be used
  if (min_required_inliers_ == min_sample_index_ &&
      obs.size() == min_required_inliers_)
  {
    // Generate the 3D point hypothesis by triangulating all the observations
    Vec3 X;
    if (track_triangulation(sfm_data, obs, X, etri_method_) &&
        track_check_predicate(obs, sfm_data, X, predicate_binding))
    {
      landmark.X = X;
      landmark.obs = obs;
      return true;
    }
    return false;
  }

  // else we perform a robust estimation since
  //  there is more observations than the minimal number of required sample.

  const IndexT nbIter = obs.size() * 2; // TODO: automatic computation of the number of iterations?

  // - Ransac variables
  Vec3 best_model = Vec3::Zero();
  FixedArray<IndexT, 32, 0 /* No init */> best_inlier_set(obs.size()); // Never larger
  size_t best_inlier_set_size = 0;
  double best_error = std::numeric_limits<double>::max();

  //--
  // Random number generation
  std::mt19937 random_generator(std::mt19937::default_seed);

  const auto& intrinsics = sfm_data.GetIntrinsics();

#if 0 // JPB WIP Revisit
  const size_t num_obs = obs.size();
  std::vector<const View*> obsViews(num_obs);
  std::vector<const IntrinsicBase*> obsCams(num_obs);
  std::vector<const std::pair<const IndexT, Pose3>*> obsPoses(num_obs);

  const auto& allPoses = sfm_data.poses;
  size_t idx = 0;
  for (const auto & obs_it : obs)
  {
    const View* __restrict view = sfm_data.views.at(obs_it.first).get();
    obsViews[idx] = view;

    if (!view) continue;
    const auto view_intrinsic = view->id_intrinsic;
    if (view_intrinsic == UndefinedIndexT) continue;

    const auto view_pose = view->id_pose;
    if (view_pose == UndefinedIndexT) continue;

    auto it = intrinsics.find(view_intrinsic);
    obsCams[idx] = it->second.get();
    auto itPose = allPoses.find(view_pose);
    obsPoses[idx] = &(*itPose);
  }
#endif

  // - Ransac loop
  Observations minimal_sample;
  minimal_sample.reserve(10); // Just a guess
  for (IndexT i = 0; i < nbIter; ++i)
  {
    FixedArray<uint32_t, 16, 0 /* No init */> samples(obs.size());
    const size_t numActualSamples = robust::UniformSample2(min_sample_index_, obs.size(), random_generator, samples.get());
    Vec3 X;
    // Hypothesis generation
    ObservationsSampler(minimal_sample, obs, samples.get(), numActualSamples);

    if (!track_triangulation(sfm_data, minimal_sample, X, etri_method_))
      continue;

    // Test validity of the hypothesis
    if (!track_check_predicate(minimal_sample, sfm_data, X, predicate_binding))
      continue;

    FixedArray<IndexT, 32, 0 /* No init */> inlier_set(obs.size()); // Never larger
    size_t inlier_cnt = 0;
    double current_error = 0.0;
    // inlier/outlier classification according pixel residual errors.
    const auto& intrinsics = sfm_data.GetIntrinsics();
    const auto& poses = sfm_data.GetPoses();
    for (const auto & obs_it : obs)
    {
      const View * view = sfm_data.views.at(obs_it.first).get();

      if (!view) continue;
      if (view->id_intrinsic == UndefinedIndexT) continue;
      if (view->id_pose == UndefinedIndexT) continue;
      auto it = intrinsics.find(view->id_intrinsic);
      if (it == intrinsics.end()) continue;
      auto itPose = poses.find(view->id_pose);
      if (itPose == poses.end()) continue;

      const IntrinsicBase & cam =  *(it->second.get());
      const Pose3& pose = itPose->second;
      if (!CheiralityTest(cam.oneBearing(obs_it.second.x), pose, X))
        continue;
      const double residual_sq = cam.residual(pose(X), obs_it.second.x).squaredNorm();
      if (residual_sq < dSquared_pixel_threshold)
      {
        inlier_set[inlier_cnt++] = obs_it.first;
        current_error += residual_sq;
      }
      else
      {
        current_error += dSquared_pixel_threshold;
      }
    }
    // Does the hypothesis:
    // - is the best one we have seen so far.
    // - has sufficient inliers.
    if (current_error < best_error &&
      inlier_cnt >= min_required_inliers_)
    {
      best_model = X;
      std::copy(std::begin(inlier_set), std::begin(inlier_set)+inlier_cnt, std::begin(best_inlier_set));
      best_inlier_set_size = inlier_cnt;
      best_error = current_error;
    }
  }

  if (best_inlier_set_size >= min_required_inliers_)
  {
    // Update information (3D landmark position & valid observations)
    landmark.X = best_model;
    for (size_t i = 0; i != best_inlier_set_size; ++i)
    {
      const auto val = best_inlier_set[i];
      landmark.obs.insert({ val, obs.at(val) }); // JPB WIP BUG  emplace(val, obs.at(val));
    }
  }
  return best_inlier_set_size;
}

} // namespace sfm
} // namespace openMVG
