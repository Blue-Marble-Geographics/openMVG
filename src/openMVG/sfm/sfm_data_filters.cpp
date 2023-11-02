// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2015 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "openMVG/sfm/sfm_data_filters.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/stl/stl.hpp"
#include "openMVG/system/logger.hpp"
#include "openMVG/tracks/union_find.hpp"

#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace openMVG {
namespace sfm {

/// List the view indexes that have valid camera intrinsic and pose.
std::set<IndexT> Get_Valid_Views
(
  const SfM_Data & sfm_data
)
{
  std::set<IndexT> valid_idx;
  for (const auto & view_it : sfm_data.GetViews())
  {
    const View * v = view_it.second.get();
    if (sfm_data.IsPoseAndIntrinsicDefined(v))
    {
      valid_idx.insert(v->id_view);
    }
  }
  return valid_idx;
}

// Remove tracks that have a small angle (tracks with tiny angle leads to instable 3D points)
// Return the number of removed tracks
IndexT RemoveOutliers_PixelResidualError
(
  SfM_Data & sfm_data,
  const double dThresholdPixel,
  const unsigned int minTrackLength
)
{
  IndexT outlier_count = 0;
  auto& structure = sfm_data.structure;
  Landmarks::iterator iterTracks = structure.begin();
  const double dThresholdPixelSquared = dThresholdPixel * dThresholdPixel;
  const auto& poses = sfm_data.GetPoses();
  const auto& intrinsics = sfm_data.GetIntrinsics();
  while (iterTracks != structure.end())
  {
    Observations & obs = iterTracks->second.obs;
    Observations::iterator itObs = obs.begin();
    size_t cnt = obs.size();
    while (cnt--)
    {
      const auto& ibObsPair = *itObs;
      const View* view = sfm_data.views.at(ibObsPair.first).get();
      const geometry::Pose3& pose = poses.find(view->id_view)->second;
      const cameras::IntrinsicBase * intrinsic = intrinsics.find(view->id_intrinsic)->second.get();
      const Vec2 residual = intrinsic->residual(pose(iterTracks->second.X), ibObsPair.second.x);
      if (residual.squaredNorm() > dThresholdPixelSquared)
      {
        ++outlier_count;
        itObs = obs.erase(itObs);
      }
      else
        ++itObs;
    }
    if (obs.empty() || obs.size() < minTrackLength)
      iterTracks = structure.erase(iterTracks);
    else
      ++iterTracks;
  }
  return outlier_count;
}

// Remove tracks that have a small angle (tracks with tiny angle leads to instable 3D points)
// Return the number of removed tracks
IndexT RemoveOutliers_AngleError
(
  SfM_Data & sfm_data,
  const double dMinAcceptedAngle
)
{
  std::unordered_map<const View*, std::pair<Mat3, const cameras::IntrinsicBase*>> poseInfo;
  poseInfo.reserve(sfm_data.views.size());
  const auto& poses = sfm_data.GetPoses();
  const auto& intrinsics = sfm_data.GetIntrinsics();
  for (const auto& it : sfm_data.views) {
    auto tmp = poses.find(it.second->id_pose);
    if (tmp != poses.end())
    {
      poseInfo.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(it.second.get()),
        std::forward_as_tuple(poses.find(it.second->id_view)->second.rotation().transpose(), intrinsics.find(it.second->id_intrinsic)->second.get())
      );
    }
  }

  auto& structure = sfm_data.structure;

  std::vector<Vec3> rays;

  IndexT removedTrack_count = 0;
  Landmarks::iterator iterTracks = structure.begin();
  const auto& views = sfm_data.GetViews();

  double dMinAcceptedAngleRadians = D2R(dMinAcceptedAngle);
  while (iterTracks != structure.end())
  {
    Observations & obs = iterTracks->second.obs;
    double max_angle = 0.0;

    rays.clear();
    size_t cnt = obs.size();
    for (auto obs_it = std::begin(obs); cnt--; ++obs_it) {
      const View* view = views.find(obs_it->first)->second.get();
      const auto& pi = poseInfo.find(view)->second;

      rays.emplace_back(
        (pi.first * pi.second->oneBearing(pi.second->get_ud_pixel(obs_it->second.x))).normalized()
      );
    }

    for (size_t i = 0, cnt = rays.size(); i != cnt; ++i)
    {
      for (size_t j = i+1; j != cnt; ++j)
      {
        const double angle = cameras::AngleBetweenRayInRadians(rays[i], rays[j]);
        max_angle = std::max(angle, max_angle);
      }
    }
    if (max_angle < dMinAcceptedAngleRadians)
    {
      iterTracks = sfm_data.structure.erase(iterTracks);
      ++removedTrack_count;
    }
    else
      ++iterTracks;
  }
  return removedTrack_count;
}

bool eraseMissingPoses
(
  SfM_Data & sfm_data,
  const IndexT min_points_per_pose
)
{
  bool removed_an_element = false;
  const Landmarks & landmarks = sfm_data.structure;

#if 0
  int num_poses = sfm_data.GetPoses().size();
  int num_views = sfm_data.GetViews().size();
  std::array<IndexT, 256> view_poses;
  std::array<IndexT, 256> map_poseid_cnts;
  std::fill_n(std::begin(map_poseid_cnts), map_poseid_cnts.size(), 0);
  bool views_and_pose_ids_compact = true;
  if (num_views < 256) {
    for (const auto& i : sfm_data.GetViews()) {
      if (i.first >= 256) {
        views_and_pose_ids_compact = false;
        break;
      }
      view_poses[i.first] = i.second->id_pose;
    }

    if (views_and_pose_ids_compact) {
      // Init with 0 count (in order to be able to remove non referenced elements)
      for (const auto& i : sfm_data.GetPoses()) {
        if (i.first >= 256) {
          views_and_pose_ids_compact = false;
          break;
        }
        map_poseid_cnts[i.first] = 0;
      }
    }
  }
  else {
    views_and_pose_ids_compact = false;
  }

  if (views_and_pose_ids_compact) {
    // Count the observation poses occurrence
    // Count occurrence of the poses in the Landmark observations
    for (const auto& lanmark_it : landmarks)
    {
      const Observations& obs = lanmark_it.second.obs;
      size_t cnt = obs.size();
      for (auto obs_it = std::begin(obs); cnt--; ++obs_it) {
        ++map_poseid_cnts[view_poses[obs_it->first]]; // Default initialization is 0
      }
    }

    auto& poses = sfm_data.poses;
    // If usage count is smaller than the threshold, remove the Pose
    for (const auto& it : map_poseid_cnts)
    {
      if (it < min_points_per_pose)
      {
        poses.erase(it);
        removed_an_element = true;
      }
    }
  } else {
#endif
    // Count the observation poses occurrence
    Hash_Map<IndexT, IndexT> map_PoseId_Count;
    map_PoseId_Count.reserve(sfm_data.GetPoses().size());
    // Init with 0 count (in order to be able to remove non referenced elements)
    for (const auto& pose_it : sfm_data.GetPoses())
    {
      map_PoseId_Count[pose_it.first] = 0;
    }

    const auto& views = sfm_data.GetViews();
    // Count occurrence of the poses in the Landmark observations
    for (const auto& lanmark_it : landmarks)
    {
      const Observations & obs = lanmark_it.second.obs;
      size_t cnt = obs.size();
      for (auto obs_it = std::begin(obs); cnt--; ++obs_it) {
        const IndexT ViewId = obs_it->first;
        const View * v = views.find(ViewId)->second.get();
        map_PoseId_Count[v->id_pose] += 1; // Default initialization is 0
      }
    }

    auto& poses = sfm_data.poses;
    // If usage count is smaller than the threshold, remove the Pose
    for (const auto& it : map_PoseId_Count)
    {
      if (it.second < min_points_per_pose)
      {
        poses.erase(it.first);
        removed_an_element = 1;
      }
    }
#if 0
  }
#endif

  return removed_an_element;
}

bool eraseObservationsWithMissingPoses
(
  SfM_Data & sfm_data,
  const IndexT min_points_per_landmark
)
{
  bool removed_an_element = false;

  int num_poses = sfm_data.GetPoses().size();
  int num_views = sfm_data.GetViews().size();
  std::array<IndexT, 256> view_ids;
  bool view_ids_compact = true;
  if (num_views < 256) {
    for (const auto& i : sfm_data.GetViews()) {
      if (i.first >= 256) {
        view_ids_compact = false;
        break;
      }
      view_ids[i.first] = i.second->id_pose;
    }
  }
  else {
    view_ids_compact = false;
  }
  if (num_poses < 256 && view_ids_compact) {
    std::array<uint64_t, 4> pose_Index;
    std::fill_n(std::begin(pose_Index), pose_Index.size(), 0);
    for (const auto& i : sfm_data.GetPoses()) {
      int pos = i.first>>6;
      int bit = i.first & 63;
      pose_Index[pos] |= (1ULL << bit);
    }

    auto& structure = sfm_data.structure;
    // For each landmark:
    //  - Check if we need to keep the observations & the track
    Landmarks::iterator itLandmarks = structure.begin();
    while (itLandmarks != structure.end())
    {
      Observations& obs = itLandmarks->second.obs;
      Observations::iterator itObs = obs.begin();
      size_t cnt = obs.size();
      while (cnt--) {
        const IndexT ViewId = itObs->first;
        auto view_id_idx = view_ids[ViewId];
        int pos = view_id_idx>>6;
        int bit = view_id_idx & 63;

        if (!(pose_Index[pos] & (1ULL << bit))) {
          itObs = obs.erase(itObs);
          removed_an_element = true;
        }
        else
          ++itObs;
      }
      if (obs.empty() || obs.size() < min_points_per_landmark)
        itLandmarks = structure.erase(itLandmarks);
      else
        ++itLandmarks;
    }

  } else {
    std::unordered_set<IndexT> pose_Index;
    pose_Index.reserve(sfm_data.GetPoses().size() * 2);
    std::transform(sfm_data.poses.cbegin(), sfm_data.poses.cend(),
      std::inserter(pose_Index, pose_Index.begin()), stl::RetrieveKey());

    auto& structure = sfm_data.structure;
    // For each landmark:
    //  - Check if we need to keep the observations & the track
    Landmarks::iterator itLandmarks = structure.begin();
    const auto& views = sfm_data.GetViews();
    while (itLandmarks != structure.end())
    {
      Observations & obs = itLandmarks->second.obs;
      Observations::iterator itObs = obs.begin();
      while (itObs != obs.end())
      {
        const IndexT ViewId = itObs->first;
        const View * v = views.find(ViewId)->second.get();
        if (pose_Index.count(v->id_pose) == 0)
        {
          itObs = obs.erase(itObs);
          removed_an_element = true;
        }
        else
          ++itObs;
      }
      if (obs.empty() || obs.size() < min_points_per_landmark)
        itLandmarks = structure.erase(itLandmarks);
      else
        ++itLandmarks;
    }
  }
  return removed_an_element;
}

/// Remove unstable content from analysis of the sfm_data structure
bool eraseUnstablePosesAndObservations
(
  SfM_Data & sfm_data,
  const IndexT min_points_per_pose,
  const IndexT min_points_per_landmark
)
{
  // First remove orphan observation(s) (observation using an undefined pose)
  eraseObservationsWithMissingPoses(sfm_data, min_points_per_landmark);
  // Then iteratively remove orphan poses & observations
  IndexT remove_iteration = 0;
  bool bRemovedContent = false;
  do
  {
    bRemovedContent = false;
    if (eraseMissingPoses(sfm_data, min_points_per_pose))
    {
      bRemovedContent = eraseObservationsWithMissingPoses(sfm_data, min_points_per_landmark);
      // Erase some observations can make some Poses index disappear so perform the process in a loop
    }
    remove_iteration += bRemovedContent ? 1 : 0;
  }
  while (bRemovedContent);

  return remove_iteration > 0;
}

/// Tell if the sfm_data structure is one CC or not
bool IsTracksOneCC
(
  const SfM_Data & sfm_data
)
{
  // Compute the Connected Component from the tracks

  // Build a table to have contiguous view index in [0,n]
  // (Use only the view index used in the observations)
  Hash_Map<IndexT, IndexT> view_renumbering;
  IndexT cpt = 0;
  const Landmarks & landmarks = sfm_data.structure;
  for (const auto & Landmark_it : landmarks)
  {
    const Observations & obs = Landmark_it.second.obs;
    for (const auto & obs_it : obs)
    {
      if (view_renumbering.count(obs_it.first) == 0)
      {
        view_renumbering[obs_it.first] = cpt++;
      }
    }
  }

  UnionFind uf_tree;
  uf_tree.InitSets(view_renumbering.size());

  // Link track observations in connected component
  for (const auto & Landmark_it : landmarks)
  {
    const Observations & obs = Landmark_it.second.obs;
    std::set<IndexT> id_to_link;
    for (const auto & obs_it : obs)
    {
      id_to_link.insert(view_renumbering.at(obs_it.first));
    }
    std::set<IndexT>::const_iterator iterI = id_to_link.cbegin();
    std::set<IndexT>::const_iterator iterJ = id_to_link.cbegin();
    std::advance(iterJ, 1);
    while (iterJ != id_to_link.cend())
    {
      // Link I => J
      uf_tree.Union(*iterI, *iterJ);
      ++iterJ;
    }
  }

  // Run path compression to identify all the CC id belonging to every item
  for (unsigned int i = 0; i < uf_tree.GetNumNodes(); ++i)
  {
    uf_tree.Find(i);
  }

  // Count the number of CC
  const std::set<unsigned int> parent_id(uf_tree.m_cc_parent.cbegin(), uf_tree.m_cc_parent.cend());
  return parent_id.size() == 1;
}

/// Keep the largest connected component of tracks from the sfm_data structure
void KeepLargestViewCCTracks
(
  SfM_Data & sfm_data
)
{
  // Compute the Connected Component from the tracks

  // Build a table to have contiguous view index in [0,n]
  // (Use only the view index used in the observations)
  Hash_Map<IndexT, IndexT> view_renumbering;
  {
    IndexT cpt = 0;
    const Landmarks & landmarks = sfm_data.structure;
    for (const auto & Landmark_it : landmarks)
    {
      const Observations & obs = Landmark_it.second.obs;
      for (const auto & obs_it : obs)
      {
        if (view_renumbering.count(obs_it.first) == 0)
        {
          view_renumbering[obs_it.first] = cpt++;
        }
      }
    }
  }

  UnionFind uf_tree;
  uf_tree.InitSets(view_renumbering.size());

  // Link track observations in connected component
  Landmarks & landmarks = sfm_data.structure;
  for (const auto & Landmark_it : landmarks)
  {
    const Observations & obs = Landmark_it.second.obs;
    std::set<IndexT> id_to_link;
    for (const auto & obs_it : obs)
    {
      id_to_link.insert(view_renumbering.at(obs_it.first));
    }
    std::set<IndexT>::const_iterator iterI = id_to_link.cbegin();
    std::set<IndexT>::const_iterator iterJ = id_to_link.cbegin();
    std::advance(iterJ, 1);
    while (iterJ != id_to_link.cend())
    {
      // Link I => J
      uf_tree.Union(*iterI, *iterJ);
      ++iterJ;
    }
  }

  // Count the number of CC
  const std::set<unsigned int> parent_id(uf_tree.m_cc_parent.cbegin(), uf_tree.m_cc_parent.cend());
  if (parent_id.size() > 1)
  {
    // There is many CC, look the largest one
    // (if many CC have the same size, export the first that have been seen)
    std::pair<IndexT, unsigned int> max_cc( UndefinedIndexT, std::numeric_limits<unsigned int>::min());
    {
      for (const unsigned int parent_id_it : parent_id)
      {
        if (uf_tree.m_cc_size[parent_id_it] > max_cc.second) // Update the component parent id and size
        {
          max_cc = {parent_id_it, uf_tree.m_cc_size[parent_id_it]};
        }
      }
    }
    // Delete track ids that are not contained in the largest CC
    if (max_cc.first != UndefinedIndexT)
    {
      const unsigned int parent_id_largest_cc = max_cc.first;
      Landmarks::iterator itLandmarks = landmarks.begin();
      while (itLandmarks != landmarks.end())
      {
        // Since we built a view 'track' graph thanks to the UF tree,
        //  checking the CC of each track is equivalent to check the CC of any observation of it.
        // So we check only the first
        const Observations & obs = itLandmarks->second.obs;
        Observations::const_iterator itObs = obs.begin();
        if (!obs.empty())
        {
          if (uf_tree.Find(view_renumbering.at(itObs->first)) != parent_id_largest_cc)
          {
            itLandmarks = landmarks.erase(itLandmarks);
          }
          else
          {
            ++itLandmarks;
          }
        }
      }
    }
  }
}

/**
* @brief Implement a statistical Structure filter that remove 3D points that have:
* - a depth that is too large (threshold computed as factor * median ~= X84)
* @param sfm_data The sfm scene to filter (inplace filtering)
* @param k_factor The factor applied to the median depth per view
* @param k_min_point_per_pose Keep only poses that have at least this amount of points
* @param k_min_track_length Keep only tracks that have at least this length
* @return The min_median_value observed for all the view
*/
double DepthCleaning
(
  SfM_Data & sfm_data,
  const double k_factor,
  const IndexT k_min_point_per_pose,
  const IndexT k_min_track_length
)
{
  using DepthAccumulatorT = std::vector<double>;
  std::map<IndexT, DepthAccumulatorT > map_depth_accumulator;

  // For each landmark accumulate the camera/point depth info for each view
  for (const auto & landmark_it : sfm_data.structure)
  {
    const Observations & obs = landmark_it.second.obs;
    for (const auto & obs_it : obs)
    {
      const View * view = sfm_data.views.at(obs_it.first).get();
      if (sfm_data.IsPoseAndIntrinsicDefined(view))
      {
        const Pose3 pose = sfm_data.GetPoseOrDie(view);
        const double depth = Depth(pose.rotation(), pose.translation(), landmark_it.second.X);
        if (depth > 0)
        {
          map_depth_accumulator[view->id_view].push_back(depth);
        }
      }
    }
  }

  double min_median_value = std::numeric_limits<double>::max();
  std::map<IndexT, double > map_median_depth;
  for (const auto & iter : sfm_data.GetViews())
  {
    const View * v = iter.second.get();
    const IndexT view_id = v->id_view;
    if (map_depth_accumulator.count(view_id) == 0)
      continue;
    // Compute median from the depth distribution
    const auto & acc = map_depth_accumulator.at(view_id);
    double min, max, mean, median;
    if (minMaxMeanMedian(acc.begin(), acc.end(), min, max, mean, median))
    {

      min_median_value = std::min(min_median_value, median);
      // Compute depth threshold for each view: factor * medianDepth
      map_median_depth[view_id] = k_factor * median;
    }
  }
  map_depth_accumulator.clear();

  // Delete invalid observations
  size_t cpt = 0;
  for (auto & landmark_it : sfm_data.structure)
  {
    Observations obs;
    for (auto & obs_it : landmark_it.second.obs)
    {
      const View * view = sfm_data.views.at(obs_it.first).get();
      if (sfm_data.IsPoseAndIntrinsicDefined(view))
      {
        const Pose3 pose = sfm_data.GetPoseOrDie(view);
        const double depth = Depth(pose.rotation(), pose.translation(), landmark_it.second.X);
        if ( depth > 0
            && map_median_depth.count(view->id_view)
            && depth < map_median_depth[view->id_view])
          obs.insert(obs_it);
        else
          ++cpt;
      }
    }
    landmark_it.second.obs.swap(obs);
  }
  OPENMVG_LOG_INFO << "#point depth filter: " << cpt << " measurements removed";

  // Remove orphans
  eraseUnstablePosesAndObservations(sfm_data, k_min_point_per_pose, k_min_track_length);

  return min_median_value;
}

} // namespace sfm
} // namespace openMVG
