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
  Landmarks & landmarks = sfm_data.structure;
  auto & views = sfm_data.views;
  auto & intrinsics = sfm_data.intrinsics;
  IndexT outlier_count = 0;
  Landmarks::iterator iterTracks = landmarks.begin();
  const double dThresholdPixelSquared = dThresholdPixel * dThresholdPixel;

  // Precisely counts outliers.  Cannot early out on known track deletion.
  while (iterTracks != landmarks.end())
  {
    Observations & obs = iterTracks->second.obs;
    Observations::iterator itObs = obs.begin();

    bool eraseTrack = false;
    while (itObs != obs.end())
    {
      const View * view = views.at(itObs->first).get();
      const geometry::Pose3 pose = sfm_data.GetPoseOrDie(view);
      const cameras::IntrinsicBase * intrinsic = intrinsics.at(view->id_intrinsic).get();
      const Vec2 residual = intrinsic->residual(pose(iterTracks->second.X), itObs->second.x);
      if (residual.squaredNorm() > dThresholdPixelSquared)
      {
        ++outlier_count;
        itObs = obs.erase(itObs);
      }
      else
        ++itObs;
    }

    if (obs.empty() || obs.size() < minTrackLength)
    {
      iterTracks = landmarks.erase(iterTracks);
    }
    else
    {
      ++iterTracks;
    }
  }
  return outlier_count;
}

void RemoveOutliers_PixelResidualErrorWithoutCount
(
  SfM_Data & sfm_data,
  const double dThresholdPixel,
  const unsigned int minTrackLength
)
{
  Landmarks & landmarks = sfm_data.structure;
  const auto & views = sfm_data.views;
  const auto & intrinsics = sfm_data.intrinsics;
  Landmarks::iterator iterTracks = landmarks.begin();
  const double dThresholdPixelSquared = dThresholdPixel * dThresholdPixel;
  while (iterTracks != landmarks.end())
  {
    Observations & obs = iterTracks->second.obs;
    Observations::iterator itObs = obs.begin();
    bool eraseTrack = false;
    if (obs.empty() || (obs.size() < minTrackLength))
    {
      eraseTrack = true;
    }

    if (!eraseTrack)
    {
      while (itObs != obs.end())
      {
        const View * view = views.at(itObs->first).get();
        const geometry::Pose3 pose = sfm_data.GetPoseOrDie(view);
        const cameras::IntrinsicBase * intrinsic = intrinsics.at(view->id_intrinsic).get();
        const Vec2 residual = intrinsic->residual(pose(iterTracks->second.X), itObs->second.x);
        if (residual.squaredNorm() > dThresholdPixelSquared)
        {
          const int cnt = obs.size();
          if ((1 == cnt) || (cnt <= minTrackLength)) {
            eraseTrack = true;
            break;
          }
          itObs = obs.erase(itObs);
        }
        else
        {
          ++itObs;
        }
      }
    }

    if (eraseTrack)
    {
      iterTracks = landmarks.erase(iterTracks);
    }
    else
    {
      ++iterTracks;
    }
  }
}

// Remove tracks that have a small angle (tracks with tiny angle leads to instable 3D points)
// Return the number of removed tracks
IndexT RemoveOutliers_AngleError
(
  SfM_Data & sfm_data,
  const double dMinAcceptedAngle
)
{
  IndexT removedTrack_count = 0;
  Landmarks::iterator iterTracks = sfm_data.structure.begin();

  // Caching projections saves a little time (20%).
  // Adding the early out saves a lot of time (3x faster).
  std::vector<std::tuple<bool, Pose3, const cameras::IntrinsicBase*, Vec2>> objJPts;
  Landmarks & landmarks = sfm_data.structure;
  const auto & views = sfm_data.views;
  const auto & intrinsics = sfm_data.intrinsics;

  while (iterTracks != landmarks.end())
      {
    Observations & obs = iterTracks->second.obs;
    double max_angle = 0.0;

    objJPts.resize( obs.size() );

    // Remember, we are reusing the store; however, we need to
    // reset the cached entries for the next iteration too.
    for (auto & i : objJPts)
    {
      std::get<0>(i) = false;
    }
    int idxI = 0;
    // WIP Could rewrite the loop to avoid the last obs calculation.
    for (Observations::const_iterator itObs1 = obs.begin();
      itObs1 != obs.end(); ++itObs1)
    {
      const View * view1 = views.at(itObs1->first).get();
      // Getting the post incurs a modest penalty (20%)
      const geometry::Pose3 pose1 = sfm_data.GetPoseOrDie(view1);
      const cameras::IntrinsicBase * intrinsic1 = intrinsics.at(view1->id_intrinsic).get();
      const Vec2 objIPt = intrinsic1->get_ud_pixel(itObs1->second.x);

      Observations::const_iterator itObs2 = itObs1;
      ++itObs2;
      int idxJ = idxI+1;
      for (; itObs2 != obs.end(); ++itObs2)
      {
        auto & cachedObjJInfo = objJPts[ idxJ ];
        if (!std::get<0>( cachedObjJInfo))
        {
          const View * view2 = views.at(itObs2->first).get();
          std::get<1>(cachedObjJInfo) = sfm_data.GetPoseOrDie(view2);
          const cameras::IntrinsicBase * intrinsic2 = intrinsics.at(view2->id_intrinsic).get();
          std::get<2>(cachedObjJInfo) = intrinsic2;
          std::get<3>(cachedObjJInfo) = intrinsic2->get_ud_pixel(itObs2->second.x);
          std::get<0>(cachedObjJInfo) = true;
        }

        const double angle = AngleBetweenRay(
          pose1, intrinsic1,
          std::get<1>(cachedObjJInfo), std::get<2>(cachedObjJInfo),
          objIPt, std::get<3>(cachedObjJInfo)
        );

        if (angle >= dMinAcceptedAngle)
        {
          goto EarlyOut;
        }
        max_angle = std::max(angle, max_angle);

        ++idxJ;
      }

      ++idxI;
    }

    if (max_angle < dMinAcceptedAngle)
    {
      iterTracks = landmarks.erase(iterTracks);
      ++removedTrack_count;
    }
    else
    {
EarlyOut:
      ++iterTracks;
    }
  }

  return removedTrack_count;
}

bool eraseMissingPoses
(
  SfM_Data & sfm_data,
  const IndexT min_points_per_pose
)
{
  // Fast path requires #poses and #views <= 256
  // Replacing hash map map_PoseId_Count with arrays makes this about 3x faster.
  // WIP Likely can be improved.
  bool removed_an_element = false;
  const Landmarks & landmarks = sfm_data.structure;
  int num_poses = sfm_data.GetPoses().size();
  int num_views = sfm_data.GetViews().size();
  std::array<IndexT, 256> view_poses;
  std::array<IndexT, 256> map_poseid_cnts {}; // WIP Avoid?
  bool views_and_pose_ids_compact = true;
  if (num_views < 256)
  {
    for (const auto& i : sfm_data.GetViews())
    {
      if (i.first >= 256)
      {
        views_and_pose_ids_compact = false;
        break;
      }
      view_poses[i.first] = i.second->id_pose;
    }

    if (views_and_pose_ids_compact)
    {
      // Init with 0 count (in order to be able to remove non referenced elements)
      for (const auto& i : sfm_data.GetPoses())
      {
        if (i.first >= 256)
        {
          views_and_pose_ids_compact = false;
          break;
        }
        map_poseid_cnts[i.first] = 0;
      }
    }
  }
  else
  {
    views_and_pose_ids_compact = false;
  }

  if (views_and_pose_ids_compact)
  {
    // Count the observation poses occurrence
    // Count occurrence of the poses in the Landmark observations
    for (const auto& lanmark_it : landmarks)
    {
      for (const auto& it : lanmark_it.second.obs)
      {
        ++map_poseid_cnts[view_poses[it.first]]; // Default initialization is 0
      }
    }

    auto& poses = sfm_data.poses;
    // If usage count is smaller than the threshold, remove the Pose
    for (auto it = std::begin(poses); it != std::end(poses);)
    {
      if (map_poseid_cnts[it->first] < min_points_per_pose)
      {
        poses.erase(it++);
        removed_an_element = true;
      }
      else
      {
        ++it;
      }
    }
  }
  else
  {
    // Count the observation poses occurrence
    Hash_Map<IndexT, IndexT> map_PoseId_Count;
    // Init with 0 count (in order to be able to remove non referenced elements)
    for (const auto & pose_it : sfm_data.GetPoses())
    {
      map_PoseId_Count[pose_it.first] = 0;
    }
    
    // Count occurrence of the poses in the Landmark observations
    for (const auto & lanmark_it : landmarks)
    {
      const Observations & obs = lanmark_it.second.obs;
      for (const auto & obs_it : obs)
      {
        const IndexT ViewId = obs_it.first;
          const View * v = sfm_data.GetViews().at(ViewId).get();
        map_PoseId_Count[v->id_pose] += 1; // Default initialization is 0
      }
    }
    // If usage count is smaller than the threshold, remove the Pose
    for (const auto & it : map_PoseId_Count)
    {
      if (it.second < min_points_per_pose)
      {
          sfm_data.poses.erase(it.first);
          removed_an_element = true;
      }
    }
  }
  return removed_an_element;
}

bool eraseObservationsWithMissingPoses
(
  SfM_Data & sfm_data,
  const IndexT min_points_per_landmark
)
{
  // Fast path requires #poses and #views <= 256
  // Replacing std::set<IndexT> pose_Index with an array makes this at least twice as fast;
  int num_poses = sfm_data.GetPoses().size();
  int num_views = sfm_data.GetViews().size();
  std::array<uint8_t, 256> view_ids;
  bool view_id_poses_compact = true;
  if (num_views < 256)
  {
    for (const auto& i : sfm_data.GetViews())
    {
      if (i.second->id_pose >= 256)
      {
        view_id_poses_compact = false;
        break;
      }
      view_ids[i.first] = i.second->id_pose;
    }
  }
  else
  {
    view_id_poses_compact = false;
  }

  bool removed_an_element = false;
  if (view_id_poses_compact)
  {
    std::array<uint64_t, 4> pose_Index {};
    for (const auto& i : sfm_data.GetPoses())
    {
      int pos = i.first>>6;
      int bit = i.first & 63;
      pose_Index[pos] |= (1ULL << bit);
    }

    // Almost all of the time taken here is just accessing semi-cached obs data.
    auto& landmarks = sfm_data.structure;
    for (auto itLandmarks = std::begin(landmarks); itLandmarks != std::end(landmarks); )
    {
      auto& landmark_obs = itLandmarks->second.obs.obs;

      bool erase_entire_landmark = false;
      if (landmark_obs.empty() || (landmark_obs.size() < min_points_per_landmark))
      {
        erase_entire_landmark = true;
      }

      if (!erase_entire_landmark)
      {
        Observations::iterator itObs = landmark_obs.begin();
        while (itObs != landmark_obs.end())
        {
          const IndexT ViewId = itObs->first;
          auto view_id_idx = view_ids[ViewId];
          int pos = view_id_idx>>6;
          int bit = view_id_idx & 63;
          bool has_pose = pose_Index[pos] & ( 1ULL << bit );
          if ( !has_pose )
          {
            removed_an_element = true;
            const int cnt = landmark_obs.size();
            if ((1 == cnt) || (cnt  <= min_points_per_landmark))
            {
              erase_entire_landmark = true;
              break;
            }

            itObs = landmark_obs.erase( itObs );
           }
           else
           {
              ++itObs;
           }
        }
      }

      if (erase_entire_landmark)
      {
        // This erase takes a significant amount of time as the landmark is an unordered_map.
        itLandmarks = landmarks.erase( itLandmarks );
      }
      else
      {
        ++itLandmarks;
      }
    }
  }
  else
  {
    std::set<IndexT> pose_Index;
    std::transform(sfm_data.poses.cbegin(), sfm_data.poses.cend(),
      std::inserter(pose_Index, pose_Index.begin()), stl::RetrieveKey());

    // For each landmark:
    //  - Check if we need to keep the observations & the track
    auto & landmarks = sfm_data.structure;
    auto & views = sfm_data.GetViews();
    Landmarks::iterator itLandmarks = landmarks.begin();
    while (itLandmarks != landmarks.end())
    {
      Observations & obs = itLandmarks->second.obs;
      Observations::iterator itObs = obs.begin();
      while (itObs != obs.end())
      {
        const IndexT ViewId = itObs->first;
        const View * v = views.at(ViewId).get();
        if (pose_Index.count(v->id_pose) == 0)
        {
          itObs = obs.erase(itObs);
          removed_an_element = true;
        }
        else
        {
          ++itObs;
        }
      }

      if (obs.empty() || obs.size() < min_points_per_landmark)
      {
        itLandmarks = landmarks.erase(itLandmarks);
      }
      else
      {
        ++itLandmarks;
      }
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
    for (const auto& obs_it : obs)
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
      const Observations& obs = Landmark_it.second.obs;
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
