// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2013 Pierre Moulon, Bruno Duisit.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef OPENMVG_EXIF_SENSOR_WIDTH_PARSE_DATABASE_HPP
#define OPENMVG_EXIF_SENSOR_WIDTH_PARSE_DATABASE_HPP

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

#include "openMVG/exif/sensor_width_database/datasheet.hpp"
#include "openMVG/stl/split.hpp"
#include "openMVG/system/logger.hpp"
#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"

// Parse the database
bool parseDatabase( const std::string& sfileDatabase, std::vector<Datasheet>& vec_database )
{
  std::ifstream iFilein( sfileDatabase );
  if ( stlplus::is_file(sfileDatabase) && iFilein)
  {
    std::string line;
    while ( iFilein.good() )
    {
      getline( iFilein, line );
      if ( !line.empty() )
      {
        if ( line[0] != '#' )
        {
          std::vector<std::string> values;
          stl::split(line, ';', values);
          if ( values.size() == 2 )
          {
            vec_database.emplace_back(
              values[0], // model
              atof( values[1].c_str() ) // sensor size
              );
          }
        }
      }
    }
    return true;
  }
  else
  {
    OPENMVG_LOG_ERROR << "Cannot read the database file: " << sfileDatabase;
    return false;
  }
}

// Retrieve camera 'Datasheet' information for the given camera model model name
//  iff it is found in the database
bool getInfo
(
  const std::string & sModel,
  const std::vector<Datasheet>& vec_database,
  Datasheet& datasheetContent,
  bool bDoExtendedLookup = false // if no exact match on camera name, should we look for unique sub-match?
)
{
  bool existInDatabase = false;

  const Datasheet refDatasheet( sModel, -1. );
  std::vector<Datasheet>::const_iterator datasheet = std::find( vec_database.begin(), vec_database.end(), refDatasheet );
  if ( datasheet != vec_database.end() )
  {
    datasheetContent = *datasheet;
    existInDatabase = true;
  }
  else if (!sModel.empty() && bDoExtendedLookup)
  {
      // Do check for contains name
      std::string model_name_lc(sModel);
      std::transform(model_name_lc.begin(), model_name_lc.end(), model_name_lc.begin(), ::tolower);
      const Datasheet* best_datasheet = NULL;
      for (const auto& cur_datasheet : vec_database)
      {
          // If the current datasheet model name contains the provided model name, keep it
          std::string datasheet_model_name_lc(cur_datasheet.model_);
          std::transform(datasheet_model_name_lc.begin(), datasheet_model_name_lc.end(), datasheet_model_name_lc.begin(), ::tolower);
          if (NULL != ::strstr(datasheet_model_name_lc.c_str(), model_name_lc.c_str()))
          {
              if (!best_datasheet || (best_datasheet->sensorSize_ == cur_datasheet.sensorSize_))
              {
                  best_datasheet = &cur_datasheet;
              }
              else
              {
                  // Multiple matches with different sensor sizes, fail
                  best_datasheet = NULL;
                  break;
              }
          }
      }

      // If we found a match, use it
      if (best_datasheet)
      {
          datasheetContent = *best_datasheet;
          existInDatabase = true;
      }
  }

  return existInDatabase;
}

#endif // OPENMVG_EXIF_SENSOR_WIDTH_PARSE_DATABASE_HPP
