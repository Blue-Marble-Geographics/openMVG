// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2012, 2013 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "openMVG/robust_estimation/rand_sampling.hpp"

#include "testing/testing.h"

#include <numeric>
#include <set>

using namespace openMVG;
using namespace openMVG::robust;

std::mt19937 random_generator(std::mt19937::default_seed);

// Assert that each time exactly N random number are picked (no repetition)
TEST(UniformSampleTestInARange, NoRepetions) {

  std::vector<uint32_t> samples;
  for (size_t total = 1; total < 500; total *= 2) { //Size of the data set
    for (size_t num_samples = 1; num_samples <= total; num_samples *= 2) { //Size of the consensus set
      UniformSample(num_samples, total, random_generator, &samples);
      const std::set<uint32_t> myset(samples.begin(), samples.end());
      CHECK_EQUAL(num_samples, myset.size());
    }
  }
}

TEST(UniformSampleTestInAVectorOfUniqueElements, NoRepetions) {

  std::vector<uint32_t> samples;
  for (size_t total = 1; total < 500; total *= 2) { //Size of the data set
    std::vector<uint32_t> vec_index(total);
    std::iota(vec_index.begin(), vec_index.end(), 0);
    for (size_t num_samples = 1; num_samples <= total; num_samples *= 2) { //Size of the consensus set
      UniformSample(num_samples, random_generator, &vec_index, &samples);
      const std::set<uint32_t> myset(samples.begin(), samples.end());
      CHECK_EQUAL(num_samples, myset.size());
    }
  }
}

TEST( RandomSequence, t1 )
{
    {
        std::cout << "std::mt19937::default_seed: " << std::mt19937::default_seed << "\n";

        std::mt19937 random_generator( std::mt19937::default_seed );

        /*
        std::mt19937::default_seed: 5489
        r0: 3499211612
        r1: 581869302
        r2: 3890346734
        */
        std::cout << "r0: " << random_generator() << "\n";
        std::cout << "r1: " << random_generator() << "\n";
        std::cout << "r2: " << random_generator() << "\n";
    }
    {
        std::mt19937 random_generator( std::mt19937::default_seed );

        std::uniform_int_distribution<uint32_t> distribution( 0, 1024 - 1 );
        auto d0 = distribution( random_generator );
        /*
        d0: 834
        */
        std::cout << "d0: " << d0 << std::endl;
#if _MSC_VER >= 1930 /* vs2022 */
		EXPECT_EQ( 834, d0 );
#elif _MSC_VER >= 1920 /* vs2019 */
		EXPECT_EQ( 860, d0 );
#else
        // TODO: add your compiler results here
#endif
    }
}

/* ************************************************************************* */
int main() { TestResult tr; return TestRegistry::runAllTests(tr);}
/* ************************************************************************* */
