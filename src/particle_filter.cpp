/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  this->num_particles = 10 ;

  std::random_device random_device;
  std::mt19937 generator(random_device());
  std::normal_distribution<double> x_distribution(x, std[0]);
  std::normal_distribution<double> y_distribution(y, std[1]);
  std::normal_distribution<double> theta_distribution(theta, std[2]);

  for(int index = 0 ; index < this->num_particles ; ++index)
  {
    auto particle = Particle() ;
    particle.id = index ;
    particle.x = x_distribution(generator) ;
    particle.y = y_distribution(generator) ;
    particle.theta = theta_distribution(generator) ;
    particle.weight = 1.0 ;

    this->particles.push_back(particle) ;
  }

  this->is_initialized = true ;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  std::cout << "Predicting with delta time " << delta_t << ", velocity " << velocity << " and yaw rate " << yaw_rate << std::endl ;

  std::random_device random_device;
  std::mt19937 generator(random_device());

  std::normal_distribution<double> x_distribution(0, std_pos[0]);
  std::normal_distribution<double> y_distribution(0, std_pos[1]);
  std::normal_distribution<double> theta_distribution(0, std_pos[2]);

  for(int index = 0 ; index < this->particles.size() ; ++index){

    double predicted_theta = this->particles[index].theta + (delta_t * yaw_rate);

    if(std::fabs(yaw_rate) > 0.00001) {

      double velocities_ratio = velocity / yaw_rate;

      this->particles[index].x +=
        velocities_ratio * (std::sin(predicted_theta) - std::sin(this->particles[index].theta)) + x_distribution(generator);

      this->particles[index].y +=
        velocities_ratio * (std::cos(this->particles[index].theta) - std::cos(predicted_theta)) + y_distribution(generator);

    } else {

      this->particles[index].x += delta_t * velocity * std::cos(this->particles[index].theta) + x_distribution(generator);
      this->particles[index].y += delta_t * velocity * std::sin(this->particles[index].theta) + y_distribution(generator);

    }

    this->particles[index].theta = predicted_theta + theta_distribution(generator);

  };

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  for (int particleIndex = 0; particleIndex < this->particles.size(); ++particleIndex) {

    std::vector<LandmarkObs> predictedObservations;

    // For each observation translate its coordinates so they are expressed as seen from particle in map coordinates
    for (auto observation : observations) {

      LandmarkObs landmarkPrediction;

      // Get magnitude and angle of landmark observation
      double magnitude = std::sqrt((observation.x * observation.x) + (observation.y * observation.y));
      double angle = this->particles[particleIndex].theta + std::atan2(observation.y, observation.x);

      landmarkPrediction.x = this->particles[particleIndex].x + (magnitude * std::cos(angle));
      landmarkPrediction.y = this->particles[particleIndex].y + (magnitude * std::sin(angle));

      predictedObservations.push_back(landmarkPrediction);

    }

    std::vector<Map::single_landmark_s> nearestLandmarks;

    // For each observation get nearest landmark
    for (auto observation : predictedObservations) {

      Map::single_landmark_s nearestLandmark = map_landmarks.landmark_list.front();

      double initial_x_distance = observation.x - nearestLandmark.x_f;
      double initial_y_distance = observation.y - nearestLandmark.y_f;

      double nearestDistance = std::sqrt(
        (initial_x_distance * initial_x_distance) + (initial_y_distance * initial_y_distance));

      for (auto landmark : map_landmarks.landmark_list) {

        double x_distance = observation.x - landmark.x_f;
        double y_distance = observation.y - landmark.y_f;
        double distance = std::sqrt((x_distance * x_distance) + (y_distance * y_distance));

        if (distance < nearestDistance) {
          nearestDistance = distance;
          nearestLandmark = landmark;
        }
      }

      nearestLandmarks.push_back(nearestLandmark);
    }

    double weight = 1.0;

    std::cout << "Calculating weight for particle " << particleIndex << std::endl;
    std::cout << "Observations size is " << observations.size() << std::endl;

    // Get new weight for the particle
    for (int observationIndex = 0; observationIndex < observations.size(); observationIndex++) {
      // Multiply weight by probability of observation given landmark position
      double scaling = 1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1]);

      double x_difference = predictedObservations[observationIndex].x - nearestLandmarks[observationIndex].x_f;
      double x_term = (x_difference * x_difference) / (2.0 * std_landmark[0] * std_landmark[0]);

      double y_difference = predictedObservations[observationIndex].y - nearestLandmarks[observationIndex].y_f;
      double y_term = (y_difference * y_difference) / (2.0 * std_landmark[1] * std_landmark[1]);

      weight *= scaling * std::exp(-(x_term + y_term));
    }

    std::cout << "Unscaled weight is " << weight << std::endl;

    this->particles[particleIndex].weight = weight;

  } // End of loop iterating over particles

  // Get total weights for rescaling
  double totalWeight = 0;

  for (int particleIndex = 0; particleIndex < this->particles.size(); ++particleIndex) {

    totalWeight += this->particles[particleIndex].weight;

  }

  std::cout << "TOTAL WEIGHT IS: " << totalWeight << std::endl;

  // Rescale weights
  for (int particleIndex = 0; particleIndex < this->particles.size(); ++particleIndex) {

    this->particles[particleIndex].weight /= totalWeight;

  }

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  std::random_device random_device;
  std::mt19937 generator(random_device());

  // Get weights
  std::vector<double> weights ;
  for(auto particle: this->particles) {
    weights.push_back(particle.weight) ;
  }

  std::discrete_distribution<> distribution(weights.begin(), weights.end());
  std::vector<Particle> resampledParticles ;

  for(int index = 0 ; index < this->particles.size() ; ++index)
  {

    int samplingIndex = distribution(generator) ;
    resampledParticles.push_back(this->particles[samplingIndex]) ;

  }

  this->particles = resampledParticles ;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
