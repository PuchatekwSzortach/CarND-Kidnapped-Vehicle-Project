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

  std::default_random_engine generator;
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

  std::default_random_engine generator;
  std::normal_distribution<double> x_distribution(0, std_pos[0]);
  std::normal_distribution<double> y_distribution(0, std_pos[1]);
  std::normal_distribution<double> theta_distribution(0, std_pos[2]);

  for(int index = 0 ; index < this->particles.size() ; ++index){

    Particle *particle = &(this->particles[index]) ;

    double predicted_theta = particle->theta + (delta_t * yaw_rate) ;
    double velocities_ratio = velocity / yaw_rate ;

    particle->x += velocities_ratio * (std::sin(predicted_theta) - std::sin(particle->theta)) + x_distribution(generator) ;
    particle->y += velocities_ratio * (std::cos(particle->theta) - std::cos(predicted_theta)) + y_distribution(generator) ;
    particle->theta = predicted_theta + theta_distribution(generator) ;

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

  std::cout << "Should be updating weights now" << std::endl ;

  std::default_random_engine generator;
  std::normal_distribution<double> landmark_x_distribution(0, std_landmark[0]);
  std::normal_distribution<double> landmark_y_distribution(0, std_landmark[1]);

  for(int particleIndex = 0 ; particleIndex < this->particles.size() ; ++particleIndex) {

    Particle *particle = &(this->particles[particleIndex]);

    // For each observation, translate its coordinates so they are expressed as seen from particle
    std::vector<LandmarkObs> predictedObservations;

    for (auto observation : observations) {
      LandmarkObs translatedLandmark;
      translatedLandmark.id = observation.id;

      // Get magnitude and angle of landmark observation
      double magnitude = std::sqrt((observation.x * observation.x) + (observation.y * observation.y));
      double angle = particle->theta + std::atan2(observation.y, observation.x);

      translatedLandmark.x = particle->x + (magnitude * std::cos(angle));
      translatedLandmark.y = particle->y + (magnitude * std::sin(angle));

      predictedObservations.push_back(translatedLandmark);

    }

    std::vector<Map::single_landmark_s> nearestLandmarks;

    // For each observation get nearest landmark
    for (auto observation : predictedObservations) {

      Map::single_landmark_s nearestLandmark = map_landmarks.landmark_list.front();
      double initial_x_distance = observation.x - nearestLandmark.x_f;
      double initial_y_distance = observation.y - nearestLandmark.y_f;
      double nearestDistance = std::sqrt((initial_x_distance * initial_x_distance) + (initial_y_distance * initial_y_distance));

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

    // Get new weight for the particle
    double weight = 1.0;

    for (int index = 0; particleIndex < observations.size(); particleIndex++)
    {
      // Multiply weight by probability of observation given landmark position
      double scaling = 1.0 / (M_PI * std_landmark[0] * std_landmark[1]) ;

      double x_difference = (predictedObservations[index].x - nearestLandmarks[index].x_f) ;
      double x_term = (x_difference * x_difference) / (2 * std_landmark[0] * std_landmark[0]) ;

      double y_difference = (predictedObservations[index].y - nearestLandmarks[index].y_f) ;
      double y_term = (y_difference * y_difference) / (2 * std_landmark[1] * std_landmark[1]) ;

      weight *= scaling * std::exp(-(x_term + y_term)) ;
    }

    particle->weight = weight ;

  }

  // Get total weights for rescaling
  double totalWeight = 0 ;

  for(int particleIndex = 0 ; particleIndex < this->particles.size() ; ++particleIndex) {

    totalWeight += this->particles[particleIndex].weight ;

  }

  // Rescale weights
  for(int particleIndex = 0 ; particleIndex < this->particles.size() ; ++particleIndex) {

    this->particles[particleIndex].weight /= totalWeight ;

  }

  std::cout << "But updates aren't done yet" << std::endl ;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

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
