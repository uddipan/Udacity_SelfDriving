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

namespace {
  
  constexpr int kNumParticles = 100;
  
  constexpr double kEps = 0.00001;
  
  default_random_engine gen;
  
} // namespace


void ParticleFilter::init(double x, double y, double theta, double variance[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  
  num_particles = kNumParticles;
  
  // define normal distribution for sensor noise
  normal_distribution<double> dist_X(0, variance[0]);
  normal_distribution<double> dist_Y(0, variance[1]);
  normal_distribution<double> dist_theta(0, variance[2]);
  
  // init particles
  for (int i = 0; i < num_particles; i++) {
    Particle p;
    // add noise
    p.x = x + dist_X(gen);
    p.y = y + dist_Y(gen);
    p.theta = theta + dist_theta(gen);
    p.weight = 1.0;
    particles.push_back(p);
  }
  
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/
  
  // define normal distribution for sensor noise
  normal_distribution<double> x_N(0, std_pos[0]);
  normal_distribution<double> y_N(0, std_pos[1]);
  normal_distribution<double> theta_N(0, std_pos[2]);
  
  for (int i=0; i < num_particles; i++) {
    if (fabs(yaw_rate) < kEps){
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    } else {
      double phi = particles[i].theta + yaw_rate * delta_t;
      particles[i].x += (velocity/yaw_rate) * ( sin(phi) - sin(particles[i].theta));
      particles[i].y += (velocity/yaw_rate) * ( cos(particles[i].theta) - cos(phi));
      particles[i].theta = phi;
    }
    
    // add noise
    particles[i].x += x_N(gen);
    particles[i].y += y_N(gen);
    particles[i].theta += theta_N(gen);
  }
}

// sets the map_id inside an observation to the nearest landmark id
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predict, std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.
  
  // Find nearest predicted landmark
  for (unsigned int i = 0; i < observations.size(); i++) {
    double dist_min = numeric_limits<double>::max();
    int idx = -1;
    
    for (unsigned int j = 0; j < predict.size(); j++) {
      double distance = dist(observations[i].x, observations[i].y,
                             predict[j].x, predict[j].y);
      
      if (distance < dist_min) {
        dist_min = distance;
        idx = predict[j].id;
      }
    }
    observations[i].id = idx;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
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
  
  for (int i=0; i < num_particles; i++) {
    // List of nearest landmarks to the particle
    vector<LandmarkObs> nearest_landmarks;
    
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      // get id and position coordinates for landmark
      float x = map_landmarks.landmark_list[j].x_f;
      float y = map_landmarks.landmark_list[j].y_f;
      int  id = map_landmarks.landmark_list[j].id_i;
      
      if (fabs(x - particles[i].x) <= sensor_range &&
          fabs(y - particles[i].y) <= sensor_range) {
        nearest_landmarks.push_back(LandmarkObs{id, x, y});
      }
    }
    
    //observations in particles co-ord system
    vector<LandmarkObs> transformed;
    double p_theta = particles[i].theta;
    for (unsigned int j = 0; j < observations.size(); j ++) {
      double t_x = cos(p_theta) * observations[j].x - sin(p_theta) * observations[j].y + particles[i].x;
      double t_y = sin(p_theta) * observations[j].x + cos(p_theta) * observations[j].y + particles[i].y;
      transformed.push_back(LandmarkObs{observations[j].id, t_x, t_y});
    }
    
    dataAssociation(nearest_landmarks, transformed);
    
    // reinitialize weight
    particles[i].weight = 1.0;
    for (unsigned int j = 0; j < transformed.size(); j++) {
      double lx,ly;
      
      int id = transformed[j].id;
      for (unsigned int k =0; k < nearest_landmarks.size(); k++) {
        if (nearest_landmarks[k].id == id) {
          lx = nearest_landmarks[k].x;
          ly = nearest_landmarks[k].y;
          break;
        }
      }
      
      // calculate weight for each particle
      double s_x = std_landmark[0];
      double s_y = std_landmark[1];
      // calculate normalization term
      double gauss_norm = (1/(2 * M_PI * s_x * s_y));
      double exponent = exp ( -(pow(lx - transformed[j].x, 2)/(2 * pow(s_x, 2)) +
                                (pow(ly - transformed[j].y, 2)/(2 * pow(s_y, 2)))));
      double obs_w = gauss_norm * exponent;
      particles[i].weight *= obs_w;
    }
    
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  
  vector<Particle> new_particles;
  
  // get all weights
  
  vector<double> weights;
  for (int i =0; i < num_particles; i ++) {
    weights.push_back(particles[i].weight);
  }
  
  // generate random starting index for resampling wheel
  uniform_int_distribution<int> uniintdist(0, num_particles-1);
  auto index = uniintdist(gen);
  
  // get max weight
  double max_weight = *max_element(weights.begin(), weights.end());
  
  // uniform random dist
  uniform_real_distribution<double> unirealdist(0.0, max_weight);
  
  double beta = 0.0;
  
  // spin resample wheel
  for (int i=0; i < num_particles; i++) {
    beta += unirealdist(gen) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }
  
  particles = new_particles;
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
