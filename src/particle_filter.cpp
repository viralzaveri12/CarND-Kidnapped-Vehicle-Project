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
	
	num_particles = 100;
	default_random_engine gen;

	normal_distribution<double> dist_x(x,std[0]);
	normal_distribution<double> dist_y(y,std[1]);
	normal_distribution<double> dist_theta(theta,std[2]);

	for(int i=0;i<num_particles;++i){
		Particle single_particle;
		single_particle.id = i;
		single_particle.x = dist_x(gen);
		single_particle.y = dist_y(gen);
		single_particle.theta = dist_theta(gen);
		single_particle.weight = 1.0;
	
		particles.push_back(single_particle);
		}
		
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;
	normal_distribution<double> dist_x(0,std_pos[0]);
	normal_distribution<double> dist_y(0,std_pos[1]);
	normal_distribution<double> dist_theta(0,std_pos[2]);

	for(int i=0; i<num_particles; ++i){
		if(fabs(yaw_rate) > 0.0001){
		particles[i].x = particles[i].x + (velocity / yaw_rate) * (sin(particles[i].theta + (yaw_rate*delta_t)) - sin(particles[i].theta));
		particles[i].y = particles[i].y + (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate*delta_t)));
		particles[i].theta = particles[i].theta + yaw_rate * delta_t;
		}
		else{
			particles[i].x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
			particles[i].y = particles[i].y + velocity * delta_t * sin(particles[i].theta);
			//particles[i].theta = particles[i].theta + yaw_rate * delta_t; #Theta depends on yaw_rate and yaw_rate for this case is constant so theta will be constant.
		}

		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for(int i =0;i<observations.size();++i){
		
		int map_id = -1;
		LandmarkObs obs = observations[i];
		//http://www.cplusplus.com/reference/limits/numeric_limits/
		double min_distance = numeric_limits<double>::max();
		
		for(int j=0;j<predicted.size();++j){
			LandmarkObs pred = predicted[j];
			double dst = dist(obs.x,obs.y,pred.x,pred.y);

			if(dst < min_distance){
				min_distance = dst;
				map_id = pred.id;
			}
		}
		observations[i].id = map_id;
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

	// Ultimately we want to update weights of particles. So we perform the following steps:
	// Trasformation, Association, and Weight Update
	// Just to understand we are going to call dataAssociation function from here so we will
	// need predicted and observations vectors(Two vectors which need to pass to dataAssociation function)
	// We will get 'predicted' vector from map landmarks using sensor_range for all the particles.
	// We will get observations by looping over all the observation points and tranforming them to
	// map coordinates. 

	for (int i=0;i<num_particles;++i){
		double x_part = particles[i].x;
		double y_part = particles[i].y;
		double theta_part = particles[i].theta;
		//Vector to hold all the map predicted landmarks
		std::vector<LandmarkObs> prediction;

		//landmark_list is a vector holding landmark_id, x and y coordinates in map. We need to store those landmarks 
		//which are near to this particle and within the sensor range.
		for(int j=0;j<map_landmarks.landmark_list.size();++j){
			//Def in map.h
			double x_landmark = map_landmarks.landmark_list[j].x_f;
			double y_landmark = map_landmarks.landmark_list[j].y_f;
			int id_landmark = map_landmarks.landmark_list[j].id_i;

			//We don't want to consider all those landmarks which are out of sensor_range(Sensor is not going to measure those)
			//So we are checking conditions and append the correct ones to prediction vector.
			if(dist(x_landmark,y_landmark,x_part,y_part) <= sensor_range){

				prediction.push_back(LandmarkObs{id_landmark,x_landmark,y_landmark});
			}
		}

		std::vector<LandmarkObs> transformed_observations;
		for(int k = 0;k<observations.size();++k){
			double x_map = x_part + cos(theta_part) * observations[k].x - sin(theta_part) * observations[k].y;
			double y_map = y_part + sin(theta_part) * observations[k].x + cos(theta_part) * observations[k].y;

			transformed_observations.push_back(LandmarkObs{observations[k].id,x_map,y_map});
		}
		// Till above point transformation (first part) is completed.
		// Below is second part Association.

		dataAssociation(prediction,transformed_observations);


		//Third part. Update weight
		//Now in the equation x and y are the observations in map coordinates and mu_x and mu_y are the
		//coordinates of nearest landmark. We have to look for nearest/correct prediction from all for current
		//transformed measurement.

		//Re-initialize the weight. 
		particles[i].weight = 1.0;
		for(int a = 0;a<transformed_observations.size();++a){
			double obs_x,obs_y,nearest_x,nearest_y;
			obs_x = transformed_observations[a].x;
			obs_y = transformed_observations[a].y;
			
			for(int b = 0;b<prediction.size();++b){
				if(transformed_observations[a].id == prediction[b].id){
					//****MISTAKE*** Do not declare here.//
					//double nearest_x = prediction[b].x;
					//double nearest_y = prediction[b].y;
					nearest_x = prediction[b].x;
					nearest_y = prediction[b].y;
				}
			}

	    double noise_x = std_landmark[0];
	    double noise_y = std_landmark[1];
	 
        // double obs_weight = ( 1/(2*M_PI*std_landmark[0]*std_landmark[1])) * exp( -( pow(nearest_x-transformed_observations[a].x,2)/(2*pow(std_landmark[0], 2)) + (pow(nearest_y-transformed_observations[a].y,2)/(2*pow(std_landmark[1], 2))) ) );
        // double obs_weight = ( 1/(2*M_PI*std_landmark[0]*std_landmark[1])) * exp( -( pow(nearest_x-ob_x,2)/(2*pow(std_landmark[0], 2)) + (pow(nearest_y-ob_y,2)/(2*pow(std_landmark[1], 2))) ) );
	  
        //****FIRST COMMENTED LINE GIVES RUN OUT TIME ERROR WHILE SECOND LINE TAKES AROUND 96.44 SYSTEM TIME
	  
	    double obs_weight = ( 1/(2*M_PI*noise_x*noise_y)) * exp( -( pow(nearest_x-obs_x,2)/(2*pow(noise_x, 2)) + (pow(nearest_y-obs_y,2)/(2*pow(noise_y, 2))) ) );
	  
	    particles[i].weight *= obs_weight;
		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::vector<Particle> resampled_particles;
	std::vector<double> weight;
	for(int i=0;i<num_particles;++i){
		weight.push_back(particles[i].weight);
	}	
	// http://www.cplusplus.com/reference/random/uniform_int_distribution/
	//Random init index
	std::default_random_engine generator;
	std::uniform_int_distribution<int> distribution(0,num_particles-1);
	int index = distribution(generator);

	double beta = 0.0;
	double weight_max = *std::max_element(weight.begin(),weight.end());
	std::uniform_real_distribution<double> real_distribution(0.0,2*weight_max);
	for (int i = 0; i < num_particles; ++i)
	{
		beta = beta + real_distribution(generator);

		while(weight[index] < beta){
			beta = beta - weight[index];
			index = (index + 1)%num_particles;
		}
		resampled_particles.push_back(particles[index]);
	}

	particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
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
