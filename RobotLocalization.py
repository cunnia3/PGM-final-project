import numpy as np
import random
import math
import copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm

def sample_normal_dist(b):
    """ Normal distribution specified by motion model slides """
    sum = 0
    for i in range(12):
        sum+=.5 * random.uniform(-b, b)
    return sum


class Particle:
    """ Abstract representation of state """
    def __init__(self):
        return
        
    def transition(self, action):
        return

class GridParticle(Particle):
    def __init__(self,true_map, x=0, y=0, ):
        self.position = np.array([x,y])
        self.position=self.position.reshape([2,1])
        self.true_map = true_map
        
    def transition(self, action):
        """ @action, numpy array [i,j] where the i and j can be -1,0,1
        movement will be blocked by the map if there is something in the way"""
        clean_action = action.reshape([2,1])
        potential_new_state = self.position + clean_action        
        
        try:
            if self.true_map.occupancy_grid[potential_new_state[0], potential_new_state[1]] > 127:
                self.position = copy.deepcopy(potential_new_state)
                return            
        except:
            # NO CHANGE, WE WENT OUT OF BOUNDS!
            return
        
        
class OdometryParticle(Particle):
    """ STATE FORMAT *AND* TRANSITION MODEL
    of odometry particle.  See:
    http://ais.informatik.uni-freiburg.de/teaching/ss11/robotics/slides/06-motion-models.pdf"""
    def __init__(self, x=0, y=0, theta=0, alpha_params=[.5, .5, .5, .5]):
        """ @x x position of robot
            @y y position of robot
            @theta angle of robot
            @alpha list of parameters for gaussian normal noise model
            """
        self.alpha = alpha_params
        self.x = x
        self.y = y
        self.theta = theta
        
    def transition(self, action):
        """ move particle according to motion model with appropriate noise
        @action = [drot1, drot2, dtrans]"""
        drot1 = action[0]
        drot2 = action[1]
        dtrans = action[2]
        drot1_h = drot1 + sample_normal_dist(self.alpha[0]* abs(drot1) + self.alpha[1]*dtrans)
        dtrans_h = dtrans + sample_normal_dist(self.alpha[2] * dtrans + self.alpha[3]*(abs(drot1) + abs(drot2)))
        drot2_h = drot2 + sample_normal_dist(self.alpha[0]*abs(drot2) + self.alpha[1]*dtrans)    
        
        self.x = self.x + dtrans_h*math.cos(self.theta + drot1_h)
        self.y = self.y + dtrans_h*math.sin(self.theta + drot1_h)
        self.theta = self.theta + drot1_h + drot2_h
        return  

        
class ParticleFilter():
    """ Particle filter implementation.  Contains list of particles and a transition model"""
    def __init__(self, starting_particle, num_particles, starting_state_dist = 'None'):
        self.particle_list = []
        self.score_list = []
        self.do_resample = False
        if starting_state_dist == 'None':  
            for i in range(num_particles):
                self.particle_list.append(copy.deepcopy(starting_particle))
             
    def update_filter(self, action,scoring_func=[], measurement=[]):
        """ Update the particle filter with an action and a measurement"""
        self._update_particles(action)
        self._measurement_score(scoring_func, measurement)
        
        if self.do_resample:
            self._resample()

             
    def _update_particles(self, action):
        """ Update the particles according to actions particles get new position """
        for particle in self.particle_list:
            particle.transition(action)
            
    def _measurement_score(self, scoring_func=[], measurement=[]):
        """  Use external measurements to assign scores to particles
        @scoring_func: the function that determines the fitness of a particle
                        SPECIAL VALUES -100: bad particle RESAMPLE THIS ONE"""

        self.score_list = [1.0/(1.0*len(self.particle_list))]*len(self.particle_list)
        
        if scoring_func == []:
            self.do_resample = False

        # If there is a scoring function, use it
        else:
            self.do_resample = False
            for i in range(len(self.particle_list)):
                score = scoring_func(self.particle_list[i],measurement)
                if score >= 0: # only resample if we get additional info
                    self.do_resample = True
                    self.score_list[i]=score
                
        # Normalize
        self.score_list = np.asarray(self.score_list)
        self.score_list = self.score_list/np.sum(self.score_list)
        return
        
    def _resample(self):
        new_particles_list = []
        
        for i in range(len(self.particle_list)):
            rand = np.random.rand()
            if_lower_than_this = 0
            for j in range(len(self.particle_list)):
                if_lower_than_this+=self.score_list[j]
                
                if rand < if_lower_than_this:
                    new_particles_list.append(copy.deepcopy(self.particle_list[j]))
                    break
                
        self.particle_list = new_particles_list
        return
       
class Sensor():
    def __init__(self):
        return
        
class RangeSensor(Sensor):
    def __init__(self, x, y, circleR):
        self.circleX = x
        self.circleY = y
        self.circleR = circleR
        
    def score_particle(self, particle, measurement):
        """@particle: robot pose particle
           @measurement: true robot distance"""
        particle_dist = math.sqrt((self.circleX - particle.x)**2 + (self.circleY - particle.y)**2)
        if measurement < self.circleR:
            return norm.pdf(measurement - particle_dist, scale=.05)
        else:
             return -1 # we have no info
