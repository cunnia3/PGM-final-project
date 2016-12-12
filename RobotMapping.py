import numpy as np
from scipy import ndimage
from scipy import misc
from RobotLocalization import *
from PIL import Image

class Map:
    def __init__(self, x_dim, y_dim):
        self.occupancy_grid = np.zeros([x_dim,y_dim])
        
    def load_from_file(self, filename):
        self.occupancy_grid = misc.imread(filename)
        self.occupancy_grid = self.occupancy_grid[:,:,0]
        
    def show_image(self):
        img = Image.fromarray(self.occupancy_grid)
        misc.imsave('learned_map.jpg', self.occupancy_grid)
        img.show()
                
class ObservationModel:
    def __init__(self):
        return
        
    def forward_sense_reality(self, true_state):
        """ Provide a measurement given a true map and the robot's true state """
        return
        
    def inverse_sense(self, map_info, state, mesurement):
        """ Provide map info given state and a measurement """
        return
    
class DiscreteLaserRangeFinderObservationModel:
    def __init__(self, true_map, noise_scale):
        """ Image-based discrete range finder"""
        self.true_map = true_map
        self.noise_scale = noise_scale
        
    def forward_sense_reality(self, input_true_state):
        """ @true_state is a np array 2x1"""
        
        sensor_readings = np.zeros([3,3])  
        true_state = input_true_state.reshape([2,1])
        
        # SENSE UPWARD
        # sensor readings is a 3x3 array of readings, the north array corresponds to north
        # reading, middle entry does nothing
        for offset_i in range(3):
            for offset_j in range(3):
                offset_column = offset_j - 1
                offset_row = offset_i - 1
                
                if offset_column == 0 and offset_row == 0:
                    continue
            
                
                try:
                    dist_i = 1
                    while 1:
                        if self.true_map.occupancy_grid[dist_i*offset_row + true_state[0,0]][dist_i*offset_column + true_state[1,0]]<2:
                            break
                            
                        else:
                            dist_i +=1
                except:
                    print 'OUT OF BOUNDS ERROR'
                    
                sensor_readings[offset_row+1, offset_column+1] = dist_i
                
        return sensor_readings
        
    def inverse_sense(self, true_state, measurement):
        """ Provide local map given true state and a measurement 
        return list of (position, p_filled) tuples
        @true_state np_array"""
        local_map = []
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue
                
                offset_column = 1-i
                offset_row = j-1
                offset = np.array([-offset_column,offset_row])
                #print i, j, offset
                
                for dist_i in range(int(measurement[i, j])):
                    position = offset*dist_i + true_state.reshape(2,)
                    local_map.append( (position, .1) )
                    
                local_map.append((offset*measurement[i, j] + true_state.reshape(2,), .9))
                
        return local_map
            
            
class OccupancyGridMapper:
    def __init__(self, x_dim, y_dim):
        self.my_map = Map(x_dim, y_dim)
        self.map_log_odds = np.zeros([x_dim, y_dim])
        
    def learn_map(self, observations):
        for observation in observations:
            map_index = observation[0]
            c = int(map_index[1])
            r = int(map_index[0])
            likelihood = observation[1]
            
            self.map_log_odds[r,c] += math.log(likelihood/(1-likelihood))            
        
        # convert likelihood map into real map
        for i in range(self.map_log_odds.shape[0]):
            for j in range(self.map_log_odds.shape[1]):
                self.my_map.occupancy_grid[i,j] = 255-(1 - 1/(1+np.exp(self.map_log_odds[i,j])))*255
            
        print self.my_map.show_image()
        return