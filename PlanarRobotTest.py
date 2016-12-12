from KalmanLocalization import *
import matplotlib.pyplot as plt
from matplotlib import animation
import copy

A = np.eye(2) # process evolution
B = np.eye(2) # control matrix
P = np.eye(2)*.1 # initial cov in state 
Q = np.eye(2)*.1 # process error cov
starting_pos = np.zeros([2,1]) # initial state

## DUMMY SENSING MODELS, WILL BE REPLACED BY FEATURES
H = np.eye(2) # sensor model
R = np.eye(2)*10 # sensor noise

kfilter = KalmanFilterLinear(A, B, H, starting_pos, P, Q, R)
feature1 = PlanarFeature(np.array([50,50]), R)
feature_list = [feature1]
my_robot = PlanarRobot(copy.deepcopy(starting_pos), kfilter, feature_list, Q, 10)

plt.figure
for i in range(100):
    my_robot.command_robot(np.ones([2,1]))
    print my_robot.kfilter.current_state_estimate 
    plt.scatter(my_robot.kfilter.current_state_estimate[0], my_robot.kfilter.current_state_estimate[1], c='r')
    plt.scatter(my_robot.true_position[0], my_robot.true_position[1], c='b')
    
animator = RobotAnimator(my_robot)
anim = animation.FuncAnimation(animator.fig, animator.animate,
                               init_func = animator.init_animation,
                               frames = 100,
                               interval=50,
                               blit=True)
          
plt.title('Simple Planar Robot Localization with Landmark') 
plt.legend(loc = 4)
plt.ylabel('Y position (m)')
plt.xlabel('X position (m)')  
plt.show()
anim.save('simple_localization.mp4', fps=20, writer="avconv", codec="libx264")