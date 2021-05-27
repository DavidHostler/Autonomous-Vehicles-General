import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers  
from keras.models import model_from_json, load_model 
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf 
import random 
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
 
class H5(gym.Env):
    """
    Description:
       A car is navigating an environment to find reach its objective through a path of other cars.
    Source:
        
        
    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       X position                0                       1
        1       X velocity                0                       1
        2       Y posiion                 0                       1
        3       Y velocity                0                       1
       
    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push car to the right 
        1     Push car forward 
         
    Reward: Provided by paper by Ganesh, et. al plus a gradient based directional derivative
    corresponding to the target coord
        All observations are assigned a uniform random value in [0,...,1]
    Episode Termination:
        Distance from path is more than 1 m  
        Episode length is greater than 250.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """
 
    def __init__(self):
         
        self.max_speed = 8
        self.max_torque = 2.
        self.xforce_max = 1
        self.yforce_max= 1
        self.dt = .05 
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = 1  
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates 
        self.force = 0
 
        self.y_threshold = 3.1
        self.x_threshold = 2.4
     
        self.high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.y_threshold * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)
         
         
        
        self.newx = None
        self.newx_dot = None
        self.newy_dot = None
        self.newy = None 
        
        #self.action_space = spaces.Discrete(2)
        self.action_space = spaces.Box(-1, +1, (4,), dtype=np.float32)
        #2 actions space
        #self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
        
        
        self.observation_space = spaces.Box(-self.high, self.high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None
        
        #Coordinates
        self.x = None
        self.y = None
        self.x_dot = None
        self.y_dot = None
        
        
        #Target 
        self.x_target = -1#0#6
        self.y_target = 10#1#8
        
        #Path along which the car does move
        
        x_arr = []
        y_arr = []
        

        path = np.linspace(0,5,10)
        
        for i in range(len(path)): 
            x_arr.append(path[i])
            #y_arr.append(path[i]**2) # y = x**2
            
            y_arr.append(path[i]*3) #y = 3x
            
        
        self.x_arr = x_arr
        self.y_arr = y_arr
        
        
        
        #Tangent to the path 
        
            
        
        
        #Traffic 
        
        x_traffic = []
        y_traffic = []
        for i in range( 100 ):
            x_traffic.append(random.uniform(0,10))
            y_traffic.append(random.uniform(0,10))
        
        self.x_traffic = []#x_traffic
        self.y_traffic = []#y_traffic
        
        
        self.nearest_obstacle = None
        self.nearest_path = None
        
        
        
    def y_path(self, x_path):
            return 3*x_path#x_path**2
        
    def tangent(self, x):
            dx = 0.001
            dy = self.y_path(x + dx) - self.y_path(x)
            return dy/dx
        
        
    def nearest_obstacle_distance(self):
        x_traffic = self.x_traffic
        y_traffic = self.y_traffic
        x_agent_numpy =  self.x + np.zeros(len(x_traffic))
        y_agent_numpy = self.y + np.zeros(len(y_traffic))
        x_dis = x_agent_numpy - np.array(x_traffic) 
        y_dis = y_agent_numpy - np.array(y_traffic) 
        distance = np.sqrt(x_dis**2 + y_dis**2)
        counter = 0
        distance_arr = []
        counter = 0
        
        x_traffic.append(self.x_target)
        y_traffic.append(self.y_target)
        
        for i in range(len(x_traffic)):
                distance_arr.append(np.sqrt((self.x - x_traffic[i])**2 + (self.y - y_traffic[i])**2))
                while (distance_arr[counter] != min(distance_arr)):
                    counter += 1
        self.nearest_obstacle = [self.x_traffic[counter], self.y_traffic[counter]]
        return min(distance_arr)  
    
    
    def nearest_path_dist(self):
        x_traffic = self.x_arr
        y_traffic = self.y_arr
        x_agent_numpy =  self.x + np.zeros(len(x_traffic))
        y_agent_numpy = self.y + np.zeros(len(y_traffic))
        x_dis = x_agent_numpy - np.array(x_traffic) 
        y_dis = y_agent_numpy - np.array(y_traffic) 
        distance = np.sqrt(x_dis**2 + y_dis**2)
        counter = 0
        distance_arr = []
        counter = 0
        
        x_traffic.append(self.x_target)
        y_traffic.append(self.y_target)
        
        for i in range(len(x_traffic)):
                distance_arr.append(np.sqrt((self.x - x_traffic[i])**2 + (self.y - y_traffic[i])**2))
                while (distance_arr[counter] != min(distance_arr)):
                    counter += 1
        self.nearest_path = [self.x_arr[counter], self.y_arr[counter]]
        return min(distance_arr)
        
    def dot(self,A,B):
        C = 0
        for i in range(len(A)): 
            C += A[i]*B[i]
        return C
    
    def cross(self, A,B):
        C_z = A[0]*B[1] - A[1]*B[0]
        return [0,0,C_z] 
    
    def grad_V(self, x, y):
        return 1/(x**2 + y**2)
    
    
    def path(self, x):
        return x  
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    
    
    def step(self, action):
        x, x_dot, y, y_dot   = self.state  # x:= x
        
         
        self.x = x
        self.x_dot = x_dot
        self.y = y
        self.y_dot = y_dot
        dt = self.dt
        
        '''
        
        action_x = np.clip(action[0], -self.xforce_max, self.xforce_max)[0]
        action_y = np.clip(action[0], -self.yforce_max, self.yforce_max)[1]
        
        
        '''
        
        
        action_x = np.clip(action[0], 0, self.xforce_max)[0]
        action_y = np.clip(action[0], 0, self.yforce_max)[1]
        
        #self.last_action = action  # for rendering
        
        #costs =  angle_normalize(x) ** 2 + .1 * x_dot ** 2 + .001 * (action_x ** 2)
        velocity = [x_dot, y_dot]
        vel_mag = np.sqrt(x_dot**2 + y_dot**2) 
        #Normalize the directional tangent vector
        
        #Path endpoints 
        x_arr = self.x_arr
        y_arr = self.y_arr
        x_f = x_arr[len(x_arr)-1]
        y_f = y_arr[len(y_arr)-1]
        
        r_path = [x_f - self.x, y_f - self.y ]/np.sqrt((x_f - self.x)**2 + (y_f - self.y)**2)
                
        #Potential reward value
        
        
        costs = abs(self.dot(velocity, [1,self.tangent( self.x )])/np.sqrt(1 + self.tangent( self.x )**2))  - \
        abs(self.cross(velocity,  [1,self.tangent( self.x )])[2]/np.sqrt(1 + self.tangent( self.x )**2)) - \
        abs(vel_mag * self.nearest_path_dist()) + \
        11*self.grad_V(self.x, self.y)* self.dot(r_path, velocity)
        #-(self.dot(velocity, [-1,10]) - self.cross(velocity, [-1,10])[2])/np.sqrt(11) #+ .001 * (action_x ** 2 + action_y)
         
        
        newx_dot = x_dot + action_x/self.total_mass * self.tau  
        newx= x+ newx_dot * self.tau
        #newx_dot = np.clip(newx_dot, -self.max_speed, self.max_speed)
        newx_dot = np.clip(newx_dot, 0, self.max_speed)
        
        
        
        
        newy_dot = y_dot + action_y/self.total_mass * self.tau  
        newy= y+ newy_dot * self.tau
        #newy_dot = np.clip(newy_dot, -self.max_speed, self.max_speed)
        newy_dot = np.clip(newy_dot, 0, self.max_speed)
        
        
        #Kowalski, Analysis
        
        self.newx = newx
        self.newx_dot = newx_dot
        self.newy_dot = newy_dot
        self.newy = newy
        
        
        #self.state = np.array([ newx, newx_dot, newy, newy_dot])
        self.state = np.array([x, x_dot, y, y_dot])
        
        
        
        #To save coord
        self.x = newx
        self.x_dot = newx_dot
        self.y = newy
        self.y_dot = newy_dot
        
        
        
        
        done = bool(
        #0.5 tends to work the best so far!
        self.nearest_path_dist() > 0.5
        )
        
        '''
        if self.nearest_obstacle != [self.x_target,self.y_target]:
            if self.nearest_obstacle_distance() < 0.01:
                done = True
            else:
                done = False
        else:
            done = True
    
        ''' 
         
         
        
        
        return self.state, -costs, done, {}

    
    def reset(self):
        '''
        x  = self.np_random.uniform(low=-self.x_threshold, high=self.x_threshold)
        y = self.np_random.uniform(low=-self.y_threshold, high=self.y_threshold)
        x_dot = self.np_random.uniform(low=-3, high=3)
        y_dot = self.np_random.uniform(low=-3, high=3)
        ''' 
        
        x  = self.np_random.uniform(low=0, high=0.1)
        y = self.np_random.uniform(low=0, high=0.1)
        x_dot = self.np_random.uniform(low=0, high=.3)
        y_dot = self.np_random.uniform(low=0, high=.3)
        
        
        self.last_action = None
        
        self.state = x, x_dot, y, y_dot
        #self.state = 0,0,0,0
        return self.state

    
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
 

 
