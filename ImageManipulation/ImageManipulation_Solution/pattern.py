import numpy as np;
import matplotlib.pyplot as plt

class Checker:
    
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = np.zeros((self.resolution,self.resolution))
        self.num_sq_X = self.resolution//self.tile_size
        self.num_sq_Y = self.resolution//self.tile_size

    def draw(self):
        white = np.ones((self.tile_size,self.tile_size))
        black = np.zeros((self.tile_size,self.tile_size))
        
        temp = np.concatenate((black,white),1)
        
        row = np.tile(temp,(self.num_sq_X-2))
        inv_row = 1-row
        
        temp = np.concatenate((row,inv_row),0)
        
        check_board = np.tile(temp,(self.num_sq_Y-2,1))        
        x,y=self.output.shape
        self.output=check_board[:x,:y]
        return np.copy(self.output)
        

    def show(self):
        plt.axis([0, self.resolution-1, 0, self.resolution-1])
        plt.imshow(self.output,cmap='gray')
        plt.show()
     


class Spectrum:
    
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = np.zeros((self.resolution,self.resolution,3))
    
    def draw(self):
        self.output[:,:,0] = np.tile(np.linspace(0,255,self.resolution),(self.resolution,1)) /255
        self.output[:,:,1] = np.tile(np.linspace(0,255,self.resolution).reshape((-1, 1)), (1,self.resolution))/255
        self.output[:,:,2] = np.tile(np.linspace(255,0,self.resolution),(self.resolution,1)) /255
        
        return np.copy(self.output)
        

    def show(self):
        plt.imshow(self.output)
        plt.show()

class Circle:
    
     def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = np.zeros((self.resolution,self.resolution))
        
     def draw(self):
        xx,yy = np.mgrid[:self.resolution,:self.resolution]  
        temp = (xx-self.position[0])**2 + (yy-self.position[1])**2        
        self.output = np.where(temp>(self.radius**2),0,1)
        return np.copy(self.output).astype('bool')

        
     def show(self):
        plt.imshow(self.output,cmap='gray')
        plt.show()
