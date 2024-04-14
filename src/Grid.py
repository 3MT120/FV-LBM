import numpy as np
import matplotlib.pyplot as plt

class lattice():
    
    def __init__(self, Nx, Ny):
        self.Nx = Nx
        self.Ny = Ny
        self.grid = np.array([[gridpoint(i*Nx + j) for i in range(Ny)] for j in range(Nx)])
        
    def set_x_values(self, xvals):
        for i, point in enumerate(self.grid.flatten('F')):
            point.x = xvals[i]
            
    def set_y_values(self, yvals):
        for i, point in enumerate(self.grid.flatten('F')):
            point.y = yvals[i]
    
    def set_type(self, types):
        for i, point in enumerate(self.grid.flatten('F')):
            point.gridpoint_type = types[i]

    def set_neighbours(self):
        for i in range(self.Nx):
            for j in range(self.Ny):
                if self.grid[i,j].gridpoint_type != 2:
                    self.grid[i,j].neighbours[0] = self.grid[(i+1)%self.Nx,(j  )%self.Ny].index
                    self.grid[i,j].neighbours[1] = self.grid[(i+1)%self.Nx,(j+1)%self.Ny].index
                    self.grid[i,j].neighbours[2] = self.grid[(i  )%self.Nx,(j+1)%self.Ny].index
                    self.grid[i,j].neighbours[3] = self.grid[(i-1)%self.Nx,(j+1)%self.Ny].index
                    self.grid[i,j].neighbours[4] = self.grid[(i-1)%self.Nx,(j  )%self.Ny].index
                    self.grid[i,j].neighbours[5] = self.grid[(i-1)%self.Nx,(j-1)%self.Ny].index
                    self.grid[i,j].neighbours[6] = self.grid[(i  )%self.Nx,(j-1)%self.Ny].index
                    self.grid[i,j].neighbours[7] = self.grid[(i+1)%self.Nx,(j-1)%self.Ny].index

#    def set_subpoints(self):
#        for i, point in enumerate(self.grid.flatten('F')):
#            
        
    
    def grid_index(self):
        return np.array([obj.index for obj in self.grid.flatten('F')])
    
    def grid_x(self):
        return np.array([obj.x for obj in self.grid.flatten('F')])
    
    def grid_y(self):
        return np.array([obj.y for obj in self.grid.flatten('F')])

    def grid_type(self):
        return np.array([obj.gridpoint_type for obj in self.grid.flatten('F')])
    
    def show_grid(self):
        
        grid_types = self.grid_type()
        colours = np.empty_like(grid_types, str)
        colours[grid_types == 0] = 'k'
        colours[grid_types == 1] = 'b'
        colours[grid_types == 2] = 'r'
        
        plt.scatter(self.grid_x(), self.grid_y(), c = colours)
        plt.gca().set_aspect('equal')

        

class gridpoint():
    def __init__(self, index):
        self.index = index
        self.x = 0
        self.y = 0
        self.gridpoint_type = 0 # 0 = normal, 1 = periodic, 2 = no-slip
        self.neighbours = np.empty(8, int)
        self.subpoints = np.empty(8, "O")
        self.lengths = np.empty(12)
        self.normals = np.empty(12, "O")
    

        
    