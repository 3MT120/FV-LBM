import numpy as np
import matplotlib.pyplot as plt

def normalize_rotate_vector(v):
    return np.array([v[1], -v[0]])/np.linalg.norm(v)

def calculate_area(a, b, c, d):
    x = [a[0], b[0], c[0], d[0]]
    y = [a[1], b[1], c[1], d[1]]
    
    area = 0.0
    for i in range(4):
        j = (i + 1) % 4
        area += x[i] * y[j] - x[j] * y[i]
    return abs(area) / 2.0


class lattice():
    
    def __init__(self, Nx, Ny):
        self.Nx = Nx
        self.Ny = Ny
        self.grid = np.array([[gridpoint(i*Nx + j) for i in range(Ny)] for j in range(Nx)])
        self.fi = np.zeros([Nx, Ny, 9])
        
        self.ci = np.array([[0,0], [1,0], [0,1], [-1,0], [0,-1], [1,1], [-1,1], [-1,-1], [1, -1]])
        self.ai = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
        self.wi = np.array([4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0])

        
    def set_x_values(self, xvals):
        for i, point in enumerate(self.grid.flatten('F')):
            point.x = xvals[i]

        self.Lx = max(xvals) - min(xvals)
            
    def set_y_values(self, yvals):
        for i, point in enumerate(self.grid.flatten('F')):
            point.y = yvals[i]
            
        self.Ly = max(yvals) - min(yvals)
    
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

    def set_subpoints(self):
        
        c = np.stack((self.grid_x(), self.grid_y()), -1)
        
        for i, point in enumerate(self.grid.flatten('F')):
            if point.gridpoint_type == 0:
                neighbours = point.neighbours
                
                point.subpoints[0] = (c[i] + c[neighbours[0]])/2
                point.subpoints[1] = (c[i] + c[neighbours[0]] + c[neighbours[1]] + c[neighbours[2]])/4
                point.subpoints[2] = (c[i] + c[neighbours[2]])/2
                point.subpoints[3] = (c[i] + c[neighbours[2]] + c[neighbours[3]] + c[neighbours[4]])/4
                point.subpoints[4] = (c[i] + c[neighbours[4]])/2
                point.subpoints[5] = (c[i] + c[neighbours[4]] + c[neighbours[5]] + c[neighbours[6]])/4
                point.subpoints[6] = (c[i] + c[neighbours[6]])/2
                point.subpoints[7] = (c[i] + c[neighbours[6]] + c[neighbours[7]] + c[neighbours[0]])/4
                
            #elif point.gridpoint_type == 1:
            #    neighbours = point.neighbours
            #    
            #    point.subpoints[0] = (c[i] + c[neighbours[0]])/2
            #    point.subpoints[1] = (c[i] + c[neighbours[0]] + c[neighbours[1]] + c[neighbours[2]])/4
            #    point.subpoints[2] = (c[i] + c[neighbours[2]])/2
            #    point.subpoints[3] = (c[i] + c[neighbours[2]] + c[neighbours[3]] + c[neighbours[4]])/4
            #    point.subpoints[4] = (c[i] + c[neighbours[4]])/2
            #    point.subpoints[5] = (c[i] + c[neighbours[4]] + c[neighbours[5]] + c[neighbours[6]])/4
            #    point.subpoints[6] = (c[i] + c[neighbours[6]])/2
            #    point.subpoints[7] = (c[i] + c[neighbours[6]] + c[neighbours[7]] + c[neighbours[0]])/4
            #    
            #    for i in range(8):
            #        if point.subpoints[i][0] > self.Lx/2:
            #            point.subpoints[i][0] -= self.Lx/2
            #        if point.subpoints[i][1] > self.Ly/2:
            #            point.subpoints[i][1] -= self.Ly/2
    
    def set_lengths(self):
        for i, point in enumerate(self.grid.flatten('F')):
            
            if point.gridpoint_type == 0:
                c = np.array([point.x, point.y])
                
                point.lengths[0] = np.sqrt(sum(np.square(point.subpoints[1] - point.subpoints[0])))
                point.lengths[1] = np.sqrt(sum(np.square(point.subpoints[2] - point.subpoints[1])))
                point.lengths[2] = np.sqrt(sum(np.square(point.subpoints[3] - point.subpoints[2])))
                point.lengths[3] = np.sqrt(sum(np.square(point.subpoints[4] - point.subpoints[3])))
                point.lengths[4] = np.sqrt(sum(np.square(point.subpoints[5] - point.subpoints[4])))
                point.lengths[5] = np.sqrt(sum(np.square(point.subpoints[6] - point.subpoints[5])))
                point.lengths[6] = np.sqrt(sum(np.square(point.subpoints[7] - point.subpoints[6])))
                point.lengths[7] = np.sqrt(sum(np.square(point.subpoints[0] - point.subpoints[7])))
                
                point.lengths[8] = np.sqrt(sum(np.square(point.subpoints[0] - c)))
                point.lengths[9] = np.sqrt(sum(np.square(point.subpoints[2] - c)))
                point.lengths[10] = np.sqrt(sum(np.square(point.subpoints[4] - c)))
                point.lengths[11] = np.sqrt(sum(np.square(point.subpoints[6] - c)))
                                                                                                                                                                                                                                      
    def set_normals(self):
        for i, point in enumerate(self.grid.flatten('F')):
            
            if point.gridpoint_type == 0:
                c = np.array([point.x, point.y])
                
                point.normals[0] = normalize_rotate_vector(point.subpoints[1] - point.subpoints[0])
                point.normals[1] = normalize_rotate_vector(point.subpoints[2] - point.subpoints[1])
                point.normals[2] = normalize_rotate_vector(point.subpoints[3] - point.subpoints[2])
                point.normals[3] = normalize_rotate_vector(point.subpoints[4] - point.subpoints[3])
                point.normals[4] = normalize_rotate_vector(point.subpoints[5] - point.subpoints[4])
                point.normals[5] = normalize_rotate_vector(point.subpoints[6] - point.subpoints[5])
                point.normals[6] = normalize_rotate_vector(point.subpoints[7] - point.subpoints[6])
                point.normals[7] = normalize_rotate_vector(point.subpoints[0] - point.subpoints[7])
                
                point.normals[8] = normalize_rotate_vector(point.subpoints[0] - c)
                point.normals[9] = normalize_rotate_vector(point.subpoints[2] - c)
                point.normals[10] = normalize_rotate_vector(point.subpoints[4] - c)
                point.normals[11] = normalize_rotate_vector(point.subpoints[6] - c)
                
    def set_areas(self):
        for i, point in enumerate(self.grid.flatten('F')):
            
            if point.gridpoint_type == 0:
                c = np.array([point.x, point.y])
                
                point.areas[0] = calculate_area(c, point.subpoints[0], point.subpoints[1], point.subpoints[2])
                point.areas[1] = calculate_area(c, point.subpoints[2], point.subpoints[3], point.subpoints[4])
                point.areas[2] = calculate_area(c, point.subpoints[4], point.subpoints[5], point.subpoints[6])
                point.areas[3] = calculate_area(c, point.subpoints[6], point.subpoints[7], point.subpoints[0])

        
    
    def grid_index(self):
        return np.array([obj.index for obj in self.grid.flatten('F')])
    
    def grid_x(self):
        return np.array([obj.x for obj in self.grid.flatten('F')])
    
    def grid_y(self):
        return np.array([obj.y for obj in self.grid.flatten('F')])

    def grid_type(self):
        return np.array([obj.gridpoint_type for obj in self.grid.flatten('F')])
    
    def show_grid(self, with_dist = False):
        
        grid_types = self.grid_type()
        colours = np.empty_like(grid_types, str)
        colours[grid_types == 0] = 'k'
        colours[grid_types == 1] = 'b'
        colours[grid_types == 2] = 'r'
            
        plt.scatter(self.grid_x(), self.grid_y(), c = colours)
        plt.gca().set_aspect('equal')
        
        if with_dist:
            for q in range(9):
                plt.quiver(self.grid_x(), self.grid_y(), self.fi[:,:,q].flatten('F') * self.ci[q, 0], self.fi[:,:,q].flatten('F') * self.ci[q, 1], scale=1, scale_units='inches')

        

    def calculate_equilibrium(self):
        f_eq = np.zeros([self.Nx, self.Ny, 9])
        
        for i in range(self.Nx):
            for j in range(self.Ny):
                if self.grid[i, j].gridpoint_type == 0:
                    
                    for q in range(9):
                        vu = self.ci[q][0] * self.grid[i, j].vx + self.ci[q][1] * self.grid[i, j].vy
                        vu2 = vu**2
                        uu = (self.grid[i, j].vx)**2 + (self.grid[i, j].vy)**2
                        self.grid[i, j].f[q] = self.wi[q] * self.grid[i, j].rho * (1 + 3.0/2.0 * vu + 9.0/2.0 * vu2 - 3.0/2.0 * uu)
                        self.fi[i, j, q] = self.wi[q] * self.grid[i, j].rho * (1 + 3.0/2.0 * vu + 9.0/2.0 * vu2 - 3.0/2.0 * uu)
        return f_eq
            
            
#    def iterate(self):
#        f_new = np.zeros([self.Nx, self.Ny, 9])
#        
#        for i in range(self.Nx):
#            for j in range(self.Ny):
#                if self.grid[i, j].gridpoint_type == 0:
#                    
#                    for q in range(9):
#                        collisions = 
        

class gridpoint():
    def __init__(self, index):
        self.index = index
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.rho = 1.0
        self.f = np.empty(9)
        self.gridpoint_type = 0 # 0 = normal, 1 = periodic, 2 = no-slip
        self.neighbours = np.empty(8, int)
        self.subpoints = np.empty(8, "O")
        self.lengths = np.empty(12)
        self.normals = np.empty(12, "O")
        self.areas = np.empty(4)
    

        
    