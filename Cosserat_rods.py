import numpy as np
from matplotlib import pyplot as plt

# Variables on elements
class CosseratRod():
  def __init__(self, number_of_elements, total_length, density, radius, direction, normal, youngs_modulus, poisson_ratio):
    #direction must be a unit vector
    self.direction = direction / np.linalg.norm(direction)

    #properties for whole rod
    self.youngs_modulus = youngs_modulus
    self.shear_modulus = self.youngs_modulus / (poisson_ratio + 1)
    self.n_elements = number_of_elements# pick some number or use the given values in snake.pdf
    self.n_nodes = self.n_elements + 1
    self.n_voronoi = self.n_elements - 1
    self.total_length = total_length
    self.density = density
    self.radius = radius

    # Element scalars
    self.reference_lengths = np.ones((self.n_elements))* self.total_length/self.n_elements # Here I took total rod length as 1
    self.current_lengths = self.reference_length.copy() #initially, current length is initial length
    self.area = np.pi * self.radius**2 #initial
    self.current_area = self.area * np.ones((self.n_elements))
    self.current_radius = self.radius * np.ones((self.n_elements))
    self.element_volume = self.area * self.current_lengths
    self.element_mass = self.element_volume * density
    self.element_dilation = np.ones((self.n_elements)) #initialize first, update later
    #moment of area
    self.I = np.zeros((3, self.n_elements))
        #moment of inertia
    self.J = np.zeros((self.n_elements))

    for element in range(self.n_elements):
        self.I[:, element] = self.current_area[element]**2/(4 * np.pi) * np.array([1, 1, 2])
        self.J[element] = self.current_length[element]**2 * self.element_mass[element] / 2

    #nodal mass
    self.mass = np.zeros((self.n_elements + 1))
    self.mass[:-1] += 0.5 * self.element_mass
    self.mass[1:] += 0.5 * self.element_mass
    
    #nodal positions
    self.position = np.zeros((3, self.n_elements + 1))
    discretize_length = np.linspace(0, self.total_length, self.n_elements + 1)
    for node in range(self.n_elements + 1):
        self.position[:, node] =  discretize_length[node] * direction

    # Element vectors
    self.directors = np.zeros((3,3, self.n_elements))
    self.tangents = self.position[:, 1:] - self.position[:, :-1]
    self.tangents /= np.linalg.norm(self.tangents, axis=0, keepdims=True)
    for idx in range(self.n_elements):
        self.d1 = normal 
        self.d3 = direction
        self.d2 = np.cross(self.d3, self.d1) # binormal
        self.directors[0, :, idx] = self.d1 # d1
        self.directors[1, :, idx] = self.d2 # d2
        self.directors[2, :, idx] = self.d3 # d3

    self.sigma = np.zeros((3, self.n_elements)) #shear strain vector, initialize first, update later
    
    # Element matrix
    self.element_bend_matrix = np.zeros((3, 3, self.n_elements))
    self.shear_matrix = np.zeros((3, 3, self.n_elements))
    
    self.alpha = 4/3
    for element in range(self.n_elements):
        B1 = self.youngs_modulus * self.I[0, element]
        B2 = self.youngs_modulus * self.I[1, element]
        B3 = self.shear_modulus * self.I[2, element]
        S1 = self.alpha * self.shear_modulus * self.current_area[element]
        S2 = self.alpha * self.shear_modulus * self.current_area[element]
        S3 = self.youngs_modulus * self.current_area[element]

        self.element_bend_matrix[:, :, element] = np.diag([B1, B2, B3])
        self.shear_matrix[:, :, element] = np.diag([S1, S2, S3])
    
    # Voronoi scalars
    self.reference_voronoi_length = (self.reference_lengths[1:] - self.reference_lengths[:-1]) / 2
    self.current_voronoi_length = self.reference_voronoi_length.copy()
    self.voronoi_dilation = np.ones((self.n_elements)) #initialize first, update later
    # Voronoi vectors
    self.kappa = np.zeros((3, self.n_voronoi)) #initialize but update later
    # Voronoi matrix

    self.bend_matrix = np.zeros((3,3,self.n_voronoi))

    for voronoi in range(self.n_voronoi):
        self.bend_matrix[:,:,voronoi] = (self.element_bend_matrix[:,:, voronoi + 1] + self.element_bend_matrix[: ,: , voronoi]) / (2 * self.reference_voronoi_length[voronoi])

def force_rule(self):
    pass
    # according to cosserate therory, solve for the linear and angular acceleartions

def update(self):
    pass
    #update current force

    #update current postion

    #update current velocity

    #update current length

    #update current dilation

    #update current tangent

    #update current shear strain

    #update current angular velocity

    #update current director

    #update current area

    #update current moment of inertia

    #update current moment of area

    #update current element bending matrix

    #update current element shear matrix

    #update current external moment

    #update current voronoi length

    #update current voronoi dilation

    #update current R = Q @ Q.T

    #update current theta = np.acos((np.trace(R) - 1) / 2)

    #update current voronoi kappa

    #update current bend matrix


# Difference operator (Delta^h operator in discreatized cosserat rod equations)
def modified_diff(t_x):
    """ Modified trapezoidal integration"""
    # Pads a 0 at the end of an array
    temp = np.pad(t_x, (0,1), 
                  'constant', 
                  constant_values=(0,0)) # Using roll calculate the diff (ghost node of 0)
    return (temp - np.roll(temp, 1))[:-1, :]

# Modified trapezoidal integration (A^h operator in discreatized cosserat rod equations)
def modified_trapz(t_x):
  """ Modified trapezoidal integration"""
  # Pads a 0 at the start of an array
  temp = np.pad(t_x, (1,0), 'constant', constant_values=(0,0))
  # Using roll calculate the integral (ghost node of 0)
  return (0.5*(temp + np.roll(temp, -1)))[1:, :]

def position_verlet(dt, x, v, force_rule):
        """Does one iteration/timestep using the Position verlet scheme
        
        Parameters
        ----------
        dt : float
            Simulation timestep in seconds
        x : float/array-like
            Quantity of interest / position of COM
        v : float/array-like
            Quantity of interest / velocity of COM
        force_rule : ufunc
            A function, f, that takes one argument and
            returns the instantaneous forcing

        Returns
        -------
        x_n : float/array-like
            The quantity of interest at the Next time step
        v_n : float/array-like
            The quantity of interest at the Next time step
        """
        temp_x = x + 0.5*dt*v
        v_n = v + dt * force_rule(temp_x)
        x_n = temp_x + 0.5 * dt * v_n
        return x_n, v_n