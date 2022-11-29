import numpy as np
from matplotlib import pyplot as plt

# Variables on elements
class CosseratRod():
  def __init__(self, number_of_elements, total_length, density, radius, direction, normal, youngs_modulus,  dt, total_time, poisson_ratio = 0.5, shear_modulus = 0):
    #direction must be a unit vector
    self.direction = direction / np.linalg.norm(direction)

    #callback parameters
    self.callback_params = {} #an empty dictionary that stores data for each time steps

    #properties for whole rod
    self.youngs_modulus = youngs_modulus
    if shear_modulus == 0:
        self.shear_modulus = self.youngs_modulus / (poisson_ratio + 1)
    else:
        self.shear_modulus = shear_modulus
    self.n_elements = number_of_elements# pick some number or use the given values in snake.pdf
    self.n_nodes = self.n_elements + 1
    self.n_voronoi = self.n_elements - 1
    self.total_length = total_length
    self.density = density
    self.radius = radius
    self.dt = dt
    self.total_time = total_time
    self.time = np.arange(0, total_time, dt)

    # Element scalars
    self.reference_lengths = np.ones((self.n_elements))* self.total_length/self.n_elements # Here I took total rod length as 1
    self.current_lengths = self.reference_lengths.copy() #initially, current length is initial length
    self.area = np.pi * self.radius**2 #initial
    self.current_area = self.area * np.ones((self.n_elements))
    self.element_volume = self.area * self.reference_lengths[0] #volume is not changing
    self.element_mass = self.element_volume * density
    self.element_dilation = np.ones((self.n_elements)) #initialize first, update later
    #moment of area
    self.I = np.zeros((3, self.n_elements))
        #moment of inertia
    self.J = np.zeros((self.n_elements))

    #initialize accelerations and velocities
    # self.acceleration = np.zeros([self.n_elements])
    # self.angular_acceleration = np.zeros([self.n_elements])

    self.velocitiy = np.zeros([self.n_elements])
    self.angular_velocity = np.zeros([self.n_elements])

    #nodal mass
    self.mass = np.zeros((self.n_elements + 1))
    self.mass[:-1] += 0.5 * self.element_mass
    self.mass[1:] += 0.5 * self.element_mass

    for element in range(self.n_elements):
        self.I[:, element] = self.current_area[element]**2/(4 * np.pi) * np.array([1, 1, 2])
        self.J[element] = self.reference_lengths[element]**2 / 4 * self.mass[element] + \
                          self.reference_lengths[element]**2 / 4 * self.mass[element + 1] #point mass on two ends of each elements




    #nodal positions
    self.angular_position = np.zeros((3, self.n_elements))
    self.position = np.zeros((3, self.n_elements + 1))
    discretize_lengths = np.linspace(0, self.total_length, self.n_elements + 1)
    for node in range(self.n_elements + 1):
        self.position[:, node] =  discretize_lengths[node] * direction

    # Element vectors
    self.Q = np.zeros((3,3, self.n_elements)) #directors
    self.tangents = self.position[:, 1:] - self.position[:, :-1]
    self.tangents /= np.linalg.norm(self.tangents, axis=0, keepdims=True)
    for idx in range(self.n_elements):
        self.d1 = normal 
        self.d3 = direction
        self.d2 = np.cross(self.d3, self.d1) # binormal
        self.Q[0, :, idx] = self.d1 # d1
        self.Q[1, :, idx] = self.d2 # d2
        self.Q[2, :, idx] = self.d3 # d3

    self.sigma = np.zeros((3, self.n_elements)) #shear strain vector, initialize first, update later
    
    # Element matrix
    self.element_B = np.zeros((3, 3, self.n_elements))
    self.S = np.zeros((3, 3, self.n_elements))
    
    self.alpha = 4/3
    for element in range(self.n_elements):
        B1 = self.youngs_modulus * self.I[0, element]
        B2 = self.youngs_modulus * self.I[1, element]
        B3 = self.shear_modulus * self.I[2, element]
        S1 = self.alpha * self.shear_modulus * self.current_area[element]
        S2 = self.alpha * self.shear_modulus * self.current_area[element]
        S3 = self.youngs_modulus * self.current_area[element]

        self.element_B[:, :, element] = np.diag([B1, B2, B3])
        self.S[:, :, element] = np.diag([S1, S2, S3])

    #external force and torque
    self.external_force = np.zeros([1, self.n_nodes])
    self.external_torque = np.zeros([1, self.n_elements])
    
    # Voronoi scalars
    self.reference_voronoi_lengths = (self.reference_lengths[1:] - self.reference_lengths[:-1]) / 2
    self.current_voronoi_lengths = self.reference_voronoi_lengths.copy()
    self.voronoi_dilation = np.ones((self.n_elements)) #initialize first, update later
    # Voronoi vectors
    self.kappa = np.zeros((3, self.n_voronoi)) #initialize but update later
    # Voronoi matrix
    self.B = np.zeros((3,3,self.n_voronoi))

    for voronoi in range(self.n_voronoi):
        self.B[:,:,voronoi] = (self.element_B[:,:, voronoi + 1] * self.reference_lengths[voronoi + 1] \
         + self.element_B[: ,: , voronoi] * self.reference_lengths[voronoi]) / (2 * self.reference_voronoi_lengths[voronoi])

def _batch_matmul(first_matrix_collection, second_matrix_collection):
    """
    This is batch matrix matrix multiplication function. Only batch
    of 3x3 matrices can be multiplied.
    Parameters
    ----------
    first_matrix_collection
    second_matrix_collection

    Returns
    -------
    Notes
    Microbenchmark results showed that for a block size of 200,
    %timeit np.einsum("ijk,jlk->ilk", first_matrix_collection, second_matrix_collection)
    8.45 µs ± 18.6 ns per loop
    This version is
    2.13 µs ± 1.01 µs per loop
    """
    blocksize = first_matrix_collection.shape[2]
    output_matrix = np.zeros((3, 3, blocksize))

    for i in range(3):
        for j in range(3):
            for m in range(3):
                for k in range(blocksize):
                    output_matrix[i, m, k] += (
                        first_matrix_collection[i, j, k]
                        * second_matrix_collection[j, m, k]
                    )

    return output_matrix

def _batch_matvec(matrix_collection, vector_collection):
    """
    This function does batch matrix and batch vector product

    Parameters
    ----------
    matrix_collection
    vector_collection

    Returns
    -------
    Notes
    -----
    Benchmark results, for a blocksize of 100, using timeit
    Python einsum: 4.27 µs ± 66.1 ns per loop
    This version: 1.18 µs ± 39.2 ns per loop
    """
    blocksize = vector_collection.shape[1]
    print(blocksize, "of vector")
    output_vector = np.zeros((3, blocksize))

    for i in range(3):
        for j in range(3):
            for k in range(blocksize):
                output_vector[i, k] += (
                    matrix_collection[i, j, k] * vector_collection[j, k]
                )
    return output_vector

def _batch_cross(first_vector_collection, second_vector_collection):
    """
    This function does cross product between two batch vectors.

    Parameters
    ----------
    first_vector_collection
    second_vector_collection

    Returns
    -------
    Notes
    ----
    Benchmark results, for a blocksize of 100 using timeit
    Python einsum: 14 µs ± 8.96 µs per loop
    This version: 1.18 µs ± 141 ns per loop
    """
    blocksize = first_vector_collection.shape[1]
    output_vector = np.empty((3, blocksize))

    for k in range(blocksize):
        output_vector[0, k] = (
            first_vector_collection[1, k] * second_vector_collection[2, k]
            - first_vector_collection[2, k] * second_vector_collection[1, k]
        )

        output_vector[1, k] = (
            first_vector_collection[2, k] * second_vector_collection[0, k]
            - first_vector_collection[0, k] * second_vector_collection[2, k]
        )

        output_vector[2, k] = (
            first_vector_collection[0, k] * second_vector_collection[1, k]
            - first_vector_collection[1, k] * second_vector_collection[0, k]
        )

    return output_vector

def _batch_matrix_transpose(input_matrix):
    """
    This function takes an batch input matrix and transpose it.
    Parameters
    ----------
    input_matrix

    Returns
    -------
    Notes
    ----
    Benchmark results,
    Einsum: 2.08 µs ± 553 ns per loop
    This version: 817 ns ± 15.2 ns per loop
    """
    output_matrix = np.empty(input_matrix.shape)
    for i in range(input_matrix.shape[0]):
        for j in range(input_matrix.shape[1]):
            for k in range(input_matrix.shape[2]):
                output_matrix[j, i, k] = input_matrix[i, j, k]
    return output_matrix

def force_rule(self, position):
    # according to cosserate therory, solve for the linear and angular acceleartions
    # we basically need to update everything in order to solve for the right values
    
    
    
    # position of the current time is given already
    self.position = position

    # with the position, we need to update tangent and the length
    self.tangents = self.position[:, 1:] - self.position[:, :-1]
    self.current_lengths = np.linalg.norm(self.tangents, axis=0, keepdims=True)
    self.tangents /= self.current_lengths

    #update current dilation 
    self.element_dilation = self.current_lengths / self.reference_lengths

    #update current voronoi lengths
    self.current_voronoi_lengths = (self.current_lengths[1:] - self.current_lengths[:-1]) / 2
    
    #update current voronoi dilation
    self.voronoi_dilation = self.current_voronoi_lengths / self.reference_voronoi_lengths
    
    #we have everything we need, now we need to update sigma.
    for element in range(self.n_elements):
        #update current shear strain sigma
        d3 = self.Q[2, :, element]
        self.sigma[element] = self.Q[:,:,element] * (self.element_dilation[element] * self.tangents - d3)
    
    #sanity check: 
    # Q depends on omega, update later
    # keep S the same
    # e is updated
    # update force externally with external methods
    # keep B the same
    # kappa depends on Q, update later
    # epsilon is updated
    # tangent is updated
    # sigma is updated
    # keep reference length the same
    

    #first solve for acceleration:
    QT_S_sigma_over_e = _batch_matvec(_batch_matmul(_batch_matrix_transpose(self.Q), self.S), self.sigma) / self.element_dilation #batch transformation
    acceleration = (delta_h(QT_S_sigma_over_e) + self.external_force) / self.mass
    
    #then solve for angular acceleration:
    B_kappa_over_epsilon_cube = _batch_matmul(self.B, self.kappa) / self.voronoi_dilation ** 3
    J_over_e = self.J / self.element_dilation
    kappa_cross_B_kappa = _batch_cross(self.kappa, _batch_matvec(self.B, self.kappa))
    kappa_cross_B_kappa_over_epsilon_cube_D = (kappa_cross_B_kappa / self.voronoi_dilation ** 3) * self.reference_voronoi_length
    Q_t_cross_S_sigma_l = _batch_cross(_batch_matvec(self.Q, self.tangents) , _batch_matvec(self.S, self.sigma)) * self.reference_length
    angular_acceleration = (delta_h(B_kappa_over_epsilon_cube) + alpha_h(kappa_cross_B_kappa_over_epsilon_cube_D) + Q_t_cross_S_sigma_l\
         + self.external_torque) / J_over_e

    return acceleration, angular_acceleration
   
def update(self):
    #update current force by specific force class
    #update current torque by specific torque class

    #position_verlet updates basically everything

    #update current Q
    #update current angular velocity
    #update current velocity
    #update current postion
    self.position, self.velocity = self.position_verlet(self.dt, self.position, self.velocity)

        #update current area
        # self.current_area = self.current_area / self.element_dilation

        #update current moment of area
        # self.I[element] = self.I[element] / self.element_dilation[element] ** 2
        
        # #update current moment of inertia
        # self.J[element] = self.current_lengths[element]**2 / 4 * self.mass[element] + \
        #                   self.current_lengths[element]**2 / 4 * self.mass[element + 1] #point mass on two ends of each elements
        
        # #update current element bending matrix
        # self.B[:,:,element] = self.B[:,:,element] / self.element_dilation[element] ** 2

        # #update current element shear matrix
        # self.S[:,:,element] = self.S[:,:,element] / self.element_dilation[element]

    #kappa
    for voronoi in range(self.n_voronoi):
        #update current R = Q @ Q.T
        R = self.Q[:,:,voronoi + 1] @ self.Q[:,:,voronoi].T

        #update current theta = np.acos((np.trace(R) - 1) / 2)
        theta = np.acos((np.trace(R) - 1) / 2)

        #update skew_symmetric matrix U and vector u associated with it
        U = R - R.T
        u = np.array([U[2,1], -U[2,0], U[1,0]])
        #update log_Q_QT
        if theta == 0:
            log_Q_QT = 0
        else:
            log_Q_QT = theta / (2 * np.sin(theta)) * u

        #update current kappa
        self.kappa[voronoi] = log_Q_QT / self.reference_voronoi_length

def make_callback(self):

    
    self.callback_params["position"].append(self.position_collection.copy())
    self.callback_params["velocity"].append(self.velocity_collection.copy())
    self.callback_params["curvature"].append(self.kappa.copy())

    return

# Difference operator (Delta^h operator in discreatized cosserat rod equations)
def delta_h(t_x):
    """ Modified trapezoidal integration"""
    # Pads a 0 at the end of an array
    temp = np.pad(t_x, (0,1), 'constant', constant_values=(0,0)) # Using roll calculate the diff (ghost node of 0)
    return (temp - np.roll(temp, 1))[:-1, :]

# Modified trapezoidal integration (A^h operator in discreatized cosserat rod equations)
def alpha_h(t_x):
  """ Modified trapezoidal integration"""
  # Pads a 0 at the start of an array
  temp = np.pad(t_x, (1,0), 'constant', constant_values=(0,0))
  # Using roll calculate the integral (ghost node of 0)
  return (0.5*(temp + np.roll(temp, -1)))[1:, :]

def position_verlet(self, dt, x, v):
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
    #update half step position
    half_step_r = x + 0.5*dt*v

    #update half step Q
    Q_t_half_dt = np.zeros([3,3,self.n_elements])
    for element in range(self.n_elements):
        Q_t_half_dt[:,:,element] = np.exp(-self.dt/2 * self.angular_velocity[element]) * self.Q[:,:,element]
    
    #obtain half step acceleration
    half_step_acceleration, half_step_angular_acceleration = self.force_rule(half_step_r)
    
    #update velocity
    v_n = v + dt * half_step_acceleration

    #update angular velocity
    self.angular_velocity = self.angular_velocity + dt * half_step_angular_acceleration

    #update position
    x_n = half_step_r + 0.5 * dt * v_n

    #update Q
    for element in range(self.n_elements):
        self.Q[:,:,element] = np.exp(-self.dt/2 * half_step_angular_acceleration) * Q_t_half_dt

    return x_n, v_n


n_elements = 1
length = 1
density = 1000
radius = 0.1
direction = np.array([0, 0, 1])
normal = np.array([0, 1, 0])
youngs_modulus = 10e7
shear_modulus = youngs_modulus * 2 / 3
dt = 0.001
total_time = 1
rod = CosseratRod(number_of_elements=n_elements, total_length=length, density=density, radius=radius, direction=direction,
    normal=normal, youngs_modulus=youngs_modulus, dt=dt, total_time= total_time, shear_modulus=shear_modulus)
