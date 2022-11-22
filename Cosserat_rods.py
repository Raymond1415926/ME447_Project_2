import numpy as np
from matplotlib import pyplot as plt

# Variables on elements
class CosseratRod():
  def __init__(self):
    n_elements = 10# pick some number or use the given values in snake.pdf
    n_nodes = n_elements + 1
    n_voronoi = n_elements - 1

    # Here I am just showing the size of these arrays. You still need to initialize them with proper values.
    # Nodal vectors has dimension (3, n_nodes)
    position = np.zeros((3,n_nodes))
    # Nodal scalars (n_nodes)
    mass = np.zeros((n_nodes))

    # Element vectors
    tangents = np.zeros((3,n_elements))
    # Element scalars
    lengths = np.zeros((n_elements))
    # Element matrix
    directors = np.zeros((3,3,n_elements))

    # Voronoi vectors
    kappa = np.zeros((3, n_voronoi))
    # Voronoi scalars
    voronoi_length = np.zeros((n_voronoi))
    # Voronoi matrix
    bend_matrix = np.zeros((3, 3, n_voronoi))

    density = 1000
    radius = 0.01
    lengths = np.ones((n_elements))* 1/n_elements # Here I took total rod length as 1
    volume = np.pi * radius**2 * lengths
    element_mass = volume * density

    # Now lets compute nodal masses. Distribute element masses on nodes
    mass = np.zeros((n_elements+1))
    mass[:-1] += 0.5 * element_mass
    mass[1:] += 0.5 * element_mass

    position = np.zeros((3, n_elements+1))
    position[2,:] = np.linspace(0, 1, n_elements+1) # I picked z direction, you can pick something as well
    # Compute element tangents.
    # Initially you can take d3 and element tangents same.
    tangents = position[:, 1:] - position[:, :-1]
    tangents /= np.linalg.norm(tangents, 
                                    axis=0, 
                                            keepdims=True)
    # Pick a normal vector direction. You have to make sure normal vector is perpendicular to tangent and unit vector.
    # I picked unit vector in y direction
    normal = np.array([0, 1., 0.])

    director = np.zeros((3,3, n_elements))
    for idx in range(n_elements):
      d1 = normal 
      d3 = tangents[:,idx]
      d2 = np.cross(d3, d1) # binormal
      directors[0, :, idx] = d1 # d1
      directors[1, :, idx] = d2 # d2
      directors[2, :, idx] = d3 # d3

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