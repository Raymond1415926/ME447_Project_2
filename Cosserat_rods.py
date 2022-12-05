import numpy as np
from matplotlib import pyplot as plt
from matrix_operators import _batch_matvec, _batch_cross, _batch_matrix_transpose, _batch_norm
import copy
from tqdm import tqdm
import matplotlib.animation as manimation
from snake_force import TimoshenkoForce, MuscleTorques
import time
from numba import njit

np.set_printoptions(linewidth = np.inf)

# the legendary cosserate solver!
class CosseratRod():

    def __init__(self, number_of_elements, total_length, density, radius,  normal, youngs_modulus, dt,
                 total_time, dissipation_constant = 0.1, direction = np.array([0,0,1]), poisson_ratio=0.5, shear_modulus=0,  \
                 fixed_BC = False, stretchable = True):
        self.count = 1
        # direction must be a unit vector
        self.stretchable = stretchable
        self.forces_torques_to_add = []
        self.direction = direction / np.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
        self.fixed_BC = fixed_BC
        # callback parameters
        self.callback_params = {}  # an empty dictionary that stores data for each time steps
        self.callback_params['positions'] = []
        self.callback_params['velocities'] = []
        self.callback_params['curvature'] = []
        # properties for whole rod
        self.youngs_modulus = youngs_modulus
        if shear_modulus == 0:
            self.shear_modulus = self.youngs_modulus / 2 / (poisson_ratio + 1)
        else:
            self.shear_modulus = shear_modulus
        self.n_elements = number_of_elements  # pick some number or use the given values in snake.pdf
        self.n_nodes = self.n_elements + 1
        self.n_voronoi = self.n_elements - 1
        self.total_length = total_length
        self.density = density
        self.radius = radius
        self.dt = dt
        self.total_time = total_time
        self.time = np.arange(0, total_time, dt)
        self.current_time = self.time[0]
        self.dissipation_constant = dissipation_constant

        # Element scalars
        self.reference_lengths = np.ones((self.n_elements)) * self.total_length / self.n_elements  #length of elements
        self.current_lengths = self.reference_lengths.copy()  # initially, current length is initial length
        self.area = np.pi * self.radius ** 2  # initial
        self.element_volume = self.area * self.reference_lengths[0]  # volume is not changing
        self.element_mass = self.element_volume * density
        self.element_dilataton = np.ones((1, self.n_elements))  # initialize first, update later


        # initialize accelerations and velocities

        self.velocities = np.zeros((3, self.n_nodes))
        self.angular_velocities = np.zeros([3, self.n_elements])


        # nodal mass
        self.mass = np.zeros((self.n_nodes))
        self.mass[:-1] += 0.5 * self.element_mass
        self.mass[1:] += 0.5 * self.element_mass

        #moment of area and moment of inertia
        self.I = self.area ** 2 / (4 * np.pi) * np.array([1, 1, 2])
        self.J = self.I * self.density * self.reference_lengths[0]
        self.J_inv_matrix = np.diag([1/self.J[0], 1/self.J[1], 1/self.J[2]])
        self.J_inv = np.zeros([3, 3, self.n_elements])
        for element in range(self.n_elements):
            self.J_inv[:,:,element] = self.J_inv_matrix
        # nodal positions
        self.positions = np.zeros((3, self.n_nodes))
        self.half_step_r = np.zeros((3, self.n_nodes))

        discretize_lengths = np.linspace(0, self.total_length, self.n_nodes)
        for node in range(self.n_nodes):
            self.positions[:, node] = discretize_lengths[node] * direction
        # Element vectors
        self.Q = np.zeros((3, 3, self.n_elements))  # directors
        self.tangents = self.positions[:, 1:] - self.positions[:, :-1]
        self.tangents = np.divide(self.tangents, self.current_lengths)
        for idx in range(self.n_elements):
            self.d1 = normal
            self.d3 = direction
            self.d2 = np.cross(self.d3, self.d1)  # binormal vectors
            self.Q[0, :, idx] = self.d1  # d1
            self.Q[1, :, idx] = self.d2  # d2
            self.Q[2, :, idx] = self.d3  # d3

        self.ref_Q = self.Q.copy()

        self.sigma = np.zeros((3, self.n_elements))  # shear strain vector, initialize first, update later

        # Element matrix
        self.element_B = np.zeros((3, 3, self.n_elements))
        self.S = np.zeros((3, 3, self.n_elements))

        self.alpha = 4 / 3
        B1 = self.youngs_modulus * self.I[0]
        B2 = self.youngs_modulus * self.I[1]
        B3 = self.shear_modulus * self.I[2]
        S1 = self.alpha * self.shear_modulus * self.area
        S2 = self.alpha * self.shear_modulus * self.area
        S3 = self.youngs_modulus * self.area
        for element in range(self.n_elements):
            self.element_B[:, :, element] = np.diag([B1, B2, B3])
            self.S[:, :, element] = np.diag([S1, S2, S3])

        # external force and torque
        self.external_forces = np.zeros([3, self.n_nodes])
        self.external_torques = np.zeros([3, self.n_elements])

        # Voronoi scalars
        self.reference_voronoi_lengths = (self.reference_lengths[1:] + self.reference_lengths[:-1]) / 2
        self.current_voronoi_lengths = self.reference_voronoi_lengths.copy()
        self.voronoi_dilataton = np.ones((self.n_voronoi))  # initialize first, update later
        # Voronoi vectors
        self.kappa = np.zeros((3, self.n_voronoi))  # initialize but update later
        # Voronoi matrix
        self.B = np.zeros((3, 3, self.n_voronoi))

        for voronoi in range(self.n_voronoi):
            self.B[:, :, voronoi] = (self.element_B[:, :, voronoi + 1] * self.reference_lengths[voronoi + 1] \
                                     + self.element_B[:, :, voronoi] * self.reference_lengths[voronoi]) / (
                                                2 * self.reference_voronoi_lengths[voronoi])
    
    def force_rule(self):
        # apply external forces and torques
        self.apply_forces_torques()

        # debug, add some fixed force or torque here



        #apply damping force and torques
        element_velocity = 0.5 * (self.velocities[:, 1:] + self.velocities[:, :-1])
        disspation_force = self.delta_h(np.multiply(self.dissipation_constant * element_velocity, self.current_lengths))
        self.external_forces = np.add(self.external_forces, disspation_force)
        self.external_torques = np.add(self.external_torques,
                                       - np.multiply(self.dissipation_constant * self.angular_velocities,\
                                        self.current_lengths))

        # transpose Q
        Qt = _batch_matrix_transpose(self.Q)

        #stretch/strain internal force
        stretch_force = np.divide(_batch_matvec(Qt, _batch_matvec(self.S, self.sigma)), self.element_dilataton)
        stretch_force = self.delta_h(stretch_force)
        #sum the force, divide by mass, get acceleration
        acceleration = np.divide(np.add(stretch_force, self.external_forces), self.mass)

        #bend/twist internal couple
        #cubic dilatation
        epsilon3 = np.power(self.voronoi_dilataton, 3)
        twist_bend_couple1 = self.delta_h(np.divide(_batch_matvec(self.B, self.kappa), epsilon3))
        twist_bend_couple2 = self.alpha_h(self.reference_voronoi_lengths * \
                             np.divide(_batch_cross(self.kappa, _batch_matvec(self.B, self.kappa)),\
                                       epsilon3))
        twist_bend_couple = np.add(twist_bend_couple1, twist_bend_couple2)
        #shear/stretch internal couple
        Q_t = _batch_matvec(self.Q, self.tangents)
        S_sigma = _batch_matvec(self.S, self.sigma)
        shear_stretch_couple = _batch_cross(Q_t, S_sigma) * self.reference_lengths
        #summing torqus
        internal_torques = np.add(twist_bend_couple, shear_stretch_couple)
        #calculate alpha
        alpha = np.multiply(np.add(internal_torques, self.external_torques), self.element_dilataton)
        alpha = _batch_matvec(self.J_inv, alpha)
        #clear forces and torques
        self.external_forces.fill(0.0)
        self.external_torques.fill(0.0)
        
        return acceleration, alpha

    def apply_BC(self):
        if self.fixed_BC:
            self.positions[:, 0] = 0
            self.angular_velocities[:, 0] = 0
            self.velocities[:, 0] = 0
            self.Q[:, :, 0] = self.ref_Q[:, :, 0]

    def add_forces_torques(self, forceobj):
        self.forces_torques_to_add.append(forceobj)

    def apply_forces_torques(self):

        for forceobj in self.forces_torques_to_add:
            try:
                forceobj.apply_forces(self)
            except AttributeError:
                pass
            try:
                forceobj.apply_torques(self)
            except AttributeError:
                pass

    def update_elements(self):
        self.tangents = np.subtract(self.positions[:, 1:], self.positions[:, :-1])
        self.current_lengths = _batch_norm(self.tangents)
        self.tangents = np.divide(self.tangents, self.current_lengths)
        self.current_voronoi_lengths = np.add(self.current_lengths[1:], self.current_lengths[:-1]) / 2

        if self.stretchable:
            # update dilatations
            self.element_dilataton = np.divide(self.current_lengths, self.reference_lengths)
            # update voronoi lengths
            self.voronoi_dilataton = np.divide(self.current_voronoi_lengths, self.reference_voronoi_lengths)
        # update sigma
        strain = np.subtract(np.multiply(self.element_dilataton, self.tangents), self.Q[2, :, :])
        self.sigma = _batch_matvec(self.Q, strain)

    def update_Q(self):
        directors = np.zeros((3, 3, self.n_elements))
        norm_omega = _batch_norm(self.angular_velocities)
        for i in range(self.n_elements):
            angle = -0.5 * self.dt * norm_omega[i]
            if norm_omega[i] == 0:
                about = np.zeros((3,))
            else:
                about = self.angular_velocities[:, i] / norm_omega[i]
            k1, k2, k3 = about
            K = np.array([[0.0, -k3, k2], [k3, 0.0, -k1], [-k2, k1, 0.0]])
            R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
            directors[:, :, i] = np.dot(R, self.Q[:,:,i])
        self.Q = directors

    def update(self):
        self.make_callback()
        self.position_verlet()
        # if(self.angular_velocities[0,-1] > np.pi/3):
        #     input("exploded")
    
    def update_kappa(self):
        curvature = np.zeros((3, self.n_voronoi))
        for i in range(self.n_voronoi):
            R = self.Q[:, :, i + 1] @ self.Q[:, :, i].T
            to_be_inv_cos = np.clip((np.trace(R) - 1.0) / 2.0, -1, 1)
            angle = np.arccos(to_be_inv_cos)
            if angle < 1e-10:
                curvature[:, i] = np.zeros(3)
            else:
                K = (R - R.T) * 0.5 / np.sin(angle) * angle
                curvature[:, i] = -np.array([-K[1, 2], K[0, 2], K[0, 1]]) / self.reference_voronoi_lengths[i]
        self.kappa = curvature

    def run(self):
        steps = int(self.total_time // self.dt)
        for step in tqdm(range(steps)):
            #update time
            self.current_time = self.time[step]
            self.update()

    def make_callback(self):
        self.callback_params["positions"].append(copy.deepcopy(self.positions))
        self.callback_params["velocities"].append(copy.deepcopy(self.velocities))
        self.callback_params["curvature"].append(copy.deepcopy(self.kappa))

    # Difference operator (Delta^h operator in discreatized cosserat rod equations)
    def delta_h(self, t_x):
        """ Modified trapezoidal integration"""
        # Pads a 0 at the end of an array
        temp = np.pad(t_x, (0, 1), 'constant',
                      constant_values=(0, 0))  # Using roll calculate the diff (ghost node of 0)
        return (temp - np.roll(temp, 1))[:-1, :]

    # Modified trapezoidal integration (A^h operator in discreatized cosserat rod equations)
    def alpha_h(self, t_x):
        """ Modified trapezoidal integration"""
        # Pads a 0 at the start of an array
        temp = np.pad(t_x, (1, 0), 'constant', constant_values=(0, 0))
        # Using roll calculate the integral (ghost node of 0)
        return (0.5 * (temp + np.roll(temp, -1)))[1:, :]

    def position_verlet(self):
        # update half step position
        self.positions = np.add(self.positions, 0.5 * self.dt * self.velocities)
        # update half step Q
        self.apply_BC()
        self.update_Q()
        self.update_elements()
        self.apply_BC()
        self.update_kappa()

        self.apply_BC()

        # obtain half step acceleration
        half_step_acceleration, half_step_angular_acceleration = self.force_rule()
        # update velocity
        self.velocities = self.velocities + self.dt * half_step_acceleration

        # update angular velocity
        self.angular_velocities = self.angular_velocities + self.dt * half_step_angular_acceleration
        self.apply_BC()

        # update position
        self.positions = np.add(self.positions, 0.5 * self.dt * self.velocities)
        self.apply_BC()
        # update Q full step
        self.update_Q()
        self.apply_BC()

# dynamic plotting, clear and updates
def dynamic_plotting(ploting_parameters: dict, every = 10,xlim = [0,3], ylim = [-1,1]):
    wait = 0.01
    figure, ax = plt.subplots()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    steps = len(ploting_parameters["positions"])
    plt.xlabel("Distance along lab frame z")
    plt.ylabel("Deflection in lab frame x")
    plt.title("Deflection of a rod under torque")
    positions = ploting_parameters["positions"]
    line1 = ax.plot(positions[0][2], positions[0][0])
    for step in tqdm(range(0, steps, every)):
        line1[0].set_xdata(positions[step][2])
        line1[0].set_ydata(positions[step][0])
        figure.canvas.draw()
        figure.canvas.flush_events()
        plt.pause(wait)  # is necessary for the plot to update for some reason

    plt.show()

# dynamic plotting, does not clear previous graph
def dynamic_plotting_v2(ploting_parameters: dict, xlim=[-1,3],ylim=[0,1],every = 10, wait = 0.001):
    figure, ax = plt.subplots()
    steps = len(ploting_parameters["positions"])
    plt.xlabel("Distance along lab frame z")
    plt.ylabel("Deflection in lab frame x")
    plt.title("Deflection of a rod under torque")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    positions = ploting_parameters["positions"]
    for step in tqdm(range(0, steps, every)):
        ax.plot(positions[step][2], positions[step][0])
        plt.pause(wait)  # is necessary for the plot to update for some reason

    plt.show()

# make videos
def plot_video(plot_params: dict, video_name="video.mp4", fps=100, xlim=(0, 4), ylim=(-.3, .2), every = 10):

    positions_over_time = np.array(plot_params["positions"])

    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
    ax = fig.add_subplot(111)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("z [m]", fontsize=16)
    ax.set_ylabel("x [m]", fontsize=16)
    rod_lines_2d = ax.plot(positions_over_time[0][2], positions_over_time[0][0])[0]
    # plt.axis("equal")
    steps = positions_over_time.shape[0]
    with writer.saving(fig, video_name, dpi=150):
        for step in tqdm(range(0, steps, 150)):
            rod_lines_2d.set_xdata(positions_over_time[step][2])
            rod_lines_2d.set_ydata(positions_over_time[step][0])
            writer.grab_frame()
    # Be a good boy and close figures
    # https://stackoverflow.com/a/37451036
    # plt.close(fig) alone does not suffice
    # See https://github.com/matplotlib/matplotlib/issues/8560/
    plt.close(plt.gcf())

"""
The timoshenko force cases
"""
# n_elements = 20
# length = 3
# density = 5e3
# radius = 0.25
# direction = np.array([0, 0, 1])
# normal = np.array([0, 1, 0])
# youngs_modulus = 1e6
# shear_modulus = 1e4
# dt = 3e-4
# total_time = 10
# dissipation_constant = 0.1
#
# rod = CosseratRod(number_of_elements=n_elements, total_length=length, density=density, radius=radius, \
#                   direction=direction,normal=normal, youngs_modulus=youngs_modulus, \
#                   dt=dt, total_time=total_time, dissipation_constant = dissipation_constant, \
#                   shear_modulus=shear_modulus, fixed_BC=True)
#
# timoshenko_force = TimoshenkoForce(applied_force= [-15,0,0])
# rod.add_forces_torques(timoshenko_force)
# rod.run()
# dynamic_plotting(rod.callback_params)
"""
Snake case
"""
# n_elements = 5 #target 49
# length = 1
# density = 5e3
# radius = 0.025
# direction = np.array([0, 0, 1])
# normal = np.array([0, 1, 0])
# youngs_modulus = 1e7
# shear_modulus = 2 * youngs_modulus / 3
# dt = 2.5e-5
# total_time = 10
# dissipation_constant = 5
# muscle_activation_period = 1
# gravity_acceleration = 9.81
# #anisotropic frictions are built in
# #threshold velocitiey is built in
# wall_stiffness = 1
# ground_dissipation = 1e-6
# lambda_m = 0.5 #wavelength, sub m to not be confused with lambda function
# b_coeffs = np.array([0, 25, 25, 25, 25, 0])
#
# snake = CosseratRod(number_of_elements=n_elements, total_length=length, density=density, radius=radius, \
#                   direction=direction,normal=normal, youngs_modulus=youngs_modulus, \
#                   dt=dt, total_time=total_time, dissipation_constant = dissipation_constant, \
#                   shear_modulus=shear_modulus, fixed_BC=False, stretchable=False)
#
# muscle_torques = MuscleTorques(b_coeff=b_coeffs, period=muscle_activation_period, direction=normal,\
#                                wave_length=lambda_m,rest_lengths=snake.reference_lengths)
# #add force to rod
# snake.add_forces_torques(muscle_torques)

# snake.run()
# dynamic_plotting(snake.callback_params)
"""
Butterfly
"""
final_time = 10

dt = 0.01 # time-step

n_elem = 4  # Change based on requirements, but be careful
n_elem += n_elem % 2
half_n_elem = n_elem // 2

origin = np.zeros((3, 1))
angle_of_inclination = np.deg2rad(45.0)

# in-plane
horizontal_direction = np.array([0.0, 0.0, 1.0]).reshape(-1, 1)
vertical_direction = np.array([1.0, 0.0, 0.0]).reshape(-1, 1)

# out-of-plane
normal = np.array([0.0, 1.0, 0.0])

total_length = 3.0
base_radius = 0.25
base_area = np.pi * base_radius ** 2
density = 5000
youngs_modulus = 1e6
poisson_ratio = 0.5
shear_modulus = 10**6
dissipation_constant = 50
positions = np.empty((3, n_elem + 1))
dl = total_length / n_elem
direction = np.array([0,0,1])
# First half of positions stem from slope angle_of_inclination
first_half = np.arange(half_n_elem + 1.0).reshape(1, -1)
positions[..., : half_n_elem + 1] = origin + dl * first_half * (
    np.cos(angle_of_inclination) * horizontal_direction
    + np.sin(angle_of_inclination) * vertical_direction
)
positions[..., half_n_elem:] = positions[
    ..., half_n_elem : half_n_elem + 1
] + dl * first_half * (
    np.cos(angle_of_inclination) * horizontal_direction
    - np.sin(angle_of_inclination) * vertical_direction
)

butterfly = CosseratRod(number_of_elements=n_elem, total_length=total_length, density=density, radius=base_radius, \
                  direction=direction,normal=normal, youngs_modulus=youngs_modulus, \
                  dt=dt, total_time=final_time, dissipation_constant = dissipation_constant, \
                  shear_modulus=shear_modulus, fixed_BC=False, stretchable=True)
butterfly.positions = positions
butterfly.tangents = butterfly.positions[:, 1:] - butterfly.positions[:, :-1]
butterfly.tangents = np.divide(butterfly.tangents, butterfly.current_lengths)
for idx in range(butterfly.n_elements):
    butterfly.d1 = normal
    butterfly.d3 = butterfly.tangents[:, idx]
    butterfly.d2 = np.cross(butterfly.d3, butterfly.d1)  # binormal vectors
    butterfly.Q[0, :, idx] = butterfly.d1  # d1
    butterfly.Q[1, :, idx] = butterfly.d2  # d2
    butterfly.Q[2, :, idx] = butterfly.d3  # d3

butterfly.run()
# plot_video(butterfly.callback_params,xlim=[-1,3],ylim=[-0.1,1.1],every=10)