import numpy as np
from matplotlib import pyplot as plt
from matrix_operators import _batch_matmul, _batch_matvec, _batch_cross, _batch_matrix_transpose, _batch_norm
import copy
from tqdm import tqdm
import matplotlib.animation as manimation
from numba import njit



np.set_printoptions(linewidth = np.inf)



# Variables on elements
class CosseratRod():

    def __init__(self, number_of_elements, total_length, density, radius, direction, normal, youngs_modulus, dt,
                 total_time, dissipation_constant = 0.1, poisson_ratio=0.5, shear_modulus=0,  fixed_BC = False):
        # direction must be a unit vector
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
    
    def force_rule(self, positions):
        
        
        #do this for milestone 1 only
        self.external_forces[2, -1] = 15000
        
        
        
        #use positions to recalculate sigma
        self.positions = positions


        self.tangents = np.subtract(self.positions[:, 1:], self.positions[: ,:-1])
        self.current_lengths = _batch_norm(self.tangents)
        self.tangents = np.divide(self.tangents, self.current_lengths)
        self.current_voronoi_lengths = np.add(self.current_lengths[1:], self.current_lengths[:-1]) / 2
        
        #update dilatations
        self.element_dilataton = np.divide(self.current_lengths, self.reference_lengths)
        #update voronoi lengths
        self.voronoi_dilataton = np.divide(self.current_voronoi_lengths, self.reference_voronoi_lengths)
        
        #update sigma
        strain = np.subtract(np.multiply(self.element_dilataton, self.tangents), self.Q[2, :, :])
        self.sigma = _batch_matvec(self.Q, strain)
        #transpose Q
        Qt = _batch_matrix_transpose(self.Q)
        
        #apply damping force and torques
        self.external_forces = np.add(self.external_forces, - self.dissipation_constant * self.velocities) 
        self.external_torques = np.add(self.external_torques, - self.dissipation_constant * self.angular_velocities)
     
        #stretch/strain internal force
        stretch_force = np.divide(_batch_matvec(Qt, _batch_matvec(self.S, self.sigma)), self.element_dilataton)
        stretch_force = self.delta_h(stretch_force)
        
        #apply external forces and torques
        self.apply_forces_torques()
        
        #sum the force, divide by mass, get acceleration
        acceleration = np.divide(np.add(stretch_force, self.external_forces), self.mass)
        
        #bend/twist internal couple
        #cubic dilatation
        epsilon3 = np.power(self.voronoi_dilataton, 3)
        twist_bent_couple1 = self.delta_h(np.divide(_batch_matvec(self.B, self.kappa), epsilon3))
        twist_bend_couple2 = np.divide(_batch_cross(self.kappa, _batch_matvec(self.B, self.kappa)),\
            epsilon3)
        twist_bend_couple2 = self.alpha_h(np.multiply(twist_bend_couple2, self.reference_voronoi_lengths))
        twist_bend_couple = np.add(twist_bent_couple1, twist_bend_couple2)
        #shear/stretch internal couple
        Qt_t = _batch_matvec(self.Q, self.tangents)
        S_sigma = _batch_matvec(self.S, self.sigma)
        shear_stretch_couple = np.multiply(_batch_cross(Qt_t, S_sigma), self.reference_lengths)
        
        #summing torqus
        internal_torques = np.add(twist_bend_couple, shear_stretch_couple)
        #calculate alpha
        alpha = np.multiply(np.add(internal_torques, self.external_torques), self.element_dilataton)
        alpha = _batch_matvec(self.J_inv, alpha)

        #clear forces and torques
        self.external_forces.fill(0.0)
        self.external_torques.fill(0.0)
        
        return acceleration, alpha

        # return acceleration, angular_acceleration

    def apply_BC(self):
        if self.fixed_BC:
            self.positions[:, 0] = 0
            self.angular_velocities[:, 0] = 0
            self.velocities[:, 0] = 0
            self.Q[:, :, 0] = self.ref_Q[:, :, 0]
            self.Q_t_half_dt[:, :, 0] = self.ref_Q[:, :, 0]

    def add_forces_torques(self, forceobj):
        self.forces_torques_to_add.append(forceobj)

    def apply_forces_torques(self):

        for forceobj in self.forces_torques_to_add:
            try:
                forceobj.apply_forces()
            except AttributeError:
                pass
            try:
                forceobj.apply_torques()
            except AttributeError:
                pass

    def rodrigues(self, dt, omega, directors):
        n_elements = omega.shape[-1]
        R = np.zeros((3, 3, n_elements))
        norm_omega = _batch_norm(omega)
        for i in range(n_elements):
            angle = -0.5 * dt * norm_omega[i]
            if norm_omega[i] < 1e-14:
                about = np.zeros((3,))
            else:
                about = omega[:, i] / norm_omega[i]
            K = np.array([[0, -about[2], about[1]], [about[2], 0, -about[0]], [-about[1], about[0], 0]])


            R[:, :, i] = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

            directors[:, :, i] = R[:, :, i] @ directors[:, :, i]
        return directors

    def update(self):
        self.make_callback()
        self.position_verlet()
        self.update_kappa()
    
    def update_kappa(self):
        """ Finds the angle and axes of rotation given any two frames

        Parameters
        ----------
        t_frameone : frame/np.array
            If frame object, then t_frame is given by the process function of
            the frame
            Else just a numpy array (vector/collection of vectors) which you
            want to find the angle of
        t_frametwo : frame/np.array
            If frame object, then t_frame is given by the process function of
            the frame
            Else just a numpy array (vector/collection of vectors) which you
            want to find the angle of
        Both obey t_frametwo = R @ t_frameone

        Returns
        -------
        angle : float
            Returns the angle about which the frame rotates
        axis : list/float like
            Returns the axis about which the frames differ
        """
        for i in range(self.n_elements - 1):
            R = self.Q[:, :, i + 1] @ self.Q[:, :, i].T
            angle = np.arccos(0.5 * np.trace(R) - 0.5 - 1e-10)
            K = (R - R.T) * 0.5 / np.sin(angle + 1e-14)
            self.kappa[:, i] = np.array([-K[1, 2], K[0, 2], -K[0, 1]]) * angle

    def run(self):
        steps = int(self.total_time // self.dt)
        for step in tqdm(range(steps)):
            rod.update()
    
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
        # update half step position
        self.half_step_r = np.add(self.positions, 0.5 * self.dt * self.velocities)
        # update half step Q
        self.Q_t_half_dt = self.rodrigues(self.dt, self.angular_velocities, self.Q)

        self.apply_BC()
        # obtain half step acceleration

        half_step_acceleration, half_step_angular_acceleration = self.force_rule(self.half_step_r)

        # update velocity
        self.velocities = self.velocities + self.dt * half_step_acceleration

        # update angular velocity
        self.angular_velocities = self.angular_velocities + self.dt * half_step_angular_acceleration

        self.apply_BC()

        # update position
        self.positions = np.add(self.half_step_r, 0.5 * self.dt * self.velocities)

        # update Q
        self.Q = self.rodrigues(self.dt, self.angular_velocities, self.Q_t_half_dt)

        self.apply_BC()



def dynamic_plotting(ploting_parameters: dict):
    wait = 0.01
    figure, ax = plt.subplots()
    ax.set_xlim([0,4])
    ax.set_ylim([-2,1])

    steps = len(ploting_parameters["positions"])
    plt.xlabel("Distance along lab frame z")
    plt.ylabel("Deflection in lab frame x")
    plt.title("Deflection of a rod under torque")
    positions = ploting_parameters["positions"]
    line1 = ax.plot(positions[0][2], positions[0][0])
    for step in tqdm(range(0, steps, 10)):
        line1[0].set_xdata(positions[step][2])
        line1[0].set_ydata(positions[step][0])
        figure.canvas.draw()
        figure.canvas.flush_events()
        plt.pause(wait)  # is necessary for the plot to update for some reason

    plt.show()

# @njit(cache = True)
def dynamic_plotting_v2(ploting_parameters: dict):
    wait = 0.0005
    figure, ax = plt.subplots()
    steps = len(ploting_parameters["positions"])
    plt.xlabel("Distance along lab frame z")
    plt.ylabel("Deflection in lab frame x")
    plt.title("Deflection of a rod under torque")
    positions = ploting_parameters["positions"]
    for step in tqdm(range(0, steps, 10)):
        ax.plot(positions[step][2], positions[step][0])
        plt.pause(wait)  # is necessary for the plot to update for some reason

    plt.show()

# @njit(cache = True)
def plot_video(
    plot_params: dict,
    video_name="video.mp4",
    fps=100,
    xlim=(0, 4),
    ylim=(-1, 1),
):  # (time step, x/y/z, node)


    positions_over_time = np.array(plot_params["positions"])

    print("plot video")
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
        for step in tqdm(range(0, steps, 5)):
            rod_lines_2d.set_xdata(positions_over_time[step][2])
            rod_lines_2d.set_ydata(positions_over_time[step][0])
            writer.grab_frame()

    # Be a good boy and close figures
    # https://stackoverflow.com/a/37451036
    # plt.close(fig) alone does not suffice
    # See https://github.com/matplotlib/matplotlib/issues/8560/
    plt.close(plt.gcf())

n_elements = 100
length = 3
density = 5e3
radius = 0.25
direction = np.array([0, 0, 1])
normal = np.array([0, 1, 0])
youngs_modulus = 1e6
shear_modulus = 1e4
dt = 3e-4
total_time = 200
dissipation_constant = 0

rod = CosseratRod(number_of_elements=n_elements, total_length=length, density=density, radius=radius, \
                  direction=direction,normal=normal, youngs_modulus=youngs_modulus, \
                  dt=dt, total_time=total_time, dissipation_constant = dissipation_constant, \
                  shear_modulus=shear_modulus, fixed_BC=True)


rod.run()
dynamic_plotting(rod.callback_params)

