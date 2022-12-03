import numpy as np
from matplotlib import pyplot as plt
from matrix_operators import _batch_matmul, _batch_matvec, _batch_cross, _batch_matrix_transpose, _batch_norm
import copy
from tqdm import tqdm

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
        # moment of area
        self.I = np.zeros((3, self.n_elements))
        # moment of inertia
        self.J = np.zeros((3, self.n_elements))

        # initialize accelerations and velocities

        self.velocities = np.zeros((3, self.n_nodes))
        self.angular_velocities = np.zeros([3, self.n_elements])


        # nodal mass
        self.mass = np.zeros((self.n_nodes))
        self.mass[:-1] += 0.5 * self.element_mass
        self.mass[1:] += 0.5 * self.element_mass

        for element in range(self.n_elements):
            self.I[:, element] = self.area ** 2 / (4 * np.pi) * np.array([1, 1, 2])
            self.J[:, element] = self.I[:, element] * self.density * self.reference_lengths[0]

        # nodal positions
        self.positions = np.zeros((3, self.n_nodes))
        discretize_lengths = np.linspace(0, self.total_length, self.n_nodes)
        for node in range(self.n_nodes):
            self.positions[:, node] = discretize_lengths[node] * direction
        # Element vectors
        self.Q = np.zeros((3, 3, self.n_elements))  # directors
        self.tangents = self.positions[:, 1:] - self.positions[:, :-1]
        self.tangents /= _batch_norm(self.tangents)
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
        for element in range(self.n_elements):
            B1 = self.youngs_modulus * self.I[0, element]
            B2 = self.youngs_modulus * self.I[1, element]
            B3 = self.shear_modulus * self.I[2, element]
            S1 = self.alpha * self.shear_modulus * self.area
            S2 = self.alpha * self.shear_modulus * self.area
            S3 = self.youngs_modulus * self.area

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

        print(self.S)
    def apply_BC (self):
        if self.fixed_BC:
            self.positions[:,0] = 0
            self.Q[:, :, 0] = self.ref_Q[:, :, 0]
            self.Q_t_half_dt[:,:,0] = self.ref_Q[:,:,0]

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

    def force_rule(self, positions):
        # according to cosserate therory, solve for the linear and angular acceleartions
        # we basically need to update everything in order to solve for the right values

        # positions of the current time is given already

        self.positions = positions
        # with the positions, we need to update tangent and the length
        self.tangents = self.positions[:, 1:] - self.positions[:, :-1]
        self.current_lengths = _batch_norm(self.tangents)
        self.tangents /= self.current_lengths


        # update current dilataton
        self.element_dilataton = self.current_lengths / self.reference_lengths
        # update current voronoi lengths
        self.current_voronoi_lengths = (self.current_lengths[1:] + self.current_lengths[:-1]) / 2

        # update current voronoi dilataton
        self.voronoi_dilataton = self.current_voronoi_lengths / self.reference_voronoi_lengths

        Qt = _batch_matrix_transpose(self.Q)
        # we have everything we need, now we need to update sigma.

        self.sigma = _batch_matvec(self.Q, self.element_dilataton * self.tangents - self.Q[2,:,:])

        # sanity check:
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

        # apply external forces and torques!
        self.apply_forces_torques()



        # do this only for timoshenko!!!!!!

        self.external_forces[0 ,-1] = - 15




        # apply dissipaton
        self.external_forces += - self.dissipation_constant * self.velocities


        self.external_torques += - self.dissipation_constant * self.angular_velocities

        # first solve for acceleration:
        Qt_S_sigma_over_e = _batch_matvec(Qt, _batch_matvec(self.S,self.sigma)) / self.element_dilataton

        acceleration = (self.delta_h(Qt_S_sigma_over_e) + self.external_forces) / self.mass

        # then solve for angular acceleration:
        dilataton_cube = self.voronoi_dilataton ** 3
        B_kappa = _batch_matvec(self.B, self.kappa)

        B_kappa_over_epsilon_cube = B_kappa / dilataton_cube

        J_over_e = np.zeros((3, self.n_elements))
        for i in range(3):
            J_over_e[i,:] = self.J[i, :] / self.element_dilataton

        kappa_cross_B_kappa = _batch_cross(self.kappa, B_kappa)
        kappa_cross_B_kappa_over_epsilon_cube_D = (kappa_cross_B_kappa / dilataton_cube) * self.reference_voronoi_lengths

        Q_t_cross_S_sigma_l = _batch_cross(_batch_matvec(self.Q, self.tangents),
                                           _batch_matvec(self.S, self.sigma)) * self.reference_lengths


        diff_B_k_over_epsilon_3 = self.delta_h(B_kappa_over_epsilon_cube)
        diff_k_cross_B_over_epsilon_3 = self.alpha_h(kappa_cross_B_kappa_over_epsilon_cube_D)

        three_terms_together = diff_k_cross_B_over_epsilon_3 + diff_k_cross_B_over_epsilon_3 + Q_t_cross_S_sigma_l
        angular_acceleration = np.zeros([3, self.n_elements])

        for element in range(n_elements):
            angular_acceleration[0, element] = three_terms_together[0, element] / J_over_e[0, element]
            angular_acceleration[1, element] = three_terms_together[1, element] / J_over_e[1, element]
            angular_acceleration[2, element] = three_terms_together[2, element] / J_over_e[2, element]


        #clear force after each time step!
        self.external_forces[:,:] = 0
        self.external_torques[:,:] = 0

        return acceleration, angular_acceleration

    def update(self):
        self.make_callback()
        # update current force by specific force class
        # update current torque by specific torque class

        # position_verlet updates basically everything

        # update current Q
        # update current angular velocity
        # update current velocity
        # update current postion
        self.positions, self.velocities = self.position_verlet()
        # kappa
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
        half_step_r = self.positions + 0.5 * self.dt * self.velocities

        # update half step Q
        self.Q_t_half_dt = self.rodrigues(self.dt, self.angular_velocities, self.Q)

        self.apply_BC()
        # obtain half step acceleration

        half_step_acceleration, half_step_angular_acceleration = self.force_rule(half_step_r)
        # update velocity
        v_n = self.velocities + self.dt * half_step_acceleration

        # update angular velocity
        self.angular_velocities = self.angular_velocities + self.dt * half_step_angular_acceleration

        # update position
        x_n = half_step_r + 0.5 * self.dt * v_n

        # update Q
        self.Q = self.rodrigues(self.dt, self.angular_velocities, self.Q_t_half_dt)

        self.apply_BC()
        return x_n, v_n

def dynamic_plotting(ploting_parameters: dict):
    wait = 0.01
    figure, ax = plt.subplots()
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

def dynamic_plotting_v2(ploting_parameters: dict):
    wait = 0.001
    figure, ax = plt.subplots()
    steps = len(ploting_parameters["positions"])
    plt.xlabel("Distance along lab frame z")
    plt.ylabel("Deflection in lab frame x")
    plt.title("Deflection of a rod under torque")
    positions = ploting_parameters["positions"]
    for step in tqdm(range(0, steps, 5)):
        ax.plot(positions[step][2], positions[step][0])
        plt.pause(wait)  # is necessary for the plot to update for some reason

    plt.show()

def plot_video_2D(plot_params: dict, video_name="video.mp4", margin=0.2, fps = 15):
    from matplotlib import pyplot as plt
    import matplotlib.animation as manimation

    t = np.array(plot_params["time"])
    positions_over_time = np.array(plot_params["position"])
    total_time = int(np.around(t[-1], 1))
    total_frames = fps * total_time
    step = round(len(t) / total_frames)

    print("creating video -- this can take a few minutes")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Twisting Rod", artist="Raymond Huang", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.axis("equal")
    plt.xlabel("Distance along lab frame z")
    plt.ylabel("Deflection in lab frame x")
    plt.title("Deflection of a rod under torque")
    rod_lines_2d = ax.plot(
        positions_over_time[0][2], positions_over_time[0][0], linewidth=3
    )[0]
    with writer.saving(fig, video_name, dpi=500):
        with plt.style.context("seaborn-whitegrid"):
            for time in range(1, len(t), step):
                rod_lines_2d.set_xdata(positions_over_time[time][2])
                rod_lines_2d.set_ydata(positions_over_time[time][0])

                writer.grab_frame()
    plt.close(fig)

n_elements = 100
length = 3
density = 5e3
radius = 0.25
direction = np.array([0, 0, 1])
normal = np.array([0, 1, 0])
youngs_modulus = 1e6
shear_modulus = 1e4
dt = 3e-4
total_time = 0.5
dissipation_constant = 0.1

rod = CosseratRod(number_of_elements=n_elements, total_length=length, density=density, radius=radius, \
                  direction=direction,normal=normal, youngs_modulus=youngs_modulus, \
                  dt=dt, total_time=total_time, dissipation_constant = dissipation_constant, \
                  shear_modulus=shear_modulus, fixed_BC=True)

rod.run()
dynamic_plotting(rod.callback_params)

