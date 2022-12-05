import numpy as np
from bspline import snake_bspline
from matrix_operators import _batch_matmul, _batch_matvec, _batch_cross, _batch_dot, _batch_product_i_k_to_ik, inplace_addition, inplace_substraction

# wall response function
class AnisotropicFricton():
    def __init__(self, wall_stiffness, dissipation_coefficient, wall_normal_direction, wall_origin):
        self.wall_stiffness = wall_stiffness
        self.dissipation_coefficient = dissipation_coefficient
        self.wall_normal_direction = wall_normal_direction
        self.wall_origin = wall_origin



    def calc_anisotropic_coefficient(self, v_mag_along_longtitude, n_elements):
        # find force projection along axis and determine where the snake is going
        threshold = 10e-8
        fric_coeff = np.zeros([n_elements])
        fwd_kinetic = 1.1019368
        is_static = np.empty([n_elements])

        for element in range(n_elements):
            # kinetic case
            if abs(v_mag_along_longtitude[element]) > threshold:
                is_static[element] = False
                # forward
                if v_mag_along_longtitude[element] > 0:
                    fric_coeff[element] = fwd_kinetic
                # backward
                else:
                    fric_coeff[element] = 1.5 * fwd_kinetic
            # static case
            else:
                is_static[element] = True
                # forward
                if v_mag_along_longtitude[element] >= 0:
                    fric_coeff[element] = 2 * fric_coeff
                else:
                    fric_coeff[element] = 1.5 * fwd_kinetic
        return fric_coeff, is_static

    def element_positions(self, positions):
        # converts node positions to element positions
        n_elements = positions.shape[1] - 1
        e_positions = np.empty([3, n_elements])

        for element in range(n_elements):
            e_positions[:, element] = 0.5 * (positions[:, element] + positions[:, element + 1])
        return e_positions

    def element_velocities(self, velocities):
        n_elements = velocities.shape[1] - 1
        e_velocities = np.empty([3, n_elements])

        for element in range(n_elements):
            e_velocities[:, element] = 0.5 * (velocities[:, element] + velocities[:, element + 1])
        return e_velocities

    def wall_response(self, wall_stiffness, dissipation_coefficient, wall_normal_direction, resultant_forces, velocities, wall_origin,
                      positions, radius):
        # project force onto normal direction
        if wall_normal_direction.shape[0] == 1: wall_normal_direction = wall_normal_direction.T
        if wall_origin.shape[0] == 1: wall_origin = wall_origin.T
        normal_force = np.dot(resultant_forces.T, wall_normal_direction).T * wall_normal_direction
        n_elements = normal_force.shape[1]
        # find penetration
        # find the projection of positions + radius into the wall
        e_positions = self.element_positions(positions)
        wall_origins = wall_origin * np.ones([1, n_elements])  # convert to array
        distance_from_plane = np.dot((e_positions - wall_origins).T, wall_normal_direction).T
        penetration = radius - distance_from_plane
        elastic_force = wall_stiffness * penetration * wall_normal_direction

        e_veloities = self.element_velocities(velocities)
        damping_force = dissipation_coefficient * np.dot(e_veloities.T, wall_normal_direction).T * wall_normal_direction
        wall_response_force = np.heaviside(penetration, 1) * (-normal_force + elastic_force - damping_force)

        return wall_response_force

    def longtitudinal_force(self, resultant_force, wall_response, velocities, tangents, wall_normal_direction):
        response_mag = np.linalg.norm(wall_response, axis = 0)

        if wall_normal_direction.shape[0] == 1: wall_normal_direction = wall_normal_direction.T
        #obtain element velocity
        e_velocities = self.element_velocities(velocities)

        n_elements = resultant_force.shape[1]
        long_force = np.zeros([3, n_elements])

        wall_normal_directions = wall_normal_direction * np.ones([1, n_elements])
        #first, we need to find how much of the tangent is parallel

        lateral_tangent_on_plane = np.cross(tangents, wall_normal_directions, axis = 0)
        longtitude_tangent_on_plane = np.cross(wall_normal_directions, lateral_tangent_on_plane, axis = 0)
       #velocity along longtitude
        v_along_longtitude = _batch_dot(longtitude_tangent_on_plane, e_velocities) * longtitude_tangent_on_plane
        v_mag_along_longtitude = np.linalg.norm(v_along_longtitude, axis = 0)

        #force along longtitude
        f_along_longtitude = _batch_dot(longtitude_tangent_on_plane, resultant_force) * longtitude_tangent_on_plane
        f_mag_along_longtitude = np.linalg.norm(f_along_longtitude, axis = 0)

        #we need to find the friction coefficients
        fric_coeff, is_static = self.calc_anisotropic_coefficient(v_mag_along_longtitude, n_elements)

        for element in range(n_elements):
            if is_static[element]:

                long_force[:, element] = -max(-f_mag_along_longtitude[element], fric_coeff[element] * response_mag[element]) * f_along_longtitude[:, element] / f_mag_along_longtitude[element]
            else:

                long_force[:, element] = -fric_coeff[element] * response_mag[element] * v_along_longtitude[:, element] / v_mag_along_longtitude[element]


        return long_force



class MuscleTorques():

    def __init__(
        self,
        b_coeff,
        period,
        wave_length,
        direction,
        rest_lengths,
    ):
        """

        Parameters
        ----------
        b_coeff: nump.ndarray
            1D array containing data with 'float' type.
            Beta coefficients for beta-spline.
        period: float
            Period of traveling wave.
        wave_length:
            length of traveling wave
        direction: numpy.ndarray
           1D (dim) array containing data with 'float' type. Muscle torque direction.

        """

        self.direction = direction  # Direction torque applied
        self.angular_frequency = 2.0 * np.pi / period
        self.wave_number = 2.0 * np.pi / wave_length


        # s is the position of nodes on the rod, we go from node=1 to node=nelem-1, because there is no
        # torques applied by first and last node on elements. Reason is that we cannot apply torque in an
        # infinitesimal segment at the beginning and end of rod, because there is no additional element
        # (at element=-1 or element=n_elem+1) to provide internal torques to cancel out an external
        # torque. This coupled with the requirement that the sum of all muscle torques has
        # to be zero results in this condition.
        self.s = np.cumsum(rest_lengths)
        self.s /= self.s[-1]

        assert b_coeff.size != 0, "Beta spline coefficient array (t_coeff) is empty"
        my_spline, ctr_pts, ctr_coeffs = snake_bspline(b_coeff)
        self.my_spline = my_spline(self.s)


    def apply_torques(self, system, time: np.float64 = 0.0):
        self.compute_muscle_torques(
            time,
            self.my_spline,
            self.s,
            self.angular_frequency,
            self.wave_number,
            self.direction,
            system.director_collection,
            system.external_torques,
        )

    def compute_muscle_torques(
        time,
        my_spline,
        s,
        angular_frequency,
        wave_number,
        direction,
        director_collection,
        external_torques,
    ):
        # From the node 1 to node nelem-1
        # Magnitude of the torque. Am = beta(s) * sin(2pi*t/T + 2pi*s/lambda + phi)
        # There is an inconsistency with paper and Elastica cpp implementation. In paper sign in
        # front of wave number is positive, in Elastica cpp it is negative.
        torque_mag = my_spline * np.sin(angular_frequency * time - wave_number * s)

        # Head and tail of the snake is opposite compared to elastica cpp. We need to iterate torque_mag
        # from last to first element.
        torque = _batch_product_i_k_to_ik(direction, torque_mag[::-1])
        inplace_addition(
            external_torques[..., 1:],
            _batch_matvec(director_collection, torque)[..., 1:],
        )
        inplace_substraction(
            external_torques[..., :-1],
            _batch_matvec(director_collection[..., :-1], torque[..., 1:]),
        )

normal = np.array([[0,0,1]])
normal = normal / np.linalg.norm(normal)
wall_origin = np.array([[0,0,0]])
radius = 1.1

positions = np.array([[0,0,0], [1,0,0], [2, 0, 0], [1,2,5]]).T


force = np.array([[0,0,-3], [0, 1, 2], [3,4,5]]).T

directions = force / np.linalg.norm(force, axis = 0)

velocities = np.array([[2,5,-3], [1,1,1], [3,5,2], [2,9,-8]]).T

wall_stiffness = 1
dissipation_coefficient = 0.2

obj = AnisotropicFricton(wall_stiffness, dissipation_coefficient, normal, wall_origin)

tangents = positions[:, 1:] - positions[:, :-1]
tangents = tangents / np.linalg.norm(tangents, axis = 0)

# print(obj.element_positions(positions), "position test")
# print(obj.element_velocities(velocities), "v test")
# print(obj.wall_response(wall_stiffness, dissipation_coefficient, normal, force, velocities, wall_origin, positions, radius))
wall_response = obj.wall_response(wall_stiffness, dissipation_coefficient, normal, force, velocities, wall_origin, positions, radius)
print(obj.longtitudinal_force(force, wall_response, velocities, tangents, normal))