import numpy as np
from bspline import snake_bspline
from matrix_operators import _batch_matvec, _batch_dot, _batch_product_i_k_to_ik, inplace_addition, inplace_substraction
import copy
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
        element_resultant_forces = 0.5 * resultant_forces[:,1:] + 0.5 * resultant_forces[:,-1]
        element_normal_force = np.dot(element_resultant_forces.T, wall_normal_direction).T * wall_normal_direction
        n_elements = element_normal_force.shape[1]
        # find penetration
        # find the projection of positions + radius into the wall
        e_positions = self.element_positions(positions)
        wall_origins = wall_origin * np.ones([1, n_elements])  # convert to array
        distance_from_plane = np.dot((e_positions - wall_origins).T, wall_normal_direction).T
        penetration = radius - distance_from_plane
        elastic_force = wall_stiffness * penetration * wall_normal_direction

        e_veloities = self.element_velocities(velocities)
        damping_force = dissipation_coefficient * np.dot(e_veloities.T, wall_normal_direction).T * wall_normal_direction
        wall_response_force = np.heaviside(penetration, 1) * (-element_normal_force + elastic_force - damping_force)

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

    def apply_interactions(self, system):
        wall_response = self.wall_response(self.wall_stiffness,self.dissipation_coefficient,self.wall_normal_direction,\
                                           system.total_forces,system.velocities,self.wall_origin,system.positions, system.radius)
        inplace_addition(system.total_forces[:,1:], wall_response * 0.5)
        inplace_addition(system.total_forces[:,:-1], wall_response * 0.5)

class MuscleTorques():

    def __init__(
        self,
        b_coeff,
        period,
        wave_length,
        direction,
        rest_lengths,
    ):
        # direction, direction of the torque applied
        # b_coefficients, torque profile
        # siniusoidal, torque magnitude traversing the snake
        # wave_length, aka lambda
        self.direction = direction  # Direction torque applied
        self.angular_frequency = 2.0 * np.pi / period
        self.wave_number = 2.0 * np.pi / wave_length
        self.s = np.cumsum(copy.deepcopy(rest_lengths))
        self.s /= self.s[-1]

        my_spline = snake_bspline(b_coeff)
        self.my_spline = my_spline(self.s)


    def apply_torques(self, system):
        #get current time
        time = system.current_time
        #calculate torque magnitude
        torque_mag = self.my_spline * np.sin(self.angular_frequency * time - self.wave_number * self.s)

        torque = _batch_product_i_k_to_ik(self.direction, torque_mag)
        inplace_addition(
            system.external_torques[:, 1:],
            _batch_matvec(system.Q, torque)[..., 1:],
        )
        inplace_substraction(
            system.external_torques[..., :-1],
            _batch_matvec(system.Q[..., :-1], torque[..., 1:]),
        )

class GravityForces():

    def __init__(self, acc_gravity=np.array([0.0, -9.80665, 0.0])):
        self.acc_gravity = acc_gravity

    def apply_forces(self, system):
        self.compute_gravity_forces(
            system.mass, system.external_forces
        )

    def compute_gravity_forces(self, mass, external_forces):
        inplace_addition(external_forces, _batch_product_i_k_to_ik(self.acc_gravity, mass))

class TimoshenkoForce():
    def __init__(self, applied_force):
        self.applied_force = applied_force

    def apply_forces(self, system):
        system.external_forces[:, -1] += self.applied_force

# normal = np.array([[0,0,1]])
# normal = normal / np.linalg.norm(normal)
# wall_origin = np.array([[0,0,0]])
# radius = 1.1
#
# positions = np.array([[0,0,0], [1,0,0], [2, 0, 0], [1,2,5]]).T
#
#
# force = np.array([[0,0,-3], [0, 1, 2], [3,4,5], [3,2,4]]).T
#
# directions = force / np.linalg.norm(force, axis = 0)
#
# velocities = np.array([[2,5,-3], [1,1,1], [3,5,2], [2,9,-8]]).T
#
# wall_stiffness = 1
# dissipation_coefficient = 0.2
#
# obj = AnisotropicFricton(wall_stiffness, dissipation_coefficient, normal, wall_origin)
#
# tangents = positions[:, 1:] - positions[:, :-1]
# tangents = tangents / np.linalg.norm(tangents, axis = 0)

# print(obj.element_positions(positions), "position test")
# print(obj.element_velocities(velocities), "v test")
# print(obj.wall_response(wall_stiffness, dissipation_coefficient, normal, force, velocities, wall_origin, positions, radius))
# wall_response = obj.wall_response(wall_stiffness, dissipation_coefficient, normal, force, velocities, wall_origin, positions, radius)