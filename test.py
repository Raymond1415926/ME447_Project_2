from cosserat_rods import *
"""
Snake
"""

run_snake(wave_length=2.673, b_coeff=[0, 28.973, 41.110, 38.567, 31.226,  0],make_video=True, run_time=5,n_elements=10)

"""
Butterfly
"""
# final_time = 40
#
# dt = 0.01 # time-step
#
# n_elem = 4  # Change based on requirements, but be careful
# n_elem += n_elem % 2
# half_n_elem = n_elem // 2
#
# origin = np.zeros((3, 1))
# angle_of_inclination = np.deg2rad(45.0)
#
# # # in-plane
# horizontal_direction = np.array([0.0, 0.0, 1.0]).reshape(-1, 1)
# vertical_direction = np.array([1.0, 0.0, 0.0]).reshape(-1, 1)
#
# # out-of-plane
# normal = np.array([0.0, 1.0, 0.0])
#
# total_length = 3.0
# base_radius = 0.25
# base_area = np.pi * base_radius ** 2
# density = 5000
# youngs_modulus = 1e4
# poisson_ratio = 0.5
# shear_modulus = youngs_modulus/1.5
# dissipation_constant = 0
# positions = np.empty((3, n_elem + 1))
# dl = total_length / n_elem
# direction = np.array([0,0,1])
# # First half of positions stem from slope angle_of_inclination
# first_half = np.arange(half_n_elem + 1.0).reshape(1, -1)
# positions[..., : half_n_elem + 1] = origin + dl * first_half * (
#     np.cos(angle_of_inclination) * horizontal_direction
#     + np.sin(angle_of_inclination) * vertical_direction
# )
# positions[..., half_n_elem:] = positions[
#     ..., half_n_elem : half_n_elem + 1
# ] + dl * first_half * (
#     np.cos(angle_of_inclination) * horizontal_direction
#     - np.sin(angle_of_inclination) * vertical_direction
# )
#
# butterfly = CosseratRod(number_of_elements=n_elem, total_length=total_length, density=density, radius=base_radius, \
#                   direction=direction,normal=normal, youngs_modulus=youngs_modulus, \
#                   dt=dt, total_time=final_time, dissipation_constant = dissipation_constant, \
#                   shear_modulus=shear_modulus, fixed_BC=False)
# butterfly.positions = positions
# butterfly.tangents = butterfly.positions[:, 1:] - butterfly.positions[:, :-1]
# butterfly.tangents = np.divide(butterfly.tangents, butterfly.current_lengths)
# for idx in range(butterfly.n_elements):
#     butterfly.d1 = normal
#     butterfly.d3 = butterfly.tangents[:, idx]
#     butterfly.d2 = np.cross(butterfly.d3, butterfly.d1)  # binormal vectors
#     butterfly.Q[0, :, idx] = butterfly.d1  # d1
#     butterfly.Q[1, :, idx] = butterfly.d2  # d2
#     butterfly.Q[2, :, idx] = butterfly.d3  # d3
# gravity = GravityForces()
# butterfly.add_forces_torques(gravity)
# butterfly.run()
# dynamic_plotting_v2(butterfly.callback_params,xlim=[-1,3],ylim=[-0.5,1.5],every=100)
"""
Extra benchmark, gravity on, center node holding up
"""
# n_elements = 10
# length = 3
# density = 5e3
# radius = 0.25
# direction = np.array([0, 0, 1])
# normal = np.array([0, 1, 0])
# youngs_modulus = 1e6
# shear_modulus = 1e4
# dt = 3e-4
# total_time = 2
# dissipation_constant = 0.1
#
# rod = CosseratRod(number_of_elements=n_elements, total_length=length, density=density, radius=radius, \
#                   direction=direction,normal=normal, youngs_modulus=youngs_modulus, \
#                   dt=dt, total_time=total_time, dissipation_constant = dissipation_constant, \
#                   shear_modulus=shear_modulus, middle_BC=True)
#
# # timoshenko_force = TimoshenkoForce(applied_force= [-15,0,0])
# gravity = GravityForces(acc_gravity = np.array([-9.81, 0.0, 0.0]))
# rod.add_forces_torques(gravity)
# rod.run()
# plot_video(rod.callback_params)