import pinocchio as pin
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from trajopt import TrajectoryOptimization
from mj_simulator import Simulator
import mujoco as mj
import mediapy as media
from pathlib import Path
import os

urdf_path = "../../robots/planar_3R/planar_3R.urdf"
mjcf_path = "../../robots/planar_3R/planar_3R.xml"

traj_opt = TrajectoryOptimization(urdf_path)

x0 = [0, 0, 0, 0, 0, 0]
x_des = [ca.pi/6, ca.pi/6, ca.pi/6, 0, 0, 0]

N = 100
T = 2

traj_opt.opti_init(N, T, 0.5, x0)

q_bounds = [-np.pi, np.pi]
v_bounds = [-2, 2]
u_bounds = [-30, 30]

traj_opt.add_bounds(q_bounds=q_bounds, v_bounds=v_bounds, u_bounds=u_bounds)

traj_opt.solve(x_des=x_des)

traj_opt.plot_trajectory_result()

x_opt, u_opt = traj_opt.result()


sim = Simulator(
    xml_path=mjcf_path,
    dt = traj_opt.dt,
    record_video=False
)

sim.run(ctrl=u_opt)


# # === Визуализация углов суставов ===
# plt.figure(figsize=(8, 5))
# for i in range(model_mjc.nq):
#     plt.plot(np.linspace(0, traj_opt.T+traj_opt.T_hold, traj_opt.N+traj_opt.N_hold), q[:, i], label=f"q_{i}")
# plt.title("Углы суставов UR5e во времени")
# plt.xlabel("Время [с]")
# plt.ylabel("Угол [рад]")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# q_actual = np.array(q)                # из MuJoCo
# q_expected = x_opt[:model_mjc.nq, :N].T   # из CasADi OCP

# import matplotlib.pyplot as plt

# for i in range(model_mjc.nq):
#     plt.figure()
#     plt.plot(q_expected[:, i], label=f"q_opt_{i}")
#     plt.plot(q_actual[:, i], "--", label=f"q_sim_{i}")
#     plt.title(f"Сустав {i}")
#     plt.xlabel("шаг")
#     plt.ylabel("угол [рад]")
#     plt.legend()
#     plt.grid(True)
#     plt.show()