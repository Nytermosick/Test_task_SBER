from mpc2simulation import SimulatorPlanar as Simulator
from mpc2simulation import MPC, PI
import numpy as np

# Пути до urdf и mjcf файлов
urdf_path = "../robots/planar_3R/planar_3R.urdf"
mjcf_path = "../robots/planar_3R/planar_3R_obstacle.xml"

# Путь до папки логгирования
log_path = "logs/planar"

# Ограничения для углов, скоростей и управляющих моментов
q_bounds = [-PI, PI]
v_bounds = [-2, 2]
u_bounds = [-40, 40]

# Шаг дискретизации
dt = 0.01

# Функция контроллера, где будет решаться MPC на каждом временном шагу
def controller(q: np.ndarray, dq: np.ndarray, u_prev,  target) -> np.ndarray:
    
    # Вектор состояния на текущем временном шагу
    x_sim = np.concatenate([q, dq])

    # Получение оптимальной траектории для текущего временного шага
    _, u_opt = mpc.mpc_step(x_0=x_sim, u_0=u_prev, x_des=np.concatenate([target, np.zeros(3)]))

    # Передача и применение первого оптимального управляющего воздействия
    u_mpc = u_opt[:, 0]

    return u_mpc

# Создание объекта MPC
mpc = MPC(robot_path=urdf_path,
            log_path=log_path,
            idx=2,
            q_bounds=q_bounds,
            v_bounds=v_bounds,
            u_bounds=u_bounds,
            dt=dt)


# Создание объекта симуляции
sim = Simulator(
    xml_path=mjcf_path,
    log_path=log_path,
    dt = dt,
    record_video=False,
)

# Получаем координаты препятствий
obstacle_coords = sim.get_obstacles_coords(count_of_obstacles=2)

# # Включаем обход препятствий
mpc.enable_obstacle_avoid(obstacle_coords=obstacle_coords, min_dist=0.3)

sim.set_controller(controller) # Установка управляющего контроллера

sim.run(sim_time=20) # Запуск симуляции

#sim.plot_results() # Построение графиков результатов симуляции
