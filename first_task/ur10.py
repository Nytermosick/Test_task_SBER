from trajopt2simulation import TrajectoryOptimization
from trajopt2simulation import Simulator
from trajopt2simulation import PI


# Путь до mjcf файла
mjcf_path = "../robots/ur10/ur10.xml"

# Путь до папки логгирования
log_path = "logs/ur10"

# Параметры задачи
x0 = [0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0] # Начальный вектор состояний манипулятора [q, v]

x_des = [-PI/2, -PI/2, PI/2, -PI/3, PI/3, PI/3,
         0, 0, 0, 0, 0, 0] # Заданный вектор состояний манипулятора [q, v]

N = 100 # Длина горизонта
T = 1 # Время прихода в заданную позицию
T_hold = 0 # Время, которое манипулятор будет держать заданную позицию после прихода в неё

T_total = T + T_hold # Общее время моделирования

# Ограничения для углов, скоростей и управляющих моментов
q_bounds = [-PI, PI]
v_bounds = [-2, 2]
u_bounds = [[-330, -330, -150, -54, -54, -54], [330, 330, 150, 54, 54, 54]]


# Создание объекта оптимизации траектории
traj_opt = TrajectoryOptimization(robot_path=mjcf_path, log_path=log_path, idx=1)

# Инициализация оптимизации
traj_opt.opti_init(N=N, T=T, T_hold=T_hold, x0=x0)

# Добавление ограничений
traj_opt.add_bounds(q_bounds=q_bounds, v_bounds=v_bounds, u_bounds=u_bounds)

# Решение оптимизационной задачи
traj_opt.solve(x_des=x_des)

# Построение графика оптимизированной траектории (/logs/optimization_result)
traj_opt.plot_trajectory_result()

# Получение результатов оптимизации
x_opt, u_opt = traj_opt.result()


# Создание объекта симуляции в Муджоко
sim = Simulator(
    xml_path=mjcf_path,
    dt = traj_opt.dt,
    record_video=True,
    log_path = log_path
)

# Запуск симуляции
sim.run(sim_time=T_total, ctrl=u_opt)


# Построение графиков результатов симуляции
sim.plot_results(x_opt)