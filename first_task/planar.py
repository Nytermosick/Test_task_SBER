from trajopt2simulation import TrajectoryOptimization
from trajopt2simulation import Simulator
from trajopt2simulation import PI



# Пути до urdf и mjcf файлов
urdf_path = "../robots/planar_3R/planar_3R.urdf"
mjcf_path = "../robots/planar_3R/planar_3R.xml"

# Путь до папки логгирования
log_path = "logs/planar"

# Параметры задачи
x0 = [0, 0, 0, 0, 0, 0] # Начальный вектор состояний манипулятора [q, v]
x_des = [PI/6, PI/6, PI/6, 0, 0, 0] # Заданный вектор состояний манипулятора [q, v]

N = 100 # Длина горизонта
T = 2 # Время прихода в заданную позицию
T_hold = 1 # Время, которое манипулятор будет держать заданную позицию после прихода в неё

T_total = T + T_hold # Общее время моделирования

# Ограничения для углов, скоростей и управляющих моментов
q_bounds = [-PI, PI]
v_bounds = [-2, 2]
u_bounds = [-30, 30]


# Создание объекта оптимизации траектории
traj_opt = TrajectoryOptimization(robot_path=urdf_path, log_path=log_path, idx=2)

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