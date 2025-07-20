import pinocchio.casadi as cpin
import pinocchio as pin
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Tuple

class TrajectoryOptimization:
    """
    Класс для оптимизации траектории манипулятора с использованием Pinocchio и CasADi.
    Поддерживает построение динамики, прямую кинематику, интеграцию по RK4,
    формулировку задачи оптимизации, добавление ограничений и визуализацию результатов.
    """

    def __init__(self, urdf_path: str) -> None:
        """
        Инициализирует класс TrajectoryOptimization.

        Args:
            urdf_path (str): Путь до urdf-модели робота.
        """

        # Инициализация модели
        model = pin.buildModelFromUrdf(urdf_path)
        self.model = cpin.Model(model)
        self.data = self.model.createData()

        # Размерности
        self.nq, self.nv, self.nu = model.nq, model.nv, model.nv
        self.nx = self.nq + self.nv

        # Определение символьных переменных
        self.q = ca.SX.sym("q", self.nq)
        self.v = ca.SX.sym("v", self.nv)
        self.u = ca.SX.sym("u", self.nu)

        self.x = ca.vertcat(self.q, self.v) # Вектор состояний

        # Построение графа вычислений для динамики манипулятора
        self.dynamics = self.calculate_dynamics()

        # Выбор интегратора
        self.integrator = self.rk4_step # Наверное можно добавить выбор и  других интеграторов (неявные?), но явный рунге-кутта тоже сойдёт

        # Построение графа вычислений для z-координат линков, чтобы задать ограничения для избежания проваливания линков под пол
        self.calculate_z = self.FK()


    def calculate_dynamics(self) -> ca.Function:
        """
        Вычисляет правую часть уравнения движения манипулятора в виде функции CasADi.

        Returns:
            ca.Function: Символьная функция f(x, u) → dx, где x = [q; v]
        """

        M = cpin.crba(self.model, self.data, self.q) # Получение матрицы инерции через алгоритм CRBA (Composite Rigid Body Algorithm)
        nle = cpin.rnea(self.model, self.data, self.q, self.v, ca.SX.zeros(self.nv)) # Получение матрицы нелинейных эффектов через RNEA (Recoursive Newton-Euler Algorithm)
        a = ca.solve(M, self.u - nle)
        dx = ca.vertcat(self.v, a) # Вектор производной состояния

        return ca.Function("f", [self.x, self.u], [dx])
    

    @staticmethod
    def rk4_step(f: ca.Function, xk: ca.SX, uk: ca.SX, dt: float) -> ca.SX:
        """
        Выполняет один шаг интеграции методом Рунге-Кутта 4-го порядка.

        Args:
            f (ca.Function): Функция правой части динамики f(x, u)
            xk (ca.SX): Состояние на текущем шаге
            uk (ca.SX): Управление на текущем шаге
            dt (float): Величина временного шага

        Returns:
            ca.SX: Состояние на следующем временном шаге
        """

        k1 = f(xk, uk)
        k2 = f(xk + dt/2 * k1, uk)
        k3 = f(xk + dt/2 * k2, uk)
        k4 = f(xk + dt * k3, uk)

        return xk + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    

    def FK(self) -> ca.Function:
        """
        Формирует CasADi-функцию, возвращающую z-координаты тел манипулятора,
        кроме базовых, в мировой системе координат.

        Returns:
            ca.Function: Функция z(q) → [z_link2, z_link3, ...]
        """

        # Собираем id фреймов линков для выражения z-координат линков
        link_frame_ids = [fid for fid, frame in enumerate(self.model.frames)
                  if frame.type == pin.FrameType.BODY]
        link_frame_ids = link_frame_ids[2:] # Вынужденный костыль, чтобы не учитывать link0 и link1, у которых z координата всегда в нуле
        
        # Вызываем полную прямую кинематику
        cpin.framesForwardKinematics(self.model, self.data, self.q)

        # Собираем z-координаты линков
        z_positions = []
        for fid in link_frame_ids:
            pos = self.data.oMf[fid].translation
            z_positions.append(pos[2])

        # Одна функция, возвращающая вектор всех нужных z-координат
        return ca.Function("link_z_positions", [self.q], [ca.vertcat(*z_positions)])


    def opti_init(self, N: int, T: int|float, T_hold: int|float = 0.5,  x0: List[int|float] = None) -> None:
        """
        Инициализирует параметры задачи оптимизации: дискретизацию, переменные,
        начальные условия и тайм-холдинг для удержания цели.

        Args:
            N (int): Количество временных интервалов
            T (int | float): Общее время движения
            T_hold (int | float): Время удержания заданного положения
            x0 (List[float], optional): Начальное состояние
        """

        if x0 is None:
            x0 = [0] * self.nx

        self.N, self.T, self.dt = N, T, T/N

        self.T_hold = T_hold # Нужно для того, чтобы манипулятор ещё 0.5с держал заданное положение после прихода в него
        self.N_hold = int(self.T_hold/self.dt)


        # Инициализация оптимизационной задачи
        self.problem = ca.Opti()

        # Инициализация decision variables
        self.U = self.problem.variable(self.nv, self.N + self.N_hold)
        self.X = self.problem.variable(self.nx, self.N + self.N_hold + 1)

        # Инициализация приближения и начальных условий
        self.problem.set_initial(self.X, 0)
        self.problem.set_initial(self.U, 0)

        self.problem.subject_to(self.X[:,0]== x0)


    def add_bounds(self, q_bounds: List[int|float], v_bounds: List[int|float], u_bounds: List[int|float]) -> None:
        """
        Добавляет ограничения на состояния и управления.

        Args:
            q_bounds (List[float]): [q_min, q_max]
            v_bounds (List[float]): [v_min, v_max]
            u_bounds (List[float]): [u_min, u_max]
        """

        q_min, q_max = q_bounds
        v_min, v_max = v_bounds
        u_min, u_max = u_bounds
        
        self.problem.subject_to(self.problem.bounded(q_min, self.X[:self.nq, :], q_max)) # Ограничения на управляющие воздействия
        self.problem.subject_to(self.problem.bounded(v_min, self.X[self.nq:, :], v_max)) # Ограничения на скорости шарниров
        self.problem.subject_to(self.problem.bounded(u_min, self.U, u_max)) # # Ограничения на положения шарниров


    def solve(self, x_des: List[int|float], q_penalty: int|float = 1e-2, v_penalty: int|float = 1e-2, u_penalty: int|float = 1e-2) -> None:
        """
        Формулирует и решает задачу оптимизации траектории.

        Args:
            x_des (List[float]): Желаемое конечное состояние [q; v]
            q_penalty (float): Вес за отклонение углов
            v_penalty (float): Вес за скорости
            u_penalty (float): Вес за управляющее воздействие и его резкие изменения
        """

        for i in range(self.N, self.N+self.N_hold+1): # Добавление ограничений на удержание заданной конфигурации
            self.problem.subject_to(self.X[:,i] == x_des)  

        cost = 0

        # Формирование ограничений на каждый шаг
        for k in range(self.N+self.N_hold):
            next_state = self.integrator(self.dynamics, self.X[:,k], self.U[:, k], self.dt) # Расчёт следующего состояния
            self.problem.subject_to(self.X[:,k+1] == next_state) # Ограничение на динамику

            q_k = self.X[:self.nq, k]
            v_k = self.X[self.nq:, k]
            u_k = self.U[:, k]

            cost += ca.sumsqr(u_k) * u_penalty + ca.sumsqr(q_k) * q_penalty + ca.sumsqr(v_k) * v_penalty

            cost += ca.sumsqr(self.X[:self.nq,k+1] - q_k) * q_penalty # Ограничение, чтобы не было лишних телодвижений

            self.problem.subject_to(self.calculate_z(q_k) >= 0.01) # Ограничение, чтобы линки не проваливались под пол
            
            if k == 0:
                continue
            cost += ca.sumsqr(self.U[:,k] - self.U[:,k-1]) * u_penalty # Ограничение на плавность управляющих моментов


        self.problem.minimize(cost)
        self.problem.solver("ipopt") # Выбор решателя
        self.sol = self.problem.solve()   # Решение оптимизационной задачи

        self.x_res = self.sol.value(self.X)
        self.u_res = self.sol.value(self.U)

    
    def result(self) -> Tuple[np.ndarray]:
        """
        Извлекает оптимальное решение после вызова solve().

        Returns:
            Tuple[np.ndarray, np.ndarray]: Оптимальные траектории состояний и управляющих воздействий
        """

        return self.x_res, self.u_res
    
    def plot_trajectory_result(self) -> None:
        """
        Строит и сохраняет графики углов, скоростей и моментов управления на одном полотне.
        Результат сохраняется в файл logs/optimization_results.png
        """

        q = self.x_res[:self.nq]
        v = self.x_res[self.nq:]
        u = self.u_res

        os.makedirs("logs", exist_ok=True)

        # Построение графиков
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # Положения
        for i in range(self.nq):
            axs[0].plot(np.linspace(0, self.T+self.T_hold, self.N+self.N_hold+1), q[i])
        axs[0].set_ylabel("Joint angles (rad)")
        axs[0].legend([f"q{i+1}" for i in range(q.shape[1])])
        axs[0].grid(True)

        # Скорости
        for i in range(self.nv):
            axs[1].plot(np.linspace(0, self.T+self.T_hold, self.N+self.N_hold+1), v[i])
        axs[1].set_ylabel("Joint velocities (rad/s)")
        axs[1].legend([f"v{i+1}" for i in range(v.shape[1])])
        axs[1].grid(True)

        # Моменты
        for i in range(self.nu):
            axs[2].plot(np.linspace(0, self.T+self.T_hold, self.N+self.N_hold), u[i])
        axs[2].set_ylabel("Torques (Nm)")
        axs[2].set_xlabel("Time (s)")
        axs[2].legend([f"u{i+1}" for i in range(u.shape[1])])
        axs[2].grid(True)

        # Общий заголовок
        fig.suptitle("Optimization Results for 3R Manipulator")

        # Сохранение в файл
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig("logs/optimization_results.png")
        plt.close()