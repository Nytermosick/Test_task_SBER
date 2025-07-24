import pinocchio as pin
import pinocchio.casadi as cpin
import casadi as ca
import numpy as np
from typing import List, Tuple

class MPC:
    """
    Класс для построения и решения задачи управления в виде MPC (Model Predictive Control)
    для манипуляторов с использованием Pinocchio + CasADi.

    Поддерживает:
    - учёт динамики манипулятора (через CRBA и RNEA),
    - интеграцию методом Рунге-Кутты,
    - ограничение по углам, скоростям и управляющим воздействиям,
    - штрафы на сглаженность, усилия, отслеживание, начальное управление,
    - ограничения на проваливание звеньев под "пол".
    """

    def __init__(self, robot_path: str, log_path: str, idx: int,
                 q_bounds: List[int|float], v_bounds: List[int|float], u_bounds: List[int|float],
                 dt: float, N: int = 10,
                 q_penalty: int|float =1e-1, v_penalty: int|float =1e-2,
                 u_penalty: int|float =1e-4, u_delta_penalty: int|float =1e-3, initial_control_penalty: int|float =1e-3,
                 tracking_penalty: int|float =1e-1, terminal_penalty: int|float =500.0) -> None:
        """
        Инициализация класса MPC с загрузкой модели, построением динамики и кинематики.

        Args:
            robot_path (str): Путь к URDF или MJCF модели робота.
            log_path (str): Папка для логирования.
            idx (int): Индекс первого звена, c которого начнут учитываться ограничения при защите от проваливания и обхода препятствий - (для планарника - 2, для ur10 - 1).
            q_bounds, v_bounds, u_bounds (List[float]): Ограничения на состояния и управления.
            dt (float): Шаг дискретизации MPC.
            N (int): Длина горизонта предсказания.
            *_penalty (float): Вес соответствующего штрафа в целевой функции.
        """

        # Инициализация MPC
        self.q_bounds = q_bounds
        self.v_bounds = v_bounds
        self.u_bounds = u_bounds

        self.dt = dt
        self.N = N

        self.penalties = {"q": q_penalty, "v": v_penalty,
                          "u": u_penalty, "u_delta": u_delta_penalty, "init_u": initial_control_penalty,
                          "tracking": tracking_penalty, "terminal": terminal_penalty}

        # Инициализация модели робота
        ext = robot_path.split(".")[-1]
        if ext == "urdf":
            model = pin.buildModelFromUrdf(robot_path)
        elif ext == "xml":
            model = pin.buildModelFromMJCF(robot_path)

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

        # Вынужденный костыль, чтобы просчитывать прямую кинематику для звеньев. Указывается тот индекс, с которого будет рассчитываться прямая кинематика,
        # чтобы учесть их z-координату в ограничениях и исключить проваливание под пол
        self.idx = idx 

        # Построение графа вычислений для z-координат линков, чтобы задать ограничения для избежания проваливания линков под пол
        self.calculate_z = self.FK()

        # Папка логгирования
        self.log_path = log_path

        # Изначально False, но можно поменять на True, вызвав функцию enable_obstacle_avoid и передав туда координаты препятствия.
        # Тогда будет учитываться обхождение препятствий в задаче оптимизации
        self.obstacle_avoid = False

    
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
        self.link_frame_ids = [fid for fid, frame in enumerate(self.model.frames)
                  if frame.type == pin.FrameType.BODY]
        self.link_frame_ids = self.link_frame_ids[self.idx:] # Вынужденный костыль, чтобы не учитывать линки, у которых z координата всегда в нуле
        
        # Вызываем полную прямую кинематику
        cpin.framesForwardKinematics(self.model, self.data, self.q)

        # Собираем z-координаты линков
        z_positions = []
        for fid in self.link_frame_ids:
            pos = self.data.oMf[fid].translation
            z_positions.append(pos[2])

        # Одна функция, возвращающая вектор всех нужных z-координат
        return ca.Function("link_z_positions", [self.q], [ca.vertcat(*z_positions)])
    
    
    def enable_obstacle_avoid(self, obstacle_coords: List[np.ndarray], min_dist: int|float =0.3) -> None:
        """
        Включает учёт препятствий в задаче MPC.

        Args:
            obstacle_coords (List[np.ndarray]): Список координат препятствий (в мировой системе координат).
            min_dist (float): Минимальное допустимое расстояние между звеньями и препятствиями.
        """

        self.obstacle_coords = obstacle_coords
        self.obstacle_avoid = True

        self.obstacle_avoid_func = self.obstacle_avoidance(min_dist)
    

    def obstacle_avoidance(self, min_distance: float) -> ca.Function:
        """
        Строит FK "на лету" и возвращает CasADi-функцию, обеспечивающую обход препятствий.
        """

        # Создаём символьную переменную для q
        q_sym = ca.SX.sym("q", self.nq)
        data_tmp = self.model.createData()

        # Строим прямую кинематику в CasADi стиле
        cpin.framesForwardKinematics(self.model, data_tmp, q_sym)

        positions = []
        for fid in self.link_frame_ids:
            cpin.updateFramePlacement(self.model, data_tmp, fid)
            pos_expr = data_tmp.oMf[fid].translation
            positions.append(pos_expr)

        all_positions = ca.vertcat(*positions)

        constraints = []
        for obs_pos in self.obstacle_coords:
            obs = ca.DM(obs_pos)
            for i in range(len(self.link_frame_ids)):
                link_pos = all_positions[3*i : 3*(i+1)]
                dist = ca.norm_2(link_pos - obs)
                constraints.append(dist - min_distance)

        return ca.Function("obstacle_avoid", [q_sym], [ca.vertcat(*constraints)])


    def mpc_step(self, x_0: np.ndarray, u_0: np.ndarray, x_des: np.ndarray) -> Tuple[np.ndarray]:
        """
        Выполняет один MPC шаг: решает задачу оптимизации и возвращает оптимальные траектории состояний и управляющих воздействий.

        Args:
            x_0 (np.ndarray): Текущее состояние (q, v).
            u_0 (np.ndarray): Предыдущее управляющее воздействие.
            x_des (np.ndarray): Желаемое состояние (q, v) в конце горизонта.

        Returns:
            Tuple[np.ndarray]: Оптимальные траектории состояний (X) и управлений (U) длиной N+1 и N соответственно.
        """

        problem = ca.Opti()

        U = problem.variable(self.nv, self.N)
        X = problem.variable(self.nx, self.N + 1)

        q_min, q_max = self.q_bounds
        v_min, v_max = self.v_bounds
        u_min, u_max = self.u_bounds

        # Ограничения на состояния и управления
        problem.subject_to(problem.bounded(q_min, X[:self.nq, :], q_max))
        problem.subject_to(problem.bounded(v_min, X[self.nq:, :], v_max))
        problem.subject_to(problem.bounded(u_min, U, u_max))

        # Начальное состояние
        problem.subject_to(X[:, 0] == x_0)

        cost = 0

        for k in range(self.N):
            next_state = self.integrator(self.dynamics, X[:, k], U[:, k], self.dt)
            problem.subject_to(X[:, k + 1] == next_state)

            q_k = X[:self.nq, k]
            v_k = X[self.nq:, k]
            u_k = U[:, k]

            # Основные штрафы
            cost += ca.sumsqr(u_k) * self.penalties["u"]
            cost += ca.sumsqr(q_k) * self.penalties["q"]
            cost += ca.sumsqr(v_k) * self.penalties["v"]
            cost += ca.sumsqr(X[:self.nq, k + 1] - q_k) * self.penalties["q"]
            cost += ca.sumsqr(X[:, k] - x_des) * self.penalties["tracking"]

            # Ограничение на пол
            problem.subject_to(self.calculate_z(q_k) >= 0.01)

            # Обход препятствий
            if self.obstacle_avoid:
                problem.subject_to(self.obstacle_avoid_func(q_k) >= 0)

            # Плавность управляющих воздействий
            if k > 0:
                cost += ca.sumsqr(U[:, k] - U[:, k - 1]) * self.penalties["u_delta"]

        # Мягкий штраф на отклонение от цели (в конце горизонта)
        cost += ca.sumsqr(X[:, self.N] - x_des) * self.penalties["terminal"]

        # Мягкий штраф на стартовое управление
        cost += ca.sumsqr(U[:, 0] - u_0) * self.penalties["init_u"]

        # Решение
        problem.minimize(cost)
        problem.solver("ipopt")
        sol = problem.solve()

        x_res = sol.value(X)
        u_res = sol.value(U)

        return x_res, u_res