import numpy as np
import os
import matplotlib.pyplot as plt
import mujoco
import mediapy as media
from typing import Dict, List, Tuple, Callable
import time
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

class SimulatorPlanar:
    """
    Класс-обёртка над MuJoCo API для симуляции и визуализации движения манипулятора.

    Позволяет выполнять пошаговую симуляцию модели, логировать состояния,
    записывать видео и сравнивать с оптимальной траекторией.
    """

    def __init__(self,
                 xml_path: str,
                 log_path: str,
                 dt: float,
                 width: int = 1920,
                 height: int = 1080,
                 record_video: bool = True) -> None:
        """
        Инициализирует симулятор MuJoCo из xml-файла.

        Args:
            xml_path (str): Путь к MJCF-файлу модели.
            dt (float): Шаг интегрирования (в секундах).
            width (int): Ширина рендера.
            height (int): Высота рендера.
            record_video (bool): Записывать ли видео симуляции.
        """
        
        # Загрузка MJCF модели
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.model.opt.timestep = dt
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4

        # Параметры моделирования
        self.dt = dt

        # Параметры видео
        self.fps = 1/dt

        self.width = width
        self.height = height

        self.frames = []

        # Создание папки логирования
        self.log_path = log_path
        if record_video:
            os.makedirs(self.log_path, exist_ok=True)

        self.record_video = record_video

        # Настройка рендера
        self.renderer = mujoco.Renderer(self.model, width=self.width, height = self.height)

        # Настройка таргета
        mocap_name = "target"
        self.mocap_id = self.model.body(mocap_name).mocapid[0]


    def target_coords(self) -> Tuple[int|float]:
        """
        Возвращает координаты (y, z) и ориентацию phi целевой точки (mocap body "target").

        Returns:
            Tuple[float, float, float]: y, z координаты и угол ориентации (в радианах)
        """

        y, z = self.data.mocap_pos[self.mocap_id].copy()[1:] # [y, z]

        quat = self.data.mocap_quat[self.mocap_id].copy()
        r = R.from_quat(quat)  # [x, y, z, w]
        euler = r.as_euler('xyz', degrees=False)  # Углы в радианах
        phi = euler[1] # Угол ориентации эндефектора

        return y, z, phi
    
    @staticmethod
    def target_IK(coords: Tuple[int|float], l1: int|float = 1, l2: int|float = 1, l3: int|float = 1) -> List[List]:
        """
        Аналитическое решение IK для планарного 3R манипулятора в плоскости yOz.

        Args:
            coords (Tuple): (y, z, phi) — координаты цели и ориентация
            l1, l2, l3 (float): длины звеньев

        Returns:
            List[List]: список допустимых решений [q1, q2, q3]
        """

        y, z, phi = coords
        y = -y # Инверсия в нужную сторону

        y_wrist = y - l3 * np.sin(phi)
        z_wrist = z - l3 * np.cos(phi)

        r2 = y_wrist**2 + z_wrist**2
        c2 = (r2 - l1**2 - l2**2) / (2 * l1 * l2)

        if abs(c2) > 1.0:
            return []  # Нет решения

        # Два возможных решения для q2
        s2_pos = np.sqrt(1 - c2**2)
        s2_neg = -s2_pos

        solutions = []

        for s2 in [s2_pos, s2_neg]:
            q2 = np.arctan2(s2, c2)
            k1 = l1 + l2 * np.cos(q2)
            k2 = l2 * np.sin(q2)
            q1 = np.arctan2(y_wrist, z_wrist) - np.arctan2(k2, k1)

            if q1 >= np.pi/2 or q1 <= -np.pi/2:
                continue 
            q3 = phi - q1 - q2

            solutions.append([q1, q2, q3])

        solutions = list(list(map(float, lst)) for lst in solutions) # Перевод в славянские флоаты
        
        return solutions
    

    @staticmethod
    def closest_solution(solutions: List[List], q_current: np.ndarray) -> np.ndarray:
        """
        Выбирает из набора IK-решений то, которое ближе к текущей конфигурации q_current.

        Args:
            solutions (List[List]): список решений IK
            q_current (np.ndarray): текущий вектор углов

        Returns:
            np.ndarray: ближайшее по норме решение, приведённое к диапазону [-π, π]
        """

        def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """
            Вычисляет разность углов с учётом цикличности (в диапазоне [-π, π]).

            Returns:
                np.ndarray: нормализованная разность
            """

            return np.arctan2(np.sin(a - b), np.cos(a - b))

        def wrap_to_pi(q: np.ndarray) -> np.ndarray:
            """
            Приводит углы к диапазону [-π, π).

            Returns:
                np.ndarray: нормализованные углы
            """

            return (q + np.pi) % (2 * np.pi) - np.pi

        best_q = None
        min_dist = float('inf')

        for q in solutions:
            q = np.array(q)
            delta = angle_diff(q, q_current)
            dist = np.linalg.norm(delta)
            if dist < min_dist:
                min_dist = dist
                best_q = q

        return wrap_to_pi(np.array(best_q))
    
    
    def get_obstacles_coords(self, count_of_obstacles: int) -> List[np.ndarray]:
        """
        Возвращает список координат препятствий с именами obstacle1, obstacle2, ..., obstacleN.
        Для этого вызывается mj_forward и читается data.xpos.

        Args:
            count_of_obstacles (int): количество препятствий (должны быть названы obstacle1, obstacle2, ...)

        Returns:
            List[np.ndarray]: список мировых координат центров препятствий
        """

        mujoco.mj_forward(self.model, self.data)
        return [self.data.xpos[self.model.body(name=f"obstacle{i+1}").id].copy() for i in range(count_of_obstacles)]

    def _save_video(self) -> None:
        """
        Сохраняет собранные кадры симуляции в видеофайл logs/vid.mp4.
        """

        if self.frames:
            print(f"Saving video to {self.log_path}/vid.mp4...")
            media.write_video(f"{self.log_path}/vid.mp4", self.frames, fps=self.fps)
            self.frames = []


    def _capture_frame(self) -> np.ndarray:
        """
        Делает снимок текущего состояния симуляции.

        Returns:
            np.ndarray: RGB кадр симуляции.
        """

        self.renderer.update_scene(self.data)
        pixels = self.renderer.render()
        return pixels.copy()


    def reset(self) -> None:
        """
        Сбрасывает состояние симуляции на начальное.
        """

        mujoco.mj_resetData(self.model, self.data)


    def get_state(self) -> Dict[str, np.ndarray]:
        """
        Возвращает текущее состояние манипулятора:
        - положения q
        - скорости dq
        - целевая конфигурация target (IK ближайшая)

        Returns:
            Dict[str, np.ndarray]: словарь с полями 'q', 'dq', 'target'
        """

        q = self.data.qpos.copy()

        target_solutions = self.target_IK(self.target_coords())
        target_best = self.closest_solution(target_solutions, q)

        state = {
            'q': q,
            'dq': self.data.qvel.copy(),
            'target': target_best
        }

        return state
    

    def _def_init_state(self, q0: List[int|float], dq0: List[int|float]) -> None:
        """
        Устанавливает начальное состояние симуляции.

        Args:
            q0 (List[float]): Начальные положения сочленений.
            dq0 (List[float]): Начальные скорости сочленений.
        """

        self.data.qpos = q0
        self.data.qvel = dq0
    
    
    def step(self, tau: np.ndarray) -> None:
        """
        Выполняет один шаг симуляции с заданным управляющим воздействием.

        Args:
            tau (np.ndarray): Вектор управляющих моментов.
        """
        
        self.data.ctrl = tau
        mujoco.mj_step(self.model, self.data)

        self.data.qvel[:] = np.clip(self.data.qvel, -2, 2) # Я так понял, на моторах в муджоко нельзя поставить ограничения по скорости, поэтому так (КОСТЫЛЬ)

    
    def set_controller(self, controller: Callable) -> None:
        """
        Устанавливает пользовательский контроллер.

        Args:
            controller (Callable): функция управления tau = f(q, dq, u_prev, target)
        """

        self.controller = controller


    def run(self, sim_time: int|float, q0: List[int|float] = None, dq0: List[int|float] = None) -> None:
        """
        Запускает интерактивную симуляцию с визуализацией, логированием и видео.

        Args:
            sim_time (float): длительность симуляции
            q0 (List[float]): начальные положения
            dq0 (List[float]): начальные скорости
        """

        viewer = mujoco.viewer.launch_passive(
            model = self.model,
            data = self.data,
            show_left_ui = False,
            show_right_ui = False
        )

        self.reset()

        # Для логирования
        self.pos = []
        self.vel = []
        self.controls = []
        self.times = []

        if q0 is None:
            q0 = [0] * self.model.nq
        if dq0 is None:
            dq0 = [0] * self.model.nv

        self._def_init_state(q0, dq0)

        # Начальная инициализация управляющих моментов
        u_prev = np.zeros(self.model.nv)

        mujoco.mjv_defaultFreeCamera(self.model, viewer.cam)

        try:
            t = 0
            start_time = time.perf_counter()

            while viewer.is_running():
                step = time.perf_counter()

                # Получение вектора состояния из симуляции
                state = self.get_state()

                tau = self.controller(state["q"], state["dq"], u_prev, state["target"])

                # Логирование
                self.pos.append(state['q'])
                self.vel.append(state['dq'])
                self.controls.append(tau)
                self.times.append(t)

                # Шаг симуляции
                self.step(tau)
                u_prev = tau

                if viewer:
                    viewer.sync()

                # Если нужно писать видео
                if self.record_video:
                    if len(self.frames) < self.fps * t:
                        self.frames.append(self._capture_frame())

                t += self.dt
                if sim_time and t >= sim_time:
                    break

                # Real-time синхронизация
                real_time = time.perf_counter() - start_time
                if t > real_time:
                    time.sleep(t - real_time)
                elif real_time - t > self.dt:
                    print(f"Warning: Simulation running slower than real-time by {real_time - t:.3f}s")

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        finally:
            if viewer:
                viewer.close()

            if self.record_video:
                self._save_video()

            self.renderer.close()
    

    def plot_results(self) -> None:
        """
        Строит графики результатов симуляции.
        """

        self.pos = np.array(self.pos)
        self.vel = np.array(self.vel)
        self.controls = np.array(self.controls)
        self.times = np.array(self.times)
        

        # Построение графиков действительных значений
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # Положения
        for i in range(self.model.nq):
            axs[0].plot(self.times, self.pos[:, i])
        axs[0].set_ylabel("Joint angles (rad)")
        axs[0].legend([f"q{i+1}" for i in range(self.pos.shape[1])])
        axs[0].grid(True)

        # Скорости
        for i in range(self.model.nv):
            axs[1].plot(self.times, self.vel[:, i])
        axs[1].set_ylabel("Joint velocities (rad/s)")
        axs[1].legend([f"v{i+1}" for i in range(self.vel.shape[1])])
        axs[1].grid(True)

        # Моменты
        for i in range(self.model.nu):
            axs[2].plot(self.times, self.controls[:, i])
        axs[2].set_ylabel("Torques (Nm)")
        axs[2].set_xlabel("Time (s)")
        axs[2].legend([f"u{i+1}" for i in range(self.controls.shape[1])])
        axs[2].grid(True)

        # Общий заголовок
        fig.suptitle("Real Results for 3R Manipulator")

        # Сохранение в файл
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"{self.log_path}/real_results.png")
        plt.close()