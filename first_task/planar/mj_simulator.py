import numpy as np
import os
import matplotlib.pyplot as plt
import mujoco
import mediapy as media
from typing import Dict, List
import time
import mujoco.viewer
    

class Simulator:
    """
    Класс-обёртка над MuJoCo API для симуляции и визуализации движения манипулятора.

    Позволяет выполнять пошаговую симуляцию модели, логировать состояния,
    записывать видео и сравнивать с оптимальной траекторией.
    """

    def __init__(self,
                 xml_path: str,
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
        if record_video:
            os.makedirs("logs", exist_ok=True)

        self.record_video = record_video

        # Настройка рендера
        self.renderer = mujoco.Renderer(self.model, width=self.width, height = self.height)


    
    def _save_video(self) -> None:
        """
        Сохраняет собранные кадры симуляции в видеофайл logs/vid.mp4.
        """

        if self.frames:
            print(f"Saving video to logs/vid.mp4...")
            media.write_video("logs/vid.mp4", self.frames, fps=self.fps)
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
        Возвращает текущие обобщённые координаты и скорости.

        Returns:
            Dict[str, np.ndarray]: Словарь с полями 'q' и 'dq'.
        """

        state = {
            'q': self.data.qpos.copy(),
            'dq': self.data.qvel.copy(),
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


    def run(self, sim_time: int|float, ctrl: np.ndarray, q0: List[int|float] = None, dq0: List[int|float] = None) -> None:
        """
        Запускает симуляцию на заданное время с управлением ctrl.

        Args:
            sim_time (float): Время симуляции в секундах.
            ctrl (np.ndarray): Массив управляющих воздействий размером (n_u, T).
            q0 (List[float], optional): Начальные положения. По умолчанию — нули.
            dq0 (List[float], optional): Начальные скорости. По умолчанию — нули.
        """

        ctrl_copy = ctrl.copy()

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

        mujoco.mjv_defaultFreeCamera(self.model, viewer.cam)

        try:
            t = 0
            start_time = time.perf_counter()

            while viewer.is_running():
                step = time.perf_counter()

                # Получение вектора состояния из симуляции
                state = self.get_state()

                tau = ctrl_copy[:, 0]
                # try:
                ctrl_copy = ctrl_copy[:, 1:]
                # except:
                #     pass

                # Логирование
                self.pos.append(state['q'])
                self.vel.append(state['dq'])
                self.controls.append(tau)
                self.times.append(t)

                # Шаг симуляции
                self.step(tau)

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
    

    def plot_results(self, x_opt: np.ndarray = None) -> None:
        """
        Строит графики результатов симуляции и сравнивает с оптимальным решением (если передано).

        Args:
            x_opt (np.ndarray, optional): Оптимальное решение траектории (q и dq).
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
        plt.savefig("logs/real_results.png")
        plt.close()


        # Если передали результаты оптимизации, то будет построен график сравнения реальных результатов и оптимальных
        if x_opt is None:
            return
        
        q_opt = x_opt[:self.model.nq].T
        v_opt = x_opt[self.model.nq:].T

        fig, axes = plt.subplots(3, 2, figsize=(20, 15))

        for i in range(3):
            # Положения
            ax_angle = axes[i, 0]
            ax_angle.plot(self.times, self.pos[:, i], label="Real", linewidth=2)
            ax_angle.plot(self.times, q_opt[:-1, i], '--', label="Optimal", linewidth=2)
            ax_angle.set_ylabel(f'Joint {i+1} Angle [rad]')
            ax_angle.grid(True)
            if i == 0:
                ax_angle.set_title("Joint Angles")
            if i == 2:
                ax_angle.set_xlabel("Time [s]")

            # Скорости
            ax_vel = axes[i, 1]
            ax_vel.plot(self.times, self.vel[:, i], label="Real", linewidth=2)
            ax_vel.plot(self.times, v_opt[:-1, i], '--', label="Optimal", linewidth=2)
            ax_vel.set_ylabel(f'Joint {i+1} Velocity [rad/s]')
            ax_vel.grid(True)
            if i == 0:
                ax_vel.set_title("Joint Velocities")
            if i == 2:
                ax_vel.set_xlabel("Time [s]")

        # Общая легенда
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=4, fontsize='large')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig("logs/difference.png")
        plt.close()