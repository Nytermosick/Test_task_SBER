import numpy as np
from pathlib import Path
import os
import pinocchio as pin
import matplotlib.pyplot as plt
import mujoco
import mediapy as media
from typing import Callable, Optional, Dict, Union, List, Any
import time
from datetime import datetime
import mujoco.viewer
    
class Simulator:
    # TODO: Сделать доку

    def __init__(self,
                 xml_path: str,
                 dt: float,
                 width: int = 1920,
                 height: int = 1080,
                 record_video: bool = True,
                 make_plots: bool = True) -> None:
        # TODO: Сделать доку
        
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


        if record_video or make_plots:
            os.makedirs("logs", exist_ok=True)

        self.record_video = record_video
        self.make_plots = make_plots

        # Настройка рендера
        self.renderer = mujoco.Renderer(self.model, width=self.width, height = self.height)


    
    def _save_video(self) -> None:
        # TODO: Сделать доку

        if self.frames:
            print(f"Saving video to logs/vid.mp4...")
            media.write_video("logs/vid.mp4", self.frames, fps=self.fps)
            self.frames = []


    def _capture_frame(self) -> np.ndarray:
        # TODO: Сделать доку
        """Capture a frame using the renderer.
        
        Returns:
            RGB image array of current scene
        """

        self.renderer.update_scene(self.data)
        pixels = self.renderer.render()
        return pixels.copy()


    def reset(self) -> None:
        #TODO: Сделать доку
        """Reset the simulation to the initial state."""

        mujoco.mj_resetData(self.model, self.data)


    def get_state(self) -> Dict[str, np.ndarray]:
        # TODO: Сделать доку
        """Get the current state of the model.
        
        Returns:
            State vector
        """

        state = {
            'q': self.data.qpos.copy(),
            'dq': self.data.qvel.copy(),
        }

        return state
    

    def _def_init_state(self, q0: List[int|float], dq0: List[int|float]) -> None:
        # TODO: Сделать доку

        self.data.qpos = q0
        self.data.qvel = dq0
    
    
    def step(self, tau: np.ndarray) -> None:
        # TODO: Сделать доку
        """Step the simulation forward.
        
        Args:
            tau: Control input
        """

        self.data.ctrl = tau
        mujoco.mj_step(self.model, self.data)


    def run(self, ctrl: np.ndarray, q0: List[int|float] = None, dq0: List[int|float] = None) -> None:
        # TODO: Сделать доку
        """Run simulation with visualization and recording.
        
        Args:
            time_limit: Maximum simulation time in seconds
            
        Raises:
            AssertionError: If controller is not set
        """

        self.N = ctrl.shape[1]

        viewer = mujoco.viewer.launch_passive(
            model = self.model,
            data = self.data,
            show_left_ui = False,
            show_right_ui = False
        )

        self.reset()

        self.pos = []
        self.controls = []

        if q0 is None:
            q0 = [0] * self.model.nq
        if dq0 is None:
            dq0 = [0] * self.model.nv

        self._def_init_state(q0, dq0)

        mujoco.mjv_defaultFreeCamera(self.model, viewer.cam)

        for k in range(self.N):
            state = self.get_state()
            tau = ctrl[:, k]

            self.pos.append(state['q'])
            self.controls.append(tau)

            self.step(tau)
            viewer.sync()

            if self.record_video:
                    self.frames.append(self._capture_frame())

        viewer.close()

        if self.record_video:
            self._save_video()
    

    def plot_results(self):
        # TODO: Сделать доку
        """Plot and save simulation results."""

        if self.log_path is None:
            self.make_log_path()

        self.pos = np.array(self.pos)
        self.controls = np.array(self.controls)
        self.times = np.array(self.times)
        
        # Joint positions plot
        plt.figure(figsize=(10, 6))
        for i in range(self.pos.shape[1]):
            plt.plot(self.times, self.pos[:, i], label=f'Joint {i+1}')
        plt.xlabel('Time [s]')
        plt.ylabel('Joint Position [rad]')
        plt.title('Joint Position over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'logs/{self.log_path}/position.png')
        plt.close()

        # Joint position errors plot
        plt.figure(figsize=(10, 6))
        for i in range(self.pos.shape[1]):
            plt.plot(self.times, 0 - self.pos[:, i], label=f'Joint {i+1}')
        plt.xlabel('Time [s]')
        plt.ylabel('Joint Position error [rad]')
        plt.title('Joint Position error over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'logs/{self.log_path}/position_error.png')
        plt.close()


        # Joint controls plot
        if self.controls.ndim == 1:
            self.controls = self.controls.reshape(-1, 1)

        for i in range(self.controls.shape[1]):
            plt.plot(self.times, self.controls[:, i], label=f'Joint {i+1}')
        plt.xlabel('Time [s]')
        plt.ylabel('Joint control signals')
        plt.title('Joint control signals over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'logs/{self.log_path}/control_signals.png')
        plt.close()