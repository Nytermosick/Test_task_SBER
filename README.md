# Trajectory Optimization & MPC for Manipulators

Этот репозиторий содержит решение тестового задания по Robotics Control. Оба задания выполнены с применением `Pinocchio`, `CasADi` и `MuJoCo`. Применялись две модели манипуляторов: самописного плоского 3R манипулятора и UR10. Обе модели представлены в форматах `URDF` и `MJCF`.

---

## Клонирование репозитория

```bash
git clone https://github.com/Nytermosick/Test_task_SBER.git
cd Test_task_SBER
```

## Установка окружения

### 1. Создание и активация виртуального окружения

```bash
python3.10 -m venv test_task_env
source test_task_env/bin/activate
```

### 2. Установка зависимостей из `requirements.txt`

```bash
pip install -r requirements.txt
```

### 3. Установка необходимых библиотек из исходников

#### Eigenpy

```bash
git clone https://github.com/stack-of-tasks/eigenpy.git
cd eigenpy
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

#### CasADi

CasADi требуется со сборкой с поддержкой `IPOPT`.

```bash
git clone https://github.com/casadi/casadi.git
cd casadi
mkdir build && cd build
cmake .. -DWITH_IPOPT=ON -DWITH_PYTHON=ON
make -j$(nproc)
make install
```

#### Pinocchio (с поддержкой CasADi)

```bash
git clone https://github.com/stack-of-tasks/pinocchio.git
cd pinocchio
mkdir build && cd build
cmake .. -DBUILD_WITH_CASADI_SUPPORT=ON -DBUILD_PYTHON_INTERFACE=ON
make -j$(nproc)
sudo make install
```

### 4. Проверка

```python
import casadi
import pinocchio
import eigenpy
```

Если ошибок нет — установка прошла успешно.

---

## Структура проекта

```
.
├── first_task/        # Задание 1: оптимальная траектория с учётом ограничений
├── second_task/       # Задание 2: MPC с адаптацией к меняющейся цели и препятствиям
├── robots/            # Папка с моделями манипуляторов
├── test_task.pdf      # Требования к тестовому заданию
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Что реализовано

В рамках выполнения задания были реализованы:

- Решение **Задания 1**: построение оптимальной траектории движения манипулятора с учётом ограничений на управляющие воздействия и динамику.
- Решение **Задания 2**: построение MPC-контроллера, который адаптирует движение к меняющейся цели и позволяет обходить препятствия.
- Модели манипуляторов (3R-планарный и UR10) оформлены в формате URDF и MJCF.
- Симуляция выполнена в **MuJoCo**, включая пошаговое управление, логирование, генерацию видео и графиков.
- Обратная кинематика реализована в аналитическом виде (для 3R) и с выбором наилучшего решения.
- Использована **символьная динамика** через `Pinocchio + CasADi` с формулировкой MPC-задачи как задачи оптимизации.
- Поддержка **ограничений на углы, скорости, ∆u, пол, препятствия**.

---

## Что не было реализовано (пока)

- **Интеграция с ROS2**: не успел из-за ограничения по времени. Готов выполнить позже.
- **Поддержка C++**: задание реализовано на Python. Освоение C++ и переписывание при необходимости — в планах.
- **Docker-окружение**: использована локальная установка. Образ Docker можно собрать дополнительно.
- В задании 2 реализован только планарный вариант. Перенос на UR10 возможен, но не успел выполнить в срок.

---

Всё выше перечисленное — в рабочем виде, протестировано и задокументировано. Широко использовалось ООП. Модульность кода позволяет легко расширять и дорабатывать проект.

## Задания

Каждое задание имеет свой `README.md` с подробным описанием внутри соответствующей папки:

- [`first_task/README.md`](first_task/README.md)
- [`second_task/README.md`](second_task/README.md)