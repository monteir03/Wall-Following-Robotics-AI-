# 🚀 Reinforcement Learning Wall Following Project

## 📖 Overview

This project explores the use of **Reinforcement Learning (RL)** algorithms for autonomous wall-following in robotics using the **Webots** simulation environment.

The main objective is to evaluate different RL algorithms, reward structures, and action spaces to determine the most effective approach for reliable and efficient wall-following behavior.

The project includes implementations and experiments using:

* Q-Learning
* PPO (Proximal Policy Optimization)
* TD3 (Twin Delayed Deep Deterministic Policy Gradient)

The experiments were conducted using an **e-puck robot** equipped with:

* Lidar sensors for object detection
* Collision sensors for obstacle avoidance

---

# 🛠️ Required Software

To run the project, you need to install:

## 1. Webots

Download and install:

* **Webots**

Official website:

```text
https://cyberbotics.com/
```

---

## 2. Python IDE (Recommended)

We strongly recommend using:

* **PyCharm**

Open the project folder in PyCharm after installation.

---

# 🐍 Python Version

This project was developed using:

```python id="f4a3qj"
Python 3.12
```

---

# 📦 Required Python Modules

Install the following dependencies:

```python id="ibjv4n"
numpy==0.29.1
gymnasium==1.26.2
stable-baselines3==2.3.2
tensorflow==2.16.1
tensorboard==2.16.2
torch==2.3.1
```

You can install them using:

```bash id="x4t9lu"
pip install numpy gymnasium stable-baselines3 tensorflow tensorboard torch
```

---

# ⚙️ Project Setup

## Step 1 — Open the Project

Open the project folder in **PyCharm**.

---

## Step 2 — Configure Webots

Make sure Webots is properly connected to the Python environment used by the project.

---

## Step 3 — Install Dependencies

Install all required Python packages listed above.

---

# 🧪 Testing the Best Model

## Using the Provided Maps

To test the trained model on the provided environments, simply run:

```python id="vb8hri"
testing.py
```

---

## Using Custom Maps

If you want to test the model on your own maps:

### Important Requirement

The **e-puck robot node** must be named:

```python id="5qxzb6"
robot
```

Inside the Webots scene tree:

* locate the `DEF` field of the e-puck node
* set its value to:

```python id="9y7km0"
DEF robot
```

This is required for the controller to correctly identify and interact with the robot.

---

# 🤖 Reinforcement Learning Approach

The project evaluates both:

* **Discrete action spaces**
* **Continuous action spaces**

Different reward strategies were tested to analyze their impact on:

* navigation smoothness
* collision avoidance
* wall-following accuracy
* learning efficiency

---

# 📊 Main Findings

Experimental results showed that:

* **PPO** achieved the best overall performance
* well-designed reward functions significantly improve learning quality
* continuous action spaces generally produce smoother robot behavior
* the combination of algorithm, reward structure, and action space strongly affects performance

---

# 📚 Abstract Summary

Wall-following is a fundamental robotics task that supports more advanced problems such as:

* autonomous navigation
* mapping
* localization

This project investigates how reinforcement learning can be used to solve the wall-following problem efficiently and reliably.

The developed system demonstrates that reinforcement learning techniques — particularly PPO with optimized rewards — can successfully guide autonomous robots in simulated environments with smooth and stable behavior.

---

# 👨‍💻 Authors

* Ana Batista
* Gonçalo Monteiro
* Mafalda Aires

---

# 🏫 Institutions

* FEUP — Faculty of Engineering, University of Porto
* FCUP — Faculty of Sciences, University of Porto
