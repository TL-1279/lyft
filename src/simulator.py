# simulator.py. tạo dữ liệu multi-agent (fast)
"""
Simple multi-agent simulator:
- N agents on a 2D plane
- Each agent has (x, y), yaw (rad), speed (m/s)
- Agents move with simple kinematic update + small noise
- Produce sequences of positions for history and future
"""

import numpy as np # type: ignore

class Agent:
    def __init__(self, x, y, yaw, speed, id):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.speed = speed
        self.id = id

    def step(self, dt=0.1, noise_scale=0.01):
        # simple constant-velocity with small steering noise
        self.x += self.speed * np.cos(self.yaw) * dt
        self.y += self.speed * np.sin(self.yaw) * dt
        # small yaw noise
        self.yaw += np.random.randn() * noise_scale
        # small speed noise
        self.speed += np.random.randn() * (noise_scale*0.5)
        return (self.x, self.y)

def simulate_scene(num_agents=10, history_steps=10, future_steps=50, dt=0.1, area=50.0, seed=None):
    """
    Returns:
      history: np.array (num_agents, history_steps, 2)
      future:  np.array (num_agents, future_steps, 2)
      meta: list of Agent metadata
    """
    if seed is not None:
        np.random.seed(seed)
    agents = []
    for i in range(num_agents):
        x = np.random.uniform(-area/2, area/2)
        y = np.random.uniform(-area/2, area/2)
        yaw = np.random.uniform(-np.pi, np.pi)
        speed = np.random.uniform(1.0, 6.0)  # m/s
        agents.append(Agent(x, y, yaw, speed, i))

    # produce history by stepping forward but store negative times (we simulate)
    history = np.zeros((num_agents, history_steps, 2), dtype=np.float32)
    for t in range(history_steps):
        for i, a in enumerate(agents):
            # store current pos, then step
            history[i, t] = (a.x, a.y)
            a.step(dt=dt)

    # now save future starting from current states
    future = np.zeros((num_agents, future_steps, 2), dtype=np.float32)
    for t in range(future_steps):
        for i, a in enumerate(agents):
            future[i, t] = (a.x, a.y)
            a.step(dt=dt)

    meta = [{"id": a.id, "speed": a.speed} for a in agents]
    return history, future, meta
