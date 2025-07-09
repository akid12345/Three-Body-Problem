import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
G = 1.0         # Gravitational constant
DT = 0.01       # Time step
STEPS = 2000    # Number of simulation steps

# Masses of the three bodies
masses = [1.0, 1.0, 1.0]

# Initial positions and velocities (x, y)
positions = np.array([
    [ 0.970, 0.0  ],
    [-0.970, 0.0  ],
    [ 0.0  , 0.0  ]
], dtype=float)

velocities = np.array([
    [0.0,  0.5],
    [0.0, -0.5],
    [0.0,  0.0]
], dtype=float)

# To store the simulation history
trajectory = np.zeros((STEPS, 3, 2))


def compute_acceleration(pos, i):
    """Compute gravitational acceleration on body i from all other bodies."""
    acc = np.zeros(2)
    for j in range(3):
        if i == j:
            continue
        r_ij = pos[j] - pos[i]
        dist = np.linalg.norm(r_ij) + 1e-5  # Avoid divide-by-zero
        acc += G * masses[j] * r_ij / dist**3
    return acc


def rk4_step(pos, vel, dt):
    """Runge-Kutta 4th order integration step."""
    accs = np.array([compute_acceleration(pos, i) for i in range(3)])
    new_vel = vel + accs * dt
    new_pos = pos + new_vel * dt
    return new_pos, new_vel


def simulate():
    """Run the simulation and populate the trajectory."""
    global positions, velocities
    for step in range(STEPS):
        trajectory[step] = positions
        positions, velocities = rk4_step(positions, velocities, DT)


def animate_simulation(save=False, filename="three_body.gif", tail_length=100):
    """Animate the trajectory with tail-fading. Save to file if specified."""
    colors = ['red', 'green', 'blue']
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_title("Three-Body Simulation")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.set_aspect('equal')

    # Plotting elements: 3 balls, and fading trails for each
    balls = [plt.plot([], [], 'o', color=c, markersize=10)[0] for c in colors]
    tail_lines = [[ax.plot([], [], '-', color=c, alpha=alpha)[0]
                   for alpha in np.linspace(0.05, 1.0, tail_length)] for c in colors]

    def init():
        for ball in balls:
            ball.set_data([], [])
        for tails in tail_lines:
            for line in tails:
                line.set_data([], [])
        return balls + [l for tails in tail_lines for l in tails]

    def update(frame):
        for i in range(3):
            x, y = trajectory[frame, i]
            balls[i].set_data(x, y)

            # Tail-fading logic: draw short fading segments
            for j in range(tail_length):
                idx = frame - j
                if idx < 1:
                    tail_lines[i][j].set_data([], [])
                else:
                    xs = trajectory[idx-1:idx+1, i, 0]
                    ys = trajectory[idx-1:idx+1, i, 1]
                    tail_lines[i][j].set_data(xs, ys)

        return balls + [l for tails in tail_lines for l in tails]

    anim = FuncAnimation(fig, update, frames=STEPS, init_func=init, blit=True, interval=10)

    if save:
        if filename.endswith(".gif"):
            anim.save(filename, writer='pillow', fps=30)
        elif filename.endswith(".mp4"):
            anim.save(filename, writer='ffmpeg', fps=30)
        print(f"Animation saved to {filename}")
    else:
        plt.show()


if __name__ == "__main__":
    simulate()
    # Toggle save to True to export animation
    animate_simulation(save=True, filename="three_body.gif", tail_length=75)
