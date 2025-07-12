import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# === Parameters ===
G = 10.0
dT = 0.01
STEPS = 2000
TAIL_LENGTH = 75
SAVE_FILENAME = "three_body.gif"  # or "three_body.mp4"

masses = [8.0, 8.0, 8.0]

# Initial positions and velocities
positions = np.array([[0, 0.1], [-2, 0], [2, 0]], dtype=float)

velocities = np.array([[0, -0.45], [0, 5], [0, -5]], dtype=float)

trajectory = np.zeros((STEPS, 3, 2))


# === Physics ===
def compute_acceleration(pos, i):
    acc = np.zeros(2)
    for j in range(3):
        if i != j:
            r = pos[j] - pos[i]
            dist = np.linalg.norm(r) + 1e-5
            acc += G * masses[j] * r / dist**3
    return acc


def rk4_step(pos, vel, dt):
    accs = np.array([compute_acceleration(pos, i) for i in range(3)])
    new_vel = vel + accs * dt
    new_pos = pos + new_vel * dt
    return new_pos, new_vel


def simulate():
    global positions, velocities
    for step in range(STEPS):
        trajectory[step] = positions
        positions, velocities = rk4_step(positions, velocities, dT)
    print("Simulation complete.")


# === Animation ===
def animate_simulation(save=False, filename="three_body.gif", tail_length=25):
    colors = ['red', 'blue', 'black']
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')

    balls = [ax.plot([], [], 'o', color=c, markersize=10)[0] for c in colors]
    tail_lines = [[
        ax.plot([], [], '-', color=c, alpha=alpha)[0]
        for alpha in np.linspace(0.05, 1.0, tail_length)
    ] for c in colors]

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
            balls[i].set_data([x], [y])
            for j in range(tail_length):
                idx = frame - j
                if idx < 1:
                    tail_lines[i][j].set_data([], [])
                else:
                    xs = trajectory[idx - 1:idx + 1, i, 0]
                    ys = trajectory[idx - 1:idx + 1, i, 1]
                    tail_lines[i][j].set_data(xs, ys)
        return balls + [l for tails in tail_lines for l in tails]

    anim = FuncAnimation(fig,
                         update,
                         frames=STEPS,
                         init_func=init,
                         blit=False,
                         interval=2)

    if save:
        if filename.endswith(".gif"):
            anim.save(filename, writer='pillow', fps=60)
        elif filename.endswith(".mp4"):
            anim.save(filename, writer='ffmpeg', fps=60)
        print(f"Saved to {filename}")
    else:
        plt.show()


# === Entry Point ===
if __name__ == "__main__":
    simulate()
    animate_simulation(save=False,
                       filename=SAVE_FILENAME,
                       tail_length=TAIL_LENGTH)
