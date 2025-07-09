import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# === Simulation Parameters ===
G = 1.0         # Gravitational constant
dT = 0.01       # Time step
STEPS = 2000    # Number of steps
TAIL_LENGTH = 75

# Masses
masses = [1.0, 1.0, 1.0]

# Initial positions (x, y)
positions = np.array([
    [ 0.970,  0.0],
    [-0.970,  0.0],
    [ 0.000,  0.0]
], dtype=float)

# Initial velocities (vx, vy)
velocities = np.array([
    [ 0.0,  0.5],
    [ 0.0, -0.5],
    [ 0.0,  0.0]
], dtype=float)

# Record trajectory for animation
trajectory = np.zeros((STEPS, 3, 2))


# === Physics ===

def compute_acceleration(pos, i):
    """Compute net gravitational acceleration on body i."""
    acc = np.zeros(2)
    for j in range(3):
        if i == j:
            continue
        r = pos[j] - pos[i]
        dist = np.linalg.norm(r) + 1e-5  # Prevent division by zero
        acc += G * masses[j] * r / dist**3
    return acc

def rk4_step(pos, vel, dt):
    """Runge-Kutta 4th order integrator."""
    accs = np.array([compute_acceleration(pos, i) for i in range(3)])
    new_vel = vel + accs * dt
    new_pos = pos + new_vel * dt
    return new_pos, new_vel

def simulate():
    """Run simulation and fill trajectory array."""
    global positions, velocities
    for step in range(STEPS):
        trajectory[step] = positions
        positions, velocities = rk4_step(positions, velocities, dT)


# === Animation ===

def animate_simulation(save=False, filename="three_body.gif", tail_length=75):
    colors = ['red', 'green', 'blue']
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_title("Three-Body Simulation")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')
    ax.grid(True)

    # Main balls
    balls = [ax.plot([], [], 'o', color=c, markersize=10)[0] for c in colors]

    # Tail segments with fading alpha
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

            for j in range(tail_length):
                idx = frame - j
                if idx < 1:
                    tail_lines[i][j].set_data([], [])
                else:
                    xs = trajectory[idx-1:idx+1, i, 0]
                    ys = trajectory[idx-1:idx+1, i, 1]
                    tail_lines[i][j].set_data(xs, ys)

        return balls + [l for tails in tail_lines for l in tails]

    anim = FuncAnimation(fig, update, frames=STEPS, init_func=init,
                         blit=False, interval=10)

    if save:
        if filename.endswith(".gif"):
            anim.save(filename, writer='pillow', fps=30)
        elif filename.endswith(".mp4"):
            anim.save(filename, writer='ffmpeg', fps=30)
        print(f"Saved animation to {filename}")
    else:
        plt.show()


# === Run Simulation ===

if __name__ == "__main__":
    simulate()
    animate_simulation(save=True, filename="three_body.gif", tail_length=TAIL_LENGTH)
