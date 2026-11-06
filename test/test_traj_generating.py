import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation


def leader_trajectory(n_frames, start, speed=2.0, rng=None):
    """
    Generate a smooth leader trajectory.

    :param n_frames: number of frames
    :param start: (x, y) starting position
    :param speed: pixels per frame
    :param rng: numpy Generator; if None, a fresh one is created
    :return: (n_frames, 2) array of (x, y)
    """
    if rng is None:
        rng = np.random.default_rng()

    initial_heading = rng.uniform(0, 2 * np.pi)

    increments = rng.normal(0, 0.05, n_frames)
    heading = initial_heading + np.cumsum(increments)
    heading = np.convolve(heading, np.ones(10) / 10, mode="same")

    dx = speed * np.cos(heading)
    dy = speed * np.sin(heading)
    x = start[0] + np.cumsum(dx)
    y = start[1] + np.cumsum(dy)
    return np.stack([x, y], axis=1)


def follower_trajectory(leader_xy, offset, jitter=0.5, rng=None):
    """
    Generate a follower trajectory around a leader.

    :param leader_xy: (N, 2) leader positions
    :param offset: (dx, dy) constant offset from the leader
    :param jitter: standard deviation of per-frame noise
    :param rng: numpy Generator; if None, a fresh one is created
    :return: (N, 2) array of (x, y)
    """
    if rng is None:
        rng = np.random.default_rng()

    noise = rng.normal(0, jitter, leader_xy.shape)
    noise = np.apply_along_axis(
        lambda c: np.convolve(c, np.ones(5) / 5, mode="same"), 0, noise
    )
    return leader_xy + offset + noise


def make_synthetic_scene(
    n_frames=200,
    n_groups=3,
    group_sizes=(3, 2, 4),
    n_singletons=2,
    frame_size=(640, 480),
    jitter=0.5,
    p_stationary=0.3,
    p_slow=0.3,
    rng=None,
):
    """
    Generate a synthetic scene of groups and singletons.

    :param n_frames: number of frames per track
    :param n_groups: number of groups
    :param group_sizes: group sizes (truncated to n_groups)
    :param n_singletons: number of singletons
    :param frame_size: (width, height) for start positions
    :param jitter: base jitter for group followers
    :param p_stationary: probability that a group is stationary
    :param p_slow: probability that a group is slow
    :param rng: numpy Generator; if None, a fresh one is created
    :return: (tracks, groups) where tracks is dict[tid -> (N, 2)]
             and groups is dict[group_id -> [tids]]
    """
    if rng is None:
        rng = np.random.default_rng()

    tracks = {}
    groups = {}
    tid = 0

    for gid, size in enumerate(group_sizes[:n_groups]):
        start = (
            rng.uniform(100, frame_size[0] - 100),
            rng.uniform(100, frame_size[1] - 100),
        )

        u = rng.random()
        if u < p_stationary:
            speed = rng.uniform(0.05, 0.3)
            group_jitter = jitter * 0.3
        elif u < p_stationary + p_slow:
            speed = rng.uniform(0.5, 1.0)
            group_jitter = jitter * 0.6
        else:
            speed = rng.uniform(1.5, 2.5)
            group_jitter = jitter

        leader = leader_trajectory(n_frames, start, speed=speed, rng=rng)
        members = [tid]
        tracks[tid] = leader
        tid += 1

        for k in range(1, size):
            angle = 2 * np.pi * k / size
            offset = (30 * np.cos(angle), 30 * np.sin(angle))
            follower = follower_trajectory(leader, offset, group_jitter, rng=rng)
            members.append(tid)
            tracks[tid] = follower
            tid += 1

        groups[gid] = members

    for _ in range(n_singletons):
        start = (
            rng.uniform(100, frame_size[0] - 100),
            rng.uniform(100, frame_size[1] - 100),
        )
        speed = rng.uniform(0.1, 2.0)
        singleton = leader_trajectory(n_frames, start, speed=speed, rng=rng)
        tracks[tid] = singleton
        tid += 1

    return tracks, groups


def plot_synthetic_scene(tracks, groups, frame_size=(640, 480), figsize=(8, 6)):
    """
    Plot all synthetic trajectories, colored by group.

    Group members share a color; singletons are drawn in gray.

    :param tracks: dict[tid -> (N, 2) array of (x, y)]
    :param groups: dict[group_id -> [tids]]
    :param frame_size: (width, height) for axes limits
    :param figsize: figure size in inches
    """
    fig, ax = plt.subplots(figsize=figsize)

    group_of = {}
    for gid, members in groups.items():
        for tid in members:
            group_of[tid] = gid

    cmap = plt.get_cmap("tab10")
    n_groups = max(len(groups), 1)

    for tid, xy in tracks.items():
        gid = group_of.get(tid)
        if gid is None:
            color = (0.5, 0.5, 0.5)
            label = None
        else:
            color = cmap(gid % n_groups)
            label = f"group {gid}" if tid == groups[gid][0] else None

        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=1.5, label=label)
        ax.scatter(xy[0, 0], xy[0, 1], color=color, marker="o", s=20, zorder=3)
        ax.scatter(xy[-1, 0], xy[-1, 1], color=color, marker="x", s=30, zorder=3)

    ax.set_xlim(0, frame_size[0])
    ax.set_ylim(frame_size[1], 0)
    ax.set_aspect("equal")
    ax.set_title("Synthetic scene (o = start, x = end)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def animate_synthetic_scene(tracks, groups, frame_size=(640, 480),
                            figsize=(8, 6), interval=50):
    """
    Animate the synthetic scene: dots move along the trajectories.

    :param tracks: dict[tid -> (N, 2) array of (x, y)]
    :param groups: dict[group_id -> [tids]]
    :param frame_size: (width, height) for axes limits
    :param figsize: figure size in inches
    :param interval: delay between frames in ms
    :return: matplotlib FuncAnimation
    """
    group_of = {}
    for gid, members in groups.items():
        for tid in members:
            group_of[tid] = gid

    cmap = plt.get_cmap("tab10")
    n_groups = max(len(groups), 1)

    tids = list(tracks.keys())
    n_frames = min(len(tracks[tid]) for tid in tids)

    colors = {}
    for tid in tids:
        gid = group_of.get(tid)
        colors[tid] = (0.5, 0.5, 0.5) if gid is None else cmap(gid % n_groups)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, frame_size[0])
    ax.set_ylim(frame_size[1], 0)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title("Synthetic scene (animated)")

    for tid in tids:
        ax.plot(tracks[tid][:, 0], tracks[tid][:, 1],
                color=colors[tid], linewidth=0.8, alpha=0.3)

    dots = {
        tid: ax.scatter([], [], color=colors[tid], s=40, zorder=3)
        for tid in tids
    }

    def update(frame_idx):
        for tid in tids:
            xy = tracks[tid]
            idx = min(frame_idx, len(xy) - 1)
            dots[tid].set_offsets(xy[idx:idx + 1])
        return list(dots.values())

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=interval, blit=False, repeat=True)
    plt.tight_layout()
    plt.show()
    return anim


if __name__ == '__main__':
    tracks, groups = make_synthetic_scene(
        n_frames=200,
        n_groups=3,
        group_sizes=(3, 2, 4),
        n_singletons=2,
    )

    plot_synthetic_scene(tracks, groups)