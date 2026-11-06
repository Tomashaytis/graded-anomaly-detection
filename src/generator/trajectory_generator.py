import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation


class TrajectoryGenerator:
    """
    Generate synthetic pedestrian trajectories with groups and singletons.
    """

    def __init__(self, rng: np.random.Generator = None):
        """
        :param rng: Numpy Generator; if None, a fresh one is created.
        """
        self._rng = rng if rng is not None else np.random.default_rng()

    def leader_trajectory(
            self,
            n_frames: int,
            start: tuple[float, float],
            speed: float = 2.0,
            stop_prob: float = 0.01,
            smooth: int = 10,
    ) -> np.ndarray:
        """
        Generate a smooth leader trajectory with time-varying speed.

        :param n_frames: Number of frames.
        :param start: (x, y) starting position.
        :param speed: Base speed (pixels per frame).
        :param stop_prob: Per-frame probability of "freezing" the speed near zero.
        :param smooth: Smoothing window for the speed signal.
        :return: (n_frames, 2) array of (x, y).
        """
        initial_heading = self._rng.uniform(0, 2 * np.pi)

        increments = self._rng.normal(0, 0.05, n_frames)
        heading = initial_heading + np.cumsum(increments)
        heading = np.convolve(heading, np.ones(smooth) / smooth, mode="same")

        speed_noise = self._rng.normal(0, 0.15, n_frames)
        speed_signal = np.cumsum(speed_noise)
        speed_signal = np.convolve(speed_signal, np.ones(smooth) / smooth, mode="same")
        speed_signal = speed_signal / (np.std(speed_signal) + 1e-6)
        speed_series = speed * (1.0 + 0.3 * speed_signal)

        freeze_frames = self._rng.random(n_frames) < stop_prob
        for f in np.where(freeze_frames)[0]:
            end = min(n_frames, f + self._rng.integers(5, 20))
            speed_series[f:end] *= 0.05

        speed_series = np.clip(speed_series, 0.0, None)

        dx = speed_series * np.cos(heading)
        dy = speed_series * np.sin(heading)
        x = start[0] + np.cumsum(dx)
        y = start[1] + np.cumsum(dy)
        return np.stack([x, y], axis=1)

    def follower_trajectory(
            self,
            leader_xy: np.ndarray,
            offset: tuple[float, float],
            jitter: float = 1.0,
            k_attract: float = 0.05,
            max_speed: float = 3.0,
    ) -> np.ndarray:
        """
        Generate a follower trajectory with its own speed and feedback.

        The follower is pulled toward (leader + offset) with strength k_attract,
        and its speed is capped by max_speed.

        :param leader_xy: (N, 2) leader positions.
        :param offset: (dx, dy) target offset from the leader.
        :param jitter: Per-frame random noise (added to acceleration).
        :param k_attract: Attraction strength toward the target position.
        :param max_speed: Maximum follower speed (pixels per frame).
        :return: (N, 2) array of (x, y).
        """
        n = len(leader_xy)
        offset_arr = np.asarray(offset, dtype=float)
        target = leader_xy + offset_arr

        pos = target[0].copy()
        vel = np.zeros(2, dtype=float)

        out = np.zeros((n, 2), dtype=float)
        out[0] = pos

        for t in range(1, n):
            desired = target[t] - pos
            accel = k_attract * desired
            accel += self._rng.normal(0, jitter * 0.1, 2)

            vel += accel

            speed = np.linalg.norm(vel)
            if speed > max_speed:
                vel = vel / speed * max_speed

            pos = pos + vel
            out[t] = pos

        return out

    def singleton_trajectory(
            self,
            n_frames: int,
            start: tuple[float, float],
            speed: float = 1.5,
            heading_noise: float = 0.005,
            speed_noise: float = 0.05,
    ) -> np.ndarray:
        """
        Generate a singleton trajectory: mostly straight or standing.

        Unlike group leaders, singletons rarely change direction and
        have small speed variation.

        :param n_frames: Number of frames.
        :param start: (x, y) starting position.
        :param speed: Base speed (pixels per frame); near zero for standing.
        :param heading_noise: Per-frame heading noise (small = straighter).
        :param speed_noise: Per-frame speed noise (small = steadier).
        :return: (n_frames, 2) array of (x, y).
        """
        initial_heading = self._rng.uniform(0, 2 * np.pi)

        increments = self._rng.normal(0, heading_noise, n_frames)
        heading = initial_heading + np.cumsum(increments)

        speed_series = speed * (1.0 + speed_noise * self._rng.normal(0, 1, n_frames))
        speed_series = np.clip(speed_series, 0.0, None)

        dx = speed_series * np.cos(heading)
        dy = speed_series * np.sin(heading)
        x = start[0] + np.cumsum(dx)
        y = start[1] + np.cumsum(dy)
        return np.stack([x, y], axis=1)

    def make_scene(
        self,
        group_sizes: tuple[int, ...],
        n_frames: int = 300,
        frame_size: tuple[int, int] = (640, 480),
        jitter: float = 1.0,
        p_stationary: float = 0.3,
        p_slow: float = 0.3,
    ) -> tuple[dict[int, np.ndarray], dict[int, list[int]]]:
        """
        Generate a synthetic scene.

        Group sizes define clusters: size 1 = singleton, size >= 2 = group.

        :param group_sizes: Sizes of clusters (1 = singleton).
        :param n_frames: Number of frames per track.
        :param frame_size: Frame size.
        :param jitter: Base jitter for group followers.
        :param p_stationary: Probability that a cluster is stationary.
        :param p_slow: Probability that a cluster is slow.
        :return: (tracks, groups); singletons are in groups as [tid].
        """
        tracks: dict[int, np.ndarray] = {}
        groups: dict[int, list[int]] = {}
        tid = 0

        for gid, size in enumerate(group_sizes):
            start = (
                self._rng.uniform(100, frame_size[0] - 100),
                self._rng.uniform(100, frame_size[1] - 100),
            )

            u = self._rng.random()
            if u < p_stationary:
                speed = self._rng.uniform(0.05, 0.3)
                group_jitter = jitter * 0.3
            elif u < p_stationary + p_slow:
                speed = self._rng.uniform(0.5, 1.0)
                group_jitter = jitter * 0.6
            else:
                speed = self._rng.uniform(1.5, 2.5)
                group_jitter = jitter

            if size == 1:
                singleton = self.singleton_trajectory(n_frames, start, speed=speed)
                tracks[tid] = singleton
                groups[gid] = [tid]
                tid += 1
                continue

            leader = self.leader_trajectory(n_frames, start, speed=speed)
            members = [tid]
            tracks[tid] = leader
            tid += 1

            for k in range(1, size):
                angle = 2 * np.pi * k / size
                radius = self._rng.uniform(30, 70)
                offset = (radius * np.cos(angle), radius * np.sin(angle))
                follower = self.follower_trajectory(leader, offset, group_jitter)
                members.append(tid)
                tracks[tid] = follower
                tid += 1

            groups[gid] = members

        return tracks, groups

    @staticmethod
    def plot(tracks, groups, frame_size=(640, 480), figsize=(8, 6)):
        """
        Plot all synthetic trajectories, colored by group.

        Group members share a color; singletons are drawn in gray.

        :param tracks: Tracks.
        :param groups: Groups.
        :param frame_size: Frame size.
        :param figsize: Figure size in inches.
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

    @staticmethod
    def animate(tracks, groups, frame_size=(640, 480),
                figsize=(8, 6), interval=50):
        """
        Animate the synthetic scene: dots move along the trajectories.

        :param tracks: Tracks.
        :param groups: Groups.
        :param frame_size: Frame size.
        :param figsize: Figure size in inches.
        :param interval: Delay between frames in ms.
        :return: Matplotlib FuncAnimation.
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