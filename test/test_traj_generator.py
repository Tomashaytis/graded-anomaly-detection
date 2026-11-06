from src.generator import TrajectoryGenerator

GROUP_SIZES = (3, 2, 4, 1, 1)


if __name__ == '__main__':
    gen = TrajectoryGenerator()
    tracks, groups = gen.make_scene(GROUP_SIZES)
    gen.animate(tracks, groups)