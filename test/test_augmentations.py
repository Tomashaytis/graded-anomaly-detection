from src.dataset import VideoLoader
from src.augmentation import *
from src.utils import create_complex_augmentations, visualize_video_augmentation

VIDEO_PATH = '../data/test.mp4'


def test():
    with VideoLoader(VIDEO_PATH) as loader:
        video_tensor = loader.to_tensor(max_frames=64, resize_size=(224, 224), sampling_strategy='uniform')

        # Composite augmentations
        visualize_video_augmentation(video_tensor, create_complex_augmentations(current_n_frames=64, current_size=224))

        # Spatial augmentations
        visualize_video_augmentation(video_tensor, RandomHorizontalFlip(p=1))
        visualize_video_augmentation(video_tensor, RandomGrayscale(p=1))
        visualize_video_augmentation(video_tensor, RandomRotation(p=1))
        visualize_video_augmentation(video_tensor, RandomResizedCrop(size=224))
        visualize_video_augmentation(video_tensor, ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1))

        # Temporal augmentations
        visualize_video_augmentation(video_tensor, TemporalRandomCrop(clip_length=32))
        visualize_video_augmentation(video_tensor, TemporalFlip(p=1))
        visualize_video_augmentation(video_tensor, TemporalStrideSample(stride=2))
        visualize_video_augmentation(video_tensor, TemporalStridePick(stride=2))
        visualize_video_augmentation(video_tensor, TemporalShift(p=1))
        visualize_video_augmentation(video_tensor, TimeMasking(p=1))
        visualize_video_augmentation(video_tensor, SpeedChange(p=1))
        visualize_video_augmentation(video_tensor, TemporalSubsample(p=1))

        # Distortion augmentations
        visualize_video_augmentation(video_tensor, GaussianNoise(p=1))
        visualize_video_augmentation(video_tensor, GaussianBlur(p=1))
        visualize_video_augmentation(video_tensor, RandomErasing(p=1))


if __name__ == '__main__':
    test()