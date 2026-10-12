import torch

from torch import transforms


def get_transforms(mode='train', input_size=172):
    """Правильные трансформации для видео с учетом batch dimension"""

    def random_horizontal_flip_video(video):
        """Случайное горизонтальное отражение для видео [C, T, H, W] или [B, C, T, H, W]"""
        if video.ndim == 4:  # [C, T, H, W]
            if torch.rand(1) < 0.5:
                return video.flip(3)  # flip width dimension
            return video
        elif video.ndim == 5:  # [B, C, T, H, W]
            if torch.rand(1) < 0.5:
                return video.flip(4)  # flip width dimension
            return video
        return None

    def center_crop_video(video, size):
        """Центральный кроп для видео"""
        if video.ndim == 4:  # [C, T, H, W]
            _, _, H, W = video.shape
            top = (H - size[0]) // 2
            left = (W - size[1]) // 2
            return video[:, :, top:top + size[0], left:left + size[1]]
        elif video.ndim == 5:  # [B, C, T, H, W]
            _, _, _, H, W = video.shape
            top = (H - size[0]) // 2
            left = (W - size[1]) // 2
            return video[:, :, :, top:top + size[0], left:left + size[1]]
        return None

    # Упрощенная версия без resize_video - будем использовать только center_crop
    if mode == 'train':
        return transforms.Compose([
            transforms.Lambda(lambda x: x.float()),
            transforms.Lambda(lambda x: random_horizontal_flip_video(x)),
            transforms.Lambda(lambda x: center_crop_video(x, (input_size, input_size))),
        ])
    else:  # test/val
        return transforms.Compose([
            transforms.Lambda(lambda x: x.float()),
            transforms.Lambda(lambda x: center_crop_video(x, (input_size, input_size))),
        ])