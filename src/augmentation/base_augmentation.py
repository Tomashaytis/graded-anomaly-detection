import torch


class BaseAugmentation:
    """
    Base class for all video augmentations.
    """
    @property
    def name(self) -> str:
        """
        Return augmentation name.
        """
        return self.__class__.__name__

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        Overload for python call method.

        :param video: video [C, T, H, W].
        :return: augmented video [C, T, H, W].
        """
        return self.apply(video)


    def apply(self, video: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations.

        :param video: video [C, T, H, W].
        :return: augmented video [C, T, H, W].
        """
        raise NotImplementedError