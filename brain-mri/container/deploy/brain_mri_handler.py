from torch.profiler import ProfilerActivity
from torchvision import transforms
from ts.torch_handler.image_classifier import ImageClassifier


class BrainHandler(ImageClassifier):
    """
    DogCatClassifer handler class. This handler extends class ImageClassifier from image_classifier.py, a
    default handler. This handler takes an image and returns the number in that image.

    Here method postprocess() has been overridden while others are reused from parent class.

    Author: Cyrill Hug / 01.17.2023
    Based on: https://github.dev/pytorch/serve/blob/master/examples/image_classifier/mnist/mnist_handler.py#L1
    """

    image_processing = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    def __init__(self):
        super(BrainHandler, self).__init__()
        self.profiler_args = {
            "activities": [ProfilerActivity.CPU],
            "record_shapes": True,
        }

    def postprocess(self, data):
        """The post process of DogCat converts the predicted output response to a label.

        Args:
            data (list): The predicted output from the Inference with probabilities is passed
            to the post-process function
        Returns:
            list : A list of dictionaries with predictions and explanations is returned
        """
        return data.tolist()
