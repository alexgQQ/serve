import functools

from torch.nn import BatchNorm2d, init
from localutils import UnetGenerator


class Pix2Pix(UnetGenerator):
    """
    Model file for the pix2pix generator model.
    This is based off of the structure defined here:
        https://github.com/pytorch/serve/blob/master/model-archiver/README.md#model-file
    And is inspired from the model structure here:
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """

    input_nc = 3
    output_nc = 3
    ngf = 64
    no_dropout = False

    def __init__(self):
        norm_layer = functools.partial(
            BatchNorm2d, affine=True, track_running_stats=True
        )
        super().__init__(
            self.input_nc, self.output_nc, 8, self.ngf, norm_layer, self.no_dropout
        )

        def init_func(m):
            init_gain = 0.02
            classname = m.__class__.__name__
            if hasattr(m, "weight") and (
                classname.find("Conv") != -1 or classname.find("Linear") != -1
            ):
                init.normal_(m.weight.data, 0.0, init_gain)
                if hasattr(m, "bias") and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif (
                classname.find("BatchNorm2d") != -1
            ):  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)

        self.apply(init_func)
