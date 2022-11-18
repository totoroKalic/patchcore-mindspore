"""OneStepCell"""
import mindspore.nn as nn

class OneStepCell(nn.Cell):
    """OneStepCell"""
    def __init__(self, network):
        super(OneStepCell, self).__init__()
        self.network = network

        self.pad = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT")
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, pad_mode="valid")

    def construct(self, img):
        output = self.network(img)

        output_one = self.pool(self.pad(output[0]))
        output_two = self.pool(self.pad(output[1]))

        return [output_one, output_two]
