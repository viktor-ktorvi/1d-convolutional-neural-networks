class Conv1dLayerSizes:
    def __init__(self, in_len, in_ch, out_ch, kernel, pool_kernel):
        self.in_len = in_len
        self.pool_kernel = pool_kernel
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel

        self.out_len = self.calcOutputLen()

    # length of the signal after 1d convolution
    def calcOutputLen(self):
        out_len = self.in_len - self.kernel_size + 1
        assert out_len > 0
        # TODO throw exception

        return self.calcPoolOutSize(out_len)

    # length of the signal after pooling
    def calcPoolOutSize(self, pool_input_len):
        out_float = pool_input_len / self.pool_kernel
        out_len = int(out_float)

        assert abs(out_len - out_float) < 1e-4
        # TODO throw exception

        return out_len


class TransposeConv1dLayerSizes:
    def __init__(self, in_len, out_len, in_ch, out_ch, stride):
        self.in_len = in_len
        self.out_len = out_len
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride

        self.kernel_size = self.calcKernelSize()

    def calcKernelSize(self):
        # Lout = (Lin - 1) * stride - 2*padding + dilation*(kernel_size - 1) + output_padding + 1
        # so the kernel size is

        kernel_size = self.out_len - self.stride * (self.in_len - 1)

        assert kernel_size > 0
        # TODO throw exception

        return kernel_size


class FullyConnectedLayerSizes:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
