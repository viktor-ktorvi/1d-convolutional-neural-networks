def calcPoolOutSize(in_len, pool_kernel):
    out_float = in_len / pool_kernel
    out = int(out_float)

    assert abs(out - out_float) < 1e-4
    # TODO throw exception

    return out


class Conv1dLayerSizes:
    def __init__(self, in_ch, out_ch, kernel):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel

    def calcOutputLen(self, in_len, pool_kernel=1):
        out_len = in_len - self.kernel_size + 1
        assert out_len > 0
        # TODO throw exception

        return calcPoolOutSize(out_len, pool_kernel)
