def calcConv1dOutSize(in_sz, kernel_sz):
    out = in_sz - kernel_sz + 1
    assert out > 0
    # TODO throw exception

    return out


def calcPoolOutSize(in_sz, pool_sz):
    out_float = in_sz / pool_sz
    out = int(out_float)

    assert abs(out - out_float) < 1e-4
    # TODO throw exception

    return out


def calcConv1dPoolOutSize(in_sz, kernel_sz, pool_sz):
    return calcPoolOutSize(calcConv1dOutSize(in_sz, kernel_sz), pool_sz)
