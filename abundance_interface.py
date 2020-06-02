from __future__ import division
import numpy as np
import abundance

def setup(options):
    number_count = abundance.NumberCount(options)
    return number_count

def execute(block, number_count):
    lnlike = float(number_count.lnlike(block))
    if np.isneginf(lnlike):
        return 1
    block.put_double('likelihoods', 'ABUNDANCE_LIKE', lnlike)
    return 0

def cleanup(config):
    pass
