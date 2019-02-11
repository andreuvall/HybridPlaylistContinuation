
import numpy as np
from lasagne.utils import floatX


def get_constant():
    """
    Keep learning rate constant
    """
    def update(lr, epoch):
        return lr

    return update


def get_stepwise(k=10, factor=0.5):
    """
    Stepwise learning rate update every k epochs
    """
    def update(lr, epoch):

        if epoch >= 0 and np.mod(epoch, k) == 0:
            return floatX(factor * lr)
        else:
            return floatX(lr)

    return update


def get_predefined(schedule):
    """
    Predefined learn rate changes at specified epochs
    :param schedule:  dictionary that maps epochs to to learn rate values.
    """

    def update(lr, epoch):
        if epoch in schedule:
            return floatX(schedule[epoch])
        else:
            return floatX(lr)

    return update


def get_linear(start_at, ini_lr, decrease_epochs):
    """ linear learn rate schedule"""

    def update(lr, epoch):
        if epoch < start_at:
            return floatX(lr)
        else:
            k = ini_lr / decrease_epochs
            return floatX(np.max([0.0, lr - k]))

    return update


if __name__ == "__main__":
    """ main """
    import matplotlib.pyplot as plt

    # test linear schedule
    lr = 0.1
    update_lr = get_linear(start_at=0, ini_lr=lr, decrease_epochs=100)
    lrs = []
    for i in xrange(120):
        lrs.append(lr)
        lr = update_lr(lr, i)

    plt.figure("Linear")
    plt.plot(lrs, "k-")
    plt.grid("on")
    plt.show()