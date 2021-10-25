import numpy as np
# Generate functions using np (wthout torch)
def gabor_fn(theta, sigma=2, Lambda=10, psi=np.pi/2, gamma=.8, center=(0, 0), size=(28, 28)):
    """Returns a gabor filter.
    Args:
        theta (float): Orientation of the sinusoid (in radian).
        sigma (float): std deviation of the Gaussian.
        Lambda (float): Sinusoid wavelengh (1/frequency).
        psi (float): Phase of the sinusoid.
        gamma (float): The ratio between sigma in x-dim over sigma in y-dim (acts
            like an aspect ratio of the Gaussian).
        center (tuple of integers): The position of the filter.
        image_size (tuple of integers): Image height and width.
        
    Returns:
        2D Numpy array: A gabor filter.
    """

    sigma_x = sigma
    sigma_y = sigma / gamma

    xmax, ymax = size
    xmax, ymax = (xmax - 1)/2, (ymax - 1)/2
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax+1), np.arange(xmin, xmax+1))

    # shift the positon
    y -= center[0]
    x -= center[1]

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)

    return gb


def dog_fn(sigma1=2, sigma2=4, amp1=4,amp2=1, center=(0, 0), size=(28, 28), offset=0):
    """Returns a DOG filter.
    Args:
            sigma1(float): std deviation of the center Gaussian.
            sigma2(float): std of the surround Gaussian
            amp1(float): amplitude of the centre
            amp2(float): amp of the surround
            center (tuple of integers): The position of the filter.
            image_size (tuple of integers): Image height and width.
            target_std:
    Returns:
        2D Numpy array: A DOG filter.
    """

    xmax, ymax = size
    xmax, ymax = (xmax - 1)/2, (ymax - 1)/2
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax+1), np.arange(xmin, xmax+1))

    # shift the positon
    y -= center[0]
    x -= center[1]


    dog = amp1*np.exp(-.5 * (x ** 2 / sigma1 ** 2 + y ** 2 / sigma1 ** 2)) - amp2*np.exp(-.5 * (x ** 2 / sigma2 ** 2 + y ** 2 / sigma ** 2))
    
    dog=dog+offset

    return dog