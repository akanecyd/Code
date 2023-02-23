import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def outlier_annotated_bland_altman_plot(data1, data2, *args, **kwargs):
    """
    Creates a Bland-Altman_plot

    """
    pat_list = None
    if 'pat_list' in kwargs:
        pat_list = kwargs['pat_list']
    plt.figure(figsize=kwargs['figsize'], dpi=80)
    
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    upper_lim = md + 1.96*sd
    lower_lim = md - 1.96*sd

    # Plot
    plt.scatter(mean, diff,alpha=kwargs['alpha']) #, *args, **kwargs
    plt.axhline(md,        color='gray', linestyle='--')
    plt.axhline(upper_lim, color='gray', linestyle='--')
    plt.axhline(lower_lim, color='gray', linestyle='--')
    
    # Add annotations
    _x = np.min(mean)
    plt.text(_x, upper_lim, '+1.96 SD: %6.3f' % (upper_lim),fontsize=12)
    plt.text(_x, lower_lim, '-1.96 SD: %6.3f' % (lower_lim),fontsize=12)
    
    if pat_list:
        idxs = []
        sorted_diff_idx = np.argsort(diff)
        idxs = np.concatenate([sorted_diff_idx[0:5], sorted_diff_idx[-5:]])
        
        for _idx in idxs:
            _x = mean[_idx]+0.1*mean[_idx]*(np.random.uniform()-0.5)
            _y = diff[_idx]+0.05*mean[_idx]*(np.random.uniform()-0.5)
            _txt = pat_list[_idx]
            plt.text(_x, _y, _txt)

        return idxs

def confidence_ellipse(x, y, ax, col, n_std=3.0, plot_mean=False, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    ellipse.set_alpha(0.2)
    ellipse.set_linewidth(5)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    if plot_mean is True:
        ax.plot(mean_x, mean_y, '*', color=col)
    return ax.add_patch(ellipse)