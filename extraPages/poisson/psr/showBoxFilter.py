import torch, torch.nn.functional as F, numpy as np
import matplotlib.pyplot as plt

# Do not worry about quadratic fit or whatever.
# Just use gaussian since it is pretty damn close.
# When estimating coarse to fine, use support dictated by 3rd box filter convolved with itself.
# But DO NOT precompute weights, just compute the gaussian gradients inline!

def do_conv(y,k):
    N = y.numel()
    y = y.view(1,1,N)
    y = F.conv1d(y, k.view(1,1,-1), padding=N//2)[0,0]
    print(y.shape)
    return y

def show_it():
    # x = torch.linspace(-4, 4, 1023)
    x = torch.linspace(-6, 6, 2047)

    box = (x.abs() < .5).float()

    # This is pretty close for 29* filter reps.
    # sig = 1.59
    # exp = 1 * (-.5*(x*x/(sig*sig))).exp() / np.sqrt(2*sig*sig*np.pi)
    # This is pretty close for 3* filter reps.
    sig = .56
    # sig = 0.02371
    exp = 1 * (-.5*(x/sig)**2).exp() / np.sqrt(2*sig*sig*np.pi)

    ys = [box/box.sum()]
    k = box / box.sum()

    for i in range(30):
        ys.append(do_conv(ys[-1], k))

        p = ys[-1]
        p = p / p.sum()
        samples = x*p
        mu = samples.mean().item()
        std = ((samples-mu) * (samples-mu)).sum().sqrt()
        print(f' - rep *{i:>2d} mu: {mu:.3f} std: {std:.5f}')
    ys = [(i,y) for (i,y) in enumerate(ys) if i in [0,1,2,3,7,15,29]]

    fig,axs = plt.subplots(len(ys), 2)
    plt.rcParams['axes.titley'] = 1.0

    for j,(i,y) in enumerate(ys):
        xx = x.cpu().numpy()
        yy = y.cpu().numpy() * (exp.max().item() / y.max().item())
        eexp = exp.cpu().numpy()

        axs[j,0].plot(xx,yy, color='b', label=str(i))
        axs[j,0].plot(xx,eexp, color='orange')

        axs[j,1].plot(xx,yy-eexp, color='r')

        axs[j,0].set_title(str(i))
        axs[j,1].set_title(str(i))

        if j < len(ys)-1:
            axs[j,0].set_xticks([])
            axs[j,1].set_xticks([])
    fig.suptitle("Repeated Box Filter vs Gaussian, and its error")
    plt.show()

show_it()
