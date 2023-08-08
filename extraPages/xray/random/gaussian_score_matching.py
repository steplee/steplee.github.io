import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn
plt.rcParams['figure.figsize'] = (10,10)
plt.style.use('dark_background')
plt.tight_layout()

lr = .1

# Just a warmup: implement example from Hyvarinen paper
def score_match_gaussian(samples):
    mu = np.zeros(3)
    info = np.eye(3)

    for epoch in range(100):
        o = samples - mu[np.newaxis]
        o1 = o.reshape(100,3,1)
        o2 = o.reshape(100,1,3)
        oo = (o1@o2).mean(0)

        d_mu = info@info@mu - info@info@samples.mean(0)
        d_info = -np.eye(3) + .5*info@oo + .5*oo@info

        mu -= d_mu * lr
        info -= d_info * lr

    print('final:')
    print('   - mu:\n', mu)
    print('   - cov:\n', np.linalg.inv(info))

    sampleMean = samples.mean(0)
    sampleCov = ((samples-sampleMean[np.newaxis]).reshape(-1,3,1) @ (samples-sampleMean).reshape(-1,1,3)).mean(0)
    sampleInfo = np.linalg.inv(sampleCov)
    print('\ntrue:')
    print('   - mu  :\n', sampleMean)
    print('   - cov:\n', sampleCov)

# Sample a gaussian mixture model
def sample_gmm(gausses, N):
    norm = sum([g[0] for g in gausses])
    weights = [g[0]/norm for g in gausses]
    ids = np.random.choice(len(gausses), size=N, p=weights)
    mus, covs = np.array([gausses[i][1] for i in ids]), np.array([gausses[i][2] for i in ids])
    x = np.random.randn(N,2)
    x = (x.reshape(N,1,2) @ covs)[:,0] + mus
    return x


# Compute a 2d gradient field from a 2d scalar field
def numerical_gradient(f):
    g = np.zeros((f.shape[0],f.shape[1],2))
    g[1:,  :, 1] += f[1:] - f[:-1]
    g[:, 1:,  0] += f[:, 1:] - f[:, :-1]

    g[:-1, :, 1] -= f[:-1] - f[1:]
    g[:, :-1, 0] -= f[:, :-1] - f[:, 1:]

    return g/2

# Works on 2d or 3d input
def eval_density_points(gausses, xy):
    density = np.zeros_like(xy[...,0])
    for alpha,mu,cov in gausses:
        d = (xy - mu)
        dd = (d * (d @ np.linalg.inv(cov).T)).sum(-1)
        density += alpha * np.exp(-.5 * dd)
    density = density / density.sum()
    return density

def eval_density_grid(gausses, S,N):
    X,Y = np.linspace(-S,S,N), np.linspace(-S,S,N)
    xy = np.stack(np.meshgrid(X,Y), -1)
    density = eval_density_points(gausses, xy)
    return xy,density

def plot_modes_and_samples(gausses, samples1, samples2,
        predXY=None,
        predGrad=None,
        wait=True,
        fig=None,
        cmesh=None,
        S=4,N=250,
        title=''):

    if fig is None:
        fig = plt.figure()
    else:
        fig.clf()
    plt.figure(fig.number)
    plt.xlim(-S,S)
    plt.ylim(-S,S)
    fig.suptitle(title)

    if 0:
        if cmesh is None:
            xy,density = eval_density_grid(gausses, S,N)
            X,Y = xy[...,0],xy[...,1]
            plt.pcolormesh(X,Y,density, cmap='inferno')
        else:
            # Does not work
            fig.axes[0].add_artist(cmesh)

        if 0:
            D = 4
            xy1 = xy[::D,::D][1:-1,1:-1]
            log_dens = np.log(density[::D,::D])
            uv = numerical_gradient(log_dens)[1:-1,1:-1]
            x,y = xy1[...,0], xy1[...,1]
            u,v = uv[...,0], uv[...,1]
            plt.quiver(x,y, u,v, color=(0,.5,0,.5), width=.002,headwidth=2,headlength=2,headaxislength=2)

    if samples1 is not None and len(samples1):
        plt.plot(samples1[:,0], samples1[:,1], '.', color=(.5,1,.5,.7), markersize=.4)
    if samples2 is not None and len(samples2):
        plt.plot(samples2[:,0], samples2[:,1], '.', color=(.1,.3,.99,.7), markersize=.4)

    if predGrad is not None:
        x,y = predXY[...,0], predXY[...,1]
        u,v = predGrad[...,0], predGrad[...,1]
        plt.quiver(x,y,u,v, color=(0.96,.05,.0,.5), width=.002,headwidth=2,headlength=2,headaxislength=2)


    if wait:
        plt.show(block=True)
    else:
        plt.show(block=False)
        plt.pause(.0001)


def learn_log_grad_then_eval_on_grid(trainXY, trainTarget, gridXY):

    trainXY = torch.from_numpy(trainXY).float().cuda()
    trainTarget = torch.from_numpy(trainTarget).float().cuda()

    net = torch.nn.Sequential(
            nn.Linear(2,32, bias=False), nn.BatchNorm1d(32), nn.ReLU(True),
            nn.Linear(32,64, bias=False), nn.BatchNorm1d(64), nn.ReLU(True),
            nn.Linear(64,64, bias=False), nn.BatchNorm1d(64), nn.ReLU(True),
            # nn.Linear(64,128, bias=False), nn.BatchNorm1d(128), nn.ReLU(True),
            # nn.Linear(128,64, bias=False), nn.BatchNorm1d(64), nn.ReLU(True),
            nn.Linear(64,2)).cuda().train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)

    pred = net(trainXY)
    loss = (pred - trainTarget).pow(2).sum(-1).mean()
    print('pre-loss',loss.item())
    for epoch in range(240):
        pred = net(trainXY)
        loss = (pred - trainTarget).pow(2).sum(-1).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        if epoch % 25 == 0:
            print('loss',loss.item())

    gridXY = torch.from_numpy(gridXY).float().cuda()
    pred = net.eval()(gridXY).detach().cpu().numpy()
    return pred


def get_gmm():
    a1,m1,c1 = 2,np.array((1,1)), np.eye(2) * np.square(.7)
    a2,m2,c2 = 1,np.array((-2,-2)), np.eye(2) * np.square(.7)
    gausses = [[a1,m1,c1], [a2,m2,c2]]

    a2,m2,c2 = 1,np.array((1,-2)), np.eye(2) * np.square(.2)
    gausses.append([a2,m2,c2])

    return gausses

def multi_mode_test():
    gausses = get_gmm()
    # gausses = gausses[:1]

    # x_data = sample_gmm(gausses, 100)

    # The grid we will eval on
    N = 100
    gridXY,_ = eval_density_grid(gausses, S=4,N=N)
    gridXY = gridXY.reshape(-1,2)


    # When you train with samples covering a large rectangle, you get good results (obviously)
    # When you train with samples coming from the GMM, you get only good results near the samples.
    for i in range(3):
        if i==0:
            trainXY,density = eval_density_grid(gausses, S=4,N=N)
            trainTarget = numerical_gradient(np.log(density))
            x_data = None
            x_data2 = None
        if i==1:
            trainXY,density = eval_density_grid(gausses, S=4,N=1000)
            selected = np.random.choice(trainXY.shape[0]*trainXY.shape[1], size=N*N, p=density.reshape(-1), replace=False)
            trainTarget = numerical_gradient(np.log(density))
            trainXY = trainXY.reshape(-1,2)[selected]
            trainTarget = trainTarget.reshape(-1,2)[selected]
            x_data = trainXY
            x_data2 = None
        if i==2:
            xy_original,density = eval_density_grid(gausses, S=4,N=1000)
            selected = np.random.choice(xy_original.shape[0]*xy_original.shape[1], size=N*N, p=density.reshape(-1), replace=False)
            # trainTarget = numerical_gradient(np.log(density))
            # trainTarget = trainTarget.reshape(-1,2)[selected]
            xy_original = xy_original.reshape(-1,2)[selected]
            trainXY = xy_original + np.random.randn(*trainXY.shape) * .8
            trainTarget = xy_original - trainXY
            x_data2 = trainXY
            x_data = xy_original
            # x_data=x_data2=None

        trainXY = trainXY.reshape(-1,2)
        trainTarget = trainTarget.reshape(-1,2)

        predGrad = learn_log_grad_then_eval_on_grid(trainXY, trainTarget, gridXY)
        predGrad = predGrad.reshape(N,N,2)

        if i == 0: title = 'Sample from support region'
        if i == 1: title = 'Sample from GMM'
        if i == 2: title = 'Sample from GMM + Denoising Score Matching'
        plot_modes_and_samples(gausses, x_data, x_data2,
                            predXY=gridXY, predGrad=predGrad, wait=i==2, title=title)



def conditional_learn_log_grad_then_eval_on_grid(trainXYN, trainTarget, gridXY, gridNoiseLevels, callback):
    import torch, torch.nn as nn

    trainXYN = torch.from_numpy(trainXYN).float()
    trainTarget = torch.from_numpy(trainTarget).float()

    dset = torch.utils.data.TensorDataset(trainXYN, trainTarget)
    g = torch.Generator().manual_seed(0)
    dloader = torch.utils.data.DataLoader(dset, batch_size=256, shuffle=True, generator=g)

    # The third input shall be the noise level.
    net = torch.nn.Sequential(
            nn.Linear(3,32, bias=False), nn.BatchNorm1d(32), nn.ReLU(True),
            nn.Linear(32,64, bias=False), nn.BatchNorm1d(64), nn.ReLU(True),
            # nn.Linear(64,64, bias=False), nn.BatchNorm1d(64), nn.ReLU(True),
            nn.Linear(64,128, bias=False), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Linear(128,64, bias=False), nn.BatchNorm1d(64), nn.ReLU(True),
            nn.Linear(64,2)).cuda().train()
    # opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    opt = torch.optim.Adam(net.parameters(), lr=1e-5)

    print(' - Begin train')
    try:
        for epoch in range(240):
            nn,loss_ = 0,0
            for x,y in dloader:
                pred = net(x.cuda())
                loss = (pred - y.cuda()).pow(2).sum(-1).mean()
                loss.backward()
                opt.step()
                opt.zero_grad()
                loss_ += loss.item()
                nn += 1
            # if epoch % 25 == 0: print('loss',loss.item())
            if epoch % 1 == 0: print('ep {:>4d}'.format(epoch),'loss', loss_/nn)
    except KeyboardInterrupt:
        pass

    gridXY = torch.from_numpy(gridXY).float().cuda()
    for n in gridNoiseLevels[::-1]:
        gridXYN = torch.cat((gridXY, torch.ones_like(gridXY)[...,:1]*n), -1)
        pred = net.eval()(gridXYN).detach().cpu().numpy()
        callback(pred, n)

def noise_conditional_test():
    torch.manual_seed(0)
    np.random.seed(0)

    gausses = get_gmm()


    N = 50
    gridXY,_ = eval_density_grid(gausses, S=4,N=N)
    gridXY = gridXY.reshape(-1,2)

    trainXYN, trainTarget = [], []
    x_data,x_data2 = None, None

    noiseLevels = np.linspace(0,2,50)

    for noise in noiseLevels:
        xy_original,density = eval_density_grid(gausses, S=4,N=1000)
        selected = np.random.choice(xy_original.shape[0]*xy_original.shape[1], size=N*N, p=density.reshape(-1), replace=False)
        xy_original = xy_original.reshape(-1,2)[selected]

        x = xy_original + np.random.randn(*xy_original.shape) * noise*noise
        x = np.concatenate((x, np.ones_like(x[...,:1])*noise), -1)
        y = xy_original - x[...,:2]
        trainXYN.append(x)
        trainTarget.append(y)

    trainXYN = np.concatenate(trainXYN, 0)
    trainTarget = np.concatenate(trainTarget, 0)

    fig = plt.figure()
    S,N=4,50
    xy,density = eval_density_grid(gausses, S,N)
    X,Y = xy[...,0],xy[...,1]
    # cmesh=plt.pcolormesh(X,Y,density, cmap='inferno')
    cmesh=None

    def callback(predGrad, noise):
        plot_modes_and_samples(gausses, None,None, fig=fig, cmesh=cmesh,N=50,
                            predXY=gridXY, predGrad=predGrad, wait=False, title='σ = {:.2f}'.format(noise))


    conditional_learn_log_grad_then_eval_on_grid(trainXYN, trainTarget, gridXY, noiseLevels, callback)

# xs = np.random.randn(100,3) * (4,1,1)
# score_match_gaussian(xs)

# multi_mode_test()
noise_conditional_test()
