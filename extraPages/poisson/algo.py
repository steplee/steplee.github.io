import torch, torch.nn.functional as F
import numpy as np, sys
torch.set_printoptions(linewidth=150, edgeitems=10)

# Vizualize the box filter (iterated three times) in 2D.
# The 3d section would suggest that Khazdan 2006 is talking about the regular (non-manhattan) box blur.
# It has roughly twice the number of non-zeros.
def viz_box():
    import cv2

    # Show box blur
    if 1:
        print('\nDoing 2d regular blur.')
        hw = 16
        img = torch.zeros((hw,hw), dtype=torch.float32).cuda()
        img[hw//2, hw//2] = 1
        imgs = [img]
        for i in range(3):
            img = F.avg_pool2d(img.unsqueeze(0).unsqueeze(0), 3, 1, padding=1)[0,0]
            imgs.append(img)
        imgs = (torch.stack(imgs, 0))

        print(imgs.view(imgs.size(0),-1).sum(1)) # Sum is preserved
        print('nonzeros per iter', (imgs>0).view(imgs.size(0),-1).sum(1))
        # imgs.sqrt_() # Easier to see.

        imgs = (imgs.cpu().numpy() * 255).astype(np.uint8)
        imgs = np.hstack(imgs)
        imgs = cv2.resize(imgs, (0,0), fx=16, fy=16, interpolation=cv2.INTER_NEAREST)
        cv2.imshow('box', imgs)

    # Show manhattan box blur. Not sure canonical name.
    if 1:
        print('\nDoing 2d manhattan blur.')
        hw = 16
        img = torch.zeros((hw,hw), dtype=torch.float32).cuda()
        img[hw//2, hw//2] = 1
        imgs = [img]
        kernel = torch.zeros((1,1,3,3)).cuda()
        for dy in range(-1,2):
            for dx in range(-1,2):
                if dy == 0 or dx == 0:
                    kernel[:,:,dy+1,dx+1] = 1
        kernel.div_(kernel.sum())
        for i in range(3):
            img = F.conv2d(img.unsqueeze(0).unsqueeze(0), kernel, stride=1, padding=1)[0,0]
            imgs.append(img)
        imgs = (torch.stack(imgs, 0))

        print(imgs.view(imgs.size(0),-1).sum(1)) # Sum is preserved
        print('nonzeros per iter', (imgs>0).view(imgs.size(0),-1).sum(1))
        # imgs.sqrt_() # Easier to see.

        imgs = (imgs.cpu().numpy() * 255).astype(np.uint8)
        imgs = np.hstack(imgs)
        imgs = cv2.resize(imgs, (0,0), fx=16, fy=16, interpolation=cv2.INTER_NEAREST)
        cv2.imshow('manhattan box', imgs)

    # Compute regular and manhattan box blur.
    if 1:
        print('\nDoing 3d regular blur.')
        hw = 16
        img = torch.zeros((hw,hw,hw), dtype=torch.float32).cuda()
        img[hw//2, hw//2, hw//2] = 1
        imgs = [img]
        kernel = torch.zeros((1,1,3,3,3)).cuda()
        for dz in range(-1,2):
            for dy in range(-1,2):
                for dx in range(-1,2):
                    kernel[:,:,dz+1,dy+1,dx+1] = 1
        kernel.div_(kernel.sum())
        for i in range(3):
            img = F.conv3d(img.unsqueeze(0).unsqueeze(0), kernel, stride=1, padding=1)[0,0]
            imgs.append(img)
        imgs = (torch.stack(imgs, 0))

        print(imgs.view(imgs.size(0),-1).sum(1)) # Sum is preserved
        print('nonzeros per iter', (imgs>0).view(imgs.size(0),-1).sum(1))

        print('\nDoing 3d manhattan blur.')
        hw = 16
        img = torch.zeros((hw,hw,hw), dtype=torch.float32).cuda()
        img[hw//2, hw//2, hw//2] = 1
        imgs = [img]
        kernel = torch.zeros((1,1,3,3,3)).cuda()
        for dz in range(-1,2):
            for dy in range(-1,2):
                for dx in range(-1,2):
                    if dy == 0 or dx == 0:
                        kernel[:,:,dz+1,dy+1,dx+1] = 1
        kernel.div_(kernel.sum())
        for i in range(3):
            img = F.conv3d(img.unsqueeze(0).unsqueeze(0), kernel, stride=1, padding=1)[0,0]
            imgs.append(img)
        imgs = (torch.stack(imgs, 0))

        print(imgs.view(imgs.size(0),-1).sum(1)) # Sum is preserved
        print('nonzeros per iter', (imgs>0).view(imgs.size(0),-1).sum(1))

    cv2.waitKey(0)

# def get_stencil(iteration=2, kind='manhattan', size=-1, normalize='l1'):
def get_stencil(iteration=2, kind='regular', size=-1, normalize='l1'):
    if size < 0: size = (iteration)*2 + 1

    img = torch.zeros((1,1,size,size,size), dtype=torch.float32).cuda()
    img[:, :, size//2, size//2, size//2] = 1
    kernel = torch.zeros((1,1,3,3,3)).cuda()
    for dz in range(-1,2):
        for dy in range(-1,2):
            for dx in range(-1,2):
                if kind == 'manhattan':
                    if dy == 0 or dx == 0:
                        kernel[:,:,dz+1,dy+1,dx+1] = 1
                elif kind == 'regular':
                        kernel[:,:,dz+1,dy+1,dx+1] = 1
                else:
                    assert False

    for i in range(iteration):
        img = F.conv3d(img, kernel, stride=1, padding=1)

    if normalize == 'l1': img.div_(img.sum())
    elif normalize == 'l2': img.div_(img.norm())
    elif normalize == 'none' or normalize == None: pass
    else: assert False

    return img[0,0]


# Sparse 3d convolutions.
def form_system(pts, nrmls):
    stencil = get_stencil()
    S = stencil.size(0)

    # Iterate over the stencil, replicate
    stencil = stencil.to_sparse().coalesce()
    for D,W in zip(stencil.indices().t(), stencil.values()):
        print(D,W)





pts   = torch.randn(512, 3).cuda()
nrmls = torch.randn(512, 3).cuda()
form_system(pts, nrmls)
viz_box()
