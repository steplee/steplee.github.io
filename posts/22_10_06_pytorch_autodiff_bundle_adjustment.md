```meta
title: AD BA
date: 2022/10/06
tags: pytorch, math, optimization, least-squares, bundle adjustment, autodiff
```
# AD BA

I had off today and decided to explore how far I could get using some of pytorch's new features in creating a simple Bundle Adjuster in pure python.

 > Code is [here](https://github.com/steplee/steplee.github.io/tree/master/extraPages/adba)

I've used the popular Ceres and g2o libraries for similar things in the past.
I even tried to code up a BA myself in C++, but my interest waned after a day or two of slow progress.
But it was more fun this time around with pytorch. Within a day I got some nice results and even built a point-and-line vizualizer for it, in the spirit of pixel shaders but using pytorch because why not?

## First try
My first initial protoype was to `torch.autograd.functional.jacobian` to get the needed jacobian matrices, pack them in a dense matrix, and solve the normal equations.
This works, but it required a lot of python loops and individual invotaions of the jacobian function. It was cripplingly slow, and there was no way to vectorize out of it.

Not only that but using dense matrices was not going to scale. So I switched to actually forming a sparse graph, and using pytorch's sparse matrix types to help out.

Pytorch's `sparse_coo_matrix` is amazingly useful. I've used it for all kinds of non-standard things, like as a replacement to an octree, where I only needed the sum of values contained in subnodes, and to vectorize the cluster-mean computation step of the kmeans algorithm. Here I used it more for it's typical use case as storing a sparse matrix.

The downside is that `torch.linalg.solve` does not support sparse matrices, so I still need to densify it. Suprisingly the densification and solve even for 3000^2 sized matrices is not a bottleneck (just a few milliseconds).
When this was a bottleneck, the plan is to directly pass the sparse representation to an actual sparse matrix solver.

## Using functorch
So calling the autograd jacobian function lots of times is slow. Luckily pytorch introduced new features in `functorch`, which add Jax-like capability. I've used Jax before to code up an EKF and it was really nice. Unfortunately at the time it was not supported on ARM, where I wanted to use it, and it required building TensorFlow, which was insufferable.

With functorch, you can transform functions using an explicit vectorization higher-order function called `vmap`. You can also get jacobian-vector products, which I'll talk about below. For now the important features are `vmap` and `jacfwd`/`jacrev`. `jacfwd` allows getting the jacobian matrices of a function, and doing it for multiple inputs if desired. Combining `vmap` and `jacfwd` is exactly what we need to do to avoid the bottleneck in the previous approach. Figuring out how to mix them properly, and how to deciper the rank-6 resulting tensor took me a while.

All in all, the full process is very simple, an with the functorch features, suprisingly efficient.
In fact the current bottleneck is filling in the sparse graph, which I currently do in a nested python loop and with appending a bunch of numbers to python lists.
This is greay news, because this step could be easily optimized (in a C snippet, or hopefully with an easy to use JIT compiler like numba)

One downside to the functorch/jax approach is that it typically requires a fixed batch size. Specifically, every camera must observed the same number of points (otherwise we can't vectorize out the cameras). This is not a huge issue though, you can just use a large enough axis length to fit the camera with the most point observations, and zero out any invalid data later. Or do that, but with a small number groups.

Next I worked on a viewer for the optimization process. I didn't want to work with the same OpenGL code I have a million times, so I worked on a pytorch cuda tensor based class. The only difficult part was copying a 2d 'distanceToLine' function from some old GLSL code and vectorizing it with pytorch.

Here is some results in a scene with a bad initialization.

![Results](/res/adba/steps.gif)


## Conjugate Gradients

Even if I allow a sparse optimizer to solve `J.T @ J`, it is still expensive to form it for large systems. The Conjugate Gradient algorithm avoids the normal equations, and I believe is the preferred method in BA.
I believe the `jvp` and `vjp` functions from functorch/jax could be used too, to avoid any large stacked-jacobian matrix.

## Robustness
A huge desiderada for a BA system is robustness. Currently this assumes no outliers and has no robust loss functions. That would probably not be hard, because of how easy everything is with AD!

# TODO: Finish writing this

