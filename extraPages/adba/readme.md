# AD-BA

Building a simple Bundle Adjuster in pytorch.

## Bundle Adjustment
Bundle Adjustment (in this case) is the process of refining the point locations and camera parameters, based on 2d point correspondances. When a camera observes a point, it generates one observation and two residuals. Cameras are not related to each other directly, but in viewing the same points, they indirectly rely on one-another. BA takes these relationships and refines an initial collection of points and poses in order to make it more consistent. This is done by formulating a least-squares problem: there are 2d point observations, 3d landmarks, and 7d poses. The 2d points are fixed, the landmarks and poses are shifted around to minimize the reprojection error (and possibly other constraints). Because camera projection and orientation updates are nonlinear, this least-squares problem cannot be solved in one step. Because there may be many points, and one camera will see a small subset of them (and there are no point-point or camera-camera relationships), the graph will be very sparse. BA software must take this into account because the full `NxN` graph would be too large to form.

### Gauss-Newton
A good way non-linear least-squares to formulate least-squares is by considering the normal equations: `J'Jx = J'res`.
`J` is a `MxN` matrix, where `M` is the number of residuals/observations, and `N` is the number of states (flattened state vector). `J` is the stacked jacobian matrix. Each row comes from the differentiating a residual w.r.t the state. Because we mostly care about pose-landmark relationships, each row will typically have only `3+7` non-zeros. So J may be very sparse.
`x` is the increment to bring us to the next estimate. Solving for it is the goal of one iteration.
`res` is the flattened vector of residuals.

For the initial implementation, I will explicitly form the approximate hessian matrix `J'J`, which will also be pretty sparse, but could have lots of fill in. Then use a sparse matrix solver for `Hx = J'res` to get `x`, apply the increment, and repeat the process.


## Simple Model
Orientations are best represented by quaternions and rotation vector increments, but to make using AD simpler, let's just use an additive quaternion-increment model. This means we can just use AD as normal to get a quaternion component increments, which we than add to the state. After that, the quaternions are no longer unit-norm and so they must be renormalized. This process introduces some error because you are stepping off the manifold then projecting back to it, but with good initializations it isn't a big issue.

Only points and poses are represented. A prior will be put on both kinds of nodes, so they should have good initializations. Note: the prior residual is not tied to the original input, but rather the last estimate. So it is more of a dampening then a prior.

No outlier rejection yet, but because of AD it will be easy to support robust cost functions etc.

## `ba.py` Aproaches
### Approach 1: Dense
This uses the `torch.autograd.functional.jacobian` function. It does not scale well.

### Approach 2: Sparse
Both versions still call `H.to_dense()` because `torch.linalg.solve` does not support sparse matrices. This could be circumvented by using `scipy`, though...

#### Version 1
This again uses the `torch.autograd.functional.jacobian` function. There must be a python loop over each observation, and then the AD function is called for every one. This linearization step takes a long time.

#### Version 2
This uses the new `functorch.jacfwd` combined with `functorch.vmap`. It is *much* more efficient. I think Jax would be even more efficient, because in that library you can JIT compile your transformed functions.
In functorch, compiling with the TorchScript compiler actually made it slower. Luckily just using vmap already made it pretty performant.

The issue with any of these approaches, and it would apply to using Jax as well, is that the python math libs work best with tensors that are as 'batched' as possible, meaning inputs have same shapes.
But in BA, some cameras may observe 10 pts and others may observe 1000, so we have to constrain the problem to some fixed number, or do some complicated business of grouping into different maximum-size bins. In either case you need to zero-pad inputs.


### Approach 3: Conjugate-Gradient
This avoids the `J.T * J` approximate hessian matrix formation. It is a fancy algorithm that would fit functorch or Jax very well, because they both provide `jvp/vjp` functions which are more efficient then even forming `J`, let alone the product.
Not even started yet.
