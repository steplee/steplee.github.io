# Relationship Between the EKF and Gauss-Newton Method

It's no secret that the EKF is related to Gauss-Newton optimization. They are both used to solve non-linear least-squares problems. Typically Kalman Filters are used in state-estimation problems where there is a time aspect. Gauss-Newton is more popular in static optimization problems like bundle adjustment and function fitting. There are academic papers exploring comparing the two, but I want to write a post to try and build more intuition. Both EKF and GN can be derived from first principles using the [Matrix Inversion Lemma](https://en.wikipedia.org/wiki/Woodbury_matrix_identity), but that is a little impractical.

The Information Filter is similar to the Kalman Filter and very similar to GN. In the KF, the state is the covariance matrix and the mean vector (the parameters of a Gaussian in `moment` form). In the Information Filter, the state is the inverse-covariance matrix and the 'information vector' (the parameters of a Gaussian in `natural` form).
Going between these representations requires a matrix inverse operation, which is `O(n^3)`. So chances are you'd like to avoid doing it often (it also loses a lot of precision which may be another concern).

Back to the optimization terminology, it turns out the inverse covariance \(P^{-1}\) is *almost* the Hessian \(\Lambda\) evaluated at the current estimate. You just need to propagate the covariance along, by just adding \( P^{-1}\) into \(\Lambda\).
Everything then falls into place and you can see the duality of the two representations.

 - Marginalization is "easy" in the EKF, but conditioning (the update step) requires inverting the \(S\) matrix.
 - Conditioning is "easy" in EIF/Gauss-Newton because you just add an outer-product to the Hessian and the \(J^T \cdot residual\) term to the information vector, but marginalization requires inverting the Hessian \(\Lambda\).

There is no free lunch -- you can't just keep making measurments/conditioning in IEF/Gauss-Newton, because you need the next mean vector in order to evaluate a new Hessian!

Coming back to Earth, sometimes it is easier for me to reason about the measurement step in GN than in the EKF formulation. It's easier to replace the measurement step while keeping the standard time step.

###### EKF Step
$$
x_1 = x_0 + PH^T(HPH^T+R)^{-1} \cdot \hat{y}
$$

###### GN Step
$$
x_1 = x_0 + (J^T\Lambda J + P^{-1})^{-1}J^T \cdot \hat{y}
$$

TODO: Finish
