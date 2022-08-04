# Ori

## RollPitchYaw
	- Jacobian for Time Step
		- To compute the Jacobian of the time step, you need to do like `matrix_to_rpy(R{rpy} @ Exp(k))` where `R{rpy}` creates a 3x3 R matrix from the RPY vector `rpy`, `Exp` applies rodrigues formulat on rotation vector `k`, and `matrix_to_rpy` converts the 3x3 R back to the rpy components.
		- The Jacobian of interest is then the derivatives of the output rpy with the inputs.
	- Jacobian for Measurement Step
		- This is where it gets interesting :)
		- You can model the magnometer measurement as just one angle, then use some atan-based function as the EKF and get jacobian for `H`. Same goes for gravity, but it gives info on a combination of roll and pitch.
		- But I think I prefer modelling not as angles, but as the original vectors. The error function must change. It is not the the residual `z - Hx`. Then the standard EKF equations don't exactly apply.
		- What we end up with is something I've seen called an "Implicit EKF".
		- The key to understanding it to consider the measurement covariance `R`.
			- Think of it like this: if your EKF has 3 states and you have one measurement function with 2 observable quantities, then your `H` matrix is 2x3 and your `R` matrix is 2x2.
			- In the Implicit EKF, `R` is the state size (3x3). This is kind of a generalization, because you could think of the original `R` being 3x3 but with rank 2.
			- In fact the `R` matrix could have rank 2, as long as `HPH'` spans the missing subspace, since `S = HPH' + R`
			- The `R` matrix is the inverse Hessian! TODO: Explain from a NLLS/GN viewpoint

	- Scrap the ekf update step :: just formulate as GN step...
