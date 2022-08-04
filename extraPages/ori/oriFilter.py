import numpy as np, os, sys
from numpy.linalg import inv as inv
import dominate, dominate.tags as D
from dominate.util import raw
import sympy, time, json
from sympy import symbols, Matrix, cos as cos_, sin as sin_, diff, simplify, pprint, count_ops, cse, Abs
from sympy.utilities.lambdify import lambdify
np.set_printoptions(suppress=True)
from matplotlib import pyplot as plt
np.random.seed(0)

'''

This will qualitatively and quantitavely evaluate different orientation state
representations for an EKF.

Simulated data is generated (gyro for time evolution, magnometer + accel as
measurements), and the filters can be run on them.
There's two measurements:
    one from acceloremeter (when low pass filtered, predicts down),
    one from magnometer (predicts north)
These are dotted with the appropriate row of the orientation matrix to
give two scalar residuals for the EKF.

Estimates from filters are evaluated via statistics and a WebGL viz.

First I will start with the roll-pitch-yaw representation.
I am using SymPy to automatically do the math (there's some nasty jacobians)

Next I will use two models based on quaternions.
    1) Additive (i.e. compose by coeff add: q <- q + dwxyz)
    2) Multiplicative (i.e. multiply q <- q * [1 dx dy dz])
Then I will use the quat+rotVec Lie theory based framework.

I expect to find that the quat+rotVec SO3 rep works best.
But I want to build up to it!

'''

########################################################################################
# Jacobians & MathUtils
########################################################################################

# Mine is a Y-X-Z system:
#    1) yaw   around y
#    2) pitch around x
#    3) roll  around z
# This is the same as the more standard ZYX, but Z and Y are swapped and Z flipped.
# This is more intuitive for the image I have in my head:
# The front of the vehicle is Z+ and at id points north
# The right-side           is X+ and at id points east
# down                     is Y+ and at id points down into Earth
r,p,y = sympy.symbols('r p y')
x,y,z = sympy.symbols('x y z')
R = Matrix([[cos_(r), -sin_(r), 0], [sin_(r), cos_(r), 0], [0, 0, 1]])
P = Matrix([[1, 0, 0], [0, cos_(p), -sin_(p)], [0, sin_(p), cos_(p)]])
# P = Matrix([[1, 0, 0], [0, cos_(p), sin_(p)], [0, -sin_(p), cos_(p)]])
Y = Matrix([[cos_(y), 0, -sin_(y)], [0, 1, 0], [sin_(y), 0, cos_(y)]])
RPY = R*P*Y

def skew(v):
    return np.array((
          0,   -v[2], v[1],
         v[2], 0,     -v[0],
        -v[1], v[0], 0)).reshape(3,3)

# Note: this breaks down at theta=0
def rodrigues(k):
    theta = np.clip(np.linalg.norm(k), 1e-9, 9999)
    K = skew(k/theta)
    return np.eye(3) + np.sin(theta) * K + (1-np.cos(theta)) * K@K

def eval_expr(expr, **kwargs):
    v = expr.subs(kwargs)
    v = np.array(v.tolist())
    v = v.astype(np.float32)
    return v

_rpy_to_R = lambdify(symbols('r p y'), RPY, 'numpy', cse=True)
def rpy_to_R(rpy):
    # return (eval_expr(RPY, r=rpy[0], p=rpy[1], y=rpy[2]))
    return _rpy_to_R(rpy[0],rpy[1],rpy[2])

def skew_sympy(v):
    return sympy.Matrix([
        [0,   -v[2], v[1]],
        [v[2], 0,     -v[0]],
        [-v[1], v[0], 0]])

def log_R(R):
    t = np.arccos((np.trace(R) - 1) * .5)
    d = np.linalg.norm((R[2,1]-R[1,2], R[0,1]-R[1,0], R[0,2]-R[2,0]))
    return np.array((R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1])) * t / d

# print(' - cycle 000', log_R(rodrigues(np.array((1e-9,0,0)))))
# print(' - cycle 001', log_R(rodrigues(np.array((1e-9,0,1)))))
# print(' - cycle 011', log_R(rodrigues(np.array((1e-9,1,1))))) sys.exit(0)

# Note: this breaks down at theta=0
def rodrigues_sympy(k):
    theta = k.norm()
    K = skew_sympy(k/theta)
    return sympy.eye(3) + sympy.sin(theta) * K + (1-sympy.cos(theta)) * K*K

# Deriving is easy, but done by hand:
#      For the two atans: write out the RPY matrix in trig form,
#      match pairs of elements so that after dividing one factor
#      disappears and the other reduces to sin/cos (i.e. tangent).
#      For the arcsin one, just use the lone sin term
# Note: this is highly dependent on below RPY formulation
def matrix_to_rpy_sympy(R):
    from sympy import asin, atan2, Matrix
    return Matrix([atan2(-R[0,1], R[1,1]), asin(R[2,1]), atan2(R[2,0],R[2,2])])
def matrix_to_rpy(R):
    return np.array((np.arctan2(-R[0,1], R[1,1]), np.arcsin(R[2,1]), np.arctan2(R[2,0], R[2,2])))

# Test via cycle
# print('cycle rpy [0 0 .5]:', matrix_to_rpy_sympy(rpy_to_R([0,0,.5])))
# print('cycle rpy [0 .1 .5]:', matrix_to_rpy_sympy(rpy_to_R([0,0.1,.5])))
# print('cycle rpy [-.1 .1 .5]:', matrix_to_rpy_sympy(rpy_to_R([-.1,0.1,.5])))
# print('cycle rpy [-.1 1.1 2.5]:', matrix_to_rpy_sympy(rpy_to_R([-.1,1.1,2.5]))); sys.exit(0)

# Return Jacobian matrices for time and measurmenet steps
# The time expr must be given k1-k3, the gyro, and kn2, its norm
# The observe expr must be given m1-m3 and g1-g3, the mag+grav observations
# Of course, both most also be given r, p, & y
def gen_rpy_code(approximate=True):
    print(' - Rpy Matrix:\n')
    pprint(RPY)
    # approximate=False

    k1,k2,k3 = sympy.symbols('k1 k2 k3')
    k = Matrix([k1,k2,k3])

    # jacobians for time step
    if 1:
        Rs = Matrix(RPY)
        Rk = rodrigues_sympy(k)
        Rp = (Rs * Rk)
        # fRp = Rp @ Matrix([1,1,1])
        fRp = matrix_to_rpy_sympy(Rp)
        Jplus_x = Matrix(diff(fRp, Matrix([r,p,y]))[:,0,:,0])

        '''
        intr,expr = cse(Jplus)
        for intr_ in intr: print(intr_)
        pprint(expr)
        print(count_ops([intr,expr],True))
        '''

        # Substitute the common expr for |k|
        Jplus_x = Jplus_x.subs(Abs(k1)**2+Abs(k2)**2+Abs(k3)**2, symbols('kn2'))
        kn2 = symbols('kn2')
        Jplus_x = Jplus_x.subs((k2)**2+(k1)**2+(k3)**2, kn2)
        # print('FULL')
        # pprint(Jplus_x)
        if approximate:
            # NOTE: This makes an *APPROXIMATION*
            #       Assume that all k are small, and therefore k**2 is zero, and so are any ka*kb pair
            for k in k1,k2,k3: Jplus_x = Jplus_x.subs(k**2, 0)
            for k in k1,k2,k3: Jplus_x = Jplus_x.subs(Abs(k)**2, 0)
            for a,b in (k1,k2),(k1,k3),(k2,k3): Jplus_x = Jplus_x.subs(a*b, 0)
            # WARNING: I added these four because I was getting NaNs.
            for k in k1,k2,k3: Jplus_x = Jplus_x.subs(sympy.sin(k**2), 0)
            Jplus_x = Jplus_x.subs(sympy.sin(sympy.sqrt(kn2)), 0)
        # print('APPROX')
        # pprint(Jplus_x)
        # sys.exit(0)
        # print(Jplus)
        # k = 0,0,.01
        # kn2 = np.array(k)@k
        # print(Jplus.subs(dict(r=0,p=0,y=0, kn2=kn2,k1=k[0], k2=k[1], k3=k[2])))
        # print(Jplus.subs(dict(r=0,p=0,y=.1, kn2=kn2,k1=k[0], k2=k[1], k3=k[2])))

    # Jacobians for measurements
    # Assume we observe (grav mag), which are the (Y Z) axes
    if 1:
        # print(' - Rpy Matrix:'); pprint(RPY)
        g = Matrix([symbols('g1 g2 g3')])
        m = Matrix([symbols('m1 m2 m3')])
        # z = Matrix([g,m])
        Rs = Matrix(RPY)
        if 0:
            fRp_g = 1. - (Rs[0,1]*g[0] + Rs[1,1]*g[1] + Rs[2,1]*g[2])
            fRp_m = 1. - (Rs[0,2]*m[0] + Rs[1,2]*m[1] + Rs[2,2]*m[2])
            # fRp_g = 1. - (Rs[1,0]*g[0] + Rs[1,1]*g[1] + Rs[1,2]*g[2])
            # fRp_m = 1. - (Rs[2,0]*m[0] + Rs[2,1]*m[1] + Rs[2,2]*m[2])
        elif 1:
            fRp_g = 1. - (Rs[:,1].T @ g.T)[0]
            fRp_m = 1. - (Rs[:,2].T @ m.T)[0]
        else:
            fRp_g = 1. - (Rs[1,:] @ g.T)[0]
            fRp_m = 1. - (Rs[2,:] @ m.T)[0]
        # fRp_g = Rs[1,:] @ g.T
        # fRp_m = Rs[2,:] @ m.T
        Jobs_g_x = Matrix(diff(fRp_g, Matrix([r,p,y])))
        Jobs_g_g = Matrix(diff(fRp_g, Matrix(g)))
        Jobs_m_x = Matrix(diff(fRp_m, Matrix([r,p,y])))
        Jobs_m_m = Matrix(diff(fRp_m, Matrix(m)))
        # print('Jobs_grav'); pprint(Jobs_g)
        # print('Jobs_mag'); pprint(Jobs_m)
        Jobs_x  = Matrix([[Jobs_m_x, Jobs_g_x]])
        Jobs_mg = Matrix([[Jobs_m_m.T, Jobs_g_g.T]])
        # return Jplus, Jobs

        Jplus_x = lambdify(symbols('r p y k1 k2 k3 kn2'), Jplus_x, 'numpy', cse=True)
        # TODO: Have combined 2 jac 2x3 for _x and _gm
        # Jobs_x = lambdify(symbols('r p y g1 g2 g3 m1 m2 m3'), Jobs_g_x, 'numpy', cse=True)
        Jobs_g_x = lambdify(symbols('r p y g1 g2 g3'), Jobs_g_x, 'numpy', cse=True)
        Jobs_g_g = lambdify(symbols('r p y g1 g2 g3'), Jobs_g_g, 'numpy', cse=True)
        Jobs_m_x = lambdify(symbols('r p y m1 m2 m3'), Jobs_m_x, 'numpy', cse=True)
        Jobs_m_m = lambdify(symbols('r p y m1 m2 m3'), Jobs_m_m, 'numpy', cse=True)
        return dict(Jplus_x=Jplus_x,
                Jobs_g_x=Jobs_g_x,
                Jobs_g_g=Jobs_g_g,
                Jobs_m_x=Jobs_m_x,
                Jobs_m_m=Jobs_m_m,

                Jobs_x  = lambdify(symbols('r p y m1 m2 m3 g1 g2 g3'), Jobs_x , 'numpy', cse=True),
                Jobs_mg = lambdify(symbols('r p y m1 m2 m3 g1 g2 g3'), Jobs_mg, 'numpy', cse=True)
                )


# gen_rpy_code(); sys.exit(0)

########################################################################################
# Simulation Data
########################################################################################

GRAVITY = 9.92

# Return tensor of [N,9] measurments (acc, gyr, mag)
def generateSimData_1(N=4000, sigmas=None):
# def generateSimData_1(N=40, sigmas=None):
    if sigmas is None: sigmas = np.ones(9)

    # Note: The R matrices should be f64 even if all else f32
    out = np.zeros((N,13),dtype=np.float32)

    rvec = np.zeros(3,dtype=np.float32) + 1e-1

    trueRs = np.zeros((N,3,3),dtype=np.float64)
    trueRs[0] = rodrigues(rvec)
    acc,gyr,mag = [np.zeros(3,dtype=np.float64) for _ in range(3)]
    # speed = .1 / N
    speed = 30 / N

    gyr_bias = np.zeros(3)
    gyr_bias[1] = 1e-3 # WARNING: kind of high!

    for i,t in enumerate(np.linspace(0,1, N)):

        if 1:
            if t < .01:
                new_gyr = 0,0,0
            elif t > 1-.01:
                new_gyr = 0,0,0
            else:
                tt = t * 9
                new_gyr = speed*np.cos(tt+np.sin(t))*np.cos(tt), speed*.5 + np.cos(tt)*speed*.1, speed*np.sin(tt*.2)*.2
        elif 1:
            if t < .01:
                new_gyr = 0,0,0
            elif t < .1:
                new_gyr = (0,speed,0) #Messed up, this should be yaw but is pitch
            elif t < .3:
                new_gyr = (speed*.2,0,0)
            elif t < .4:
                new_gyr = (0,-speed,0)
            elif t < .5:
                new_gyr = (-speed*.2,0,0)
            elif t < .92:
                new_gyr = (0,speed*3.2,0)
            else:
                new_gyr = (0,0,0)
        elif 0:
            new_gyr = (0,0,0)
        else:
            if t < .1:
                new_gyr = 0,0,0
            elif t < .6:
                new_gyr = (0,speed,0)
            else:
                new_gyr = 0,0,0

        trueRs[i] = trueRs[i-1 if i-1 >= 0 else 0] @ rodrigues(new_gyr)
        # R = rodrigues((rvec+new_gyr)*.5)
        R = trueRs[i]
        rvec = rvec + new_gyr
        # rvec = log_R(R)
        # print(rvec)
        # gyr[:] = new_gyr + np.random.randn(3) * 1e-3 #* (i<N-9)
        gyr[:] = new_gyr + gyr_bias + np.random.randn(3) * 1e-2 #* (i<N-9)
        # gyr[:] = new_gyr + np.random.randn(3) * 1e-99
        mag[:] = R @ (0,0,1)
        acc[:] = R @ (0,GRAVITY,0)
        out[i] = t, *acc, *gyr, *mag, *rvec
        # print('SIM R:\n',R)

    return out, trueRs

########################################################################################
# Filters
########################################################################################

class BaseFilter():
    def __init__(self, name):
        self.name = name
        self.estimates = []
        self.covs = []

    def step(self, vals):
        raise NotImplementedError()
    def getRotAndCov(self, i):
        raise NotImplementedError()
    def getRpy(self, i):
        raise NotImplementedError()

class RpyFilter(BaseFilter):
    def __init__(self, approx):
        super().__init__('Rpy' + ('Approx' if approx else ''))
        self.lastTime = 0

        self.Jacobians = gen_rpy_code(approx)

        import inspect
        # print(' - source:\n', (inspect.getsource(self.Jplus_))

    def reset(self, rv):
        # self.rpy = np.array([1e-9,1e-9,1e-9],dtype=np.float32)
        if np.linalg.norm(rv) < 1e-8:
            self.rpy = np.array([1e-9,1e-9,1e-9],dtype=np.float32)
        else:
            self.rpy = matrix_to_rpy(rodrigues(rv))
        print(' - reset with rv',rv,'-> rpy', self.rpy)
        self.cov = np.eye(3,dtype=np.float32)

    def getRotAndCov(self, i):
        return rpy_to_R(self.estimates[i]), self.covs[i]
    def getRpy(self, i):
        return self.estimates[i]

    def step(self, vals, truth):
        t, acc, gyr, mag, rvec = vals[0], vals[1:4], vals[4:7], vals[7:10], vals[10:]
        dt = self.lastTime - t

        # ---------------------------------
        # Time Step: spin according to gyro
        # ---------------------------------
        if 0:
            # Slow interpreted sympy version
            F = eval_expr(self.Jplus,
                    r=self.rpy[0], p=self.rpy[1], y=self.rpy[2],
                    k1=gyr[0], k2=gyr[1], k3=gyr[2], kn2=gyr@gyr).T
        else:
            Jplus_x = self.Jacobians['Jplus_x']
            # Faster sympy->numpy version
            # print('RPY GYR, |GYR|^2', self.rpy, gyr, gyr@gyr)
            F = Jplus_x(self.rpy[0], self.rpy[1], self.rpy[2], gyr[0], gyr[1], gyr[2], gyr@gyr+1e-16).T


        print('gyr:', gyr)
        print('F:\n',F)
        self.rpy = matrix_to_rpy(rpy_to_R(self.rpy) @ rodrigues(gyr)) % (2. * np.pi)
        # self.rpy = (self.rpy + F @ gyr) % (2 * np.pi)
        Q = np.eye(3) * 1e-2 # TODO:
        # Q = np.eye(3) * 1e-0 # TODO:
        self.cov = F @ self.cov @ F.T + Q

        u_grav = acc / np.linalg.norm(acc)
        u_mag = mag / np.linalg.norm(mag)

        # ---------------------------------
        # Measurement Step: observe mag/grav
        # ---------------------------------
        # Seperate
        if 0:
            # R_m = 1 # information for magnometer
            # R_m = np.eye(2) * 1e1 # TODO:
            R_m = 1e-3 # TODO:
            # R_m = 1e-6 # TODO:
            # R_m = 1e20 # TODO:
            R_g = R_m

            # Make gravity step
            Jobs_g_x = self.Jacobians['Jobs_g_x']
            Jobs_g_g = self.Jacobians['Jobs_g_g']
            H = Jobs_g_x(*self.rpy, *u_grav).T
            L = Jobs_g_g(*self.rpy, *u_grav)

            y = 1. - rpy_to_R(self.rpy)[:,1] @ u_grav
            S = H @ self.cov @ H.T + L @ L.T * R_g
            K = self.cov @ H.T @ np.linalg.inv(S)
            dx = (K * -y).reshape(-1)
            print('innov:',y,'grav',u_grav,'yplus', rpy_to_R(self.rpy)[:,1])
            print('grav dx:',dx)
            self.rpy = (self.rpy + dx) % (2 * np.pi)
            self.cov = self.cov - K @ H @ self.cov

            # Make magnometer step
            Jobs_m_x = self.Jacobians['Jobs_m_x']
            Jobs_m_m = self.Jacobians['Jobs_m_m']
            H = Jobs_m_x(*self.rpy, *u_mag).T
            L = Jobs_m_m(*self.rpy, *u_mag)
            # print('H:\n',H)
            # print('L:\n',L)
            y = 1. - rpy_to_R(self.rpy)[:,2] @ u_mag
            S = H @ self.cov @ H.T + L @ L.T * R_m
            K = self.cov @ H.T @ np.linalg.inv(S)
            dx = (K * -y).reshape(-1)
            print('innov:',y,'mag',u_mag,'zplus', rpy_to_R(self.rpy)[:,2])
            # print('K:\n',K)
            print('mag  dx:',dx)
            self.rpy = (self.rpy + dx) % (2 * np.pi)
            self.cov = self.cov - K @ H @ self.cov

        # Combined
        elif 0:
            R_m = 1e-4
            # R_m = 1e30
            R_g = R_m
            R_mg = np.eye(2) * (R_m, R_g)
            # R_mg = np.eye(3) * (R_m) # ??????????????
            # Something is seriously wrong, this should be 2d but must be three to fit L
            # I need to step back and think about the implicit EKF:
            #    Is there two observations, or three?
            #          Currently, it is two, one each for z=m,g in [1 - R[1/2] . g]
            #
            #          BUT, it makes more sense for H to be identtiy, then have three measurements,
            #          which is like normal gauss-newton.
            #          The catch is that the rpy state is linearized, so

            # Okay to put it brielfy before going to sleep:
            #  Right now the Implicit EKF is taking H to be 2x3,
            #  because there are two measurments (the dot products with m and g)
            # BUT
            #  H should be 3x3 because it is *vital* we don't lose the third
            #  subspace when we do form HPH'
            #  Then the 'measurement' is in the natural rpy vector space.
            # BUT
            #  Does the fact that we are linearized destroy this?
            #  Well the KF equations are altered to the point that I should
            #  start from a NLLS perspective and find out.

            # No, actually it's not vital, it happens all the time, but K handlse properly

            Jobs_x  = self.Jacobians['Jobs_x']
            Jobs_mg = self.Jacobians['Jobs_mg']
            H = Jobs_x (*self.rpy, *u_mag, *u_grav).T
            # L = Jobs_mg(*self.rpy, *u_mag, *u_grav)
            L = np.eye(2)

            print('H',H.shape)
            print('L',L.shape)
            # print('R truncated', rpy_to_R(self.rpy).T[1:].shape)
            print('obs', np.stack((u_mag,u_grav),0).shape)
            # Yes, you read it right. Element-wise mult, then sum. Can't express as matmul?
            # y = np.ones((2)) - (rpy_to_R(self.rpy).T[1:] * np.stack((u_mag,u_grav),0)).sum(1)
            y = 1. - np.array((
                rpy_to_R(self.rpy)[:,2] @ u_mag,
                rpy_to_R(self.rpy)[:,1] @ u_grav))
            S = H @ self.cov @ H.T + L @ R_mg @ L.T
            K = self.cov @ H.T @ np.linalg.inv(S)
            dx = (K @ -y).reshape(-1)
            print('innov:',y)
            print('grav',u_grav,'yplus', rpy_to_R(self.rpy)[:,1])
            print('mag ',u_mag,'zplus', rpy_to_R(self.rpy)[:,2])
            print('  dx:',dx)
            self.rpy = (self.rpy + dx) % (2 * np.pi)
            self.cov = self.cov - K @ H @ self.cov

        elif 1:
            # GN approach
            Jobs_x = self.Jacobians['Jobs_x']

            Alpha = np.eye(2) * 1e2
            P = self.cov

            J = Jobs_x (*self.rpy, *u_mag, *u_grav).T
            res = 1. - np.array((
                rpy_to_R(self.rpy)[:,2] @ u_mag,
                rpy_to_R(self.rpy)[:,1] @ u_grav))

            Hess = (J.T @ Alpha @ J + inv(P))
            # print(' - J:\n', J)
            # print(' - P^-1:\n', inv(P))
            # print(' - J\'AJ:\n', J.T@Alpha@J)
            print(' - Hess:\n', Hess, 'JtJ eigenvals',np.linalg.eig(J.T@Alpha@J)[0])

            P = inv(Hess)
            print(' - Init  P:\n',self.cov)
            print(' - Final P:\n',P)
            dx = P @ (J.T @ Alpha @ res)
            # print(' - grad:', J.T@Alpha@res, 'res:', res)
            # print(' - dx:',dx)

            self.cov = P
            self.rpy += -dx


        # print('cov:\n', self.cov)
        # print('rpy:', self.rpy)
        # print('true R:\n', truth)
        # print('pred R:\n', rpy_to_R(self.rpy))
        # if np.linalg.norm(gyr) > 1e-6: input()


        self.estimates.append(np.copy(self.rpy))
        self.covs.append(np.copy(self.cov))

        self.lastTime = t


########################################################################################
# Generate Web Page
########################################################################################

def writeWebPage(valss, trueRs, filterOutputs):
    outDir = os.getcwd()
    outHtml = 'ori.html'
    outJson = 'simAndFilterData.json'
    if 'extraPages' not in outDir: outDir = os.path.join(outDir, 'extraPages')
    if 'ori' not in outDir[-5:]: outDir = os.path.join(outDir, 'ori')
    outPathHtml = os.path.join(outDir,outHtml)
    outPathJson = os.path.join(outDir,outJson)

    simRpys = []
    for R in trueRs:
        simRpys.append(matrix_to_rpy(R))
    simRpys = np.array(simRpys)

    # Generate json data and save to file
    jobj = dict(simData=valss.reshape(-1).tolist(), simRpys=simRpys.tolist(),simDataShape=valss.shape)
    jobj['outputs'] = {}
    for i,(name,fo) in enumerate(filterOutputs.items()):
        jobj['outputs'][name] = {}
        for dataKey,dataVals in fo.items():
            jobj['outputs'][name][dataKey+'Shape'] = dataVals.shape
            jobj['outputs'][name][dataKey] = dataVals.tolist()

    with open(outPathJson, 'w') as fp:
        json.dump(jobj, fp)


    with open(outPathHtml, 'w') as fp:
        with dominate.document(title='ori') as doc:
            doc.body.add(D.script(type='text/javascript', src='runOri.bundled.js'))

            fp.write(doc.render())
    print(' - wrote',outPathHtml)
    print(' - wrote',outPathJson)

def run_filter(filt, valss, trueRs):
    for i, vals in enumerate(valss):
        print(i, '/', len(valss))
        filt.step(vals, trueRs[i])
        # if i > 20: sys.exit(0)
        # time.sleep(1)

    rpys = []
    errs = []
    sigmas = []
    rpys = []
    plt.title('Err/Sig')
    for i,vals in enumerate(valss):
        time, acc, gyr, mag, rvec = vals[0], vals[1:4], vals[4:7], vals[7:10], vals[10:]
        estRot,estCov = filt.getRotAndCov(i)
        estRpy = filt.getRpy(i)
        estVec = log_R(estRot)
        sig = np.sqrt(np.cbrt(np.linalg.eig(estCov)[0].prod()))

        d = estVec-rvec
        # d = (estVec%(2*np.pi))-(rvec%(2*np.pi))
        d = d % (2*np.pi)
        d = np.minimum(d, 2*np.pi-d)
        # print(d)
        err = np.linalg.norm(d)
        # err = np.linalg.norm((estVec%(2*np.pi)) - (rvec%(2*np.pi)))
        if i % 1 == 0: print('err',err, 'from',estVec, rvec,'sigma',sig,d)
        rpys.append(estRpy)
        errs.append(err)
        sigmas.append(sig)
    plt.plot(errs, color='r', label='err')
    plt.plot(sigmas, color='y', label='sig')
    plt.legend()
    plt.savefig('err_'+filt.name+'.jpg')
    plt.clf()

    rpys = np.array(rpys)
    errs = np.array(errs)
    sigmas = np.array(sigmas)
    return dict(rpys=rpys, errs=errs, sigmas=sigmas)



if __name__ == '__main__':

    valss,trueRs = generateSimData_1()

    filterOutputs = {}
    filt = RpyFilter(True)
    filt.reset(valss[0, -3:] + (0,.2,.2))
    filterOutputs[filt.name] = run_filter(filt, valss, trueRs)
    # filt = RpyFilter(True)
    # filt.reset(valss[0, -3:])
    # filterOutputs[filt.name] = run_filter(filt, valss, trueRs)

    writeWebPage(valss, trueRs, filterOutputs)
