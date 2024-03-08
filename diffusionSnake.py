# Diffusion Snakes for image Segmentation from Creemers 2002 (PHD-Thesis)
import numpy as np
from scipy.interpolate import make_interp_spline

# TODO: Next: find u-function .. see p.14 2.1 and/or p.64 and/ or slides acp-03a
def u_func_full(image, mask, lamb):
    # blur only mask part it
    return image

def u_func_simple(image, mask):
    # take average of mask part
    return image

def get_mask(image, contour):
    # TODO return in_mask, out_mask, contour_mask
    return image, image, image

def circle_spline(n_nodes=100, degree=2, scale=2):
    """
    degree: of spline, so 2 is quadratic, 3 is qubic
    scale: of cicle in radius
    """
    theta = 2 * np.pi * np.linspace(0, 1, n_nodes)
    y = np.c_[np.cos(theta), np.sin(theta)]         # y are the input datapoints. Here = def cycle
    cs = make_interp_spline(theta, y, k=degree)
    cs.c = scale *cs.c
    return cs

def get_normal(curve, point):
    dydx = curve(point, 1)
    return np.array(dydx[1], -dydx[0])

# Diffusion Snake
# Minimizing the full Munford-Shah Functional 2.11 on p.19 by performing a gradient decent step on each step-call
class DiffusionSnake:    
    def __init__(self, image, contour):
        self.I = image
        self.C = contour
        self.u = u_init(image, contour)
        def u_init(self, image, contour):
            return image


    def step(self):
        # 1. Curve Step = Euler-Lagrange equation 2.25
        # 2. u-Step

        # see p.6 for spline def, he used
        return self.I, self.C, self.u

    

# Simplified Diffusion Snake
# in the catoon limit the image approximation u is constant in every region. Therefor the Diffusion Snake becomes more simple
# Minimizing simplified Mumford-Shah Functional 2.13 on p.20
class SimplifiedDiffusionSnake:    
    def __init__(self, image, spline_nodes=100, spline_degree=2, spline_scale=1):
        self.I = image
        # conture parameters. Save for design matrix
        self.x, self.k, self.C = circle_spline(n_nodes=spline_nodes, degree=spline_degree, scale=spline_scale)
        self.u = u_func_simple(image, self.C)
        self.tau = 0.25 # Has to be leq 0.25 for stability, see p.27
        self.v


    def step(self):
        # 1. Curve Step = Euler-Lagrange equation 2.25 or 2.30
        I, C, u = self.I, self.C, self.u
        in_mask, out_mask, c_mask = get_mask(self.I, self.C)
        e = (I - u)^2                   # XXX Only this has to be ajusted for DS for Curve-Step (p.24, 2.26)
        e_plus = e * out_mask   
        e_minus = e * in_mask
        B = C.design_matrix(self.x, C.t, self.k).inv    # need inverse or not?

        control_points = C.c
        for m in range(len(control_points)):
            point = control_points[m]
            # Spline-Def Creemers p. 22
            # corresponds to 
            # Spline-Def SciPy https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html
            # Creemers C: [0,1] -> Omega,  C(s) = sum (p_n * B_n (s))
            # SciPy: S(x) = sum (c_j * B_j (x))
            # Creemers p.24f. C(s,t): s = , t = timestep    # for p.24f.
            normal = get_normal(C, point)
            dx = 
            dy = 
        # 2. u-Step

        # see p.6 for spline def, he used
        return self.I, self.C, self.u



# In section 3.5 Creemers propose to add an E_shape(z) to the minimization term (p.56)
# XXX: Implement, if there is time