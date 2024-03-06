# Diffusion Snakes for image Segmentation from Creemers 2002 (PHD-Thesis)


# TODO: Next: find u-function


# Diffusion Snake
# Minimizing the full Munford-Shah Functional 2.11 on p.19 by performing a gradient decent step on each step-call
# TODO: Next
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
        return self.I, self.C, self.u

    

# Simplified Diffusion Snake
# in the catoon limit the image approximation u is constant in every region. Therefor the Diffusion Snake becomes more simple
# Minimizing simplified Mumfor-Shah Functional 2.13 on p.20
class SimplifiedDiffusionSnake:    
    def __init__(self, image, contour):
        self.I = image
        self.C = contour
        self.u = u_init(image, contour)
        def u_init(self, image, contour):
            # for the simplified Diffusion Snake u is pice-wise constant in each region (inside and outside the contour)
            return image


    def step(self):
        # 1. Curve Step = Euler-Lagrange equation 2.25
        # 2. u-Step
        return self.I, self.C, self.u



# In section 3.5 Creemers propose to add an E_shape(z) to the minimization term (p.56)
# XXX: Implement, if there is time