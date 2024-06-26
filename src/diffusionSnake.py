import numpy as np
from src.cicleBspline import Spline
from skimage import io, color


def error(f,u,C,lambd,v):
    futerm=0.5*np.sum((f-u)**2)

    mask=C.get_mask(f.shape,steps=200)
    du_dx, du_dy = np.gradient(u)
    duterm = lambd**2*0.5*np.sum((du_dx**2 + du_dy**2)*(mask!=1))

    cnormterm=v*np.sum(C.spline(np.linspace(0,1,1000,endpoint=False),1)**2)
    #print(futerm,duterm,cnormterm)
    return futerm+duterm+cnormterm


def generate_points_in_circle(num_points,radius=1,center=[(0,0)]):
    theta = -np.linspace(0, 2 * np.pi, num_points,endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)
    return np.stack((x, y), axis=-1)*radius+center


def readimageasnp(image_path):
    return io.imread(image_path,as_gray=True).T


class DiffusionSnake:
    r""" Diffusion Snakes for image Segmentation from Creemers 2002 (PHD-Thesis) 
        https://cvg.cit.tum.de/_media/spezial/bib/cremers_dissertation.pdf
        Minimizing the full Munford-Shah Functional (p.19, 2.11), if mode is "full"
        or the simplified Functional at the catoon limit (p.20, 2.13), if mode is "simple"
        by performing a gradient decent step on each step-call
    """    
    def __init__(self, image_path, v, n_points, alpha, respace=True, init_size=0.3, mode="full", lambd=2, tau=0.25, u_iterations=4):
        r"""Init function
            #Args:
                image_path: the image-volume to segment
                v: Determines how much the length of the spline is penalized
                n_points: number of controllpoints for spline
                alpha: learning rate for gradient decent
                respace: bool or int: if True controllpoints will be respaced equidistant every 20 steps. If int, every respace steps
                init_size: size of initial spline radius in part per input image
                mode: "full" or "simple". Use full Munford-Shah Functional for Minimization, if mode is "full". 
                Use simplified Functional at the catoon limit, if mode is "simple"
                --- Only for full mode ---
                lambd: Blur-factor for u-function
                tau: stepsize, tau > 0.25 leads to instability
                u_iterations: Determines how strongly u is adapted to the new curve in each step
        """
        assert(tau <= 0.25)
        assert(init_size <= 1.0 and init_size > 0.0)

        self.f = readimageasnp(image_path)
        self.u = self.f

        size=np.array(self.f.shape)
        controllpoints=generate_points_in_circle(n_points,size*init_size,size/2)+np.random.rand(n_points,2)*1
        self.C=Spline(c=controllpoints,k=2)
        self.B=self.C.designmatrix(wrap=True)
        self.Binv=np.linalg.inv(self.B)

        self.lambd, self.v, self.alpha, self.tau, self.u_iterations, self.respace = lambd, v, alpha, tau, u_iterations, respace
        self.n_step = 0

        if respace is True: self.respace = 20
        else: self.respace = respace

        if mode == "simple": 
            self.u_func = self.u_e_simple
            self.si_range = range(0,20)  #range random
        elif mode == "full":   
            self.u_func = self.u_e_full
            self.si_range = range(4,20)  #.. # do not start at 0, becaus discontinuity mess up gradient 
        else:                  
            raise(NameError("Mode has to be 'full' or 'simple'"))


    def u_e_simple(self):
        r""" Computes the e and u part for the simplyfied diffusion snake (p.20, 2.13)
        u is picewise constant here. I.e. it is the current average of each segmented region (inside and outside the spline) 
        #Args:
            f: the image-volume
            spline: current spline
        """
        f, spline = self.f, self.C

        mask=spline.get_mask(f.shape,steps=200)
        in_mask, out_mask = (mask==2),(mask==0) # inside/outside spline without pixels on spline
        
        # u_in/u_out is average of each region
        u_in, u_out = in_mask.astype(float), out_mask.astype(float),
        u_in *= np.sum(f*in_mask) / np.sum(in_mask) 
        u_out*= np.sum(f*out_mask)/ np.sum(out_mask)
        u = u_in + u_out

        # "energy" outside, inside
        e_p = (f*out_mask - u_out)**2
        e_m = (f*in_mask  - u_in )**2
        return mask, u, e_p, e_m


    def u_e_full(self):
        r""" Computes the e and u part for the full diffusion snake (p.19, 2.11)
        f: the image-volume
        spline: current spline
        lambd: Blur-factor
        tau: stepsize, tau > 0.25 leads to instability
        iterations: determines how strongly u is adapted to the new curve in each step. I.e. how many u-steps are performed in each curve-step
        """
        f, spline, uk, lambd, tau, iterations = self.f, self.C, self.u, self.lambd, self.tau, self.u_iterations   

        mask=spline.get_mask(shape=f.shape)
        in_mask, out_mask, spline_mask = (mask==2),(mask==0),(mask==1)  # inside/outside spline without pixels on spline and only pixels on spline
        w = spline_mask!=1                                              # pixels not on spline

        pw = np.pad(w, pad_width=1, constant_values=0)
        smid=slice(1,-1);sup=slice(2,None);sdown=slice(None,-2)            
        neighbourslices=[(sup,smid),(sdown,smid),(smid,sup),(smid,sdown)]    # under, over, right, left
        for i in range(iterations):
            puk = np.pad(uk, pad_width=1, constant_values=0)    # padded uk
            uk=((1-tau*sum((w*pw[j]) for j in neighbourslices))*uk          
                +tau*sum((w*pw[j]) * puk[j] for j in neighbourslices)       # uk=u_(k+1)    # no sqrt see 3.27 but is not needed
                +(tau/lambd**2)*f
                )/(1 + tau/lambd**2)
        u = uk
        
        # "energy" outside, inside (2.26)
        u_out, u_in = u*out_mask, u*in_mask
        grad_x, grad_y = np.gradient(u)
        grad_x_out, grad_y_out = grad_x*out_mask, grad_y*out_mask
        grad_x_in,  grad_y_in  = grad_x*in_mask,  grad_y*in_mask 

        e_p = (f*out_mask - u_out)**2 + lambd**2 * (grad_x_out**2 + grad_y_out**2)
        e_m = (f*in_mask  - u_in )**2 + lambd**2 * (grad_x_in**2  + grad_y_in**2)
        return mask, u, e_p, e_m

    def step(self):
        r""" Let Diffusion Snake performce a single iteration p.25 (2.30)
        """
        mask, u, ep, em = self.u_func() # depends on mode simple or full

        x=self.C.xbmax() #compute best x

        #compute normals and nodes
        normals=self.C.normals(x).T # normals point outside
        normals = normals / np.linalg.norm(normals, axis=1)[:,None] # Normalize each row vector
        s=self.C.spline(x)

        # determine e of si aka the next pixel outside/inside spline in normal direction
        N=len(s)#number of points
        esiplus =np.zeros(N)#outside
        esiminus=np.zeros(N)#inside
        for i,(si,ni) in enumerate(zip(s,normals)):
            for d in self.si_range:
                x,y=self.inbounds(*(si+ni*d/2).astype(int))
                if mask[x,y]==0:        # if reached outside
                    esiplus[i]=ep[x,y]
                    break
            else:                       # fall-back-case: This is reached sometimes depending on hyperparameters, input-image, state of convergence and self.si_range
                # print(":(")
                x,y=self.inbounds(*si.astype(int))
                esiplus[i]=ep[x,y]
            for d in self.si_range:               
                x,y=self.inbounds(*(si-ni*d/2).astype(int))
                if mask[x,y]==2:        # if reached inside
                    esiminus[i]=em[x,y]
                    break
            else:                       # fall-back-case
                # print(":(")
                x,y=self.inbounds(*si.astype(int))
                esiminus[i]=em[x,y]

        ep, em = esiplus, esiminus
        
        # gradients
        gradients=np.zeros(s.shape)
        sumterm=(ep-em)[:,None]*normals + self.v*(np.roll(s, 1,axis=0)-2*s+np.roll(s, -1,axis=0))
        gradients=np.dot(self.Binv, sumterm)
        c_new=self.C.c+gradients*self.alpha

        # update
        self.C.set_c(c_new)
        self.u = u
        self.n_step+=1

        # respace points
        if self.respace and self.n_step%self.respace==0:
            self.respacepoints()
    
    def inbounds(self, x, y):
        x = min(self.f.shape[0]-1, x)
        y = min(self.f.shape[1]-1, y)
        return x, y

    def draw(self):
        r"""Draws the current Diffusion Snake on a pixel-plane of initial input image size
        #Returns:
            mask: 
            x, y: The coordinates of the 
        """
        x, y = self.C.draw(self.f.shape, steps=1000)
        return self.u, x, y
    
    def respacepoints(self,steps=1000):
        r""""respace controll points equvidisttly on spline
        """
        curve_points = self.C.spline(np.linspace(0, 1, steps))
        #print(np.linalg.norm(curve_points[0]-curve_points[-1]))# first and last are he same d.h. shoild be 0

        distances = np.linalg.norm(curve_points[1:] - curve_points[:-1], axis=1)
        cumulative_distances = np.cumsum(distances)
        normalized_distances = np.insert(cumulative_distances / cumulative_distances[-1], 0, 0)
        dists=normalized_distances

        x=np.linspace(0, 1, len(self.C.points),endpoint=False)
        index=np.searchsorted(dists,x)
        x0,x1=dists[index-1],dists[index]
        y0,y1=curve_points[index-1],curve_points[index]
        y=(y0*(x1-x)[:,None]+y1*(x-x0)[:,None])/(x1-x0)[:,None]# linear interpolation
        points=y

        self.C.setpoints(points)