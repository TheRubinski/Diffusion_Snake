import numpy as np
from src.cicleBspline import Spline
from skimage import io, color
from matplotlib import pyplot as plt


def u_full(f,spline,u=None,lambd=10,tau=0.25,iterations=1):
    if u is not None:
        uk=u
    else:
        uk=f

    mask,*_=spline.draw(np.zeros(f.shape,np.uint8),drawinside=False,steps=200)
    w=mask!=1
    pw = np.pad(w, pad_width=1, constant_values=0)
    
    smid=slice(1,-1);sup=slice(2,None);sdown=slice(None,-2)
    neighbourslices=[(sup,smid),(sdown,smid),(smid,sup),(smid,sdown)]

    for i in range(iterations):
        puk = np.pad(uk, pad_width=1, constant_values=0)#padded uk

        uk=(
            (1-tau*sum(np.sqrt(w*pw[j])for j in neighbourslices))*uk
            +tau*sum(np.sqrt(w*pw[j])*puk[j]for j in neighbourslices)#u_k=u_(k+1)
            +(tau/lambd**2)*f
            )/(1+tau/lambd**2)
    return uk


def u_e_full():
    raise(NotImplementedError)
    return


def generate_points_in_circle(num_points,radius=1,center=[(0,0)]):
    theta = -np.linspace(0, 2 * np.pi, num_points,endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)
    return np.stack((x, y), axis=-1)*radius+center


def error(f,u,C,lambd,v):
    futerm=0.5*np.sum((f-u)**2)

    mask,*_=C.draw(np.zeros(f.shape,np.uint8),drawinside=False,steps=200)
    du_dx, du_dy = np.gradient(u)
    duterm = lambd**2*0.5*np.sum((du_dx**2 + du_dy**2)*(mask!=1))

    cnormterm=v*np.sum(C.spline(np.linspace(0,1,1000,endpoint=False),1)**2)
    #print(futerm,duterm,cnormterm)
    return futerm+duterm+cnormterm


def u_e_simple(f, spline):
    in_mask, out_mask, _ = spline.get_masks(f, steps=200)
    u_in, u_out = in_mask.astype(float), out_mask.astype(float),
    u_in *= np.sum(f*in_mask) / np.sum(in_mask) 
    u_out*= np.sum(f*out_mask)/ np.sum(out_mask)
    u = u_in + u_out
    # "energy" outside, inside
    e_p = np.power(((f*out_mask) - u_out), 2)
    e_m = np.power(((f*in_mask)  - u_in),  2)
    return u, u_in, u_out, e_p, e_m


def readimageasnp(image_path):
    image = io.imread(image_path)
    f = color.rgb2gray(image)
    return f


class DiffusionSnake:
    r""" Diffusion Snakes for image Segmentation from Creemers 2002 (PHD-Thesis) 
        Minimizing the full Munford-Shah Functional (p.19, 2.11), if mode is "full"
        or the simplified Functional at the catoon limit(p.20, 2.13), if mode is "simple"
        by performing a gradient decent step on each step-call
    """    
    def __init__(self, image_path, lambd, v, n_points, alpha, mode="full"):
        r"""Init function
            #Args:
                image_path:
                lambd: 
                v:
                n_points: number of controllpoints for spline
                alpha: learning rate for gradient decent
                mode: "full" or "simple". Use full Munford-Shah Functional for Minimization, if mode is "full".
                    Use simplified Functional at the catoon limit, if mode is "simple"
        """
        self.f = readimageasnp(image_path)
        self.u = self.f

        size=np.array(self.f.shape)
        controllpoints=generate_points_in_circle(n_points,size/3,size/2)+np.random.rand(n_points,2)*1
        self.C=Spline(c=controllpoints,k=2)
        self.B=self.C.designmatrix(wrap=True)
        self.Binv=np.linalg.inv(self.B)

        self.lambd, self.v, self.alpha = lambd, v, alpha
        if   mode == "simple": self.u_func = u_e_simple
        elif mode == "full":   self.u_func = u_e_full
        else:                  raise(NameError("Mode has to be 'full' or 'simple'"))


    def step(self):
        r""" Let Diffusion Snake performce a single iteration
        """
        u, u_in, u_out, ep, em = self.u_func(self.f,self.C)
        # e=error(f,u,C,lambd,v)
        
        x=self.C.xbmax() #compute best x

        #compute normals and nodes
        normals=self.C.normals(x).T # normals point outside
        # Normalize each row vector
        normals = normals / np.linalg.norm(normals, axis=1)[:,None]
        #print(normals)


        s=self.C.spline(x)
        N=len(s)#number of points

        gradients=np.zeros(s.shape)
        #compute mask 0=outside 1=spline 2=inside

        mask,*_=self.C.draw(np.zeros(self.f.shape))

        # e-part-optimizer from XXX
        esiplus =np.zeros(N)#outside
        esiminus=np.zeros(N)#inside
        for i,(si,ni) in enumerate(zip(s,normals)):
            for d in range(0,30):
                x,y=(si+ni*d/2).astype(int)
                if mask[x,y]==0:
                    esiplus[i]=ep[x,y]
                    break
            else:
                print(":(")
                x,y=si.astype(int)
                esiplus[i]=ep[x,y]
            
            for d in range(0,30):
                x,y=(si-ni*d/2).astype(int)
                if mask[x,y]==2:
                    esiminus[i]=em[x,y]
                    break
            else:
                print(":(")
                x,y=si.astype(int)
                esiminus[i]=em[x,y]
        # esi=esiplus-esiminus

        
        sumterm=(esiplus-esiminus)[:,None]*normals + self.v*(np.roll(s, 1,axis=0)-2*s+np.roll(s, -1,axis=0))
        gradients=np.dot(self.Binv, sumterm)
        c_new=self.C.c+gradients*self.alpha

        self.C.set_c(c_new)
        self.u = u
        return
    

    def draw(self):
        r"""Draws the current Diffusion Snake on a pixel-plane of initial input image size
        #Returns:
            mask: 
            x, y: The coordinates of the 
        """
        _, x, y = self.C.draw(np.zeros(self.f.shape,np.uint8), steps=1000) # XXX: Könnte es sein, dass diese steps der Grund für die Missing-Pixels sind?

        return self.u, x, y

    def get_controlpoints(self):
        r"""
        #Returns: current controllpoints. For Plotting/Debugging"""