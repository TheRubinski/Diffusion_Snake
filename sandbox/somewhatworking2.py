import numpy as np
from bsplineclass import Spline
from skimage import io, color
from matplotlib import pyplot as plt





def uiter(f,spline,u=None,lambd=10,tau=0.25,iterations=1):

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









# Load the image
image_path = './sample_images/snail1.png'
image = io.imread(image_path)
f = color.rgb2gray(image)
u=f


size=np.array(f.shape)
n_points = 100

controllpoints=generate_points_in_circle(n_points,size/3,size/2)+np.random.rand(n_points,2)*1
C=Spline(c=controllpoints,k=2)

#print(C.xbmax())
#print(C.designmatrix(wrap=True))
B=C.designmatrix(wrap=True)
binv=np.linalg.inv(B)
mask,x,y=C.draw(np.zeros(f.shape,np.uint8))



lambd,v=5,0.1



from matplotlib import animation
fig=plt.figure()
uplt=plt.imshow(u)
Cplt,=plt.plot(x,y,"-b")
u=uiter(f,C,u,lambd,iterations=100)
delta_w_neu=0
step=0
print_step = plt.text(1,5,"Step: "+str(step))


def animate(frame):
    global u,delta_w_neu,C,step, print_step,controllpoints
    
    print_step.set_text("Step: "+str(step)) # = plt.text(1,5,"Step: "+str(step))

    u=uiter(f,C,u,lambd,iterations=4)
    #u, u_in, u_out, e_p, e_m=u_e_simple(f,C)
    
    e=error(f,u,C,lambd,v)
    #print(e)

    

    #compute best x
    x=C.xbmax()

    #compute normals and nodes
    normals=C.normals(x).T#normals point outside
    # Normalize each row vector
    normals = normals / np.linalg.norm(normals, axis=1)[:,None]
    #print(normals)


    s=C.spline(x)
    N=len(s)#number of points



    gradients=np.zeros(s.shape)

    #compute e
    grad_x, grad_y = np.gradient(u)
    e = (f - u) ** 2+lambd ** 2 * (grad_x ** 2 + grad_y ** 2)
    ep=e
    em=e
    #ep=e_p
    #em=e_m
    #e=uiter(e,C,e,lambd,iterations=1)
    
    #compute mask 0=outside 1=spline 2=inside

    mask,*_=C.draw(np.zeros(f.shape))


    esiplus =np.zeros(N)#outside
    esiminus=np.zeros(N)#inside
    #print(list(zip(s,normals)))
    for i,(si,ni) in enumerate(zip(s,normals)):
        #ni=ni/np.linalg.norm(ni)
        for d in range(4,30):
            x,y=(si+ni*d/2).astype(int)
            #print(x,y,mask[x,y])
            if mask[x,y]==0:
                #print(x,y)
                esiplus[i]=ep[x,y]
                break
        else:
            print(":(")
            x,y=si.astype(int)
            esiplus[i]=ep[x,y]
        
        for d in range(4,30):
            x,y=(si-ni*d/2).astype(int)
            if mask[x,y]==2:
                esiminus[i]=em[x,y]
                break
        else:
            print(":(")
            x,y=si.astype(int)
            esiminus[i]=em[x,y]
    
    esi=esiplus-esiminus
    #print(esi)



    sumterm=(esiplus-esiminus)[:,None]*normals + v*(np.roll(s, 1,axis=0)-2*s+np.roll(s, -1,axis=0))
    gradients=np.dot(binv, sumterm)
    """for m in range(N):
        #gradients[m]=sum([binv[m,i]*( esi[i]*normals[i] + v*(s[(i-1)%N]-2*s[i]+s[(i+1)%N]) ) for i in range(N)])#this line works
        #print(np.all(B-B.T<0.001))
        #gradients[m]=sum([binv[m,i]*( -esi[i]*normals[i] ) for i in range(N)])


        #vs=v*(np.roll(s, 1,axis=0)-2*s+np.roll(s, -1,axis=0))
        #gradients[m]=sum([binv[m,i]*( esi[i]*normals[i] + vs[i] ) for i in range(N)])
        gradients[m]=sum([binv[m,i]*( sumterm[i] ) for i in range(N)])
        print(sumterm.shape)
        #gradients[m]=sum([binv[m,i]*(  -v*(s[(i-1)%N]-2*s[i]+s[(i+1)%N]) ) for i in range(N)])"""
    

    controllpoints=controllpoints+gradients*0.4
    #print(gradients)

  









    
    #controllpoints=controllpoints-gradients*0.2
    #controllpoints=controllpoints-delta_w_neu
    C.set_c(controllpoints)#new_variable=old_variable−learning_rate*gradient
    
    mask,x,y=C.draw()
    uplt.set_array(u.T)
    Cplt.set_data(x,y)

    step += 1

    return[uplt,Cplt,print_step]

animate(0)
animate(1)
plt.pause(0.5)
anim = animation.FuncAnimation(fig, animate, interval=10,cache_frame_data=False,blit=True)
plt.show()



    
