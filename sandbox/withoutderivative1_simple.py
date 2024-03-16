import numpy as np
from bsplineclass import Spline
from skimage import io, color
from matplotlib import pyplot as plt





def generate_points_in_circle(num_points,radius=1,center=[(0,0)]):
    theta = np.linspace(0, 2 * np.pi, num_points,endpoint=False)
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


def u_simple(f, spline):
    in_mask, out_mask, _ = spline.get_masks(np.zeros(f.shape,np.uint8), steps=200)
    u_in, u_out = in_mask.astype(float), out_mask.astype(float),
    u_in *= np.sum(f*in_mask) / np.sum(in_mask) 
    u_out*= np.sum(f*out_mask)/ np.sum(out_mask)
    #return (u_in + u_out)
    return u_in + u_out # flip because of plot grows down 







# Load the image
image_path = './sample_images/snale1.png'
image = io.imread(image_path)
f = color.rgb2gray(image)
u=f


size=np.array(f.shape)
n_points = 100
C=Spline(generate_points_in_circle(n_points,size/3,size/2)) 

mask,x,y=C.draw(np.zeros(f.shape,np.uint8))



lambd, v = 10, 0.000005



from matplotlib import animation
fig=plt.figure()
uplt=plt.imshow(u)
Cplt,=plt.plot(x,y,"-b")
# u=uiter(f,C,u,lambd,iterations=100)
u = u_simple(f,C)
delta_w_neu=0
step=0
print_step = plt.text(1,5,f"Step: {step}")
def animate(frame):
    global u,delta_w_neu,C,step, print_step
    
    print_step.set_text(f"Step: {step}") # = plt.text(1,5,"Step: "+str(step))

    # u=uiter(f,C,u,lambd,iterations=4)
    u=u_simple(f,C)
    e=error(f,u,C,lambd,v)
    #print(e)

    

    """#chatgptcode
    gradients = []
    for i, arg in enumerate(args):
        args_plus_h = list(args)
        args_minus_h = list(args)

        args_plus_h[i] = arg + h
        args_minus_h[i] = arg - h

        partial_derivative = (f(*args_plus_h) - f(*args_minus_h)) / (2 * h)
        gradients.append(partial_derivative)"""

    h=1
    gradients=np.zeros(C.c.shape)
    for index in np.ndindex(C.c.shape):
        pointsplus =np.copy(C.c)
        pointsminus=np.copy(C.c)
        pointsplus[index] +=h
        pointsminus[index]-=h
        Cplus =Spline(c=pointsplus )
        Cminus=Spline(c=pointsminus)
        uplus =u_simple(f, Cplus)
        uminus=u_simple(f, Cminus)
        ep=error(f,uplus,Cplus,lambd,v)
        em=error(f,uminus,Cminus,lambd,v)
        if ep<e:
            e=ep
            u=uplus
            c=Cplus
        if em<e:
            e=em
            u=uminus
            c=Cminus

        gradients[index]=(ep-em)/2*h
    #print(gradients)
    #glen=np.sqrt(gradients[:,0]**2+gradients[:,1]**2)[:,None]
    #gradients=gradients/glen*np.sqrt(glen)/np.min(np.sqrt(glen))
    #gradients=gradients/np.sqrt(glen)/4


    eta = 0.1 # 0.2 # 2  # Lernrate
    alpha = 0.8  # Momentum

    delta_w_neu = (1 - alpha) * eta * gradients + alpha * delta_w_neu
    #print(delta_w_neu)

    #C.setpoints(C.points-delta_w_neu)
    C.set_c(C.c-gradients*eta)#new_variable=old_variable−learning_rate*gradient

    mask,x,y=C.draw()
    Cplt.set_data(x,y)

    overlay = u * f
    uplt.set_data(overlay)

    
    step += 1

    return[uplt,Cplt,print_step]
plt.pause(2)
anim = animation.FuncAnimation(fig, animate, interval=10,cache_frame_data=False,blit=True)
plt.show()



    
