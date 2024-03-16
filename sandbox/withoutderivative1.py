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













# Load the image
image_path = './sample_images/snale1.png'
image = io.imread(image_path)
f = color.rgb2gray(image)
u=f


size=np.array(f.shape)
n_points = 20 #100 got also stuck # XXX was 20, a lot faster, but got stuck at snail-gap
C=Spline(generate_points_in_circle(n_points,size/3,size/2))

mask,x,y=C.draw(np.zeros(f.shape,np.uint8))



lambd,v=10,0.000001



from matplotlib import animation
fig=plt.figure()
uplt=plt.imshow(u)
Cplt,=plt.plot(x,y,"-b")
u=uiter(f,C,u,lambd,iterations=100)
delta_w_neu=0
step=0
print_step = plt.text(1,5,"Step: "+str(step))
def animate(frame):
    global u,delta_w_neu,C,step, print_step
    
    print_step.set_text("Step: "+str(step)) # = plt.text(1,5,"Step: "+str(step))

    u=uiter(f,C,u,lambd,iterations=4)
    
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
    gradients=np.zeros(C.points.shape)
    for index in np.ndindex(C.points.shape):
        pointsplus =np.copy(C.points)
        pointsminus=np.copy(C.points)
        pointsplus[index] +=h
        pointsminus[index]-=h
        Cplus =Spline(pointsplus )
        Cminus=Spline(pointsminus)
        uplus =uiter(f,Cplus ,u,lambd,iterations=2)
        uminus=uiter(f,Cminus,u,lambd,iterations=2)
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
    print(gradients)
    glen=np.sqrt(gradients[:,0]**2+gradients[:,1]**2)[:,None]
    #gradients=gradients/glen*np.sqrt(glen)/np.min(np.sqrt(glen))
    #gradients=gradients/np.sqrt(glen)/4


    eta = 2  # Lernrate
    alpha = 0.5  # Momentum

    #delta_w_neu = (1 - alpha) * eta * gradients + alpha * delta_w_neu
    #print(delta_w_neu)

    C.setpoints(C.points-gradients*0.2)#new_variable=old_variable−learning_rate*gradient
    #uplt.set_array(u)
    mask,x,y=C.draw()
    Cplt.set_data(x,y)

    step += 1

    return[uplt,Cplt,print_step]
plt.pause(0.5)
anim = animation.FuncAnimation(fig, animate, interval=10,cache_frame_data=False,blit=True)
plt.show()



    
