import numpy as np
from src.cicleBspline import Spline
from skimage import io, color
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt





def generate_points_in_circle(num_points):
    theta = np.linspace(0, 2 * np.pi, num_points,endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)
    return np.stack((x, y), axis=-1)


def u_simple(f, spline):
    in_mask, out_mask, _ = spline.get_masks(np.zeros(f.shape,np.uint8), steps=200)
    u_in, u_out = in_mask.astype(float), out_mask.astype(float),
    u_in *= np.sum(f*in_mask) / np.sum(in_mask) 
    u_out*= np.sum(f*out_mask)/ np.sum(out_mask)
    u = u_in + u_out
    return u, u_in, u_out 


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
image_path = './sample_images/rect_4.png'
image = io.imread(image_path)
f = color.rgb2gray(image)
print(f.shape)
#f = gaussian_filter(f, 3)           # XXX to simulate full DS ... does help a little maybe ...
u=f


spline_points=20
s_points = int(spline_points *1.5) #100     # dono if ok that way or what is better
size=np.array(f.shape)
k=2
C=Spline((generate_points_in_circle(spline_points)*size/6+size/2), k=k)
mask,x,y=C.draw(np.zeros(f.shape,np.uint8))
lambd, v = 10, 0.00001


from matplotlib import animation
fig=plt.figure()
uplt=plt.imshow(u)
Cplt,=plt.plot(x,y,"-b")
# pointsplt,=plt.plot([],[], 'or')#plot points
# pointsplt.set_data(x, y)

u,*_ = u_simple(f,C)
step=0
print_step = plt.text(1,5,"Step: "+str(step))

print(C.spline.extrapolate)

def animate(frame):
    global u_in,u_out,C,step, print_step
    
    print_step.set_text("Step: "+str(step))


    u, u_in, u_out, e_p, e_m = u_e_simple(f,C)
    # u, u_in, u_out = u_simple(f, C)
    # e_p, e_m = np.power((f-u_out),2), np.power((f-u_in),2)  # "energy" outside, inside

    s = np.linspace(0, 1, s_points,endpoint=False)
    nx, ny=C.normals(s)
    B_Bold = C.designmatrix(s)
    C_points = C.c

    s = C.spline(s).astype(int)                 # pixel coordinates for s-points
    gradients=np.zeros(C_points.shape)          # we are looking for gradients for each control point in x_m and y_m direction
    for m in range(C_points.shape[0]):          # for each control point ... but we have more in B_Bold
        for i in range(s.shape[0]):             # for each point si on spline
            # print("m, i :", m, i)               # XXX is ok now. Since m, i max = (20, 40) and B_Bold.shape = (40, 20 + k)
            si = s[i]
            si_next = s[0]  if i == s.shape[0]-1 else s[i+1]               # exception for last
            si_prev = s[-1] if i == 0            else s[i-1]               # exception for first

            e_part = (e_p[*si] - e_m[*si])

            # for x, y
            ex, ey = e_part*nx[i], e_part*ny[i]
            vx, vy = v*(si_prev[0] - 2*si[0] + si_next[0]), v*(si_prev[1] - 2*si[1] + si_next[1])       # p.25 --> is x and y here 

            B_part = B_Bold[i,m] #Bold_B = # Matrix contains the spline basis function_s evaluated at the nodes si: Normal_B_ij = Normal_B_i(sj) ...

            gradients[m][1] += B_part * (ex + vx)
            gradients[m][0] += B_part * (ey + vy)


    # XXX NEXT: über die k+1 ersten Punkte in C.points doppelt drüber gehen ?
            # since, with spline_points = 10, s_points = 20, k=2: 
            # B_Bold.shape = (20, 12) = (i, m+k)
            # t:       (15,) [-0.25 -0.15  0.    0.05  0.15  0.25  0.35  0.45  0.55  0.65  0.75  0.85  1.    1.05  1.15]
            # t_trunk: (12,) [-0.25 -0.15  0.    0.05  0.15  0.25  0.35  0.45  0.55  0.65  0.75  0.85]
            # und dafür nur auf C.points und nicht auf C_points arbeiten
            

    eta = 0.001 # 0.2 # 2  # Lernrate
    C.set_c(C_points+gradients*eta)

    mask,x,y=C.draw()
    Cplt.set_data(x,y)
    # pointsplt.set_data(y, x)

    overlay = (u_in + u_out) * f
    uplt.set_data(overlay)

    step += 1

    return[uplt,Cplt,print_step]#,pointsplt]
plt.pause(2)
anim = animation.FuncAnimation(fig, animate, interval=10,cache_frame_data=False,blit=True)
plt.show()



    
