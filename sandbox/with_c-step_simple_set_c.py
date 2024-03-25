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
image_path = './sample_images/rectangular.png'
image = io.imread(image_path)
f = color.rgb2gray(image)
#f = gaussian_filter(f, 3)           # XXX to simulate full DS ... does help a little maybe ...
u=f


spline_points=20
s_points = int(spline_points *2) #100     # dono if ok that way or what is better
size=np.array(f.shape)
k=2
C=Spline((generate_points_in_circle(spline_points)*size/6+size/2), k=k)
mask,x,y=C.draw(np.zeros(f.shape,np.uint8))

lambd, v = 10, 0.00001


from matplotlib import animation
fig=plt.figure()
uplt=plt.imshow(u)
Cplt,=plt.plot(x,y,"-b")
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

    # B_Bold = np.zeros([C.points.shape[0], s.shape[0]])      # should be this shape ...
    # print(B_Bold.shape)                                         # (20, 100)
    B_Bold = C.designmatrix(s)
    # print(B_Bold.shape, "\n")                                     # (s_points, spline_points +k +1)
    # print(C.spline.t, C.spline.t.shape)                         # (27,)         # 27 - (k+1) = 23       # k = 3     # XXX NEXT: Find additional points in t, remove them, use
    # print(C.spline(C.spline.t), C.spline(C.spline.t).shape)     # (27, 2) --> 7 extra Punkte (die ersten 7 sind auch die letzten 7 !)
    # print(C.points, C.points.shape)                             # (20, 2) 
                                                                  # also werden von t -> B (k+1) Punkte entfernt. 
                                                                  # XXX: welche?-> Die k+1 letzten  (über alle übrigen müssen wir iterieren bzw. wir müssen die 20 richtigen in B finden)
                                                                  # XXX: Was ist mit den übrigen 3 die mehr sind?
    # print(C.spline.t.shape, C.spline.t)
    # t_trunk = C.spline.t[:-k-1]    # remove k+1 last elements
    # print(t_trunk.shape, t_trunk)
    C_points = C.c#C_points = C.spline.c # = C.spline(t_trunk)

    # print(B_Bold)
    # print(C.points.shape)
    # print(C_points.shape, "\n")
    # print(C_points, "\n")


    s = C.spline(s).astype(int)                 # pixel coordinates for s-points
    gradients=np.zeros(C_points.shape)          # we are looking for gradients for each control point in x_m and y_m direction
    for m in range(C_points.shape[0]):          # for each control point ... but we have more in B_Bold
        for i in range(s.shape[0]):             # for each point si on spline
            #print("m, i :", m, i)               # i  is ok, but m is not XXX
            si = s[i]
            si_next = s[0]  if i == s.shape[0]-1 else s[i+1]               # exception for last
            si_prev = s[-1] if i == 0            else s[i-1]               # exception for first

            e_part = (e_p[*si] - e_m[*si])

            # for x, y
            ex, ey = e_part*nx[i], e_part*ny[i]
            vx, vy = v*(si_prev[0] - 2*si[0] + si_next[0]), v*(si_prev[1] - 2*si[1] + si_next[1])       # p.25 --> is x and y here 

            B_part = B_Bold[i,m] #Bold_B = # Matrix contains the spline basis function_s evaluated at the nodes si: Normal_B_ij = Normal_B_i(sj) ...

            gradients[m][0] += B_part * (ex + vx)
            gradients[m][1] += B_part * (ey + vy)     # Konvergiert besser ohne v-Part ... aber immer noch nicht richtig




    # XXX NEXT: über die k+1 ersten Punkte in C.points doppelt drüber gehen !
            # since, with spline_points = 10, s_points = 20, k=2: 
            # B_Bold.shape = (20, 12) = (i, m+k)
            # t:       (15,) [-0.25 -0.15  0.    0.05  0.15  0.25  0.35  0.45  0.55  0.65  0.75  0.85  1.    1.05  1.15]
            # t_trunk: (12,) [-0.25 -0.15  0.    0.05  0.15  0.25  0.35  0.45  0.55  0.65  0.75  0.85]
            # und dafür nur auf C.points und nicht auf C_points arbeiten

    # XXX ODER: B-Matrix nur mit t-k-1 Punkten generieren
            


    eta = 0.001 # 0.2 # 2  # Lernrate
    # C.setpoints(C_points+gradients*eta)#new_variable=old_variable−learning_rate*gradient
    C.set_c(C_points+gradients*eta)
    #print(C_points)

    mask,x,y=C.draw()
    Cplt.set_data(x,y)

    overlay = (u_in + u_out) * f
    uplt.set_data(overlay)

    
    step += 1

    return[uplt,Cplt,print_step]
plt.pause(2)
anim = animation.FuncAnimation(fig, animate, interval=10,cache_frame_data=False,blit=True)
plt.show()



    
