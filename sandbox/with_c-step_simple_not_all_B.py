import numpy as np
from bsplineclass import Spline
from skimage import io, color
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







# Load the image
image_path = './sample_images/circle.png'
image = io.imread(image_path)
f = color.rgb2gray(image)
u=f


spline_points = 4
s_points = spline_points *2     # dono if ok that way or what is better
size=np.array(f.shape)
C=Spline((generate_points_in_circle(spline_points)*size/3+size/2), k=2)
mask,x,y=C.draw(np.zeros(f.shape,np.uint8))

lambd, v = 10, 0.000001


from matplotlib import animation
fig=plt.figure()
uplt=plt.imshow(u)
Cplt,=plt.plot(x,y,"-b")
u,*_ = u_simple(f,C)
step=0
print_step = plt.text(1,5,"Step: "+str(step))
def animate(frame):
    global u_in,u_out,C,step, print_step
    
    print_step.set_text("Step: "+str(step))


    u,u_in,u_out=u_simple(f,C)
    e_p, e_m = np.power((f-u_out),2), np.power((f-u_in),2)  # "energy" outside, inside

    s = np.linspace(0, 1, s_points,endpoint=False)
    nx, ny=C.normals(s)

    B_Bold = np.zeros([C.points.shape[0], s.shape[0]])      # should be this shape ...
    print(B_Bold.shape)                                         # (20, 100)
    B_Bold = C.designmatrix(s)
    # print(C.spline.t, C.spline.t.shape)                         # (27,)         # 27 - (k+1) = 23       # k = 3     # XXX NEXT: Find additional points in t, remove them, use
    # print(C.spline(C.spline.t), C.spline(C.spline.t).shape)     # (27, 2) --> 7 extra Punkte (die ersten 7 sind auch die letzten 7 !)
    # print(C.points, C.points.shape)                             # (20, 2) 
                                                                  # also werden von t -> B (k+1) Punkte entfernt. 
                                                                  # XXX: welche?-> Die k+1 letzten  (über alle übrigen müssen wir iterieren bzw. wir müssen die 20 richtigen in B finden)
                                                                  # XXX: Was ist mit den übrigen 3 die mehr sind?
    print(B_Bold.shape)                                   # (100, 23)
    print(B_Bold)
    C_points = C.spline(C.spline.t)
    print(C.points.shape)
    print(C.points)
    print(C_points.shape)
    print(C_points, "\n")


    s = C.spline(s).astype(int)                 # pixel coordinates for s-points
    # print(s.shape)
    gradients=np.zeros(C.points.shape)          # we are looking for gradients for each control point in x_m and y_m direction
    for m in range(C.points.shape[0]):          # for each control point ... but we have more in B_Bold
        for i in range(s.shape[0]):             # for each point si on spline
            #print("m, i :", m, i)               # i  is ok, but m is not XXX, since B has i * (m+k+1) elements
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



    eta = 0.001 # 0.2 # 2  # Lernrate
    C.setpoints(C.points+gradients*eta)#new_variable=old_variable−learning_rate*gradient

    mask,x,y=C.draw()
    Cplt.set_data(x,y)

    overlay = (u_in + u_out) * f
    uplt.set_data(overlay)

    
    step += 1

    return[uplt,Cplt,print_step]
plt.pause(2)
anim = animation.FuncAnimation(fig, animate, interval=10,cache_frame_data=False,blit=True)
plt.show()



    
