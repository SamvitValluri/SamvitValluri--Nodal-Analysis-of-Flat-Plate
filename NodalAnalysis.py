import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
#All units are taken to be in SI units
#Inputted Geometric Properties
plate_length = 0.1524
cyl_rad = 0.01905

#Inputted Material Properties
#Aluminium Values
#k_mat = 236 * (10**-3 * 3)     #Thermal Conductivity
#rho = 2710 * (10**-3 * 3)**3   #Density
#cp = 0.903 * 10**3             #specific heat
#eps = 0.04                     #Emissivity

#Copper Values
k_mat = 386 #* (10**-3 * 3)
rho = 8960 #* (10**-3 * 3)**3
cp = 0.38 * 10**3
eps = 0.03

#Steel Values
#k_mat = 52 * (10**-3 * 3)
#rho = 7870 * (10**-3 * 3)**3
#cp = 0.472* 10**3
#eps = 0.25
alpha = (k_mat) / ((rho)*cp)      #Diffusivity
#Inputted inital and boundary conditions
u_initial = 22
u_cyl_temp = 22
u_inf = 22
qdot = 1.5
#Inputted Mesh Size
Mesh_Size = 50      #The total number of elements will be the square of the Mesh_Size
#Time
time_seconds = 5     #The number of seconds that the simulation will calculate for

#Making sure values are proportional if delta_x = 1
conv_factor = Mesh_Size / plate_length   #Changes the values of any parameter with length such that delta_x = 1
k_mat = k_mat / conv_factor
rho = rho / conv_factor**3
alpha = alpha * conv_factor **2
delta_x = 1                                   # Do not change
delta_t = (delta_x ** 2)/(4*alpha)            #Calculated to prevent instability from forward euler method
max_iter_time = int(time_seconds / delta_t)    #The number of iterations
gamma = (alpha * delta_t) / (delta_x ** 2)    #Intermediate Variable

#Convection and Radiation Analysis
#Function defining the free convection coeffcient as a Function of Temperature
def h(T):
    if T < 40:
        return(6)
    if T < 80:
        return(7)
    if T < 100:
        return(8)
    else:
        return(10)
#Function defining the h equivalent for radiation as a Function of Temperature
def h_rad(T,T_inf):
    if T == T_inf:
        return(0)
    return(eps * 5.67 * 10**(-8) / conv_factor**2 * (T**4 - T_inf**4) / (T - T_inf))

# Initialize solution: the grid of u(k, i, j)
u = np.empty((max_iter_time, Mesh_Size, Mesh_Size))
# Set the initial condition
u.fill(u_initial)
#Defining the cylinders location
cyl_rad = cyl_rad * conv_factor
cyl_coords = []
for x in range(Mesh_Size):
    for y in range(Mesh_Size):
        if ((x-Mesh_Size/2)**2 + (y-Mesh_Size/2)**2 < cyl_rad**2):
            cyl_coords.append((x,y))
            #u[:,x,y] = u_cyl_temp
#Nodal Calulation
def calculate(u):
    for k in range(0, max_iter_time-1, 1):
        for i in range(1, Mesh_Size-1, delta_x):
            for j in range(1, Mesh_Size-1, delta_x):
                if (i,j) in cyl_coords:
                    u[k + 1, i, j] = gamma * (u[k][i+1][j] + u[k][i-1][j] + u[k][i][j+1] + u[k][i][j-1] - 4*u[k][i][j] + qdot * delta_x**2 / k_mat -(h(u[k,i,j]) + h_rad(u[k,i,j],u_inf))*(1/conv_factor**2) * delta_x / k_mat * (u[k,i,j] - u_inf)) + u[k][i][j]
                else:
                    u[k + 1, i, j] = gamma * (u[k][i+1][j] + u[k][i-1][j] + u[k][i][j+1] + u[k][i][j-1] - 4*u[k][i][j] - (h(u[k,i,j]) + h_rad(u[k,i,j],u_inf))*(1/conv_factor**2)*delta_x / k_mat * (u[k,i,j] - u_inf)) + u[k][i][j]
    return u
#Contour Plot at specific time
def plotheatmap(u_k, k):
    # Clear the current plot figure
    plt.clf()
    plt.title("Temperature Distribution for Copper")
    plt.xlabel("x")
    plt.ylabel("y")

    # This is to plot u_k (u at time-step k)
    plt.pcolormesh(u_k, cmap=plt.cm.jet, vmin=20, vmax=120)
    plt.colorbar()
    return plt
# Do the calculation here
u = calculate(u)
#Gif of result
def animate(k):
    plotheatmap(u[k], k)
anim = animation.FuncAnimation(plt.figure(), animate, frames=max_iter_time, repeat=False)
anim.save("heat_equation_solution.gif", fps = max_iter_time / time_seconds)
print("Done!")
