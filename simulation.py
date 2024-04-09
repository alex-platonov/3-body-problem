#Importing modules
import streamlit as st
import scipy as sci
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.integrate
from IPython.display import HTML
from matplotlib.animation import PillowWriter

#warnings.filterwarnings('ignore')

st.title('Three-Body Problem Simulation')
st.image('https://raw.githubusercontent.com/alex-platonov/3-body-problem/main/problem.jpg')
st.write('Having watched the new Netflix series I entertained the idea of making a little simulation to this classic problem from the field of celestial mechanics.')

st.subheader('General Description of the Problem')
st.write('The three-body problem in celestial mechanics is a classical problem that aims to determine the motions of three celestial bodies based on their initial positions, velocities, and mutual gravitational attraction. This problem encapsulates the complexity and unpredictability inherent in a system where three bodies exert gravitational forces on each other, influencing their paths through space.')

st.subheader('Formal Description of the Problem')
st.write('Given three celestial bodies of masses �1m1​, �2m2​, and �3m3​, with initial positions and velocities in a three-dimensional space, the problem involves solving the equations of motion derived from Newtons laws of motion and universal gravitation. The gravitational force between each pair of bodies is proportional to the product of their masses and inversely proportional to the square of the distance between them. The challenge lies in predicting the trajectories of these bodies over time.')

st.subheader('Problematics')
st.markdown("""
- Nonlinearity: The equations of motion for the three-body problem are inherently nonlinear due to the inverse square law of gravity, leading to complex behaviors and sensitivity to initial conditions.
- No General Solution: Unlike the two-body problem, there is no general closed-form solution for the three-body problem, making exact predictions impossible for arbitrary initial conditions.
- Chaos: Small changes in initial conditions can lead to vastly different outcomes, a hallmark of chaotic systems.
""", unsafe_allow_html=True)

st.subheader('Possible Solutions')
st.markdown("""
- Numerical Simulation: The most common approach is using numerical methods to approximate the positions and velocities of the bodies at discrete time intervals.
- Specialized Solutions: For specific initial conditions or configurations, like the Lagrange points or the Eulerian solutions for collinear bodies, exact solutions exist.
- Regularization Techniques: Techniques to simplify the mathematical formulation and reduce computational errors in near-collision scenarios.
""", unsafe_allow_html=True)

st.subheader('Special Cases and Approximation')
st.markdown("""
- Restricted Three-Body Problem: Assumes one of the bodies has negligible mass, simplifying calculations and allowing for the analysis of Lagrange points.
- Lagrange Points: Positions in an orbital configuration where a small object affected only by gravity can theoretically be stationary relative to two larger objects.
""", unsafe_allow_html=True)

st.subheader('Historical Context')
st.write('The three-body problem has been studied for centuries, with significant contributions from mathematicians like Newton, Euler, Lagrange, and Poincaré. Henri Poincaré work in the late 19th century revealed the problem inherent complexity and laid the groundwork for chaos theory.')

st.subheader('Implementation')
st.write('This simulation is implemented in Python, using numerical methods to approximate the complex interactions between three celestial bodies. The code relies on scientific libraries like NumPy and Matplotlib for calculations and visualizations, respectively.')

st.subheader('References')
st.write('"Modelling the Three Body Problem in Classical Mechanics using Python" by Gaurav Deshmukh published in "Towards Data Science", 2019 https://towardsdatascience.com/modelling-the-three-body-problem-in-classical-mechanics-using-python-9dc270ad7767')
st.write('"The three body problem" B. L. Badger https://blbadger.github.io/3-body-problem.html')


st.subheader('Simulation')
st.write('Please define the graviational constraint used for non-dimensionalization, making the equations of motion simpler to solve numerically or used already supplied values. Default value is 6.67408e-11 Gravitational constant in N-m2/kg2') 
G = st.number_input('Enter G value:', value=6.67408e-11)  # Default value is 6.67408e-11 Gravitational constant in N-m2/kg2

st.write('I am keen to leave the following reference quantities unchanged for experimentation purposes')
st.markdown("""
Reference quantities for non-dimensionalization
m_nd = 1.989e+30  # Reference mass (mass of the sun), kg
r_nd = 5.326e+12  # Reference distance, m
v_nd = 30000  # Reference velocity, m/s
t_nd = 79.91 * 365.25 * 24 * 3600  # Reference time, s
""", unsafe_allow_html=True)

st.markdown("""
The following reference quantities are to be used for non-dimensionalization
m_nd = 1.989e+30  # Reference mass (mass of the sun), kg
r_nd = 5.326e+12  # Reference distance, m
v_nd = 30000  # Reference velocity, m/s
t_nd = 79.91 * 365.25 * 24 * 3600  # Reference time, s
""", unsafe_allow_html=True)

# Reference quantities for non-dimensionalization
m_nd = 1.989e+30  # Reference mass (mass of the sun), kg
r_nd = 5.326e+12  # Reference distance, m
v_nd = 30000  # Reference velocity, m/s
t_nd = 79.91 * 365.25 * 24 * 3600  # Reference time, s

st.markdown("""
# Net constants derived from reference quantities, used in equations of motion
K1 = G * t_nd * m_nd / (r_nd**2 * v_nd)
K2 = v_nd * t_nd / r_nd
""", unsafe_allow_html=True)

# Net constants derived from reference quantities, used in equations of motion
K1 = G * t_nd * m_nd / (r_nd**2 * v_nd)
K2 = v_nd * t_nd / r_nd

st.write('Please define masses of the three bodies, in units relative to the reference mass:')
m1 = st.number_input('Enter mass for Star 1:', value=1.1)  # Default value is 1.1
m2 = st.number_input('Enter mass for Star 2:', value=0.907)  # Default value is 0.907
m3 = st.number_input('Enter mass for Star 3:', value=1.425)  # Default value is 1.425

st.write('Please define initial position vectors for the three bodies, in reference units as comma separated values.')
# Define initial position vectors for the three bodies, in reference units
# r1 = [-0.5, 1, 0]  # Initial position of Star 1
# r2 = [0.5, 0, 0.5]  # Initial position of Star 2
# r3 = [0.2, 1, 1.5]  # Initial position of Star 3

# For Star 1
r1_x = st.number_input('Enter x position for Star 1:', value=-0.5)
r1_y = st.number_input('Enter y position for Star 1:', value=1.0)
r1_z = st.number_input('Enter z position for Star 1:', value=0.0)

# Combine individual components into a position vector for Star 1
r1 = [r1_x, r1_y, r1_z]

# For Star 2
r2_x = st.number_input('Enter x position for Star 2:', value=0.5)
r2_y = st.number_input('Enter y position for Star 2:', value=0.0)
r2_z = st.number_input('Enter z position for Star 2:', value=0.5)

# Combine individual components into a position vector for Star 2
r2 = [r2_x, r2_y, r2_z]

# For Star 3
r3_x = st.number_input('Enter x position for Star 3:', value=0.2)
r3_y = st.number_input('Enter y position for Star 3:', value=1.0)
r3_z = st.number_input('Enter z position for Star 3:', value=1.5)

# Combine individual components into a position vector for Star 3
r3 = [r3_x, r3_y, r3_z]

# Convert position vectors from lists to numpy arrays for easier mathematical operations
r1 = np.array(r1)
r2 = np.array(r2)
r3 = np.array(r3)

st.write('To calculate the center of mass (COM) position, considering the masses and positions of the three bodies the following equation is used: r_com = (m1 * r1 + m2 * r2 + m3 * r3) / (m1 + m2 + m3)')
# Calculate the center of mass (COM) position, considering the masses and positions of the three bodies
r_com = (m1 * r1 + m2 * r2 + m3 * r3) / (m1 + m2 + m3)

st.write('Please define initialelocities for the three bodies, in reference units.')
# Define initial velocities for the three bodies, in reference units
#v1 = [0.02, 0.02, 0.02]  # Initial velocity of Star 1
#v2 = [-0.05, 0, -0.1]  # Initial velocity of Star 2
#v3 = [0, -0.03, 0]  # Initial velocity of Star 3

# For Star 1
v1_x = st.number_input('Enter x velocity for Star 1:', value=0.02)
v1_y = st.number_input('Enter y velocity for Star 1:', value=0.02)
v1_z = st.number_input('Enter z velocity for Star 1:', value=0.02)

# Combine individual components into a velocity vector for Star 1
v1 = [v1_x, v1_y, v1_z]

# For Star 2
v2_x = st.number_input('Enter x velocity for Star 2:', value=-0.05)
v2_y = st.number_input('Enter y velocity for Star 2:', value=0.0)
v2_z = st.number_input('Enter z velocity for Star 2:', value=-0.1)

# Combine individual components into a velocity vector for Star 2
v2 = [v2_x, v2_y, v2_z]

# For Star 3
v3_x = st.number_input('Enter x velocity for Star 3:', value=0.0)
v3_y = st.number_input('Enter y velocity for Star 3:', value=-0.03)
v3_z = st.number_input('Enter z velocity for Star 3:', value=0.0)

# Combine individual components into a velocity vector for Star 3
v3 = [v3_x, v3_y, v3_z]

# Convert velocity vectors from lists to numpy arrays for easier operations
v1 = np.array(v1)
v2 = np.array(v2)
v3 = np.array(v3)

st.write(' To calculate the velocity of the center of mass (COM), considering the masses and velocities of the three bodies the following equation is used: v_com = (m1 * v1 + m2 * v2 + m3 * v3) / (m1 + m2 + m3)')
# Calculate the velocity of the center of mass (COM), considering the masses and velocities of the three bodies
v_com = (m1 * v1 + m2 * v2 + m3 * v3) / (m1 + m2 + m3)

# Function that defines the equations governing the motion of the three bodies
# The function takes the current state (positions and velocities) of the bodies, the time, and the masses
# It returns the derivatives of the state (velocities and accelerations)
def ThreeBodyEquations(w, t, G, m1, m2):
    # Unpack the input array into position and velocity vectors for each body
    r1 = w[:3]
    r2 = w[3:6]
    r3 = w[6:9]
    v1 = w[9:12]
    v2 = w[12:15]
    v3 = w[15:18]

# Compute distances between the bodies
    r12 = sci.linalg.norm(r2 - r1)  # Distance between Star 1 and Star 2
    r13 = sci.linalg.norm(r3 - r1)  # Distance between Star 1 and Star 3
    r23 = sci.linalg.norm(r3 - r2)  # Distance between Star 2 and Star 3
    
    # Compute the acceleration of each body due to the gravitational attraction from the other two bodies
    dv1bydt = K1 * m2 * (r2 - r1) / r12**3 + K1 * m3 * (r3 - r1) / r13**3
    dv2bydt = K1 * m1 * (r1 - r2) / r12**3 + K1 * m3 * (r3 - r2) / r23**3
    dv3bydt = K1 * m1 * (r1 - r3) / r13**3 + K1 * m2 * (r2 - r3) / r23**3
    
    # Velocity of each body is simply the derivative of its position
    dr1bydt = K2 * v1
    dr2bydt = K2 * v2
    dr3bydt = K2 * v3
    
    # Combine all derivatives into a single array to return
    r12_derivs = np.concatenate((dr1bydt, dr2bydt))
    r_derivs = np.concatenate((r12_derivs, dr3bydt))
    v12_derivs = np.concatenate((dv1bydt, dv2bydt))
    v_derivs = np.concatenate((v12_derivs, dv3bydt))
    derivs = np.concatenate((r_derivs, v_derivs))
    return derivs

# Initial conditions: Flatten the initial position and velocity arrays into a single array for the ODE solver
init_params = np.array([r1, r2, r3, v1, v2, v3])  # Initial positions and velocities
init_params = init_params.flatten()  # Flatten to make it a 1D array for the solver
time_span = np.linspace(0, 20, 1000)  # Simulate for 20 units of time with 1000 points in between

# Use scipy's ODE integrator to solve the three body equations
three_body_sol = sci.integrate.odeint(ThreeBodyEquations, init_params, time_span, args=(G, m1, m2))

# Extract solutions for each star's position over time
r1_sol = three_body_sol[:, :3]
r2_sol = three_body_sol[:, 3:6]
r3_sol = three_body_sol[:, 6:9]

# Plot the orbits of the three bodies.
# Set up the figure for plotting the orbits of the three bodies
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection="3d")

# Plot the orbit of each star using the solutions obtained from the ODE solver
ax.plot(r1_sol[:, 0], r1_sol[:, 1], r1_sol[:, 2], color="mediumblue", label="Star 1")
ax.plot(r2_sol[:, 0], r2_sol[:, 1], r2_sol[:, 2], color="red", label="Star 2")
ax.plot(r3_sol[:, 0], r3_sol[:, 1], r3_sol[:, 2], color="gold", label="Star 3")

# Mark the final positions of each star with a larger dot
ax.scatter(r1_sol[-1, 0], r1_sol[-1, 1], r1_sol[-1, 2], color="darkblue", marker="o", s=80)
ax.scatter(r2_sol[-1, 0], r2_sol[-1, 1], r2_sol[-1, 2], color="darkred", marker="o", s=80)
ax.scatter(r3_sol[-1, 0], r3_sol[-1, 1], r3_sol[-1, 2], color="goldenrod", marker="o", s=80)

# Label the axes and set a title for the plot
ax.set_xlabel("x-coordinate", fontsize=14)
ax.set_ylabel("y-coordinate", fontsize=14)
ax.set_zlabel("z-coordinate", fontsize=14)
ax.set_title("Visualization of orbits of stars in a 3-body system", fontsize=14)
ax.legend(loc="upper left", fontsize=14)  # Add a legend
ig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection="3d")

#Create new arrays for animation, this gives you the flexibility
#to reduce the number of points in the animation if it becomes slow
#Currently set to select every 4th point
r1_sol_anim = r1_sol[::1,:].copy()
r2_sol_anim = r2_sol[::1,:].copy()
r3_sol_anim = r3_sol[::1,:].copy()

# Initial markers for planets
head1 = [ax.scatter(r1_sol_anim[0,0], r1_sol_anim[0,1], r1_sol_anim[0,2], color="darkblue", marker="o", s=80, label="Star 1")]
head2 = [ax.scatter(r2_sol_anim[0,0], r2_sol_anim[0,1], r2_sol_anim[0,2], color="darkred", marker="o", s=80, label="Star 2")]
head3 = [ax.scatter(r3_sol_anim[0,0], r3_sol_anim[0,1], r3_sol_anim[0,2], color="goldenrod", marker="o", s=80, label="Star 3")]

# Function to animate the orbits
def Animate(i, head1, head2, head3):
    # Remove old markers
    head1[0].remove()
    head2[0].remove()
    head3[0].remove()
    
    # Plot the orbits
    ax.plot(r1_sol_anim[:i,0], r1_sol_anim[:i,1], r1_sol_anim[:i,2], color="mediumblue")
    ax.plot(r2_sol_anim[:i,0], r2_sol_anim[:i,1], r2_sol_anim[:i,2], color="red")
    ax.plot(r3_sol_anim[:i,0], r3_sol_anim[:i,1], r3_sol_anim[:i,2], color="gold")
    
    # Plot the current markers
    head1[0] = ax.scatter(r1_sol_anim[i-1,0], r1_sol_anim[i-1,1], r1_sol_anim[i-1,2], color="darkblue", marker="o", s=100)
    head2[0] = ax.scatter(r2_sol_anim[i-1,0], r2_sol_anim[i-1,1], r2_sol_anim[i-1,2], color="darkred", marker="o", s=100)
    head3[0] = ax.scatter(r3_sol_anim[i-1,0], r3_sol_anim[i-1,1], r3_sol_anim[i-1,2], color="goldenrod", marker="o", s=100)
    return head1, head2, head3,

# Additional settings for the plot
ax.set_xlabel("x-coordinate", fontsize=14)
ax.set_ylabel("y-coordinate", fontsize=14)
ax.set_zlabel("z-coordinate", fontsize=14)
ax.set_title("Visualization of orbits of stars in a 3-body system\n", fontsize=14)
ax.legend(loc="upper left", fontsize=14)

# Creating the animation
repeatanim = animation.FuncAnimation(fig, Animate, frames=800, interval=10, blit=False, fargs=(head1, head2, head3))

# Displaying the animation in the notebook
gif_path = "three_body_problem.gif"
repeatanim.save(gif_path, writer=PillowWriter(fps=20))
st.image(gif_path)

#EOF
