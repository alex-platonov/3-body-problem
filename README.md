# Three-Body Problem Simulation

## General Description of the Problem
The three-body problem in celestial mechanics is a classical problem that aims to determine the motions of three celestial bodies based on their initial positions, velocities, and mutual gravitational attraction. This problem encapsulates the complexity and unpredictability inherent in a system where three bodies exert gravitational forces on each other, influencing their paths through space.

## Formal Description of the Problem
Given three celestial bodies of masses �1m1​, �2m2​, and �3m3​, with initial positions and velocities in a three-dimensional space, the problem involves solving the equations of motion derived from Newton's laws of motion and universal gravitation. The gravitational force between each pair of bodies is proportional to the product of their masses and inversely proportional to the square of the distance between them. The challenge lies in predicting the trajectories of these bodies over time.

## Problems
- Nonlinearity: The equations of motion for the three-body problem are inherently nonlinear due to the inverse square law of gravity, leading to complex behaviors and sensitivity to initial conditions.
- No General Solution: Unlike the two-body problem, there is no general closed-form solution for the three-body problem, making exact predictions impossible for arbitrary initial conditions.
- Chaos: Small changes in initial conditions can lead to vastly different outcomes, a hallmark of chaotic systems.

## Possible Solutions
- Numerical Simulation: The most common approach is using numerical methods to approximate the positions and velocities of the bodies at discrete time intervals.
- Specialized Solutions: For specific initial conditions or configurations, like the Lagrange points or the Eulerian solutions for collinear bodies, exact solutions exist.
- Regularization Techniques: Techniques to simplify the mathematical formulation and reduce computational errors in near-collision scenarios.

## Special Cases and Approximation
- Restricted Three-Body Problem: Assumes one of the bodies has negligible mass, simplifying calculations and allowing for the analysis of Lagrange points.
- Lagrange Points: Positions in an orbital configuration where a small object affected only by gravity can theoretically be stationary relative to two larger objects.

## Historical Context
The three-body problem has been studied for centuries, with significant contributions from mathematicians like Newton, Euler, Lagrange, and Poincaré. Henri Poincaré's work in the late 19th century revealed the problem's inherent complexity and laid the groundwork for chaos theory.

## Cultural Context
The three-body problem has transcended its scientific origins to influence culture and literature, notably inspiring the Chinese science fiction novel "The Three-Body Problem" by Liu Cixin. This work, and others like it, highlight the human fascination with the cosmos and the challenges of understanding its dynamics.

## Implementation
This simulation is implemented in Python, using numerical methods to approximate the complex interactions between three celestial bodies. The code relies on scientific libraries like NumPy and Matplotlib for calculations and visualizations, respectively.

## References
- "Modelling the Three Body Problem in Classical Mechanics using Python" by Gaurav Deshmukh published in "Towards Data Science", 2019
   https://towardsdatascience.com/modelling-the-three-body-problem-in-classical-mechanics-using-python-9dc270ad7767
- "The three body problem" B. L. Badger
   https://blbadger.github.io/3-body-problem.html

