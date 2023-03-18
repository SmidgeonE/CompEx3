# $$
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

G = 6.67E-11
Mearth = 5.972E24
Mmoon = 7.348E22
Rearth = 6371000
Msat = 1


def accelX(x, y):
    return (-G * Mearth * x) / ((x**2 + y**2)**(3/2))

def accelY(x, y):
    return (-G * Mearth * y) / ((x**2 + y**2)**(3/2))

def EvalRK4(initVel=[0,0], initDisp=[0,0],
            numPoints=1000, simTime=9000,
            plotXY=True, plotE=True,
            modelMoon=False):
    h = simTime / numPoints
    vx, vy, x, y = np.zeros(numPoints), np.zeros(numPoints), np.zeros(numPoints), np.zeros(numPoints)
    t = np.linspace(0, simTime, numPoints)
    vx[0], vy[0], x[0], y[0] = initVel[0], initVel[1], initDisp[0], initDisp[1]

    for i, dt in enumerate(t):
        if dt == t[-1]:
            continue
        # k arrays are 2d 4x4 arrays that go 1,2,3,4 then kx, ky, kvx, kvy
        k = np.zeros((4, 4))
        k[0] = np.array([vx[i], vy[i], accelX(x[i], y[i]), accelY(x[i], y[i])])

        for j in range(k.shape[0]-1):
            if j == range(k.shape[0]-1)[-1]:
                k[j + 1] = np.array([vx[i] + h * k[j][2], vy[i] + h * k[j][3],
                                     accelX(x[i] + h * k[j][0], y[i] + h * k[j][1]),
                                     accelY(x[i] + h * k[j][0], y[i] + h * k[j][1])])
            else:
                k[j+1] = np.array([vx[i] + h * k[j][2] / 2, vy[i] + h * k[j][3] / 2,
                                 accelX(x[i] + h * k[j][0] / 2, y[i] + h * k[j][1] / 2),
                                 accelY(x[i] + h * k[j][0] / 2, y[i] + h * k[j][1] / 2)])


        multArray = np.array([1, 2, 2, 1])
        x[i+1] = x[i] + (h/6)*np.sum(k[:,0]*multArray)
        y[i+1] = y[i] + (h/6)*np.sum(k[:,1]*multArray)
        vx[i + 1] = vx[i] + (h / 6) * np.sum(k[:, 2] * multArray)
        vy[i + 1] = vy[i] + (h / 6) * np.sum(k[:, 3] * multArray)
    if plotXY:
        plt.figure(0)
        earth = np.asarray(Image.open(os.getcwd()+"/earth.PNG"))
        plt.imshow(earth, extent=(-Rearth, Rearth, -Rearth, Rearth), alpha=0.9)
        plt.plot(x, y, label='Trajectory')
        plt.scatter(x[0], y[0], marker='x', color='red', label='Initial Point')
        lim = np.max([np.abs(x), np.abs(y)])
        border = 1.2
        lim *= border
        order = int(np.log10(lim))
        plt.ylabel('y / 10x'+str(order)+'m')
        plt.xlabel('x / 10x'+str(order)+'m')
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
        plt.legend()
        plt.show()

    if plotE:
        # Calculating Total Potential
        totalPotential = 0.5 * (vx ** 2 + vy ** 2) + (G * Mearth) / ((vx ** 2 + vy ** 2) ** 0.5)
        plt.figure(1)
        plt.plot(t, totalPotential, color='red', label='Total Energy')
        plt.ylabel("Total Potential Energy / $J\,kg^{-1}$")
        plt.xlabel("Time / s")
        plt.legend()
        plt.show()

print('------\nWelcome to Exercise 4\n\nObjectives:\n'
      '--Using computational tools and knowledge to solve a real world orbital problem.'
      '\n--Utilising Runge-Kutta algorithms to computationally solve orbital motion equations.')


# Loops contains the UI that loops until the user quits.

while True:
    answer = input('\nPlease enter one of the following letters, or q to quit:\n'
                   ' --(a) Satellite orbit around Earth (Circular)\n'
                   ' --(b) Satellite orbit around Earth (Eccentric)\n'
                   ' --(c) Satellite orbit around Earth (Comet Passing Fast)\n'
                   ' --(d) Satellite orbit around Earth (Comet Passing Slow)\n'
                   ' --(e) Moon\n')

    if answer == 'a':
        EvalRK4([0, 8300], [6700000, 0])

    if answer == 'b':
        EvalRK4([0, 9500], [6700000, 0], simTime=18000)

    if answer == 'c':
        EvalRK4([-24000, 11000], [20000000, 0], simTime=3000)

    if answer == 'd':
        EvalRK4([-3000, 5900], [15000000, 0], simTime=50000, plotE=False)

    if answer == 'e':
        EvalRK4([0, 8300], [6700000, 0], modelMoon=True)




    elif answer != 'q':
        print("That was not a valid answer")

    else:
        print("Thank you")
        break

# $$
