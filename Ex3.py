# $$
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

G = 6.67E-11
Mearth = 5.972E24
Mmoon = 7.348E22
Rearth = 6371000
Rmoon = 1737500
Msat = 1
moonY = 384400000
moonX = 0


def accelX(x, y, shouldModelMoon=False):
    if shouldModelMoon:
        return (-G * Mearth * x) / ((x ** 2 + y ** 2) ** (3 / 2)) +\
                 (-G * Mmoon * (x-moonX)) / (((x - moonX) ** 2 + (y-moonY) ** 2) ** (3 / 2))
    else:
        return (-G * Mearth * x) / ((x**2 + y**2)**(3/2))

def accelY(x, y, shouldModelMoon=False):
    if shouldModelMoon:
        return (-G * Mearth * y) / ((x ** 2 + y ** 2) ** (3 / 2)) +\
                 (-G * Mmoon * (y-moonY)) / (((x - moonX) ** 2 + (y-moonY) ** 2) ** (3 / 2))
    else:
        return (-G * Mearth * y) / ((x ** 2 + y ** 2) ** (3 / 2))

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
        k[0] = np.array([vx[i], vy[i], accelX(x[i], y[i], shouldModelMoon=modelMoon), accelY(x[i], y[i], shouldModelMoon=modelMoon)])

        for j in range(k.shape[0]-1):
            if j == range(k.shape[0]-1)[-1]:
                k[j + 1] = np.array([vx[i] + h * k[j][2], vy[i] + h * k[j][3],
                                     accelX(x[i] + h * k[j][0], y[i] + h * k[j][1], shouldModelMoon=modelMoon),
                                     accelY(x[i] + h * k[j][0], y[i] + h * k[j][1], shouldModelMoon=modelMoon)])
            else:
                k[j+1] = np.array([vx[i] + h * k[j][2] / 2, vy[i] + h * k[j][3] / 2,
                                 accelX(x[i] + h * k[j][0] / 2, y[i] + h * k[j][1] / 2, shouldModelMoon=modelMoon),
                                 accelY(x[i] + h * k[j][0] / 2, y[i] + h * k[j][1] / 2, shouldModelMoon=modelMoon)])


        multArray = np.array([1, 2, 2, 1])
        x[i+1] = x[i] + (h/6)*np.sum(k[:,0]*multArray)
        y[i+1] = y[i] + (h/6)*np.sum(k[:,1]*multArray)
        vx[i + 1] = vx[i] + (h / 6) * np.sum(k[:, 2] * multArray)
        vy[i + 1] = vy[i] + (h / 6) * np.sum(k[:, 3] * multArray)

    # Now checking if it hits any of the planets

    firstEarthHit = np.argmax(x**2+y**2<Rearth**2)
    firstMoonHit = 0
    if modelMoon:
        firstMoonHit = np.argmax((x-moonX)**2+(y-moonY)**2<Rmoon**2)

    if firstEarthHit > 0 and firstMoonHit == 0:
        print("It has hit the earth!")
        x, y, vy, vx, t = x[:firstEarthHit], y[:firstEarthHit], \
                          vy[:firstEarthHit], vx[:firstEarthHit], t[:firstEarthHit]
    elif firstMoonHit > 0 and firstEarthHit == 0:
        print("It has hit the moon!")
        x, y, vy, vx, t = x[:firstMoonHit], y[:firstMoonHit], \
                          vy[:firstMoonHit], vx[:firstMoonHit], t[:firstMoonHit]
    elif firstEarthHit > firstMoonHit and firstMoonHit != 0:
        print("It has hit the moon!")
        x, y, vy, vx, t = x[:firstMoonHit], y[:firstMoonHit], \
                          vy[:firstMoonHit], vx[:firstMoonHit], t[:firstMoonHit]
    elif firstEarthHit < firstMoonHit and firstEarthHit != 0:
        print("It has hit the earth!")
        x, y, vy, vx, t = x[:firstEarthHit], y[:firstEarthHit], \
                          vy[:firstEarthHit], vx[:firstEarthHit], t[:firstEarthHit]


    if plotXY:
        plt.figure(0)

        earth = np.asarray(Image.open(os.getcwd()+"/earth.PNG"))
        plt.imshow(earth, extent=(-Rearth, Rearth, -Rearth, Rearth), alpha=0.9)

        if modelMoon:
            moon = np.asarray(Image.open(os.getcwd()+"/moon.PNG"))
            plt.imshow(moon, extent=(-Rmoon, Rmoon, -Rmoon+moonY, Rmoon+moonY), alpha=0.9)

        plt.plot(x, y, label='Trajectory', color='lightblue', lw=0.5)
        plt.scatter(x[0], y[0], marker='x', color='green', label='Initial Point', s=12)

        if len(x) != numPoints:
            print("Plotting crash shite")
            # If the length of the x array isnt the numPoints parameter, it must have crashed.
            plt.plot(x[-1], y[-1], marker='x', color='red', label='Crash Site')

        lim = np.max([np.abs(x), np.abs(y)])
        border = 1
        lim *= border
        order = int(np.log10(lim))
        centre = (np.mean(x), np.mean(y))
        plt.ylabel('y / 10x'+str(order)+'m')
        plt.xlabel('x / 10x'+str(order)+'m')
        plt.xlim(centre[0]-lim, centre[0]+lim)
        plt.ylim(centre[1]-lim, centre[1]+lim)
        # if plt.ylim()[0] > -Rearth:
        #     plt.ylim(border * np.min(np.append(y, -Rearth)), border * np.max(y))
        # if -plt.xlim()[0]+plt.xlim()[0] < 0.25*(-plt.ylim()[0]+plt.ylim()[1]):
        #     plt.xlim(plt.xlim()[0]*3, plt.xlim()[1]*3)
        plt.legend()
        plt.show()

    if plotE:
        # Calculating Total Potential
        totalPotential = 0.5 * (vx ** 2 + vy ** 2) + (G * Mearth) / ((x ** 2 + y ** 2) ** 0.5)
        if modelMoon:
            totalPotential = 0.5 * (vx ** 2 + vy ** 2) + \
                             (G * Mearth) / ((x ** 2 + y ** 2) ** 0.5) + \
                             (G * Mmoon) / (((x-moonX) ** 2 + (y-moonY) ** 2) ** 0.5)

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
                   ' --(e) Satellite orbit around Earth (Circular) (With Moon Modeled)\n'
                   ' --(f) Satellite orbit around Moon and Earth\n'
                   ' --(g) Satellite orbit around Moon and Earth (Crash on Moon)\n')

    if answer == 'a':
        EvalRK4([0, 8300], [6700000, 0])

    if answer == 'b':
        EvalRK4([0, 9500], [6700000, 0], simTime=18000)

    if answer == 'c':
        EvalRK4([-24000, 11000], [20000000, 0], simTime=3000)

    if answer == 'd':
        EvalRK4([-3000, 5900], [15000000, 0], simTime=50000)

    if answer == 'e':
        EvalRK4([0, 8300], [6700000, 0], modelMoon=True)

    if answer == 'f':
        EvalRK4([10796, 0], [0, -6700000], modelMoon=True, simTime=1559200, numPoints=50000)

    if answer == 'g':
        EvalRK4([10793, 0], [0, -6700000], modelMoon=True, simTime=959200, numPoints=50000)

    elif answer != 'q':
        print("That was not a valid answer")

    else:
        print("Thank you")
        break

# $$


