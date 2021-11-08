import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.integrate as igr
import matplotlib.animation as animation
from sklearn.linear_model import LinearRegression

matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
xmin = -40
ymin = xmin
xmax = -xmin
ymax = -xmin
r = (35,0,0) 
v = (0,15.6,0)
M = 10000
m = 1
sm = M + m
xMain, yMain, zMain = [],[],[]
G = 1
Oga = v[1]/r[0]
OgaOld = Oga
plummerScaleFactor = 11.6
fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()
xMain.append(r[0])
#xMain.append(0)
yMain.append(r[1])
#yMain.append(0)
zMain.append(r[2])
#zMain.append(0)
L1x = 0
L2x = 0

tfin=30
tstep=0.1
numSteps = int(tfin / tstep)
fl = open("outputFile2.txt", "r")
xOther, yOther = [],[]
for x in fl:
    q = x.split("|")
    xOther.append(float(q[0]))
    yOther.append(float(q[1]))
sc = ax.scatter(xMain,yMain, s=80, marker="o")
sco = ax.scatter(xOther, yOther, s=5, marker="o")
plt.xlim(xmin,xmax)
plt.ylim(ymin, ymax)
ax2.set_xlim([-100, 100])
ax2.set_ylim([-100, 100])


def calcEnergy(xp, yp, zp, vx, vy, vz, xm, ym, zm, ind):
    ogCur = ogVals[ind]
    dist = length(xp, yp, zp)
    if dist == 0:
        return 0
    vxNew = vx - (ogCur * yp)
    vyNew = vy + (ogCur * xp)
    vzNew = vz
    ke = 0.5 * ((vxNew ** 2) + (vyNew ** 2) + (vzNew ** 2))
    pot = plummerPot(xp,yp,zp)
    return ke + pot

def calcAngMom(xp, yp, zp, vx, vy, vz, xm, ym, zm, ind):
    ogCur = ogVals[ind]
    dist = length(xp, yp, zp)
    if dist == 0:
        return 0
    vxNew = vx - (ogCur * yp)
    vyNew = vy + (ogCur * xp)
    vzNew = vz
    xarray = np.array([xp, yp, zp])
    varray = np.array([vxNew, vyNew, vzNew])
    crs = np.cross(xarray, varray)
    p = 0
    for k in crs:
        p += k**2
    return np.sqrt(p)

def calcGradient(egs, angmoms):
    egsRes = np.array(egs).reshape((-1, 1))
    model = LinearRegression().fit(egsRes, angmoms)
    print(model.score(egsRes, angmoms))
    return model.coef_

def calcGradients(xpos, ypos, zpos, vx, vy, vz, ind):
    xposin = []
    yposin = []
    zposin = []
    vxin = []
    vyin = []
    vzin = []
    xposout = []
    yposout = []
    zposout = []
    vxout = []
    vyout = []
    vzout = []
    engsin = []
    angMomsin = []
    engsout = []
    angMomsout = []
    for q in range(numPartsPerTick * (ind + 1)):
        if q % numPartsPerTick == 0:
            xposin.append(xpos[q])
            yposin.append(ypos[q])
            zposin.append(zpos[q])
            vxin.append(vx[q])
            vyin.append(vy[q])
            vzin.append(vz[q])
        elif q % numPartsPerTick == 1:
            xposout.append(xpos[q])
            yposout.append(ypos[q])
            zposout.append(zpos[q])
            vxout.append(vx[q])
            vyout.append(vy[q])
            vzout.append(vz[q])
    #print("arraylengths " + str(len(xposin)) + "," + str(len(xposout)))
    angMomTot = 0
    engTot = 0
    for q in range(len(xposin)):
        engIn = calcEnergy(xposin[q], yposin[q], zposin[q], vxin[q], vyin[q], vzin[q], rVals[ind], 0, 0, ind)
        engOut = calcEnergy(xposout[q], yposout[q], zposout[q], vxout[q], vyout[q], vzout[q], rVals[ind], 0, 0, ind)
        angIn = calcAngMom(xposin[q], yposin[q], zposin[q], vxin[q], vyin[q], vzin[q], rVals[ind], 0, 0, ind)
        angOut = calcAngMom(xposout[q], yposout[q], zposout[q], vxout[q], vyout[q], vzout[q], rVals[ind], 0, 0, ind)
        engsin.append(engIn)
        angMomsin.append(angIn)
        engsout.append(engOut)
        angMomsout.append(angOut)
        engTot += engIn + engOut
        angMomTot += angIn + angOut
    engAv = engTot / (numPartsPerTick * (ind + 1))
    angAv = angMomTot / (numPartsPerTick * (ind + 1))
    for q in range(len(xposin)):
        engsin[q] -= engAv
        engsout[q] -= engAv
        angMomsin[q] -= angAv
        angMomsout[q] -= angAv 
    #ax2.scatter(engsin, angMomsin)
    #ax2.scatter(engsout, angMomsout)
    sc3.set_offsets(np.c_[np.concatenate([engsin, engsout]), np.concatenate([angMomsin, angMomsout])])

    gradin = calcGradient(engsin, angMomsin)
    gradout = calcGradient(engsout, angMomsout)
    return (gradin, gradout)

def distance(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1-z2)**2)

def massDistTerm(x1, y1, z1, x2, y2, z2, ma):
    return -G * ma / np.sqrt((distance(x1, y1, z1, x2, y2, z2)**2) + (0.2**2))

def length(x,y, z):
    return np.sqrt((x**2) + (y**2) + (z**2))

def plummerPot(x, y, z):
    return -G * M /np.sqrt((length(x,y,z)**2) + (plummerScaleFactor**2))

def gradPlummer(xp,yp,zp):
    dl = xmin / 5000
    return ((plummerPot(xp + dl,yp, zp) - plummerPot(xp, yp, zp))/dl, (plummerPot(xp,yp + dl, zp) - plummerPot(xp, yp, zp))/dl, (plummerPot(xp,yp, zp + dl) - plummerPot(xp, yp, zp))/dl)

def effPot(xp, yp, zp, index):
    return massDistTerm(xMain[0], yMain[0], zMain[0], xp, yp, zp, m)  + plummerPot(xp,yp,zp) - crossPotTerm(xp, yp, zp, index)

def crossPotTerm(xp, yp, zp, index):
    return 0.5 * ((xp**2) + (yp**2)) * (ogVals[index]**2)

def gradEffPot(xp, yp, zp, index):
    dl = xmin / 5000
    return ((effPot(xp + dl,yp, zp, index) - effPot(xp, yp, zp, index))/dl, (effPot(xp,yp + dl, zp, index) - effPot(xp, yp, zp, index))/dl, (effPot(xp,yp, zp + dl, index) - effPot(xp, yp, zp, index))/dl)

def dOgdt(index):
    return (ogVals[index] - ogVals[index-1]) / tstep

def eulerForce(xp, yp, zp, index):
    if index == 0: 
        return (0,0,0)
    dot = dOgdt(index)
    #print(str(dot))
    #return (0,0,0)
    return (dot * yp, -dot * xp, 0)

def corForce(vx, vy, vz, index):
    return (2 * ogVals[index] * vy, -2 * ogVals[index] * vx, 0)


def fIn(u,t): #u[0], u[1], u[2] are x,y,z | u[3],u[4],u[5] are vx vy vz
    xNew = u[3]
    yNew = u[4]
    zNew = u[5]
    gdPl = gradPlummer(u[0], u[1], u[2])
    vxNew = -gdPl[0]
    vyNew = -gdPl[1]
    vzNew = -gdPl[2]
    return (xNew, yNew, zNew, vxNew, vyNew, vzNew)

def f(u,t): #u[0], u[1], u[2] are x,y,z | u[3],u[4],u[5] are vx vy vz
    index = int(t/tstep)
    if index >= numSteps:
        index = numSteps - 1
    xMain[0] = rVals[index]

    #print(str(t))

    xNew = u[3]
    yNew = u[4]
    zNew = u[5]
    gdep = gradEffPot(u[0], u[1], u[2], index)
    eforce = eulerForce(u[0], u[1], u[2], index)
    cforce = corForce(u[3], u[4], u[5], index)
    vxNew = -gdep[0] + cforce[0] + eforce[0]
    vyNew = -gdep[1] + cforce[1] + eforce[1]
    vzNew = -gdep[2] + cforce[2] + eforce[2]
    return (xNew, yNew, zNew, vxNew, vyNew, vzNew)

def LocateLagrangePoints(rCur, index):
    global L1x, L2x
    xStep = 0.01
    xMain[0] = rCur
    L1x = rCur + xStep
    L2x = rCur - xStep
    haveFoundL1 = False
    haveFoundL2 = False
    while haveFoundL1 == False:
        if effPot(L1x + xStep, 0, 0, index) > effPot(L1x, 0, 0, index):
            L1x += xStep
        else:
            haveFoundL1 = True
    while haveFoundL2 == False:
        if effPot(L2x - xStep, 0, 0, index) > effPot(L2x, 0, 0, index):
            L2x -= xStep
        else:
            haveFoundL2 = True
#def f(u, t):
#    return (u[1], -gradEffPot(u[0], u[2], u[4])[0] + (2 * Oga * u[3]), u[3],  -gradEffPot(u[0], u[2], u[4])[1] - (2 * Oga * u[1]), u[5], )

tVals = np.arange(0,tfin, tstep)
uMIn = (r[0], r[1], r[2], v[0],v[1],v[2])
uMOut = igr.odeint(fIn, uMIn, tVals)
rVals = np.zeros(len(tVals))
vVals = np.zeros(len(tVals))
vParVals = np.zeros(len(tVals))
ogVals = np.zeros(len(tVals))
for i in range(len(tVals)):
    rVals[i] = length(uMOut[i, 0], uMOut[i, 1], uMOut[i, 2])
    vVals[i] = length(uMOut[i, 3], uMOut[i, 4], uMOut[i, 5])
    xCur = uMOut[i, 0]
    yCur = uMOut[i, 1]
    vxCur = uMOut[i, 3]
    vyCur = uMOut[i, 4]
    vParVals[i] = np.sqrt((((xCur * vxCur) + (yCur * vyCur)) ** 2)/ ((xCur ** 2) + (yCur ** 2)))
    vTan = np.sqrt((vVals[i]**2) - ((((xCur * vxCur) + (yCur * vyCur)) ** 2)/ ((xCur ** 2) + (yCur ** 2))))
    ogVals[i] = vTan/rVals[i]
delta = 0.1
xgrid = np.arange(xmin, xmax, delta)
ygrid = np.arange(ymin, ymax, delta)
contourvals = np.arange(-40, 0, 10 * delta)
X, Y = np.meshgrid(xgrid, ygrid)
Z = effPot(X,Y, 0, 0)

numPartsPerLP = 1
numPartsPerTick = numPartsPerLP * 2 #factor of 2 for both lagrange points
numParts = numPartsPerTick * numSteps
#L1x = 13
#L2x = 17

CS = ax.contour(X, Y, Z, contourvals, colors='k', alpha = 0.3)
XPOS = []
YPOS = []
ZPOS = []
for i in range(int(numSteps)):
    for j in range(numPartsPerLP):
        LocateLagrangePoints(rVals[i], i)
         
        XPOS.append(L1x)
        XPOS.append(L2x)
        YPOS.append(0)
        YPOS.append(0)
        ZPOS.append(0)
        ZPOS.append(0)

tVals = np.arange(0, tfin, tstep)
vrange = 0.6
XVEL = np.random.uniform(low=-vrange, high=vrange, size=(numParts,))
YVEL = np.random.uniform(low=-vrange, high=vrange, size=(numParts,))
ZVEL = np.random.uniform(low=-vrange, high=vrange, size=(numParts,))
xposArray = np.zeros((numParts, numSteps))
yposArray = np.zeros((numParts, numSteps))
zposArray = np.zeros((numParts, numSteps))
vxArray = np.zeros((numParts, numSteps))
vyArray = np.zeros((numParts, numSteps))
vzArray = np.zeros((numParts, numSteps))

for i in range(numParts):
    index = int(i/numPartsPerTick)
    initTime = index * tstep
    count = int((tfin - initTime)/tstep)
    if count + index > numSteps:
        count -=1
    if count + index < numSteps:
        count += 1
    
    custTVals = np.linspace(initTime,tfin, count)
    zeros = np.zeros(index)
    uin = (XPOS[i], YPOS[i], ZPOS[i], XVEL[i] + vParVals[index], YVEL[i], ZVEL[i])
    uout = igr.odeint(f, uin, custTVals)
    xposArray[i] = np.concatenate([zeros,uout[:, 0]])
    yposArray[i] = np.concatenate([zeros,uout[:, 1]])
    zposArray[i] = np.concatenate([zeros,uout[:, 2]])
    vxArray[i] = np.concatenate([zeros,uout[:, 3]])
    vyArray[i] = np.concatenate([zeros,uout[:, 4]])
    vzArray[i] = np.concatenate([zeros,uout[:, 5]])
    print(str(i))
sc2 = ax.scatter(XPOS, YPOS, s=5)
sc3 = ax2.scatter(XPOS, YPOS, s=5)

def animate2(i):
    return 0

def animate(i):
    #sc2.set_offsets(np.c_[np.concatenate(XPOS),np.concatenate(YPOS)])
    if i < numSteps:
        
        grads = calcGradients(xposArray[:, i], yposArray[:, i], zposArray[:, i], vxArray[:, i], vyArray[:, i], vzArray[:, i], i)
        print("gradients : " + str(grads[0]) + "," + str(grads[1]))
    if i >= numSteps:       
        return 0
    #sc3.set_offsets(np.c_[xposArray[:,i],xposArray[:,i]])
    sc2.set_offsets(np.c_[xposArray[:,i],yposArray[:,i]])
    sc.set_offsets(np.c_[rVals[i],yMain])
    

ani = animation.FuncAnimation(fig, animate, 
                frames=2000, interval=200, repeat=True) 
ani2 = animation.FuncAnimation(fig2, animate2, 
                frames=2000, interval=200, repeat=True) 

plt.show()