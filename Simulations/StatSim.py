from operator import xor
from warnings import catch_warnings
import math
import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import scipy.stats as ss
import scipy.integrate as igr
#import matplotlib.animation as animation
#from matplotlib import ticker, cm
from scipy.optimize import curve_fit
import emcee
#import os
import h5py
from multiprocessing import Pool
#import time
import threading

#os.environ["OMP_NUM_THREADS"] = "1"

G = 1
mainMass = 10000
mainScaleRadius = 11.6
satMass = 1
satScaleRadius = 0.2
tFinal = 29.5
tInit = 0
tStep = 0.0625


tVals = np.arange(tInit, tFinal, tStep)
numSteps = int((tFinal - tInit) / tStep)
numParticlesPerStepPerLP = 10
numParticles = int(numSteps * 2 * numParticlesPerStepPerLP)
particleData = np.zeros((6, numParticles, numSteps))
progenitorInit = np.array((15, 0, 0, 0, 8.16, 0)) #initial values of the problem

def GetDispersion(r, rp):
    return np.sqrt(G * satMass / (6 * np.sqrt((satScaleRadius**2) + (DistanceAlt(r, rp)**2))))

def DistanceC(x1, y1, z1, x2, y2, z2):
    return np.sqrt(((x1 - x2)**2) + ((y1 - y2)**2) + ((z1-z2)**2))

def Distance(r1, r2): #calc distance between two points
    if r1.ndim == 1:
        return np.sqrt(((r1[0] - r2[0]) ** 2) + ((r1[1] - r2[1]) ** 2) + ((r1[2] - r2[2]) ** 2))
    else:
        return np.sqrt(((r1[:,0] - r2[0]) ** 2) + ((r1[:,1] - r2[1]) ** 2) + ((r1[:,2] - r2[2]) ** 2))

def DistanceAlt(r1, r2): #calc distance between two points
    if r1.ndim == 1:
        return np.sqrt(((r1[0] - r2[0]) ** 2) + ((r1[1] - r2[1]) ** 2) + ((r1[2] - r2[2]) ** 2))
    else:
        return np.sqrt(((r1[0,:] - r2[0]) ** 2) + ((r1[1,:] - r2[1]) ** 2) + ((r1[2,:] - r2[2]) ** 2))

def PlummerPot(r, rp, M, a): #gets plummer potential (centred at rp with mass M and scale length a) at point r
    return -G * M /np.sqrt((Distance(r, rp)**2) + (a**2))

def PlummerPotAlt(r, rp, M, a): #gets plummer potential (centred at rp with mass M and scale length a) at point r
    return -G * M /np.sqrt((DistanceAlt(rp, r)**2) + (a**2))

def CombinedPot(r, rp1, M1, a1, rp2 , M2, a2): #gets sum of two plummers at r
    return PlummerPot(r, rp1, M1, a1) + PlummerPot(r, rp2, M2, a2)

def EffPot(r, rp1, M1, a1, rp2, M2, a2):
    return CombinedPot(r, rp1, M1, a1, rp2, M2, a2) + CrossTerm(r)

def CrossTerm(r):
    return np.cross(r, vecOgVals)

def CalcInitData(rp2, rp1, M1, a1, M2, vMean, numStepsT):
    rVals = DistanceAlt(rp2, np.array([0,0,0]))
    trad = GetTidalRadius(rp1, M1, a1, rVals, M2)  
    L1 = rp2 * ((rVals - trad) / rVals)
    L2 = rp2 * ((rVals + trad) / rVals)
    dispData = GetDispersion(L1, rp2)
    dispData = np.tile(dispData, (numParticlesPerStepPerLP, 1)).flatten('F')
    L1x = np.tile(L1[0], (numParticlesPerStepPerLP, 1)).flatten('F')
    L1y = np.tile(L1[1], (numParticlesPerStepPerLP, 1)).flatten('F')
    L1z = np.tile(L1[2], (numParticlesPerStepPerLP, 1)).flatten('F')
    L2x = np.tile(L2[0], (numParticlesPerStepPerLP, 1)).flatten('F')
    L2y = np.tile(L2[1], (numParticlesPerStepPerLP, 1)).flatten('F')
    L2z = np.tile(L2[2], (numParticlesPerStepPerLP, 1)).flatten('F')
    vxMean = np.tile(vMean[0,:], (numParticlesPerStepPerLP, 1)).flatten('F')
    vyMean = np.tile(vMean[1,:], (numParticlesPerStepPerLP, 1)).flatten('F')
    vzMean = np.tile(vMean[2,:], (numParticlesPerStepPerLP, 1)).flatten('F')
    L1vels = np.random.normal(np.zeros((3, numStepsT * numParticlesPerStepPerLP)), dispData)
    L2vels = np.random.normal(np.zeros((3, numStepsT * numParticlesPerStepPerLP)), dispData)
    return np.array([L1x, L1y, L1z, L1vels[0] + vxMean, L1vels[1] + vyMean, L1vels[2] + vzMean, L2x, L2y, L2z, L2vels[0] + vxMean, L2vels[1] + vyMean, L2vels[2] + vzMean])

def GetTidalRadius(rp1, M1, a1, rp2, M2):
    return np.power(M2/(3 * M1), 0.333) * DistanceC(0,0,0, rp2,0,0)
    tempNum = G * M2 / ((ogVals**2) - d2Pot(rp2, rp1, M1, a1))
    try:
        mpf = np.power(tempNum, 0.33)
        return mpf
    except:
        print("heh")
        
    

def dPot(r, rp1, M1, a1):
    dl = 0.0001
    dr = r * dl
    return (PlummerPot(r + dr, rp1, M1, a1) - PlummerPot(r, rp1, M1, a1)) / dl

def d2Pot(r, rp1, M1, a1):
    dl = 0.0001
    dr = r * dl
    return (dPot(r + dr, rp1, M1, a1) -dPot(r, rp1, M1, a1)) / dl


def PlummerPotC(x, y, z , rpx, rpy, rpz, M, a):
    return -G * M /np.sqrt((DistanceC(x, y, z, rpx, rpy, rpz)**2) + (a**2))

def GradPlummerC(x, y, z, rpx, rpy, rpz, M, a):
    dl = 0.01
    return ((PlummerPotC(x + dl,y,z, rpx,rpy,rpz, M, a),PlummerPotC(x, y+dl,z, rpx,rpy,rpz, M, a),PlummerPotC(x,y,z+dl, rpx,rpy,rpz, M, a))  - PlummerPotC(x,y,z, rpx,rpy,rpz, M, a))/dl



def GradPlummerAlt(r, rp, M, a):
    dl = 0.0001
    return ((PlummerPotAlt(r + np.array([dl,0,0]), rp, M, a),PlummerPotAlt(r + np.array([0,dl,0]), rp, M, a),PlummerPotAlt(r + np.array([0,0,dl]), rp, M, a))  - PlummerPotAlt(r, rp, M, a))/dl

def GradPlummer(r, rp, M, a):
    dl = 0.0001
    return ((PlummerPot(r + np.array([dl,0,0]), rp, M, a),PlummerPot(r + np.array([0,dl,0]), rp, M, a),PlummerPot(r + np.array([0,0,dl]), rp, M, a))  - PlummerPot(r, rp, M, a))/dl

def ProgenitorPropogator(u,t): #u[0], u[1], u[2] are x,y,z | u[3],u[4],u[5] are vx vy vz
    xNew = u[3]
    yNew = u[4]
    zNew = u[5]
    gdPl = GradPlummer(np.array((u[0], u[1], u[2])), np.array((0,0,0)), mainMass, mainScaleRadius)
    vxNew = -gdPl[0]
    vyNew = -gdPl[1]
    vzNew = -gdPl[2]
    return (xNew, yNew, zNew, vxNew, vyNew, vzNew)

def GradEffPot(xp, yp, zp, index):
    return 0
    #dl = xmin / 5000
    #return ((EffPot(xp + dl,yp, zp, index) - EffPot(xp, yp, zp, index))/dl, (EffPot(xp,yp + dl, zp, index) - EffPot(xp, yp, zp, index))/dl, (EffPot(xp,yp, zp + dl, index) - EffPot(xp, yp, zp, index))/dl)

def ZeroFilter(dat):
    return np.abs(np.sign(dat))

def dOgdt(index):
    return (ogVals[index] - ogVals[index-1]) / tStep

def eulerForce(xp, yp, zp, index):
    if index == 0: 
        return (0,0,0)
    dot = dOgdt(index)
    #return (0,0,0)
    return (dot * yp, -dot * xp, 0)

def corForce(vx, vy, vz, index):
    return (2 * ogVals[index] * vy, -2 * ogVals[index] * vx, 0)

def AngularMomentum(r, v):
    return np.cross(r.transpose(), v.transpose())

def ModAngularMomentum(r,v):
    return Distance(AngularMomentum(r,v), np.array([0,0,0]))

def InstantaneousVectorOmega(r,v):
    angmom = AngularMomentum(r,v)
    rsq = DistanceAlt(r, np.array([0,0,0])) **2
    return  (angmom.transpose()/rsq).transpose() 

def InstantaneousOmega(r, v):
    angmom = ModAngularMomentum(r,v)
    rsq = DistanceAlt(r, np.array([0,0,0])) **2
    return  angmom/rsq 

def InstantaneousDeltaOmega():
    dogVals = np.zeros(len(ogVals))
    for q in range(len(ogVals)):
        if q != 0:
            dogVals[q] = (ogVals[q] - ogVals[q-1]) / tStep
    return dogVals


def ParticlePropogator(u,t,mainMassP,mainScaleRadiusP,numStepsP, progenitorDataP): #u is all the xvals, then yvals, then z, then vx, then vy, then vz
    index = int(t/tStep)
    #zeros = ZeroFilter(u)
    partCount = index * numParticlesPerStepPerLP
    totCount = numParticlesPerStepPerLP * numStepsP
    zeros = np.zeros(totCount - (partCount + numParticlesPerStepPerLP))
    x1New = np.concatenate((u[3*totCount:(3*totCount) + partCount + numParticlesPerStepPerLP], zeros))
    y1New = np.concatenate((u[4*totCount:(4*totCount) + partCount + numParticlesPerStepPerLP], zeros))
    z1New = np.concatenate((u[5*totCount:(5*totCount) + partCount + numParticlesPerStepPerLP], zeros))
    x2New = np.concatenate((u[9*totCount:(9*totCount) + partCount + numParticlesPerStepPerLP], zeros))
    y2New = np.concatenate((u[10*totCount:(10*totCount) + partCount + numParticlesPerStepPerLP], zeros))
    z2New = np.concatenate((u[11*totCount:(11*totCount) + partCount + numParticlesPerStepPerLP], zeros))
    gdep1Sat = GradPlummerC(u[0:partCount + numParticlesPerStepPerLP], u[1*totCount:(1*totCount) + partCount + numParticlesPerStepPerLP], u[2*totCount:(2*totCount) + partCount + numParticlesPerStepPerLP], progenitorDataP[index,0], progenitorDataP[index,1], progenitorDataP[index,2], satMass, satScaleRadius)
    gdep1Main = GradPlummerC(u[0:partCount + numParticlesPerStepPerLP], u[1*totCount:(1*totCount) + partCount + numParticlesPerStepPerLP], u[2*totCount:(2*totCount) + partCount + numParticlesPerStepPerLP], 0,0,0, mainMassP, mainScaleRadiusP)
    gdep2Sat = GradPlummerC(u[6*totCount:(6*totCount) + partCount + numParticlesPerStepPerLP], u[7*totCount:(7*totCount) + partCount + numParticlesPerStepPerLP], u[8*totCount:(8*totCount) + partCount + numParticlesPerStepPerLP], progenitorDataP[index,0], progenitorDataP[index,1], progenitorDataP[index,2], satMass, satScaleRadius)
    gdep2Main = GradPlummerC(u[6*totCount:(6*totCount) + partCount + numParticlesPerStepPerLP], u[7*totCount:(7*totCount) + partCount + numParticlesPerStepPerLP], u[8*totCount:(8*totCount) + partCount + numParticlesPerStepPerLP], 0,0,0, mainMassP, mainScaleRadiusP)
    totalgdep1 = -(gdep1Sat + gdep1Main)
    totalgdep2 = -(gdep1Sat + gdep1Main)
    if index > 0:
        vx1New = np.concatenate((-gdep1Sat[0,:] -gdep1Main[0,:], zeros))
        vy1New = np.concatenate((-gdep1Sat[1,:] -gdep1Main[1,:], zeros))
        vz1New = np.concatenate((-gdep1Sat[2,:] -gdep1Main[2,:], zeros))
        vx2New = np.concatenate((-gdep2Sat[0,:] -gdep2Main[0,:], zeros))
        vy2New = np.concatenate((-gdep2Sat[1,:] -gdep2Main[1,:], zeros))
        vz2New = np.concatenate((-gdep2Sat[2,:] -gdep2Main[2,:], zeros))
        return np.concatenate((x1New, y1New, z1New, vx1New, vy1New, vz1New, x2New, y2New, z2New, vx2New, vy2New, vz2New))
    else:
        vx1New = np.concatenate((np.array([-gdep1Sat[0] -gdep1Main[0]]).flatten(), zeros))
        vy1New = np.concatenate((np.array([-gdep1Sat[1] -gdep1Main[1]]).flatten(), zeros))
        vz1New = np.concatenate((np.array([-gdep1Sat[2] -gdep1Main[2]]).flatten(), zeros))
        vx2New = np.concatenate((np.array([-gdep2Sat[0] -gdep2Main[0]]).flatten(), zeros))
        vy2New = np.concatenate((np.array([-gdep2Sat[1] -gdep2Main[1]]).flatten(), zeros))
        vz2New = np.concatenate((np.array([-gdep2Sat[2] -gdep2Main[2]]).flatten(), zeros))
        return np.concatenate((x1New, y1New, z1New, vx1New, vy1New, vz1New, x2New, y2New, z2New, vx2New, vy2New, vz2New))

def RK4Iteration(func, u, t,mainMassP,mainScaleRadiusP,numStepsP, progenitorDataP):
    k1 = func(u, t,mainMassP,mainScaleRadiusP,numStepsP, progenitorDataP)
    k2 = func(u + (tStep * (k1 / 2)), t + (tStep/2),mainMassP,mainScaleRadiusP,numStepsP, progenitorDataP)
    k3 = func(u + (tStep * (k2 / 2)), t + (tStep/2),mainMassP,mainScaleRadiusP,numStepsP, progenitorDataP)
    k4 = func(u + (tStep * k3), t + (tStep/2),mainMassP,mainScaleRadiusP,numStepsP, progenitorDataP)
    qim = u + ((tStep/6) * (k1 + (2*k2) + (2*k3) + k4))
    return qim

def RK4(func, u, t,mainMassP,mainScaleRadiusP,numStepsP, progenitorDataP):
    output = np.zeros((len(u), len(t)))
    for q in range(len(t)):
        if q == 0:
            output[:,0] = u
        else:
            output[:,q] = RK4Iteration(func, output[:,q-1], t[q],mainMassP,mainScaleRadiusP,numStepsP, progenitorDataP)
    return output

def CalcColors(tm1):
    tout = np.zeros(len(tm1))
     
    #for q in range(len(tout)):
    #    if tm1[q] == 0:
    #        tout[q] = 0.5
    #    elif tm1[q] < 0:
    #        tout[q] = 0.5 + (-0.5 * (tm1[q]/min(tm1)))
    #    else:
    #        tout[q] = 0.5 + (0.5 * (tm1[q]/max(tm1)))
    if max(tm1) == min(tm1):
        return tout
    mmax = max(max(tm1), -min(tm1))
    return (1 + (tm1 / mmax)) / 2
    #return (tm1 - min(tm1)) / (max(tm1) - min(tm1))

def CalcGaussianMidPoint(pd, mpd, rv, sp, nb):
    dictCounter = np.zeros(nb+1)
    numAddedCounter = np.zeros(nb+1)
    indT = mpd / sp
    inds = indT - indT.min()
    for i in range(len(inds)):
        dictCounter[int(inds[i])] += rv[i]
        numAddedCounter[int(inds[i])] += 1
    maxindices = []
    meanArray = np.zeros(nb + 1)
    for q in range(len(meanArray)):
        if numAddedCounter[q] > 0:
            meanArray[q] = dictCounter[q]/numAddedCounter[q]
    spacing = 10
    for q in range(len(meanArray)):
        if q >= spacing and q <= len(meanArray) - (spacing + 1):
            maxim = True
            for p in range(-spacing, spacing+1):
                if meanArray[p+q] > meanArray[q]:
                    maxim = False
            if maxim == True and meanArray[q] > 0 and np.abs((q + indT.min()) * sp) < 4.5:
                maxindices.append(q)
    maxPhases = (maxindices + indT.min()) * sp
    return maxPhases

def CalcGaussianSubMidPoint(pd, mpd, rv, sp, nb):
    dictCounter = np.zeros(nb+1)
    numAddedCounter = np.zeros(nb+1)
    indT = mpd / sp
    inds = indT - indT.min()
    for i in range(len(inds)):
        dictCounter[int(inds[i])] += rv[i]
        numAddedCounter[int(inds[i])] += 1
    minindices = []
    meanArray = np.zeros(nb + 1)
    for q in range(len(meanArray)):
        if numAddedCounter[q] > 0:
            meanArray[q] = dictCounter[q]/numAddedCounter[q]
    spacing = 10
    for q in range(len(meanArray)):
        if q >= spacing and q <= len(meanArray) - (spacing + 1):
            minim = True
            for p in range(-spacing, spacing+1):
                if meanArray[p+q] < meanArray[q]:
                    minim = False
            if minim == True and meanArray[q] > 0 and np.abs((q + indT.min()) * sp) < 4.5:
                minindices.append(q)
    minPhases = (minindices + indT.min()) * sp
    return minPhases

def GaussianToFit(pd, mn, sg, pref):
    #val = ((pref/sg) * np.exp(-0.5 * (((pd-mn)/sg) **2)))
    #val = sg * np.power(pd - mn, 2) + pref
    val = (sg * np.abs(pd - mn)) + pref
    return val

def GaussianSubToFit(pd, mn, sg, pref):
    #val = ((pref/sg) * np.exp(-0.5 * (((pd-mn)/sg) **2))) + 15
    val = sg * np.power(pd - mn, 2) + pref
    return val

def CalcGaussianParams(phases, pd, rv, rg):
    gaussParams = []
    angsUsed = []
    for pf in range(len(phases)):
        p = phases[pf]
        pdc = []
        rvc = []
        for q in range(len(pd)):
            if np.abs(pd[q] - p) <= rg:
                pdc.append(pd[q])
                rvc.append(rv[q])
        try:
            tdat, covd = curve_fit(GaussianToFit, pdc, rvc, p0 = np.array((p, -10, 12)))
            add = True
            if np.sign(tdat[1]) == 1:
                continue
            for q in range(len(angsUsed)):
                if np.abs(tdat[0] - angsUsed[q]) < 0.5:
                    add = False
                    break
            
            if add == True:
                angsUsed.append(tdat[0])
                gaussParams.append(tdat)
        except:
            gaussParams.append(np.array((1000, 1000, 1000)))
    return gaussParams

def CalcGaussianSubParams(phases, pd, rv, rg):
    gaussParams = []
    angsUsed = []
    for pf in range(len(phases)):
        p = phases[pf]
        pdc = []
        rvc = []
        for q in range(len(pd)):
            if np.abs(pd[q] - p) <= rg:
                pdc.append(pd[q])
                rvc.append(rv[q])
        try:
            tdat, covd = curve_fit(GaussianSubToFit, pdc, rvc, p0 = np.array((p, 1, 15)))
            add = True
            if np.sign(tdat[1]) == -1:
                continue
            for q in range(len(angsUsed)):
                if np.abs(tdat[0] - angsUsed[q]) < 0.5:
                    add = False
                    break
            
            if add == True:
                angsUsed.append(tdat[0])
                gaussParams.append(tdat)
        except:
            gaussParams.append(np.array((1000, 1000, 1000)))
    return gaussParams


def CalcAngles(x,y,z,xp,yp,zp):
    cosines = ((x*xp) + (y*yp) + (z*zp)) / (DistanceC(0,0,0,x,y,z) * DistanceC(0,0,0,xp,yp,zp))
    return np.arccos(cosines) * np.sign(phaseData)

def CalcPhaseValues(x, y, z, xp, yp, zp, numStepsT):
    numParticlesT = 2 * numStepsT * numParticlesPerStepPerLP
    output = np.zeros((numParticlesT, numStepsT))
    colorData = np.zeros((numParticlesT, numStepsT))
    for q in range(numStepsT):
        if q > 0:
            zf = ZeroFilter(x[:,q] - x[:,q-1])
            wm = zf.sum()
            output[:, q] = output[:, q-1] + ((np.sqrt((x[:,q]** 2) + (y[:,q] ** 2) + (z[:,q]**2)) - np.sqrt((xp[q]**2) + (yp[q]**2) + (zp[q]**2))) * zf)
        colorData[:,q] = CalcColors(output[:,q])
        #for m in range(numParticles):
        #    phase = 0
        #    for p in range(q):
        #        phase += np.sqrt((x[m,p]**2) + (y[m,p]**2) + (z[m,p]**2)) - np.sqrt((xp[p]**2) + (yp[p]**2) + (zp[p]**2))
        #    output[m,q] = phase
    return output, colorData

def PositionVelocityToAngle(x,y,z,vx,vy,vz,M1,a1):

    return 3

def CalcAngle(x, y, z):
    cosa = x / np.sqrt((x**2) + (y**2) + (z**2))
    q = np.arccos(cosa)
    s = np.sign(y)
    a = np.abs(s)
    sa = a * ((1 + s) / 2)
  
    q = (q * sa) + (((2 * math.pi) - q) * (1-sa))
    return q

def CalcAngleBT(x,y,z,phase,progAngle):
    angles = CalcAngle(x,y,z)
    adjangles = angles - progAngle
    angledatarray = np.array((phase, adjangles, np.arange(0, len(adjangles), 1)))
    sortedArray = angledatarray[ :, angledatarray[0].argsort()]
    boundaryindex = 0
    for q in range(len(sortedArray[0, :])):
        #msgfn = sortedArray[1,q]
        if sortedArray[0, q] > 0:
            boundaryindex = q
            break
    lowav = 0
    for q in range(boundaryindex):
        ind = boundaryindex - (q + 1)
        ange = sortedArray[1,ind]
        fidangle = False
        while fidangle == False:
            if np.abs(lowav - ange) > np.abs(lowav - (ange + (2*math.pi))):
                ange += 2*math.pi
            elif np.abs(lowav - ange) > np.abs(lowav - (ange - (2*math.pi))):
                ange -= 2*math.pi
            else:
                fidangle = True
        sortedArray[1,ind] = ange
        numsum = np.min((q+1, 10))
        lowav = 0
        for j in range(numsum):
            indsum = ind + j
            lowav += sortedArray[1,indsum]
        if numsum > 0:
            lowav /= numsum
        else:
            lowav = 0
    highav = 0
    for q in range(len(adjangles) - (boundaryindex+1)):
        ind = boundaryindex + q
        ange = sortedArray[1,ind]
        fidangle = False
        while fidangle == False:
            if np.abs(highav - ange) > np.abs(highav - (ange + (2*math.pi))):
                ange += 2*math.pi
            elif np.abs(highav - ange) > np.abs(highav - (ange - (2*math.pi))):
                ange -= 2*math.pi
            else:
                fidangle = True
        sortedArray[1,ind] = ange
        numsum = np.min((q+1, 10))
        highav = 0
        for j in range(numsum):
            indsum = ind - j
            highav += sortedArray[1,indsum]
        if numsum > 0:
            highav /= numsum
        else:
            highav = 0
    finarray = sortedArray[ :, sortedArray[2].argsort()]
    return finarray[1,:]



def CalcAngleMM(x,y,z,phase,progAngle):
    return CalcAngleBT(x,y,z,phase,progAngle)
    """
    angles = CalcAngle(x,y,z)
    adjangles = angles - progAngle
    #newadjangles = ((adjangles + math.pi) % (2 * math.pi)) - math.pi
    for m in range(10):
        linphase, linangle = [],[]
        for q in range(len(phase)):
            if np.abs(phase[q]) < 300 * (m+1):
                linphase.append(phase[q])
                linangle.append(adjangles[q])
        df, hmf = curve_fit(strln, linphase, linangle, p0=np.array((-0.01,0)))
        tang = strln(phase, df[0], df[1])
        for q in range(len(phase)):
            tangq = tang[q]
            ag = adjangles[q]
            fidangle = False
            while fidangle == False:
                if np.abs(tangq - ag) > np.abs(tangq - (ag + (2*math.pi))):
                    ag += 2*math.pi
                elif np.abs(tangq - ag) > np.abs(tangq - (ag - (2*math.pi))):
                    ag -= 2*math.pi
                else:
                    fidangle = True
            adjangles[q] = ag
    
    return adjangles, df, angles
    """
def strln(x, a, b):
    return (a * x) + b

def CalcAngleFull(x,y,z,phase, progAngle):
    angles = CalcAngle(x,y,z)
    #angles -= progAngle
    boundaryPhases = []
    for q in range(len(phase)):
        if angles[q] < 0.03 and CheckBoundaryPhases(phase[q], boundaryPhases) == True:
            boundaryPhases.append(phase[q])
    for q in range(len(phase)):
        for j in range(len(boundaryPhases)):
            if boundaryPhases[j] < phase[q] and angles[q] < 0.03 and np.abs(phase[q] - boundaryPhases[j]) < 150:
                boundaryPhases[j] = phase[q]
                break
    numOffsets = np.zeros(len(boundaryPhases))
    for m in range(len(boundaryPhases)):
        num = 0
        for l in range(len(boundaryPhases)):
            if np.abs(boundaryPhases[l]) <= np.abs(boundaryPhases[m]) and np.sign(boundaryPhases[l]) == np.sign(boundaryPhases[m]):
                num += -np.sign(boundaryPhases[m])
        numOffsets[m] = num
    for q in range(len(phase)):
        offset = 0
        for j in range(len(boundaryPhases)):
            if np.abs(phase[q] - boundaryPhases[j]) < np.abs(phase[q]) and np.sign(boundaryPhases[j]) == np.sign(phase[q]) and np.abs(numOffsets[j]) > np.abs(offset) :
                #if np.abs(boundaryPhases[j] - phase[q]) < 100 and angles[q] > 2:
                #    continue
                if np.abs(boundaryPhases[j] - phase[q]) < 200 and angles[q] > 4.5 and np.sign(boundaryPhases[j]) == -1:
                    continue
                elif  np.abs(boundaryPhases[j] - phase[q]) < 200 and angles[q] < 2 and np.sign(boundaryPhases[j]) == 1:
                    continue
                offset = numOffsets[j]
            #elif np.abs(boundaryPhases[j] - phase[q]) < 100 and angles[q] > 4 and np.sign(boundaryPhases[j]) == np.sign(phase[q]) and np.abs(numOffsets[j]) > np.abs(offset) :
            #    offset = numOffsets[j]
            #elif np.abs(boundaryPhases[j] - phase[q]) < 100 and angles[q] < 1 and np.sign(boundaryPhases[j]) == np.sign(phase[q]) and np.abs(numOffsets[j]) > np.abs(offset) :
            #    offset = numOffsets[j]
        angles[q] += offset * math.pi * 2

    return angles
    
def CheckBoundaryPhases(p, phases):
    for q in range(len(phases)):
        if np.abs(p-phases[q]) < 150:
            return False
    return True

def CalcEnergyValues(x,y,z,xp1,yp1,zp1,M1,a1, vx,vy,vz):
    egs = PlummerPotC(x,y,z,xp1,yp1,zp1,M1,a1) + (0.5 * ((vx**2) + (vy**2) + (vz**2)))
    return egs - np.mean(egs)

def CalcOtherEnergyValues(x,y,z,xp1,yp1,zp1,M1,a1,vx,vy,vz, me):
    egs = PlummerPotC(x,y,z,xp1,yp1,zp1,M1,a1) + (0.5 * ((vx**2) + (vy**2) + (vz**2)))
    return egs - me

def CalcEnergyMean(x,y,z,xp1,yp1,zp1,M1,a1,vx,vy,vz):
    egs = PlummerPotC(x,y,z,xp1,yp1,zp1,M1,a1) + (0.5 * ((vx**2) + (vy**2) + (vz**2)))
    return np.mean(egs)

def CalcAngularMomentumValues(x,y,z,vx,vy,vz):
    ags = DistanceC(0,0,0,(y*vz) - (z*vy), (z*vx) - (x*vz), (x*vy) - (y*vx))
    return ags - np.mean(ags)

def CalcOtherAngularMomentumValues(x,y,z,vx,vy,vz, ma):
    ags = DistanceC(0,0,0,(y*vz) - (z*vy), (z*vx) - (x*vz), (x*vy) - (y*vx))
    return ags - ma

def CalcAngularMomentumMean(x,y,z,vx,vy,vz):
    ags = DistanceC(0,0,0,(y*vz) - (z*vy), (z*vx) - (x*vz), (x*vy) - (y*vx))
    return np.mean(ags)

def CalcELMidPoints(egs, ams, phs):
    ct = len(egs)
    rg = (max(phs) - min(phs)) / 50 
    lowegs, lowams, highegs, highams = [],[],[],[]
    for q in range(ct):
        if phs[q] < -rg:
            lowegs.append(egs[q])
            lowams.append(ams[q])
        elif phs[q] > rg:
            highegs.append(egs[q])
            highams.append(ams[q])
    return np.array((np.mean(lowegs), np.mean(lowams))),np.array((np.mean(highegs), np.mean(highams)))
    

def CalcQualityFactor(data):
    x = data[0]
    vy = data[1]
    Ms = data[2]
    a = data[3]
    
    if Ms < 0 or Ms > 25000:
        return -np.infty
    if a < 0 or a > 20:
        return -np.infty
    if x < 0 or x > 25:
        return -np.infty
    if vy < 0 or vy > 25:
        return -np.infty
    tValsT = np.arange(tInit, tFinal + (tStep * 20), tStep)
    numStepsT = int((tFinal + (tStep * 20) - tInit) / tStep)
    totCountT = numParticlesPerStepPerLP * numStepsT
    progenitorInitT = np.array((x, 0, 0, 0, vy, 0)) 
    progenitorDataT = igr.odeint(ProgenitorPropogator, progenitorInitT, tValsT)
    #ogVals = InstantaneousOmega(np.array([progenitorDataT[:,0], progenitorDataT[:,1], progenitorDataT[:,2]]), np.array([progenitorDataT[:,3], progenitorDataT[:,4], progenitorDataT[:,5]]))
    qmmmqq1 = np.array([progenitorDataT[:,0], progenitorDataT[:,1], progenitorDataT[:,2]])
    qmmmqq2 = np.array([progenitorDataT[:,3], progenitorDataT[:,4], progenitorDataT[:,5]])
    lagPointsT = CalcInitData(qmmmqq1, np.array([0,0,0]), Ms, a, satMass, qmmmqq2, numStepsT)
    initPositionsT = lagPointsT.flatten()
    particleDataT = RK4(ParticlePropogator, initPositionsT, tValsT, Ms, a, numStepsT, progenitorDataT)
    xDataT = np.concatenate((particleDataT[0:totCountT,:], particleDataT[6*totCountT:7*totCountT,:]))
    yDataT = np.concatenate((particleDataT[1*totCountT:2 * totCountT,:], particleDataT[7*totCountT:8*totCountT,:]))
    zDataT = np.concatenate((particleDataT[2*totCountT:3 * totCountT,:], particleDataT[8*totCountT:9*totCountT,:]))
    vxDataT = np.concatenate((particleDataT[3*totCountT:4 * totCountT,:], particleDataT[9*totCountT:10*totCountT,:]))
    vyDataT = np.concatenate((particleDataT[4*totCountT:5 * totCountT,:], particleDataT[10*totCountT:11*totCountT,:]))
    vzDataT = np.concatenate((particleDataT[5*totCountT:6 * totCountT,:], particleDataT[11*totCountT:12*totCountT,:]))
    #energyDataT = CalcEnergyValues(xDataT,yDataT,zDataT, 0,0,0,mainMass,mainScaleRadius, vxDataT, vyDataT, vzDataT)
    #angMomDataT = CalcAngularMomentumValues(xDataT,yDataT,zDataT,vxDataT,vyDataT,vzDataT)
    phaseDataT, colorDataT = CalcPhaseValues(xDataT, yDataT, zDataT, progenitorDataT[:,0], progenitorDataT[:,1], progenitorDataT[:,2], numStepsT)
    rDataT = DistanceC(0,0,0,xDataT,yDataT,zDataT)
    vDataT = DistanceC(0,0,0,vxDataT, vyDataT, vzDataT)
    progRDataT = DistanceC(0,0,0,progenitorDataT[:,0], progenitorDataT[:,1], progenitorDataT[:,2])
    haveFoundIdeal = False
    while haveFoundIdeal == False:
        rc = progRDataT[numStepsT - 1]
        sgn = np.sign((progenitorDataT[numStepsT - 1,0] * progenitorDataT[numStepsT - 1,3]) + (progenitorDataT[numStepsT - 1,1] * progenitorDataT[numStepsT - 1,4]) + (progenitorDataT[numStepsT - 1,2] * progenitorDataT[numStepsT - 1,5]))
        if (rc - progRDataO) / progRDataO <0.04 and progDataSignO == sgn:
            haveFoundIdeal = True
        else:
            numStepsT-=1
            if numStepsT <= 0:
                numStepsT = int((tFinal + (tStep * 20) - tInit) / tStep)
                break
    #angdatT = CalcAngle(xDataT, yDataT, zDataT)
    fullangdatT = CalcAngleMM(xDataT[:,numStepsT-1], yDataT[:,numStepsT-1], zDataT[:,numStepsT-1], phaseDataT[:,numStepsT-1], CalcAngle(progenitorDataT[numStepsT-1,0], progenitorDataT[numStepsT-1,1], progenitorDataT[numStepsT-1,2]))
    numangbinsT = 400
    fullangdatsepT = (fullangdatT.max() - fullangdatT.min())/numangbinsT
    angmeandatT = np.zeros(numangbinsT+1)
    angmeancountT = np.zeros(numangbinsT+1)
    igrsT = np.round((fullangdatT-fullangdatT.min())  / fullangdatsepT)
    angdataltT =  np.round(fullangdatT  / fullangdatsepT) * fullangdatsepT
    angmeanangleT = np.zeros(numangbinsT+1)
    for q in range(len(angdataltT)):
        index = int(igrsT[q] + 0.01)
        angmeandatT[index] += rDataT[q, numStepsT-1]
        angmeancountT[index] +=1
    for m in range(len(angmeandatT)):
        angmeanangleT[m] = fullangdatT.min() + (m * fullangdatsepT)
        if angmeancountT[m] == 0:
            continue
        angmeandatT[m] /= angmeancountT[m]

    accrT,accaT =[],[]
    for m in range(len(angmeandatT)):
        if np.abs(angmeanangleT[m]) < 4:
            accrT.append(angmeandatT[m])
            accaT.append(angmeanangleT[m])
    gmpsT = CalcGaussianMidPoint(fullangdatT, angdataltT, rDataT[:,numStepsT-1], fullangdatsepT, numangbins)
    gpamsT = CalcGaussianParams(gmpsT, fullangdatT, rDataT[:,numStepsT-1], gaussRangeO)
    gsubmpsT = CalcGaussianSubMidPoint(fullangdatT, angdataltT, rDataT[:,numStepsT-1], fullangdatsepT, numangbins)
    gsubpamsT = CalcGaussianSubParams(gsubmpsT, fullangdatT, rDataT[:,numStepsT-1], gaussRangeO)
    mma = 3.5
    qf = 1
    gam = []
    gah = []
    gbm = []
    gbh = []
    for q in range(len(gpamsT)):
        gpT = gpamsT[q]

        if np.abs(gpT[0]) > mma:
            continue
        gam.append(gpT[0])
        gah.append(gpT[2])
    for q in range(len(gsubpamsT)):
        gpT = gsubpamsT[q]
        if np.abs(gpT[0]) > mma:
            continue
        gbm.append(gpT[0])
        gbh.append(gpT[2])
    freqT, ampT = CalcFreqAmp(gam, gah, gbm, gbh)
    if freqT == np.infty or ampT == np.infty or freqO == np.infty or ampO == np.infty:
        return -np.infty
    veldaT = CalcMMVels(gam, gbm, fullangdatT, vDataT[:,numStepsT-1], fullangdatsepT)
    qf *= np.abs((freqT - freqO) / freqO) * np.abs((ampT - ampO) / ampO)
    qfex = np.product(np.abs((veldaT - veldaO) / veldaO))
    qf *= np.power(qfex, 0.5)
    sldfT, oddT = curve_fit(strln, fullangdatT, rDataT[:,numStepsT-1], p0=np.array((-0.1,6)))
    qf *= np.abs((sldfT[0] - sldf[0]) / sldf[0]) * np.abs((sldfT[1] - sldf[1]) / sldf[1])
    qfo = np.log(1/qf)
    #bruhT = np.fft.rfft(accrT)
    #reT = np.real(bruhT) 
    #qf = np.sum(np.abs((reT[0:10] -re[0:10]) / re[0:10]))
    #print("x="+str(x) + " vy=" + str(vy) + " M=" + str(Ms) + " a=" + str(a) + " qf=" + str(qfo))
    return qfo
     
def CalcMMVels(gam, gbm, angdat, velVals, angsep):
    gaml, gamh, gbml, gbmh = -1000, 1000, -1000, 1000
    val, vah, vbl, vbh = 0,0,0,0
    valc, vahc, vblc, vbhc = 0,0,0,0
    for q in range(len(gam)):
        if gam[q] > gaml and np.sign(gam[q]) == np.sign(gaml):
            gaml = gam[q]
        elif gam[q] < gamh and np.sign(gam[q]) == np.sign(gamh):
            gamh = gam[q]
    for q in range(len(gbm)):
        if gbm[q] > gbml and np.sign(gbm[q]) == np.sign(gbml):
            gbml = gbm[q]
        elif gbm[q] < gbmh and np.sign(gbm[q]) == np.sign(gbmh):
            gbmh = gbm[q]
    for a in range(len(angdat)):
        ang = angdat[a]
        if np.abs(ang - gaml) < angsep:
            val += velVals[a]
            valc += 1
        if np.abs(ang - gamh) < angsep:
            vah += velVals[a]
            vahc += 1
        if np.abs(ang - gbml) < angsep:
            vbl += velVals[a]
            vblc += 1
        if np.abs(ang - gbmh) < angsep:
            vbh += velVals[a]
            vbhc += 1
    if valc > 0:
        val /= valc
    if vahc > 0:
        vah /= vahc
    if vblc > 0:
        vbl /= vblc
    if vbhc > 0:
        vbh /= vbhc
    return np.array((val, vah, vbl, vbh))
    
     

def CalcFreqAmp(gam, gah, gbm, gbh):
    freq = 1000
    disp = 1000
    for q in range(len(gam)):
        cla = 1000
        indj = -1
        for p in range(len(gam)):
            if np.abs(gam[p] - gam[q]) < freq and p != q:
                freq = np.abs(gam[p] - gam[q])
        
        for j in range(len(gbm)):
            if np.abs(gbm[j] - gam[q]) < cla:
                cla = np.abs(gbm[j] - gam[q])
                indj = j
        if indj == -1:
            return np.infty, np.infty
        if disp > np.abs(gah[q] - gbh[int(indj)]):
            disp = np.abs(gah[q] - gbh[int(indj)])           

    for q in range(len(gbm)):
        for p in range(len(gbm)):
            if np.abs(gbm[p] - gbm[q]) < freq and p != q:
                freq = np.abs(gbm[p] - gbm[q])
    return freq, disp
            

currIndex = 0
lastT = 0
progenitorData = igr.odeint(ProgenitorPropogator, progenitorInit, tVals)
#rVals = DistanceAlt(np.array([progenitorData[:,0], progenitorData[:,1], progenitorData[:,2]]), np.array([0,0,0]))
#vecOgVals = InstantaneousVectorOmega(np.array([progenitorData[:,0], progenitorData[:,1], progenitorData[:,2]]), np.array([progenitorData[:,3], progenitorData[:,4], progenitorData[:,5]]))
ogVals = InstantaneousOmega(np.array([progenitorData[:,0], progenitorData[:,1], progenitorData[:,2]]), np.array([progenitorData[:,3], progenitorData[:,4], progenitorData[:,5]]))
#dOgDtVals = InstantaneousDeltaOmega()
lagPoints = CalcInitData(np.array([progenitorData[:,0], progenitorData[:,1], progenitorData[:,2]]), np.array([0,0,0]), mainMass, mainScaleRadius, satMass, np.array([progenitorData[:,3], progenitorData[:,4], progenitorData[:,5]]), numSteps)
initPositions = lagPoints.flatten()
particleData = RK4(ParticlePropogator, initPositions, tVals, mainMass, mainScaleRadius, numSteps, progenitorData)

xmin = -20
ymin = xmin
xmax = -xmin
ymax = -xmin
"""
fig, ax = plt.subplots(1,2)
#fig2, ax2 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()
fig1s, ax1s = plt.subplots(3,2)
"""
totCount = numParticlesPerStepPerLP * numSteps
xData = np.concatenate((particleData[0:totCount,:], particleData[6*totCount:7*totCount,:]))
yData = np.concatenate((particleData[1*totCount:2 * totCount,:], particleData[7*totCount:8*totCount,:]))
zData = np.concatenate((particleData[2*totCount:3 * totCount,:], particleData[8*totCount:9*totCount,:]))
vxData = np.concatenate((particleData[3*totCount:4 * totCount,:], particleData[9*totCount:10*totCount,:]))
vyData = np.concatenate((particleData[4*totCount:5 * totCount,:], particleData[10*totCount:11*totCount,:]))
vzData = np.concatenate((particleData[5*totCount:6 * totCount,:], particleData[11*totCount:12*totCount,:]))
energyData = CalcEnergyValues(xData,yData,zData, 0,0,0,mainMass,mainScaleRadius, vxData, vyData, vzData)
angMomData = CalcAngularMomentumValues(xData,yData,zData,vxData,vyData,vzData)
phaseData, colorData = CalcPhaseValues(xData, yData, zData, progenitorData[:,0], progenitorData[:,1], progenitorData[:,2], numSteps)
angleData = CalcAngles(xData, yData, zData, progenitorData[:,0], progenitorData[:,1], progenitorData[:,2])
rData = DistanceC(0,0,0,xData,yData,zData)
tstdatst = (progenitorData[numSteps-1,0], progenitorData[numSteps-1,1], progenitorData[numSteps-1,2])
numBins = 75
phaseDataSep = (phaseData.max() - phaseData.min())/numBins
phaseDataAlt = np.floor(phaseData/phaseDataSep) * phaseDataSep
#gmps = CalcGaussianMidPoint(phaseData[:,numSteps-1], phaseDataAlt[:, numSteps-1], rData[:,numSteps-1], phaseDataSep, numBins)
#gsubmps = CalcGaussianSubMidPoint(phaseData[:,numSteps-1], phaseDataAlt[:, numSteps-1], rData[:,numSteps-1], phaseDataSep, numBins)
gaussRange = phaseDataSep * 10
lowGaussRange = phaseDataSep * 5
#gpams = CalcGaussianParams(gmps, phaseData[:,numSteps-1], rData[:,numSteps-1], gaussRange)
#gsubpams = CalcGaussianSubParams(gsubmps, phaseData[:,numSteps-1], rData[:,numSteps-1], lowGaussRange)



initOffsetX = np.concatenate((xData[0:numParticlesPerStepPerLP,0], 100 + np.zeros(totCount- numParticlesPerStepPerLP), xData[totCount:totCount + numParticlesPerStepPerLP,0], 100 + np.zeros(totCount - numParticlesPerStepPerLP)))
initOffsetY = np.concatenate((yData[0:numParticlesPerStepPerLP,0], 100 + np.zeros(totCount- numParticlesPerStepPerLP), yData[totCount:totCount + numParticlesPerStepPerLP,0], 100 + np.zeros(totCount - numParticlesPerStepPerLP)))
energyMean = CalcEnergyMean(xData[:,numSteps - 1],yData[:,numSteps - 1],zData[:,numSteps - 1], 0,0,0,mainMass,mainScaleRadius, vxData[:,numSteps - 1], vyData[:,numSteps - 1], vzData[:,numSteps - 1])
angMomMean = CalcAngularMomentumMean(xData[:,numSteps - 1],yData[:,numSteps - 1],zData[:,numSteps - 1],vxData[:,numSteps - 1],vyData[:,numSteps - 1],vzData[:,numSteps - 1])



fl = open("outputFileFinRRR.txt", "r")
xOther, yOther, zOther, phaseOther, vxOther, vyOther, vzOther = [],[], [], [], [], [], []
i = 0
progXDataO = (0,0,0)
haveProg = False
for x in fl:
    q = x.split("|")
    if haveProg == False:
        progXDataO = (float(q[0]), float(q[1]), float(q[2]))
        haveProg = True
        continue
    
    xOther.append(float(q[0]))
    yOther.append(float(q[1]))
    zOther.append(float(q[2]))
    phaseOther.append(float(q[3]))
    
    #i = (i+1)%3
    vxOther.append(float(q[4]))
    vyOther.append(float(q[5]))
    vzOther.append(float(q[6]))
xOther = np.array(xOther)
yOther = np.array(yOther)
zOther = np.array(zOther)
phaseOther = np.array(phaseOther)
vxOther = np.array(vxOther)
vyOther = np.array(vyOther)
vzOther = np.array(vzOther)
rOther = DistanceC(0,0,0,xOther,yOther,zOther)
vDataO = DistanceC(0,0,0,vxOther,vyOther,vzOther)
numBinsO =400
phaseDataSepO = (phaseOther.max() - phaseOther.min())/numBinsO
phaseDataAltO = np.floor(phaseOther/phaseDataSepO) * phaseDataSepO
#gmpsO = CalcGaussianMidPoint(phaseOther, phaseDataAltO, rOther, phaseDataSepO, numBinsO)
gaussRangeO = phaseDataSepO * 10
gaussRangeO = gaussRange
lowGaussRangeO = gaussRange * 0.5
#gpamsO = CalcGaussianParams(gmpsO, phaseOther, rOther, gaussRangeO)
#gsubmpsO = CalcGaussianSubMidPoint(phaseOther, phaseDataAltO, rOther, phaseDataSepO, numBinsO)
#gsubpamsO = CalcGaussianSubParams(gsubmpsO, phaseOther, rOther, lowGaussRangeO)
#energyOther = CalcEnergyValues(xOther, yOther, zOther, 0,0,0,mainMass,mainScaleRadius,vxOther,vyOther,vzOther)
#angmomOther = CalcAngularMomentumValues(xOther,yOther,zOther,vxOther,vyOther,vzOther)
#lowmpsO, highmpsO = CalcELMidPoints(energyOther, angmomOther, phaseOther)
minAngle = 1000
minInd = 0


progRDataO = DistanceC(0,0,0,progXDataO[0], progXDataO[1], progXDataO[2])
#progXDataO = (1.82, 4.58, 0)
angdat = CalcAngle(xOther, yOther, zOther)
#progang = CalcAngle()
#fullangdat = CalcAngleFull(xOther, yOther, zOther, phaseOther, 0)
fullangdat = CalcAngleMM(xOther, yOther, zOther, phaseOther, CalcAngle(progXDataO[0], progXDataO[1], progXDataO[2]))
for q in range(len(vxOther)):
    if np.abs(fullangdat[q]) < minAngle:
        minAngle = fullangdat[q]
        minInd = q
progVDataO = (vxOther[minInd], vyOther[minInd], vzOther[minInd])
progDataSignO = np.sign((vxOther[minInd] * progXDataO[0]) +  (vyOther[minInd] * progXDataO[1]) + (vzOther[minInd] * progXDataO[2]))
#xdl = np.linspace(-500, 500, 100)
#ydl = strln(xdl, ld[0], ld[1])
sldf, odd = curve_fit(strln, fullangdat, rOther, p0=np.array((-0.1,6)))
xdla = np.linspace(-10, 10, 100)
ydla = strln(xdla, sldf[0], sldf[1])
numangbins = 400
fullangdatsepO = (fullangdat.max() - fullangdat.min())/numangbins
gaussRangeO = fullangdatsepO * 5

angmeandat = np.zeros(numangbins+1)
angmeancount = np.zeros(numangbins+1)
igrs = np.round((fullangdat-fullangdat.min())  / fullangdatsepO)
angdatalt =  np.round(fullangdat  / fullangdatsepO) * fullangdatsepO
angmeanangle = np.zeros(numangbins+1)
for q in range(len(angdatalt)):
    index = int(igrs[q] + 0.01)
    angmeandat[index] += rOther[q]
    angmeancount[index] +=1
for m in range(len(angmeandat)):
    angmeanangle[m] = fullangdat.min() + (m * fullangdatsepO)
    if angmeancount[m] == 0:
        continue
    angmeandat[m] /= angmeancount[m]
gmpsO = CalcGaussianMidPoint(fullangdat, angdatalt, rOther, fullangdatsepO, numangbins)
gpamsO = CalcGaussianParams(gmpsO, fullangdat, rOther, gaussRangeO)
gsubmpsO = CalcGaussianSubMidPoint(fullangdat, angdatalt, rOther, fullangdatsepO, numangbins)
gsubpamsO = CalcGaussianSubParams(gsubmpsO, fullangdat, rOther, gaussRangeO)


mmaO = 3.5
gamO = []
gahO = []
gbmO = []
gbhO = []
for q in range(len(gpamsO)):
    gpO = gpamsO[q]

    if np.abs(gpO[0]) > mmaO:
        continue
    gamO.append(gpO[0])
    gahO.append(gpO[2])
for q in range(len(gsubpamsO)):
    gpO = gsubpamsO[q]
    if np.abs(gpO[0]) > mmaO:
        continue
    gbmO.append(gpO[0])
    gbhO.append(gpO[2])

freqO, ampO = CalcFreqAmp(gamO, gahO, gbmO, gbhO)
veldaO = CalcMMVels(gamO, gbmO, fullangdat, vDataO, fullangdatsepO)
accr,acca =[],[]
for m in range(len(angmeandat)):
    if np.abs(angmeanangle[m]) < 4:
        accr.append(angmeandat[m])
        acca.append(angmeanangle[m])
#angmeandat = angmeandat[minzero+1:maxzero-1] 
#angmeanangle = angmeanangle[minzero+1:maxzero-1]


bruh = np.fft.rfft(accr)
re = np.abs(np.real(bruh)) 
xvasd = np.linspace(0, numangbins, len(re))

def ThreadMethod(dat, index):
    global qfdat
    qfdat[index] = CalcQualityFactor(dat)


def CQFS(data):
    global qfdat
    threads = list()
    for index in range(nwalkers):
        xtt = threading.Thread(target=ThreadMethod, args=(data[index],index))
        threads.append(xtt)
        xtt.start()

    for index, thread in enumerate(threads):
        thread.join()
    return qfdat

def MCMCMethod(dat):
    return CalcQualityFactor(dat)

ndim, nwalkers = 4, 12
qfdat = np.zeros(nwalkers)
pool = Pool()
max_n = 100000
index = 0
autocorr = np.empty(max_n)
old_tau = np.inf
filename = "mcstatbt.h5"
backend = emcee.backends.HDFBackend(filename)
#backend.reset(nwalkers, ndim)
p0 = (np.random.rand(nwalkers, ndim) * np.array((10, 10, 20000, 10))) + \
      np.array((8, 6, 6000, 8))
sampler = emcee.EnsembleSampler(nwalkers, ndim, MCMCMethod, pool=pool, backend=backend)

#state = sampler.run_mcmc(p0, 100)
for sample in sampler.run_mcmc(p0, max_n, progress=True):
    if sampler.iteration % 100:
        continue
    tau = sampler.get_autocorr_time(tol=0)
    autocorr[index] = np.mean(tau)
    index += 1
    converged = np.all(tau * 100 < sampler.iteration)
    converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
    if converged:
        break
    old_tau = tau

"""
#scELo = ax[1].scatter(energyOther, angmomOther, s=5, marker="o", alpha = 0.25, color="b")
#scpo = ax2.scatter(phaseOther, fullangdat, s=5, marker="o", color="k") 
scpo = ax4.plot(xvasd, re/2, color="k") 

sc1s = {}
for q in range(3):
    for p in range(2):
        sc1s[q,p] = ax1s[q,p].scatter(initOffsetX,initOffsetY, s=5, marker="o", alpha = 1, cmap = 'RdPu')

#sc10a = ax1s[1,0].scatter(initOffsetX,initOffsetY, s=5, marker="o", alpha = 1, cmap = 'RdPu')
#sc20a = ax1s[2,0].scatter(initOffsetX,initOffsetY, s=5, marker="o", alpha = 1, cmap = 'RdPu')
#sc01a = ax1s[0,1].scatter(initOffsetX,initOffsetY, s=5, marker="o", alpha = 1, cmap = 'RdPu')
#sc11a = ax1s[1,1].scatter(initOffsetX,initOffsetY, s=5, marker="o", alpha = 1, cmap = 'RdPu')
#sc21a = ax1s[2,1].scatter(initOffsetX,initOffsetY, s=5, marker="o", alpha = 1, cmap = 'RdPu')  

scngngn = ax3.scatter(phaseOther, fullangdat, s=5, marker="o")
scap = ax3.scatter(np.zeros(numParticles), np.zeros(numParticles), s=5, marker="o")
#scngngn = ax3.scatter(phaseOther, ggl, s=5, marker="o")
#gjgjg = ax3.plot(xdl, ydl)
gjgjg = ax2.plot(xdla, ydla)
scpo = ax2.scatter(fullangdat, rOther, s=5, marker="o", color="k") 
scpo = ax2.scatter(angmeanangle, angmeandat, s=5, marker="o", color="r") 

scc = ax[0].scatter(0,0,s=10,marker="x")
sco = ax[0].scatter(xOther, yOther, s=5, marker="o", alpha = 0.05, color="b")

scxy = ax[0].scatter(initOffsetX,initOffsetY, s=5, marker="o", alpha = 1, cmap = 'RdPu')

scp = ax[0].scatter(progenitorData[0,0], progenitorData[0,1], s=10, marker="o")

scEL = ax[1].scatter(np.zeros(numParticles), np.zeros(numParticles), s=5, alpha = 0.5, marker="o")

scrpd = ax2.scatter(np.zeros(numParticles), np.zeros(numParticles), s=5, marker="o")

qfaco = CalcQualityFactor(np.array((progenitorInit[0], progenitorInit[4], mainMass, mainScaleRadius)))

text1a = ax[0].text(0.01, 0.99, 'qf = ' + str(qfaco),
        verticalalignment='top', horizontalalignment='left',
        transform=ax[0].transAxes,
        color='red', fontsize=15)


for q in range(len(gmps)):
    gdat = gpams[q]
    pvs = np.linspace(gdat[0] - gaussRange, gdat[0] + gaussRange, 30)
    yvs = GaussianToFit(pvs, gdat[0], gdat[1], gdat[2])
    sca = ax2.plot(pvs, yvs, color="k")

for q in range(len(gsubmps)):
    gdat = gsubpams[q]
    pvs = np.linspace(gdat[0] - gaussRange, gdat[0] + gaussRange, 30)
    yvs = GaussianSubToFit(pvs, gdat[0], gdat[1], gdat[2])
    sca = ax2.plot(pvs, yvs, color="r")


for q in range(len(gmpsO)):
    gdat = gpamsO[q]
    pvs = np.linspace(gdat[0] - gaussRangeO, gdat[0] + gaussRangeO, 30)
    yvs = GaussianToFit(pvs, gdat[0], gdat[1], gdat[2])
    sca = ax2.plot(pvs, yvs, color="y")

for q in range(len(gsubmpsO)):
    gdat = gsubpamsO[q]
    pvs = np.linspace(gdat[0] - gaussRangeO, gdat[0] + gaussRangeO, 30)
    yvs = GaussianSubToFit(pvs, gdat[0], gdat[1], gdat[2])
    sca = ax2.plot(pvs, yvs, color="y")

circles = {}
for q in range(3):
    for p in range(2):
        ax1s[q,p].set_xlim(xmin,xmax)
        ax1s[q,p].set_ylim(ymin, ymax)
        ax1s[q,p].set_xlabel("x")
        ax1s[q,p].set_ylabel("y")
        circles[q,p] = plt.Circle((0,0), 1, fill=False)
        ax1s[q,p].add_artist(circles[q,p])
ax[0].set_xlim(xmin,xmax)
ax[0].set_ylim(ymin, ymax)
ax[1].set_xlim([-100, 100])
ax[1].set_ylim([-100, 100])
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[1].set_xlabel("$E-E_0$")
ax[1].set_ylabel("$L-L_0$")
ax2.set_xlim([-12,12])
ax2.set_ylim([0, 20])
ax3.set_xlim([-800,800])
ax3.set_ylim([-15, 15])
ax2.set_xlabel("Angle")
ax2.set_ylabel("r")
ax4.set_xlim([0, 400])
ax4.set_ylim([0,100])

def FunctionToFit():
    return 3



def GetQualityFactors():
    global mainMass, mainScaleRadius, ogVals
    mVals = np.arange(mmin, mmax, mstep)
    aVals = np.arange(amin, amax, astep)
    rVals = np.arange(rmin, rmax, rstep)
    vVals = np.arange(vmin, vmax, vstep)
    outputFactors = np.zeros((len(mVals), len(aVals), len(rVals), len(vVals)))
    for mc in range(len(mVals)):
        for ac in range(len(aVals)):
            for rc in range(len(rVals)):
                for vc in range(len(vVals)):
                    print(str(mVals[mc]) + " " + str(aVals[ac]) + " " + str(rVals[rc]) + " " + str(vVals[vc]))
                    mainMass = mVals[mc]
                    mainScaleRadius = aVals[ac]
                    if mainMass == 10000 and int(mainScaleRadius) == 11:
                        print("this should be best")
                    progenitorInitT = np.array((rVals[rc], 0, 0, 0, vVals[vc], 0)) 
                    progenitorDataT = igr.odeint(ProgenitorPropogator, progenitorInitT, tVals)
                    ogVals = InstantaneousOmega(np.array([progenitorDataT[:,0], progenitorDataT[:,1], progenitorDataT[:,2]]), np.array([progenitorDataT[:,3], progenitorDataT[:,4], progenitorDataT[:,5]]))
                    lagPointsT = CalcInitData(np.array([progenitorDataT[:,0], progenitorDataT[:,1], progenitorDataT[:,2]]), np.array([0,0,0]), mainMass, mainScaleRadius, satMass, np.array([progenitorDataT[:,3], progenitorDataT[:,4], progenitorDataT[:,5]]))
                    initPositionsT = lagPointsT.flatten()
                    particleDataT = RK4(ParticlePropogator, initPositionsT, tVals)
                    xDataT = np.concatenate((particleDataT[0:totCount,:], particleDataT[6*totCount:7*totCount,:]))
                    yDataT = np.concatenate((particleDataT[1*totCount:2 * totCount,:], particleDataT[7*totCount:8*totCount,:]))
                    zDataT = np.concatenate((particleDataT[2*totCount:3 * totCount,:], particleDataT[8*totCount:9*totCount,:]))
                    vxDataT = np.concatenate((particleDataT[3*totCount:4 * totCount,:], particleDataT[9*totCount:10*totCount,:]))
                    vyDataT = np.concatenate((particleDataT[4*totCount:5 * totCount,:], particleDataT[10*totCount:11*totCount,:]))
                    vzDataT = np.concatenate((particleDataT[5*totCount:6 * totCount,:], particleDataT[11*totCount:12*totCount,:]))
                    energyDataT = CalcEnergyValues(xDataT,yDataT,zDataT, 0,0,0,mainMass,mainScaleRadius, vxDataT, vyDataT, vzDataT)
                    angMomDataT = CalcAngularMomentumValues(xDataT,yDataT,zDataT,vxDataT,vyDataT,vzDataT)
                    phaseDataT, colorDataT = CalcPhaseValues(xDataT, yDataT, zDataT, progenitorDataT[:,0], progenitorDataT[:,1], progenitorDataT[:,2])
                    lowmpsT, highmpsT = CalcELMidPoints(energyDataT[:,numSteps-1], angMomDataT[:,numSteps-1], phaseDataT[:,numSteps-1])
                    elqf = np.abs(np.product((lowmpsT - lowmpsO)/lowmpsO)) * np.abs(np.product((highmpsT - highmpsO)/highmpsO))
                    #outputFactors[mc,ac] = elqf
                    
                    rDataT = DistanceC(0,0,0,xDataT,yDataT,zDataT)
                    numBinsT =150
                    phaseDataSepT = (phaseDataT.max() - phaseDataT.min())/numBinsT
                    phaseDataAltT = np.floor(phaseDataT/phaseDataSepT) * phaseDataSepT
                    gmpsT = CalcGaussianMidPoint(phaseDataT[:,numSteps-1], phaseDataAltT[:, numSteps-1], rDataT[:,numSteps-1], phaseDataSepT, numBinsT)
                    gsubmpsT = CalcGaussianSubMidPoint(phaseDataT[:,numSteps-1], phaseDataAltT[:, numSteps-1], rDataT[:,numSteps-1], phaseDataSepT, numBinsT)
                    gaussRangeT = phaseDataSepT * 10
                    gaussRangeT = gaussRange
                    gaussLowerRangeT = phaseDataSepT * 5
                    gpamsT = CalcGaussianParams(gmpsT, phaseDataT[:,numSteps-1], rDataT[:,numSteps-1], gaussRangeT)
                    gsubpamsT = CalcGaussianParams(gsubmpsT, phaseDataT[:,numSteps-1], rDataT[:,numSteps-1], gaussLowerRangeT)
                    #qfact = 0
                    lowApDat = np.array((-1000000,0,0))
                    highApDat = np.array((1000000,0,0))
                    lowApDatT = np.array((-1000000,0,0))
                    highApDatT = np.array((1000000,0,0))
                    lowSubApDat = np.array((-1000000,0,0))
                    highSubApDat = np.array((1000000,0,0))
                    lowSubApDatT = np.array((-1000000,0,0))
                    highSubApDatT = np.array((1000000,0,0))
                    for num in range(len(gpamsO)):
                        c = gpamsO[num][0]
                        if np.sign(c) == -1 and c > lowApDat[0]:
                            lowApDat = gpamsO[num]
                        elif np.sign(c) == 1 and c < highApDat[0]:
                            highApDat = gpamsO[num]
                    for num in range(len(gpamsT)):
                        cT = gpamsT[num][0]
                        if np.sign(cT) == -1 and cT > lowApDatT[0]:
                            lowApDatT = gpamsT[num]
                        elif np.sign(cT) == 1 and cT < highApDatT[0]:
                            highApDatT = gpamsT[num]
                    for num in range(len(gsubpamsO)):
                        cs = gsubpamsO[num][0]
                        if np.sign(cs) == -1 and cs > lowSubApDat[0]:
                            lowSubApDat = gsubpamsO[num]
                        elif np.sign(cs) == 1 and cs < highSubApDat[0]:
                            highSubApDat = gsubpamsO[num]
                    for num in range(len(gsubpamsT)):
                        csT = gsubpamsT[num][0]
                        if np.sign(csT) == -1 and csT > lowSubApDatT[0]:
                            lowSubApDatT = gsubpamsT[num]
                        elif np.sign(csT) == 1 and csT < highSubApDatT[0]:
                            highSubApDatT = gsubpamsT[num]
                    phd = highApDat[0] - lowApDat[0]
                    phdT = highApDatT[0] - lowApDatT[0]
                    phdsub = highSubApDat[0] - lowSubApDat[0]
                    phdsubT = highSubApDatT[0] - lowSubApDatT[0]
                    qfact =  ((phd - phdT) / phd) * ((lowApDat[1] - lowApDatT[1]) / lowApDat[1]) * ((highApDat[1] - highApDatT[1]) / highApDat[1]) * ((lowApDat[2] - lowApDatT[2]) / lowApDat[2]) * ((highApDat[2] - highApDatT[2]) / highApDat[2])
                    qsubfact =  ((phdsub - phdsubT) / phdsub) * ((lowSubApDat[1] - lowSubApDatT[1]) / lowSubApDat[1]) * ((highSubApDat[1] - highSubApDatT[1]) / highSubApDat[1]) * ((lowSubApDat[2] - lowSubApDatT[2]) / lowSubApDat[2]) * ((highSubApDat[2] - highSubApDatT[2]) / highSubApDat[2])
                    qtot = qfact * qsubfact
                    #outputFactors[mc, ac] = elqf * np.abs(np.product((lowApDat - lowApDatT) / lowApDat) * np.product((highApDat - highApDatT) / highApDat))
                    outputFactors[mc,ac, rc, vc] = math.log(np.abs(qtot), 10)
            
            
    return outputFactors

mmin = 9000
mmax = 13000
mstep = 1000
amin = 11.6
amax = 12.1
astep = 0.5
rmin = 8
rmax = 16
rstep = 1
vmin = 7.16
vmax = 9.66
vstep = 0.5


fig4, ax4 = plt.subplots()
mValsM = np.arange(mmin, mmax, mstep)
aValsM = np.arange(amin, amax, astep)
rValsM = np.arange(rmin, rmax, rstep)
vValsM = np.arange(vmin, vmax, vstep)
Xm, Ym = np.meshgrid(rValsM, vValsM)
qfs = np.zeros((len(mValsM), len(aValsM), len(rValsM), len(vValsM)))

#cs = ax4.plot(mValsM, qfs[:,0])


amin, mmin,  rmin, vmin, qfmax = 0,0,0,0,0
aminind,mminind, vminind, rminind = 0,0,0,0
for m in range(len(mValsM)):
    for a in range(len(aValsM)):
        for r in range(len(rValsM)):
            for v in range(len(vValsM)):
                qfs[m,a,r,v] = CalcQualityFactor((rValsM[r],vValsM[v],mValsM[m],aValsM[a]))
                if qfs[m,a,r,v] >  qfmax:
                    amin = aValsM[a]
                    mmin = mValsM[m]
                    mminind = m
                    aminind = a
                    vminind = v
                    rminind = r
                    rmin = rValsM[r]
                    vmin = vValsM[v]
                    qfmax = qfs[m,a, r, v]

print("Correct values are m = " + str(mmin) + ", a = " + str(amin) + ", r = " + str(rmin) + ", v = " + str(vmin))
cs = ax4.contourf(Xm, Ym, qfs[:,aminind,:,vminind].T, 20, cmap=cm.PuBu_r)
cbar = fig4.colorbar(cs)


def animate2(i):
    return 0    
def animate1s(i):
    return 0  

def animate(i):
    i = numSteps - 1
    if i < numSteps or True:
        partCount = i * numParticlesPerStepPerLP
        totCount = numParticlesPerStepPerLP * numSteps
        xd = np.concatenate((xData[0:partCount + numParticlesPerStepPerLP,i], 100 + np.zeros(totCount-(partCount + numParticlesPerStepPerLP)), xData[totCount:totCount+partCount + numParticlesPerStepPerLP,i], 100 + np.zeros(totCount - (partCount + numParticlesPerStepPerLP))))
        yd = np.concatenate((yData[0:partCount + numParticlesPerStepPerLP,i], 100 + np.zeros(totCount-(partCount + numParticlesPerStepPerLP)), yData[totCount:totCount+partCount + numParticlesPerStepPerLP,i], 100 + np.zeros(totCount - (partCount + numParticlesPerStepPerLP))))
        zd = np.concatenate((zData[0:partCount + numParticlesPerStepPerLP,i], 100 + np.zeros(totCount-(partCount + numParticlesPerStepPerLP)), zData[totCount:totCount+partCount + numParticlesPerStepPerLP,i], 100 + np.zeros(totCount - (partCount + numParticlesPerStepPerLP))))
        rd = np.concatenate((rData[0:partCount + numParticlesPerStepPerLP,i], -100 + np.zeros(totCount-(partCount + numParticlesPerStepPerLP)), rData[totCount:totCount+partCount + numParticlesPerStepPerLP,i], -100 + np.zeros(totCount - (partCount + numParticlesPerStepPerLP))))
        pd = np.concatenate((phaseData[0:partCount + numParticlesPerStepPerLP,i], -100 + np.zeros(totCount-(partCount + numParticlesPerStepPerLP)), phaseData[totCount:totCount+partCount + numParticlesPerStepPerLP,i], -100 + np.zeros(totCount - (partCount + numParticlesPerStepPerLP))))
        scxy.set_offsets(np.c_[xd, yd])
        tm1 = phaseData[:,i] #- min(phaseData[:,i])
        colmap = cm.get_cmap('jet_r', 120)
        #for q in range(len(tm2)):
        #    scuff.append((2 * max(tm2[q] - 0.5, 0), max(1 - ( 2 * abs(tm2[q] - 0.5)), 0), max(2 * (0.5-tm2[q]), 0), 1))
        scuff2 = colmap(colorData[:,i])
        scxy.set_color(scuff2)
        scEL.set_color(scuff2)
        #sc.set_color(["r"] * numParticles)
        #scEL.set_color(["r"] * numParticles)
        ad = CalcAngleMM(xd, yd, zd, pd, CalcAngle(progenitorData[i,0], progenitorData[i,1], progenitorData[i,2]))
        scEL.set_offsets(np.c_[np.concatenate((energyData[0:partCount + numParticlesPerStepPerLP,i], 100 + np.zeros(totCount-(partCount + numParticlesPerStepPerLP)), energyData[totCount:totCount+partCount + numParticlesPerStepPerLP,i], 100 + np.zeros(totCount-(partCount + numParticlesPerStepPerLP)))), np.concatenate((angMomData[0:partCount + numParticlesPerStepPerLP,i], 100 + np.zeros(totCount-(partCount + numParticlesPerStepPerLP)), angMomData[totCount:totCount+partCount + numParticlesPerStepPerLP,i], 100 + np.zeros(totCount-(partCount + numParticlesPerStepPerLP))))])
        scp.set_offsets(np.c_[progenitorData[i,0], progenitorData[i,1]])
        scrpd.set_offsets(np.c_[ad, rd])
        scrpd.set_color(scuff2)
        scap.set_offsets(np.c_[pd, ad])
        for q in range(3):
            maxn = (numSteps * (q+1)) / 3
            if i < maxn:
                sc1s[q,0].set_offsets(np.c_[xd, yd])
                sc1s[q,1].set_offsets(np.c_[xd, yd])
                sc1s[q,0].set_color(scuff2)
                sc1s[q,1].set_color(scuff2)
                ax1s[q,1].set_xlim(progenitorData[i,0] - 3,progenitorData[i,0] + 3)
                ax1s[q,1].set_ylim(progenitorData[i,1] - 3, progenitorData[i,1] + 3)
                trad = GetTidalRadius(0, mainMass, mainScaleRadius, DistanceC(0,0,0,progenitorData[i,0], progenitorData[i,1], 0), satMass)
                circles[q,0].set_radius(trad)
                circles[q,0].set_center([progenitorData[i,0], progenitorData[i,1]])
                circles[q,1].set_radius(trad)
                circles[q,1].set_center([progenitorData[i,0], progenitorData[i,1]])
            

ani = animation.FuncAnimation(fig, animate, 
                frames=int(tFinal/tStep), interval=2, repeat=False) 
ani2 = animation.FuncAnimation(fig2, animate2, 
                frames=int(tFinal/tStep), interval=2, repeat=False) 
ani2 = animation.FuncAnimation(fig1s, animate1s, 
                frames=int(tFinal/tStep), interval=2, repeat=False) 

fig.set_figheight(6)
fig.set_figwidth(12)
f = "fanimaaha.gif" 
writergif = animation.ImageMagickWriter(fps=30)
ani.save(f, writer=writergif)


plt.show()
"""
