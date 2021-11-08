import unsio.input as uns_in 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.integrate as igr
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from sklearn.linear_model import LinearRegression

nbody = 10000
tmax1 = 0.25
tmax2 = 30
tStep = 0.0625
numSteps = 1 + int(tmax2 / tStep)
xmin = -40
ymin = -40
xmax = 40
ymax = 40
xmina = -200
ymina = -120
xmaxa = -75
ymaxa = 120
satMass = 1
satMasses = np.zeros(numSteps)
mainMass = 10000
comLoc = (0,0,0)
comLocs = np.zeros((3,numSteps))
simname="/home/elliot/NemoProjectStuff/NemoProject/ffa3.nemo" 
components="all" 
times="all" 
float32=True
plummerScaleFactor = 0.2
energyErrorArray=[0]
initEn = 0
angmomErrorArray=[0]
initAngMom = 0
comDistArray=[0]
hillRadArray=[0]
massArray=[satMass]
tArray=[0]
#fig = plt.figure(constrained_layout=True)
fig1, ax1 = plt.subplots(1, 2)
fig2, ax2 = plt.subplots(1, 2)
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()
ax5 = ax3.twinx()
ax6 = ax4.twinx()

phaseVals = np.zeros((nbody, numSteps))
trueCols = np.zeros((nbody, numSteps))
escs = np.zeros(nbody)
cols = ["g"] * nbody
ax3.set_xlabel('t', fontsize=20)
ax4.set_xlabel('t', fontsize=20)
#ax5.set_xlabel('t')
#ax6.set_xlabel('t')
ax3.set_ylabel('Mass fraction Retained', color='r', fontsize=20)
ax4.set_ylabel('$\Delta E/E$', color='r', fontsize=20)
ln3, = ax3.plot(tArray, massArray, color='r', label='Mass fraction retained')
ln4, = ax4.plot(tArray, energyErrorArray, color='r')
ln5, = ax5.plot(tArray, hillRadArray, color='tab:blue', label='Hill Radius')
ln5a, = ax5.plot(tArray, comDistArray, color='g', label='Distance of CoM to host galaxy')
ln6, = ax6.plot(tArray, angmomErrorArray, color='tab:blue')
ax3.tick_params(axis='y', labelcolor='r')
ax4.tick_params(axis='y', labelcolor='r')
satmass1 = 0
satmass2 = 0
color = 'tab:blue'
ax5.set_ylabel('Distance', color=color, fontsize=20)
ax6.set_ylabel('$\Delta L/L$', color=color, fontsize=20)  # we already handled the x-label with ax1
#ln2, = ax6.plot(tArray, angmomErrorArray, color=color)
#ln3, = ax2.plot(tArray, hillRadArray, color='g')
ax5.tick_params(axis='y', labelcolor=color)
ax6.tick_params(axis='y', labelcolor=color)
plt.legend()
fig3.tight_layout() 
fig4.tight_layout() 
ax1[0].set_xlim([xmin,xmax])
ax1[0].set_ylim([ymin,ymax])
ax1[1].set_xlim([xmin,xmax])
ax1[1].set_ylim([ymin,ymax])
ax2[0].set_xlim([xmina,xmaxa])
ax2[0].set_ylim([ymina,ymaxa])
ax2[1].set_xlim([xmina,xmaxa])
ax2[1].set_ylim([ymina,ymaxa])
ax1[0].set_xlabel(r'$x$', fontsize=20)
ax1[0].set_ylabel(r'$y$', fontsize=20)
ax1[1].set_xlabel(r'$x$', fontsize=20)
ax1[1].set_ylabel(r'$y$', fontsize=20)
ax2[0].set_xlabel(r'$L-L_0$', fontsize=20)
ax2[0].set_ylabel(r'$E-E_0$', fontsize=20)
ax2[1].set_ylabel(r'$L-L_0$', fontsize=20)
ax2[1].set_xlabel(r'$E-E_0$', fontsize=20)
SMALL_SIZE = 12
MEDIUM_SIZE = 16
ax1[0].tick_params(axis='both', which='major', labelsize=MEDIUM_SIZE)
ax1[0].tick_params(axis='both', which='minor', labelsize=SMALL_SIZE)
ax1[1].tick_params(axis='both', which='major', labelsize=MEDIUM_SIZE)
ax1[1].tick_params(axis='both', which='minor', labelsize=SMALL_SIZE)
ax2[0].tick_params(axis='both', which='major', labelsize=MEDIUM_SIZE)
ax2[0].tick_params(axis='both', which='minor', labelsize=SMALL_SIZE)
ax2[1].tick_params(axis='both', which='major', labelsize=MEDIUM_SIZE)
ax2[1].tick_params(axis='both', which='minor', labelsize=SMALL_SIZE)
ax3.tick_params(axis='both', which='major', labelsize=MEDIUM_SIZE)
ax3.tick_params(axis='both', which='minor', labelsize=SMALL_SIZE)
ax4.tick_params(axis='both', which='major', labelsize=MEDIUM_SIZE)
ax4.tick_params(axis='both', which='minor', labelsize=SMALL_SIZE)
ax5.tick_params(axis='both', which='major', labelsize=MEDIUM_SIZE)
ax5.tick_params(axis='both', which='minor', labelsize=SMALL_SIZE)
ax6.tick_params(axis='both', which='major', labelsize=MEDIUM_SIZE)
ax6.tick_params(axis='both', which='minor', labelsize=SMALL_SIZE)
text1a = ax1[0].text(0.01, 0.99, 't = 0.25',
        verticalalignment='top', horizontalalignment='left',
        transform=ax1[0].transAxes,
        color='red', fontsize=15)
text1b = ax1[1].text(0.01, 0.99, 't = 0.25',
        verticalalignment='top', horizontalalignment='left',
        transform=ax1[1].transAxes,
        color='red', fontsize=15)
text2a = ax2[0].text(0.01, 0.99, 't = 0.25',
        verticalalignment='top', horizontalalignment='left',
        transform=ax2[0].transAxes,
        color='red', fontsize=15)
text2b = ax2[1].text(0.01, 0.99, 't = 0.25',
        verticalalignment='top', horizontalalignment='left',
        transform=ax2[1].transAxes,
        color='red', fontsize=15)
ax3.set_xlim([0,30])
ax4.set_xlim([0,30])
ax5.set_xlim([0,30])
ax6.set_xlim([0,30])
ax3.set_ylim([0.85, 1])
ax4.set_ylim([-0.001, 0.002])
ax5.set_ylim([0, 35])
ax6.set_ylim([-0.03, 0])
#ax2.set_ylim([0.8,1])
#ln, = ax2.plot(tArray,massArray)
gradLower = 0
gradHigher = 0
SATXPOS = np.zeros((nbody, numSteps))
SATYPOS = np.zeros((nbody, numSteps))
SATZPOS = np.zeros((nbody, numSteps))
velsArray = np.zeros((nbody * 3, numSteps))
totenArray = np.zeros(numSteps)
potenArray = np.zeros((nbody,numSteps))

sc1 = ax1[0].scatter(SATXPOS[:,0], SATYPOS[:,0], c=cols, alpha=1, s=0.5)
sc2 = ax1[1].scatter(SATXPOS[:,0], SATYPOS[:,0], c=cols, alpha=1, s=0.5)
sc3 = ax2[0].scatter(SATXPOS[:,0], SATYPOS[:,0], c=cols, alpha=0.8, s=0.5)
sc4 = ax2[1].scatter(SATXPOS[:,0], SATYPOS[:,0], c=cols, alpha=0.8, s=0.5)
sc5 = ax2[1].scatter([0,0], [0,0], c="k", alpha=1, s=3)
#sc3 = ax.scatter(comLoc[0], comLoc[1], alpha=0, c="b", s=10)
circle1 = plt.Circle((0,0), 1, fill=False)
circle2 = plt.Circle((0,0), 1, fill=False)
ax1[0].add_artist(circle1)
ax1[1].add_artist(circle2)
my_in = uns_in.CUNS_IN(simname,"all","all",float32)



def calcCOMBest(xpos, ypos, zpos):
    pos = (0,0,0)
    counter = 0
    for i in range(xpos.size):
        if escs[i] == 0:
            pos = (pos[0] + xpos[i], pos[1] + ypos[i], pos[2] + zpos[i])
            counter += 1
    return (pos[0]/counter, pos[1]/counter, pos[2]/counter)

def calcCOM(xpos,ypos, zpos):
    pos = (0,0,0)
    for i in range(xpos.size):
        pos = (pos[0] + xpos[i], pos[1] + ypos[i], pos[2] + zpos[i])
    return (pos[0]/nbody, pos[1]/nbody, pos[2]/nbody)

def calcJacCOM(xpos, ypos, zpos, fac):
    pos = (0,0,0)
    numadded = 0
    jacRad = calcJacRad(satMass, comLoc)
    for i in range(xpos.size):
        if calcDistance(comLoc, (xpos[i], ypos[i], zpos[i])) <= jacRad * fac:
            pos = (pos[0] + xpos[i], pos[1] + ypos[i], pos[2] + ypos[i])
            numadded += 1
    return (pos[0]/numadded, pos[1]/numadded, pos[2]/numadded)

def calcJacRad(satm, com):
    mr = satm/(3 * 100)
    r0 = np.sqrt(com[0]**2 + com[1]**2 + com[2]**2)
    return np.float_power(mr, 0.3333) * r0

def calcDistance(tup1, tup2):
    return np.sqrt((tup1[0] - tup2[0])**2 + (tup1[1] - tup2[1])**2 + (tup1[2] - tup2[2]) ** 2)

def checkInJacRad(xpos, ypos, zpos):
    jacRad = calcJacRad(satMass, comLoc)
    return calcDistance(comLoc, (xpos, ypos, zpos)) <= jacRad

def calcMassRetained(xpos,ypos, zpos):
    numadded = 0
    jacRad = calcJacRad(satMass, comLoc)
    for i in range(xpos.size):
        if calcDistance(comLoc, (xpos[i], ypos[i], zpos[i])) <= jacRad:
            numadded += 1
    return numadded / nbody
def calcKE(vels):
    velNew = np.reshape(vels, (-1, 3))
    satxVel = velNew[0:nbody, 0]
    satyVel = velNew[0:nbody, 1]
    satzVel = velNew[0:nbody, 2]
    ke=0
    for i in range(nbody):
        ke += (1/(2 * nbody)) * (satxVel[i] **2 + satyVel[i]**2 + satzVel[i]**2)
    return ke

def sumPlummerPot(xPos, yPos, zPos):
    tot = 0
    for i in range(xPos.size):
        tot += plummerPot(xPos[i], yPos[i], zPos[i])
    return tot/nbody

def length(x,y, z):
    return np.sqrt((x**2) + (y**2) + (z**2))

def calcAngMoms(xPos, yPos, zPos, vels):
    velNew = np.reshape(vels, (-1, 3))
    satxVel = velNew[0:nbody, 0]
    satyVel = velNew[0:nbody, 1]
    satzVel = velNew[0:nbody, 2]
    angs = np.zeros(xPos.size)
    
    for i in range(xPos.size):
        angCurr = np.cross([xPos[i], yPos[i], zPos[i]], [satxVel[i], satyVel[i], satzVel[i]])
        angs[i] = length(angCurr[0], angCurr[1], angCurr[2])
        #angs[i] = np.abs((xPos[i] * satyVel[i]) - (yPos[i] * satxVel[i]))
    return angs

def calcActualGradientInt(egsLow, angmomsLow, egsHigh, angmomsHigh):
    lowEAv = 0
    lowAAv = 0
    highEAv = 0
    highAAv = 0
    for q in range(len(egsLow)):
        lowEAv += egsLow[q] / len(egsLow)
        lowAAv += angmomsLow[q] / len(egsLow)
    for q in range(len(egsHigh)):
        highEAv += egsHigh[q] / len(egsHigh)
        highAAv += angmomsHigh[q] / len(egsHigh)
    sc5.set_offsets(np.c_[[lowEAv, highEAv], [lowAAv, highAAv]])
    return (highAAv - lowAAv) / (highEAv - lowEAv)

def calcActualGradient(xpos, ypos,zpos, vels, totEn, intPots):
    angs = calcNormalisedAngMoms(xpos, ypos, zpos, vels)
    engs = calcNormalisedEnergies(xpos, ypos, zpos, vels, totEn, intPots)
    angsIn = []
    angsOut = [] 
    engsIn = []
    engsOut = []
    for q in range(angs.size):
        if escs[q] == -1:
            angsIn.append(angs[q])
            engsIn.append(engs[q])
        elif escs[q] == 1:
            angsOut.append(angs[q])
            engsOut.append(engs[q])
    return calcActualGradientInt(engsIn, angsIn, engsOut, angsOut)

def calcGradient(egs, angmoms):
    egsRes = np.array(egs).reshape((-1, 1))
    model = LinearRegression().fit(egsRes, angmoms)
    print(model.score(egsRes, angmoms))
    return model.coef_

def calcGradients(xpos, ypos,zpos, vels, totEn, intPots):
    angs = calcNormalisedAngMoms(xpos, ypos, zpos, vels)
    engs = calcNormalisedEnergies(xpos, ypos, zpos, vels, totEn, intPots)
    angsIn = []
    angsOut = [] 
    engsIn = []
    engsOut = []
    for q in range(angs.size):
        if escs[q] == -1:
            angsIn.append(angs[q])
            engsIn.append(engs[q])
        elif escs[q] == 1:
            angsOut.append(angs[q])
            engsOut.append(engs[q])
    gradIn = calcGradient(engsIn, angsIn)
    gradOut = calcGradient(engsOut, angsOut)
    return (gradIn, gradOut)

def calcNormalisedAngMoms(xPos, yPos, zPos, vels):
    angmoms = calcAngMoms(xPos, yPos, zPos, vels)
    mean = np.sum(angmoms)/angmoms.size
    newAngMoms = np.zeros(angmoms.size)
    for i in range(angmoms.size):
        newAngMoms[i] = angmoms[i]-mean
    return newAngMoms

def calcNormalisedEnergies(xPos, yPos, zPos, vels, totEn, intPots):
    velNew = np.reshape(vels, (-1, 3))
    satxVel = velNew[0:nbody, 0]
    satyVel = velNew[0:nbody, 1]
    satzVel = velNew[0:nbody, 2]
    ens = np.zeros(xPos.size)
    for i in range(ens.size):
        extPot = plummerPot(xPos[i], yPos[i], zPos[i]) / 2
        totPot = (intPots[i] / 2) + extPot
        totPot = extPot * 2
        kinEn = 0.5 * (satxVel[i] **2 + satyVel[i]**2 + satzVel[i]**2)
        ens[i] = (totPot + kinEn) - totEn
    return ens

def plummerPot(xpos, ypos, zpos):
    r2 = xpos**2 + ypos**2 + zpos**2
    return -mainMass / np.sqrt(r2 + plummerScaleFactor**2)

def WriteToFile(vels, time):
    expFile = open("outputFileFin.txt", "w")
    index = int(time/tStep)
    velNew = np.reshape(vels, (-1, 3))
    satxVel = velNew[0:nbody, 0]
    satyVel = velNew[0:nbody, 1]
    satzVel = velNew[0:nbody, 2]
    expFile.write(str(comLocs[0,index]) + "|" + str(comLocs[1,index]) + "|" + str(comLocs[2,index]) + "\n")
    for i in range(nbody):
        if escs[i] != 0:
            expFile.write(str(SATXPOS[i,index]) + "|" + str(SATYPOS[i,index]) + "|" + str(SATZPOS[i,index]) + "|" + str(phaseVals[i, index]) + "|" + str(satxVel[i]) + "|" + str(satyVel[i]) + "|" + str(satzVel[i]) + "\n")
    expFile.close()

def updatePhases(time):
    global phaseVals
    index = int(time/tStep)
    c1 = calcDistance((0,0,0), (SATXPOS[:,index], SATYPOS[:,index], SATZPOS[:,index]))
    c2 = calcDistance((0,0, 0), comLoc)
    c3 = c1 - c2
    if index != 0:
        phaseVals[:,index] = phaseVals[:,index-1] +  (c3 * abs(escs))
    else:
        phaseVals[:,index] = c3 * abs(escs)

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
    return (tm1 - min(tm1)) / (max(tm1) - min(tm1))

def updateCols(time):
    global cols
    index = int(time/tStep)
    for i in range(nbody):
        if checkInJacRad(SATXPOS[i,index], SATYPOS[i,index], SATZPOS[i,index]) == False and escs[i] == 0:
            cols[i] = "r"
            if calcDistance((0,0,0), (SATXPOS[i,index], SATYPOS[i,index], SATZPOS[i,index])) < calcDistance((0,0, 0), comLoc):
                cols[i] = "r"
                escs[i] = -1
            #    expFile.write(str(time) + "|1\n")
            else:
                cols[i] = "b"
                escs[i] = 1
            #    expFile.write(str(time) + "|-1\n")


def animate2(i):
    return 0
def animate3(i):
    return 0
def animate4(i):
    return 0

def animate1(i):
    global gradLower, gradHigher, satmass1, satmass2
    timex = tArray[i]
    if timex < tmax1:
        sc1.set_offsets(np.c_[SATXPOS[:,i], SATYPOS[:,i]])
        sc1.set_color(cols)
        #sc3.set_offsets(np.c_[calcNormalisedEnergies(SATXPOS[:,i], SATYPOS[:,i], SATZPOS[:,i], velsArray[:,i], totenArray[i], potenArray[:,i]), calcNormalisedAngMoms(SATXPOS[:,i], SATYPOS[:,i], SATZPOS[:,i], velsArray[i])])
        #sc3.set_color(cols)
        #circle1.set_radius(calcJacRad(satMasses[i], comLoc[i]))
        #circle1.set_center(comLoc[i])
        #satmass1 = satMasses[i]
        text1a.set_text('t = ' + str(timex) + ', remnant mass = ' + str(satmass1))
        
        text2a.set_text('t = ' + str(timex) + ', remnant mass = ' + str(satmass1))

    if timex <= tmax2:
        #updateCols(timex)
        #updatePhases(timex)
        #WriteToFile(velsArray[i])
        #trueCols = CalcColors(phaseVals)
        #colmap = cm.get_cmap('jet_r', 120)
        #scuff2 = colmap(trueCols)
        #sc2.set_color(scuff2)
        sc2.set_offsets(np.c_[SATXPOS[:,i], SATYPOS[:,i]])
        #sc4.set_color(scuff2)
        #sc4.set_offsets(np.c_[calcNormalisedEnergies(SATXPOS[:,i], SATYPOS[:,i], SATZPOS[:,i], velsArray[:,i], totenArray[i], potenArray[:,i]), calcNormalisedAngMoms(SATXPOS[:,i], SATYPOS[:,i], SATZPOS[:,i], velsArray[:,i])])
        #grad = calcActualGradient(SATXPOS[:,i], SATYPOS[:,i], SATZPOS[:,i], velsArray[:,i], totenArray[i], potenArray[:,i])
        #gradLower = grad
        #gradHigher = grad
        #circle2.set_radius(calcJacRad(satMasses[i], comLocs[i]))
        #circle2.set_center(comLoc[i])
        satmass2 = satMasses[i]
        text1b.set_text('t = ' + str(timex) + ', remnant mass = ' + str(satmass2))
        text2b.set_text('t = ' + str(timex) + ', remnant mass = ' + str(satmass2) + "\n gradients = " + str(gradLower))

        
    ln3.set_xdata(tArray)
    ln4.set_xdata(tArray)
    ln5.set_xdata(tArray)
    ln5a.set_xdata(tArray)
    ln6.set_xdata(tArray)
        #ln3.set_xdata(tArray)
    ln3.set_ydata(massArray)
    ln4.set_ydata(energyErrorArray)
    ln5.set_ydata(hillRadArray)
    ln5a.set_ydata(comDistArray)
    ln6.set_ydata(angmomErrorArray)
        #ln3.set_ydata(hillRadArray)
        
        #ax.set_title("A plot of all the particles of the satellite galaxy in Energy-Momentum space. $E_0$ and $L_0$ are the mean Energy and Momentum")
        #

haveFinished = False
numIt = 0
while haveFinished == False:
    if my_in.nextFrame("") and numIt < numSteps:
        print(str(numIt))
        status,poss=my_in.getData("all","pos")
        status,timex=my_in.getData("time")
        #status,masses=my_in.getData("mass")
        status,vel=my_in.getData("all","vel")
        status,poten=my_in.getData("all","pot")
        possNew = np.reshape(poss, (-1, 3))
        SATXPOS[:,numIt] = possNew[0:nbody, 0]
        SATYPOS[:,numIt] = possNew[0:nbody, 1]
        SATZPOS[:,numIt] = possNew[0:nbody, 2]
        velNew = np.reshape(vel, (-1,3))
        tArray.append(timex)
        """
        extPot = sumPlummerPot(SATXPOS[:,numIt], SATYPOS[:,numIt], SATZPOS[:,numIt])
        intPot =  np.sum(poten)/nbody
        totPot = 0.5 * (intPot + extPot)
        totKin = calcKE(vel)
        totEn = totPot + totKin
        totAngMom = np.sum(calcAngMoms(SATXPOS[:,numIt], SATYPOS[:,numIt], SATZPOS[:,numIt], vel))
        if initEn == 0:
            initEn = totEn
            energyErrorArray.append((totEn - initEn)/initEn)
        else:
            energyErrorArray.append((totEn - initEn)/initEn)
        if initAngMom == 0:
            initAngMom = totAngMom
            angmomErrorArray.append((totAngMom - initAngMom)/(initAngMom))
        else:
            angmomErrorArray.append((totAngMom - initAngMom)/(initAngMom))
        """
        if comLoc != (0, 0, 0):
            comLoc = calcCOMBest(SATXPOS[:,numIt], SATYPOS[:,numIt], SATZPOS[:,numIt])
            
        else:
            comLoc = calcCOM(SATXPOS[:,numIt], SATYPOS[:,numIt], SATZPOS[:,numIt]) 
        comLocs[:,numIt] = comLoc
        """
        satMass = calcMassRetained(SATXPOS[:,numIt], SATYPOS[:,numIt], SATZPOS[:,numIt])
        if (comDistArray.count == 1):
            comDistArray.clear()
            comDistArray.append(calcDistance((0,0,0), comLoc))
        comDistArray.append(calcDistance((0,0,0), comLoc))
        if (hillRadArray.count == 1):
            hillRadArray.clear()
            hillRadArray.append(calcJacRad(satMass, comLoc))
        hillRadArray.append(calcJacRad(satMass, comLoc))
        
        massArray.append(satMass)
        tArray.append(timex)
        satMasses[numIt] = satMass
        totenArray[numIt] = totEn
        velsArray[:,numIt] = vel
        potenArray[:,numIt] = poten
        """
        velsArray[:,numIt] = vel
        if timex <= tmax2:
            updateCols(timex)
            updatePhases(timex)
        if numIt == numSteps - 1:
            WriteToFile(vel, timex)
        numIt += 1
    else:
        haveFinished = True  

ani1 = animation.FuncAnimation(fig1, animate1, 
                frames=2000, interval=2, repeat=False) 
ani2 = animation.FuncAnimation(fig2, animate2, 
                frames=2000, interval=1, repeat=True) 
ani3 = animation.FuncAnimation(fig3, animate3, 
                frames=2000, interval=1, repeat=True) 
ani4 = animation.FuncAnimation(fig4, animate4, 
                frames=2000, interval=1, repeat=True) 
#f = "animanemoa.gif" 
#writergif = animation.ImageMagickFileWriter(fps=30)
#ani1.save(f, writer=writergif)
plt.show()
#new = np.reshape(a, (-1, ncols))

