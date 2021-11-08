import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import matplotlib.animation as animation
import numpy as np
from scipy.optimize import curve_fit

def mf(m, c, r):
    return r * np.power(np.power(m/c, 0.666) - 1,0.5)

mmin = 5000
mmax = 15000
mstep = 500
amin = 5.6
amax = 15.6
astep = 0.5

mValsM = np.arange(mmin, mmax, mstep)
aValsM = np.arange(amin, amax, astep)
Xm, Ym = np.meshgrid(mValsM, aValsM)
qfs = np.zeros((len(mValsM), len(aValsM)))
fl = open("griddatadd.txt", "r")
for x in fl:
    q = x.split("|")
    m = int((float(q[0]) - mmin) / mstep)
    a = int((float(q[1]) - amin) / astep)
    if qfs[m,a] < float(q[2]):
        qfs[m,a] = q[2]
xok = []
aok = []
for m in range(len(mValsM)):
    for a in range(len(aValsM)):
        if qfs[m,a] > 11:
            xok.append(mValsM[m])
            aok.append(aValsM[a])

tdat, covd = curve_fit(mf, xok, aok, p0 = np.array((2612, 12.84)))
#tdat = (3612,10.84)
gs = tdat[0]
rv = tdat[1]
xml = np.arange(mmin, mmax, mstep)
yal =  mf(xml, gs, rv)
fig4, ax4 = plt.subplots()
cs = ax4.contourf(Xm, Ym, qfs.T, 20, cmap=cm.get_cmap("jet"))
ln = ax4.plot(xml, yal, c="k")
ax4.set_ylim([6,14])
ax4.set_xlim([6000,14000])
#cs = ax4.contourf(Xm, Ym, qfs, 20, cmap=cm.PuBu_r)
cbar = fig4.colorbar(cs)
plt.show()