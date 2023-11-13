import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#
#=== Parameters
#

# let's just let the well be size 1
L = 1
# we could use hbar or just set it to one?
#hbar = sp.constants.hbar
hbar = 1
# and same for mass?
# m = sp.constants.m_p   # hydrogen atom?
m = 1

# resolution of functions
npoints = 1000
# time between plots
plot_interval_sec = 0.05  # I think ~0.02 is the minimum allowed by plot refresh
initial_plot_pause = 0.01  # can pause longer to get screenshot ready
# fraction of the Nth coefficient period for deltat
deltat_TN_factor = 0.333  # 1 , 1/10?

# number of energy coefficients to get
Ncoeff = 20

# number of timepoints 
Ntimepoints = 300

#
# For the video on youtube: https://youtu.be/gncTK3M-Vac
# I used:
#
#   Ncoeff = 20    Ntimepoints = 450    npoints = 1000
#   initial_plot_pause = 10    plot_interval_sec = 0.05
#   deltat_TN_factor = 1
#
# This had a period of 400 steps (back to initial density)
# or t~1.3
#

# turn on debugging to get intermediate output
debugging = True

#
#=== our infinite square well solution
#
def get_x():
    return np.linspace(0, L, num=npoints)
def get_phi(n):
    return np.sqrt(2/L) * np.sin(n * np.pi / L * xvec)
def get_E(n):
    return (n*np.pi*hbar/L)**2 / 2 / m


#
#=== let's choose a gaussian-like initial state, but use it to modulate
#    the sine wave, so that we get the correct boundary conditions:
#
def get_psi():
    # parameters
    x0 = L/4  # center it on the left side of the box
    sigmax = L/10 # fairly narrow width
    # unnormalized function
    psi_un = np.exp( - (xvec-x0)**2 / (2 * sigmax**2) ) \
        * np.sin(np.pi * xvec / L)   # the sin forces zeros at the boundaries
    # normalize: Int[psi^2] = 1
    integ = np.trapz(np.multiply(psi_un, psi_un), xvec)
    A = 1/np.sqrt(integ)
    if debugging:
        print("\t normalization constant: A =", A)
    # normalized function
    return A * psi_un

def get_ck(n, psi):
    return np.trapz(np.multiply(get_phi(n), psi), xvec)

#===========
# Main code
#===========


#
#=== get the initial psi
#

# get vector of x values [0,L] for use in every subroutine
xvec = get_x() 

# construct the initial wavefunction
psi = get_psi()
maxf = np.max(np.multiply(psi, psi))
if debugging:
    # plot it for user
    pass
    #plt.plot(xvec, psi)
    #plt.show()
#
#=== get coefficients in energy expansion
#
c = np.zeros(Ncoeff)
for n in range(Ncoeff):
    c[n] = get_ck(n+1, psi)
    if debugging:
        print("\tn =", n+1, "c[n] = ", c[n])

#
#=== Time dynamics
#        
#
#--- set up a reasonable delta t:
#
#    let's find the frequency associated to our highest energy state
#    in the energy decomposition.  Then we'll use that associated
#    period as our deltat.
#
EN = get_E(Ncoeff)
fN = EN / (hbar * 2 * np.pi)  # E = hf , f = E/h
TN = 1 / fN
deltat = TN * deltat_TN_factor
if debugging:
    print("TN =", TN)
#
#--- get psi(x, t) at range of times
#
# set up plot
fig, ax = plt.subplots(figsize=(6,4))
# loop over timepoints
for i in range(Ntimepoints):
    # current time
    t = i*deltat
    # construct approximate psi(t)
    psi_of_t = np.zeros(npoints) + 0j
    for n in range(Ncoeff):
        psi_of_t += c[n] * get_phi(n+1) * np.exp(1j * get_E(n+1)/hbar * t)
    # get pdf = psi^*  psi  (casting to real avoids a warning in plt.plot())
    f = np.real(np.multiply(np.conjugate(psi_of_t), psi_of_t))
    # plot
    if debugging:
        print("i =", i, "t =", t)
        ax.plot(xvec, f)
        ax.set_ylim(0,1.1*maxf)
        ax.set_xlabel("x")
        ax.set_ylabel("probability density function")
        ax.set_title("plot " + str(i) + "/" + str(Ntimepoints) + " at t=" + f"{t:.3f}")
        fig.canvas.draw()  # draw() instead of show() allows for update without user intervention
        # pause a bit longer at the start
        if i==0:
            plt.pause(initial_plot_pause)
        else:
            plt.pause(plot_interval_sec)
        ax.clear()  # if you don't clear you'll see all prior plots
