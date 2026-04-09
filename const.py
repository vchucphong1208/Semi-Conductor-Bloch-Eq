#1. Parameter
import numpy as np

chi0 = 1.0
T2 = 100       # (fs) dephasing time T2
wt = 50        # (fs) width of laser pulse
N = 80         # chosen number of discrete energy in
dt = 2         # (fs) time step
tm = 200       # (fs) max. time
emax = 320     # (meV) max. energy
d0 = 100       # (meV) detuning energy
Er = 4.2       # (meV) Bound energy
h = 658.5      # (meV.fs) Planck constant (h-bar)
de = emax / N 
#2. Initial Conditions
t = -3 * wt    # initial time
# Tạo mảng số phức 2 hàng, N+1 cột chứa toàn số 0
fp = np.zeros((2, N + 1), dtype='complex_')
