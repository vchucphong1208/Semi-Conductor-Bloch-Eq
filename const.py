import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# ==========================================
# 1. HẰNG SỐ VÀ ĐIỀU KIỆN BAN ĐẦU (Đồng bộ theo Slide 4 PDF)
# ==========================================
chi0 = 0.001        # Cường độ xung (có thể thay 0.1 -> 2)
T2 = 200.0         # (fs) Thời gian mất pha (Sửa lại 200fs theo PDF)
wt = 25.0          # (fs) Bề rộng xung laser (Sửa lại 25fs theo PDF)
N = 100            # Số điểm chia năng lượng (Sửa lại 100 theo PDF)
dt = 2.0           # (fs) Bước thời gian
tm = 500.0         # (fs) Thời gian tối đa
emax = 300.0       # (meV) Năng lượng tối đa
d0 = 30.0          # (meV) Năng lượng Detuning
Er = 4.2           # (meV) Năng lượng Rydberg (liên kết)
h = 658.5          # (meV.fs) Hằng số Planck rút gọn
de = emax / N      # (meV) Mức chia năng lượng
t0 = -3 * wt       # (fs) Thời gian bắt đầu
