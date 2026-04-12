from const import *
from def_func import *
import numpy as np
import matplotlib.pyplot as plt 

# ==========================================
# 6. VẼ ĐỒ THỊ BÁO CÁO (Slide 6)
# ==========================================
# Đồ thị 1: Các đại lượng vĩ mô N(t) và P(t)
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(history_t, history_N_mac, 'b-', label='N(t) (Mật độ electron)')
ax1.set_xlabel('Thời gian t (fs)')
ax1.set_ylabel('N(t)', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(history_t, history_P_mac, 'r-', label='P(t) (Độ phân cực)')
ax2.set_ylabel('P(t)', color='r')
ax2.tick_params('y', colors='r')

plt.title('Sự tiến hóa của Mật độ hạt N(t) và Độ phân cực P(t)')
fig.tight_layout()

#Đồ thị 2: Đồ thị 3 chiều (Bề mặt) cho f_e
#Tạo lưới tọa độ Không gian (Năng lượng) và Thời gian
T_mesh, E_mesh = np.meshgrid(history_t, n_arr * de, indexing='ij')

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(T_mesh, E_mesh, history_fe, cmap='viridis', edgecolor='none')
ax.set_xlabel('Thời gian t (fs)')
ax.set_ylabel('Năng lượng (meV)')
ax.set_zlabel(r'Xác suất chiếm chỗ $f_e$')
ax.set_title('Đồ thị 3D: Phân bố Electron theo Thời gian và Năng lượng')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
