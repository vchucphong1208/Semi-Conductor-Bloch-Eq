from const import *
from def_func import *
import numpy as np
import matplotlib.pyplot as plt 

# ==========================================
# 4. VÒNG LẶP RK4 & LƯU TRỮ DỮ LIỆU
# ==========================================
Y = np.zeros((2, N), dtype=complex)
t_current = t0

# Mảng lưu dữ liệu theo thời gian
history_t = []
history_epsilon = []
history_fe = []      # Lưu ma trận f_e
history_fh = []      # Lưu ma trận f_h
history_p_abs = []   # Lưu ma trận |p|
history_N_mac = []   # Mật độ hạt N(t)
history_P_mac = []   # Độ phân cực vĩ mô P(t)

print(f"Bắt đầu chạy thuật toán RK4 từ t = {t0} fs đến {tm} fs...")
epsilon = n_arr * de  # Cập nhật epsilon theo năng lượng để lưu vào history_epsilon
while t_current <= tm:
    # 4.1 Thu thập dữ liệu hiện tại
    fe_curr = np.real(Y[0])
    fh_curr = np.imag(Y[0])
    p_abs_curr = np.abs(Y[1])
    
    history_t.append(t_current)
    history_epsilon.append(epsilon)
    history_fe.append(fe_curr)
    history_fh.append(fh_curr)
    history_p_abs.append(p_abs_curr)
    
    # 4.2 Tính đại lượng vĩ mô (Slide 6)
    N_mac = np.sum(np.sqrt(n_arr) * fe_curr)
    P_mac = np.sum(np.sqrt(n_arr) * p_abs_curr)
    history_N_mac.append(N_mac)
    history_P_mac.append(P_mac)
    
    # 4.3 Giải RK4
    k1 = compute_F(t_current, Y) * dt
    k2 = compute_F(t_current + dt/2, Y + k1/2) * dt
    k3 = compute_F(t_current + dt/2, Y + k2/2) * dt
    k4 = compute_F(t_current + dt, Y + k3) * dt
    
    Y = Y + (k1 + 2*k2 + 2*k3 + k4) / 6
    t_current += dt

print("Hoàn thành tính toán RK4!")

# Chuyển list thành array để dễ xử lý
history_t = np.array(history_t)
history_epsilon=np.array(history_epsilon)

history_fe = np.array(history_fe)     # Kích thước (số_bước_thời_gian, N)
history_fh = np.array(history_fh)
history_p_abs = np.array(history_p_abs)
history_N_mac = np.array(history_N_mac)
history_P_mac = np.array(history_P_mac)
filename = "SBE_Full_Evolution.txt"
with open(filename, "w") as f:
    # 1. Ghi tiêu đề (Header) - Căn lề phải 15 khoảng trống cho mỗi cột
    header = f"{'t(fs)':>15} {'epsilon':>15} {'Re[Y0]':>15} {'Im[Y0]':>15} {'Abs[Y1]':>15}\n"
    f.write(header)
    f.write("-" * 75 + "\n") # Đường kẻ phân cách cho đẹp

    # 2. Vòng lặp chính
    # Duyệt qua từng bước thời gian (M bước)
    for i in range(len(history_t)):
        t_val = history_t[i]
        
        # Với mỗi t, duyệt qua toàn bộ N điểm năng lượng
        for j in range(len(n_arr)):
            eps_val = history_epsilon[i, j]  # Hoặc n_arr[j] * de
            re_y0   = history_fe[i, j]
            im_y0   = history_fh[i, j]       # Lưu ý: fh bạn đang gán là Im(Y[0])
            abs_y1  = history_p_abs[i, j]
            
            # Ghi dòng dữ liệu với định dạng số khoa học (scientific notation)
            line = f"{t_val:15f} {eps_val:15.6f} {re_y0:15.6e} {im_y0:15.6e} {abs_y1:15.6e}\n"
            f.write(line)
        
        # (Tùy chọn) Thêm một dòng trống giữa các khối thời gian để dễ quan sát bằng mắt
        f.write("\n")


# ==========================================
# 5. LƯU DỮ LIỆU RA FILE (Slide 6)
# ==========================================
# Ghi t, N(t), P(t) ra file txt
data_macro = np.column_stack((history_t, history_N_mac, history_P_mac))
np.savetxt("SBE_Macro_Results.txt", data_macro, header="t(fs) N(t) P(t)", fmt='%.6f', comments='')
print("Đã lưu dữ liệu vĩ mô ra file 'SBE_Macro_Results.txt'.")

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