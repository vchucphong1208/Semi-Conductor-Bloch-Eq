import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# ==========================================
# 1. HẰNG SỐ VÀ ĐIỀU KIỆN BAN ĐẦU
# ==========================================
chi0 = 0.1        # Cường độ xung (có thể thay 0.1 -> 2)
T2 = 200.0         # (fs) Thời gian mất pha (Sửa lại 200fs theo PDF)
wt = 50.0          # (fs) Bề rộng xung laser (Sửa lại 25fs theo PDF)
N = 100            # Số điểm chia năng lượng (Sửa lại 100 theo PDF)
dt = 2.0           # (fs) Bước thời gian
tm = 500.0         # (fs) Thời gian tối đa
emax = 300.0       # (meV) Năng lượng tối đa
d0 = 100.0          # (meV) Năng lượng Detuning
Er = 4.2           # (meV) Năng lượng Rydberg (liên kết)
h = 658.5          # (meV.fs) Hằng số Planck rút gọn
de = emax / N      # (meV) Mức chia năng lượng
t0 = -3 * wt       # (fs) Thời gian bắt đầu    

# ==========================================
# 2. TIỀN XỬ LÝ
# ==========================================
n_arr = np.arange(1, N + 1)
n_grid, n1_grid = np.meshgrid(n_arr, n_arr, indexing='ij')

epsilon_const = 1e-10 
tu_so = np.sqrt(n_grid) + np.sqrt(n1_grid)
mau_so = np.abs(np.sqrt(n_grid) - np.sqrt(n1_grid)) + epsilon_const

G_matrix = (1 / np.sqrt(n_grid * de)) * np.log(tu_so / mau_so)
coef_E = (np.sqrt(Er) / np.pi) * de

# ==========================================
# 3. HÀM TÍNH ĐẠO HÀM (MA TRẬN F)
# ==========================================
def compute_F(t, Y):
    Y1 = Y[0] 
    Y2 = Y[1] 
    
    fe_plus_fh = np.real(Y1) + np.imag(Y1)
    
    En_arr = coef_E * (G_matrix @ fe_plus_fh)
    
    pulse_term = 0.5 * (h * np.sqrt(np.pi) / wt) * chi0 * np.exp(-(t**2) / (wt**2))
    Omega_R_arr = (1 / h) * (pulse_term + coef_E * (G_matrix @ Y2))
    
    F = np.zeros((2, N), dtype=complex)
    
    a_arr = -2 * np.imag(Omega_R_arr * np.conj(Y2))
    F[0] = a_arr + 1j * a_arr
    
    term1 = -(1j / h) * (n_arr * de - d0 - En_arr) * Y2
    term2 = 1j * (1 - fe_plus_fh) * Omega_R_arr
    term3 = - Y2 / T2 
    F[1] = term1 + term2 + term3
    
    return F

# ==========================================
# 4. VÒNG LẶP RK4 & LƯU TRỮ DỮ LIỆU
# ==========================================
def SoDE(chi0, wt, d0, T2):
    Y = np.zeros((2, N), dtype=complex)
    t_current = t0

    history_t = []
    history_epsilon = []
    history_fe = []      
    history_fh = []      
    history_p_abs = []   
    history_N_mac = []   
    history_P_mac = []   

    print(f"Bắt đầu chạy thuật toán RK4 từ t = {t0} fs đến {tm} fs...")
    eps_fixed = n_arr * de
    while t_current <= tm:
        fe_curr = np.real(Y[0])
        fh_curr = np.imag(Y[0])
        p_abs_curr = np.abs(Y[1])
    
        history_t.append(t_current)
        history_epsilon.append(eps_fixed)
        history_fe.append(fe_curr)
        history_fh.append(fh_curr)
        history_p_abs.append(p_abs_curr)
    
        N_mac = np.sum(np.sqrt(n_arr) * fe_curr)
        P_mac = np.sum(np.sqrt(n_arr) * p_abs_curr)
        history_N_mac.append(N_mac)
        history_P_mac.append(P_mac)
    
        k1 = compute_F(t_current, Y) * dt
        k2 = compute_F(t_current + dt/2, Y + k1/2) * dt
        k3 = compute_F(t_current + dt/2, Y + k2/2) * dt
        k4 = compute_F(t_current + dt, Y + k3) * dt
    
        Y = Y + (k1 + 2*k2 + 2*k3 + k4) / 6
        t_current += dt

    print("Hoàn thành tính toán RK4!")

    return (np.array(history_t), np.array(history_epsilon), np.array(history_fe), 
            np.array(history_fh), np.array(history_p_abs), np.array(history_N_mac), np.array(history_P_mac))

def xuat_file_text(t, eps, fe, fh, p_abs, Nt, Pt):
    filename = f"fe_fh_p-chi0={chi0}_wt={wt}_d0={d0}_T2={T2}.txt"
    with open(filename, "w") as f:
        f.write(f"chi0={chi0}, wt={wt}, d0={d0}, T2={T2}\n")
        header = f"{'t(fs)':>15} {'epsilon':>15} {'Re[Y0]':>15} {'Im[Y0]':>15} {'Abs[Y1]':>15}\n"
        f.write(header)
        f.write("-" * 75 + "\n")
        for i in range(len(t)):
            for j in range(len(n_arr)):
                line = f"{t[i]:15.6f} {eps[i, j]:15.6f} {fe[i, j]:15.6e} {fh[i, j]:15.6e} {p_abs[i, j]:15.6e}\n"
                f.write(line)
            f.write("\n")
    print(f"Đã xuất dữ liệu vào file {filename}")

# ==========================================
# 6. VẼ ĐỒ THỊ
# ==========================================
def Ve_do_thi_2D_Nt(ds_ketqua, tieu_de, ten_file_anh):
    plt.figure(figsize=(10, 6))
    for ket_qua in ds_ketqua:
        nhan = ket_qua['label']
        data = ket_qua['data']
        t_vals = data[0]
        Nt_vals = data[1]
        plt.plot(t_vals, Nt_vals, label=nhan, linewidth=2)
    plt.title(tieu_de)
    plt.xlabel("t (fs)")
    plt.ylabel("N(t)")
    plt.legend()
    plt.grid(True)
    plt.savefig(ten_file_anh, dpi=300)

def Ve_do_thi_2D_Pt(ds_ketqua, tieu_de, ten_file_anh):
    plt.figure(figsize=(10, 6))
    for ket_qua in ds_ketqua:
        nhan = ket_qua['label']
        data = ket_qua['data']
        t_vals = data[0]
        Pt_vals = data[2]
        plt.plot(t_vals, Pt_vals, label=nhan, linewidth=2)
    plt.title(tieu_de)
    plt.xlabel("t (fs)")
    plt.ylabel("P(t)")
    plt.legend()
    plt.grid(True)
    plt.savefig(ten_file_anh, dpi=300)

def Ve_do_thi_3D(tieude, t_arr, fe, fh, p_abs):
    T_mesh, E_mesh = np.meshgrid(t_arr, n_arr * de, indexing='ij')
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(T_mesh, E_mesh, fe, cmap='viridis')
    ax.set_xlabel('Thời gian t (fs)')
    ax.set_ylabel('Năng lượng (meV)')
    ax.set_zlabel('Phân bố fe')
    ax.set_title("fe " + tieude)
    plt.savefig("fe_" + tieude + ".png")
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(T_mesh, E_mesh, fh, cmap='plasma')
    ax.set_xlabel('Thời gian t (fs)')
    ax.set_ylabel('Năng lượng (meV)')
    ax.set_zlabel('Phân bố fh')
    ax.set_title("fh " + tieude)
    plt.savefig("fh_" + tieude + ".png")
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(T_mesh, E_mesh, p_abs, cmap='viridis')
    ax.set_xlabel('Thời gian t (fs)')
    ax.set_ylabel('Năng lượng (meV)')
    ax.set_zlabel('Độ phân cực abs(p)')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title("abs(p)" + tieude)
    plt.savefig("abs(p)" + tieude + ".png")
# ==========================================
# MAIN PROGRAMME
# ==========================================
# 1. Thử số của thầy
t, eps, fe, fh, p_abs, Nt, Pt = SoDE(chi0, wt, d0, T2)
Nt_Pt = np.array([t, Nt, Pt])
Ve_do_thi_2D_Nt([{'label': 'N(t)', 'data': Nt_Pt}], "Sự tiến hóa của Mật độ N(t)", "N_t.png")
Ve_do_thi_2D_Pt([{'label': 'P(t)', 'data': Nt_Pt}], "Sự tiến hóa của Độ phân cực P(t)", "P_t.png")
Ve_do_thi_3D(" ", t, fe, fh, p_abs)

# 2. Thay đổi wt
wt = 50.0
t, eps, fe, fh, p_abs, Nt, Pt = SoDE(chi0, wt, d0, T2)
Ve_do_thi_3D("wt_50", t, fe, fh, p_abs)

# 3. Thay đổi chi0
chi0_vals = [0.1, 0.5, 1.0, 1.5, 2.0,]
ket_qua = []
for c in chi0_vals:
    chi0 = c
    t_val, eps_val, fe_val, fh_val, p_val, Nt_val, Pt_val = SoDE(chi0, wt, d0, T2)
    ket_qua.append({'label': f"chi0 = {chi0}", 'data': np.array([t_val, Nt_val, Pt_val])})
    xuat_file_text(t_val, eps_val, fe_val, fh_val, p_val, Nt_val, Pt_val)

Ve_do_thi_2D_Nt(ket_qua, "Sự tiến hóa của Mật độ N(t) theo chi0", "N_t_chi0.png")
Ve_do_thi_2D_Pt(ket_qua, "Sự tiến hóa của Độ phân cực P(t) theo chi0", "P_t_chi0.png")

# 4. Thay đổi T2
T2_vals = [50.0, 100.0, 200.0, 300.0]
ket_qua_T2 = []
for v in T2_vals:
    T2 = v
    t_val, eps_val, fe_val, fh_val, p_val, Nt_val, Pt_val = SoDE(chi0, wt, d0, T2)
    ket_qua_T2.append({'label': f"T2 = {T2}", 'data': np.array([t_val, Nt_val, Pt_val])})

Ve_do_thi_2D_Nt(ket_qua_T2, "Sự tiến hóa của Mật độ N(t) theo T2", "N_t_T2.png")
Ve_do_thi_2D_Pt(ket_qua_T2, "Sự tiến hóa của Độ phân cực P(t) theo T2", "P_t_T2.png")
plt.show()
