from const import *
import numpy as np

# ==========================================
# 2. TIỀN XỬ LÝ (PRE-COMPUTATION) - BÍ QUYẾT TĂNG TỐC CODE
# Thay vì dùng vòng lặp for, ta tính sẵn ma trận G kích thước NxN.
# ==========================================
n_arr = np.arange(1, N + 1)
# Tạo lưới tọa độ để tính tương tác giữa mọi cặp (n, n1)
n_grid, n1_grid = np.meshgrid(n_arr, n_arr, indexing='ij')

# Xử lý điểm kỳ dị: thêm epsilon để không bao giờ bị chia cho 0 khi n = n1
epsilon = 1e-10 
tu_so = np.sqrt(n_grid) + np.sqrt(n1_grid)
mau_so = np.abs(np.sqrt(n_grid) - np.sqrt(n1_grid)) + epsilon

# Tính sẵn ma trận G (Tương đương hàm g(n, n1) nhưng tính cho mọi điểm cùng lúc)
G_matrix = (1 / np.sqrt(n_grid * de)) * np.log(tu_so / mau_so)

# Hệ số chung hay dùng trong En và Omega_R
coef_E = (np.sqrt(Er) / np.pi) * de

# ==========================================
# 3. HÀM TÍNH ĐẠO HÀM (MA TRẬN F)
# ==========================================
def compute_F(t, Y):
    """
    Tất cả các phép toán ở đây đều là phép tính trên mảng (Array Operations).
    Nhanh hơn gấp hàng trăm lần so với việc dùng vòng lặp for.
    """
    Y1 = Y[0] # Chứa f_e ở phần thực, f_h ở phần ảo
    Y2 = Y[1] # Chứa độ phân cực p
    
    # f_e + f_h
    fe_plus_fh = np.real(Y1) + np.imag(Y1)
    
    # Tính En(Y) bằng phép nhân ma trận (Toán tử @ trong NumPy)
    En_arr = coef_E * (G_matrix @ fe_plus_fh)
    
    # Tính Omega_R(Y)
    pulse_term = 0.5 * (h * np.sqrt(np.pi) / wt) * chi0 * np.exp(-(t**2) / (wt**2))
    Omega_R_arr = (1 / h) * (pulse_term + coef_E * (G_matrix @ Y2))
    
    # Khởi tạo ma trận F (2 hàng, N cột)
    F = np.zeros((2, N), dtype=complex)
    
    # Tính F1 (Đạo hàm của Y1)
    # a = -2 * Im[Omega_R * p*]
    a_arr = -2 * np.imag(Omega_R_arr * np.conj(Y2))
    F[0] = a_arr + 1j * a_arr
    
    # Tính F2 (Đạo hàm của Y2)
    term1 = -(1j / h) * (n_arr * de - d0 - En_arr) * Y2
    term2 = 1j * (1 - fe_plus_fh) * Omega_R_arr
    term3 = - Y2 / T2
    F[1] = term1 + term2 + term3
    
    return F
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
