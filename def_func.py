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
