from const import *
from def_func import *
import numpy as np
# Import thư viện vẽ đồ thị (nếu cần)
import matplotlib.pyplot as plt 

# ==========================================
# 1. ĐỊNH NGHĨA HÀM TÍNH MA TRẬN F
# ==========================================
def compute_F(t_curr, Y):
    """
    Tính ma trận F(t, Y) kích thước 2 x N theo chiến lược trong ảnh.
    Y[0] tương đương với mảng Y1 (chứa fe + i*fh).
    Y[1] tương đương với mảng Y2 (chứa độ phân cực p).
    """
    Y1 = Y[0] # Lấy hàng 1
    Y2 = Y[1] # Lấy hàng 2

    # Khởi tạo ma trận F toàn số 0 với kích thước giống Y (2 hàng, N cột)
    F = np.zeros((2, N), dtype=complex)

    # Chạy n từ 1 đến N
    for n in range(1, N + 1):
        # Index trong Python chạy từ 0, nên vị trí cột tương ứng là n - 1
        idx = n - 1 

        # Gọi hàm từ def_func.py
        # Truyền đúng các hằng số h, T2, de (Δε), d0 (Δ0) từ const.py
        F[0, idx] = F1_n(Y2, n, t_curr, N)
        F[1, idx] = F2_n(Y1, Y2, n, t_curr, N, h, T2, de, d0)

    return F

# ==========================================
# 2. THIẾT LẬP VÀ GIẢI RK4
# ==========================================

# Khởi tạo trạng thái ban đầu Y tại t = -3 * wt
# Ma trận 2 hàng, N cột (sửa lại từ N+1 của const.py để khớp với def_func)
Y = np.zeros((2, N), dtype=complex)

# Lấy thời gian bắt đầu từ const.py
current_t = t

# (Tùy chọn) Tạo mảng để lưu kết quả vẽ đồ thị
history_t = []
history_fe_n1 = [] # Lưu sự tiến hóa của f_e tại k/n = 1

print(f"Bắt đầu mô phỏng từ t = {current_t} fs đến {tm} fs...")

# Vòng lặp thời gian
while current_t <= tm:

    # --- Lưu dữ liệu (để plot) ---
    history_t.append(current_t)
    # Phần thực của Y[0, 0] chính là f_e tại n=1
    history_fe_n1.append(np.real(Y[0, 0])) 

    # --- THUẬT TOÁN RK4 ---
    # k1
    k1 = compute_F(current_t, Y) * dt

    # k2 (tính tại t + dt/2, Y + k1/2)
    k2 = compute_F(current_t + dt/2, Y + k1/2) * dt

    # k3 (tính tại t + dt/2, Y + k2/2)
    k3 = compute_F(current_t + dt/2, Y + k2/2) * dt

    # k4 (tính tại t + dt, Y + k3)
    k4 = compute_F(current_t + dt, Y + k3) * dt

    # Cập nhật giá trị Y mới
    Y = Y + (k1 + 2*k2 + 2*k3 + k4) / 6

    # Cập nhật thời gian tiến tới bước tiếp theo
    current_t += dt

    # (In tiến độ mỗi 100 bước để biết code đang chạy)
    if int(current_t) % 20 == 0:
        print(f"Đang giải tại t = {current_t:.1f} fs")

print("Hoàn thành mô phỏng!")

# ==========================================
# 3. VẼ ĐỒ THỊ KIỂM TRA (TÙY CHỌN)
# ==========================================
plt.figure(figsize=(8, 5))
plt.plot(history_t, history_fe_n1, color='b', label=r'$f_e$ tại $n=1$')
plt.xlabel('Thời gian t (fs)')
plt.ylabel('Xác suất chiếm chỗ (Occupation probability)')
plt.title('Sự tiến hóa của hàm phân bố electron theo thời gian')
plt.grid(True)
plt.legend()
plt.show()