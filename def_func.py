from const import *
import numpy as np

def g(n,n1):
    return (1/np.sqrt(n*de))*np.log((np.sqrt(n)+np.sqrt(n1))/(np.sqrt(n)-np.sqrt(n1)))
  
def En(Y1, n_index, N):
    """
    Y1: Mảng các giá trị Y_{1, n1}
    n_index: n hiện tại (1-based index như trong công thức)
    """
    # Tạo mảng n1 từ 1 đến N
    n1_array = np.arange(1, N + 1)
    # Tính g(n, n1) cho tất cả n1
    g_values = np.array([g(n_index,n1) for n1 in n1_array])
    # Re[Y] + Im[Y]
    y_terms = np.real(Y1) + np.imag(Y1)
    summation = np.sum(g_values * y_terms)
    coef = (np.sqrt(Er) / np.pi) * de
    return coef * summation
  
def OmegaRabi(Y2, n_index, t, N):
    """
    Y2: Mảng các giá trị Y_{2, n1}
    t: thời gian hiện tại
    """
    n1_array = np.arange(1, N + 1)
    g_values = np.array([g(n_index,n1) for n1 in n1_array])
    
    # Phần tổng Sigma
    summation = np.sum(g_values * Y2)
    term_pulse = (1/2) * ( (h * np.sqrt(np.pi)) / wt ) * chi0 * np.exp(-(t**2 / wt**2))
    
    coef = (np.sqrt(Er) / np.pi) * de
    
    return (1 / h) * (term_pulse + coef * summation)

def df_dt(p_array, n_index, t, N):
    """
    Tính đạo hàm của hàm phân bố f_{j, k}(t) theo thời gian.
    Vì phương trình của f_e và f_h giống hệt nhau ở vế phải, ta dùng chung 1 hàm.
    
    p_array: Mảng chứa toàn bộ các giá trị phân cực (p) đóng vai trò Y2.
    n_index: Chỉ số k hiện tại (1-based index).
    t: Thời gian hiện tại.
    """
    # Tính Omega Rabi hiệu dụng tại vị trí n_index
    Omega_R = OmegaRabi(p_array, n_index, t, N)
    
    # Lấy giá trị p tại vị trí k hiện tại. 
    # Do n_index tính từ 1 đến N, index trong mảng Python sẽ là n_index - 1
    p_k = p_array[n_index - 1]
    
    # Công thức: -2 * Im[ Omega_R(t) * p*(t) ]
    # np.conj() dùng để lấy số phức liên hợp (p*)
    return -2 * np.imag(Omega_R * np.conj(p_k))


def dp_dt(f_e_array, f_h_array, p_array, n_index, t, N, hbar, T2):
    """
    Tính đạo hàm của độ phân cực vi mô p_k(t) theo thời gian.
    
    f_e_array: Mảng phân bố electron (Y1 cho e_e).
    f_h_array: Mảng phân bố lỗ trống (Y1 cho e_h).
    p_array: Mảng độ phân cực (Y2 cho OmegaRabi).
    hbar: Hằng số Planck rút gọn.
    T2: Thời gian mất pha (Dephasing time).
    """
    # 1. Tính các năng lượng hiệu dụng e_e và e_h bằng hàm En của bạn
    e_e = En(f_e_array, n_index, N)
    e_h = En(f_h_array, n_index, N)
    
    # 2. Tính Omega Rabi hiệu dụng
    Omega_R = OmegaRabi(p_array, n_index, t, N)
    
    # Lấy các giá trị cụ thể tại vị trí k hiện tại (index = n_index - 1)
    idx = n_index - 1
    p_k = p_array[idx]
    f_e_k = f_e_array[idx]
    f_h_k = f_h_array[idx]
    
    # 3. Tính toán 3 thành phần của phương trình đạo hàm p_k
    
    # Thành phần 1: -(i / hbar) * [e_e(t) + e_h(t)] * p_k(t)
    term1 = -(1j / hbar) * (e_e + e_h) * p_k
    
    # Thành phần 2: + i * [1 - f_{e,k}(t) - f_{h,k}(t)] * Omega_R(t)
    term2 = 1j * (1 - f_e_k - f_h_k) * Omega_R
    
    # Thành phần 3: - p_k(t) / T2
    term3 = - (p_k / T2)
    
    # Tổng hợp các thành phần lại
    return term1 + term2 + term3
