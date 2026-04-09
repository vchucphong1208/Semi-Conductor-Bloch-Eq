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


def F1_n(Y2, n_index, t, N):
    """
    Tính F_{1,n} = complex(a, a)
    Y2: Mảng chứa các giá trị Y_2 (đóng vai trò như độ phân cực p)
    """
    # Tính Omega Rabi hiệu dụng tại vị trí n_index
    Omega_R = OmegaRabi(Y2, n_index, t, N)
    
    # Lấy giá trị Y_{2,n} hiện tại (index = n_index - 1)
    idx = n_index - 1
    Y2_n = Y2[idx]
    
    # Tính a = -2 * Im[ Omega_R * Y2_n* ]
    a = -2 * np.imag(Omega_R * np.conj(Y2_n))
    
    # Trả về complex(a, a) tương đương với a + a*j trong Python
    return a + 1j * a


def F2_n(Y1, Y2, n_index, t, N, hbar, T2, delta_eps, delta_0):
    """
    Tính F_{2,n}
    Y1: Mảng các giá trị Y_1 (có phần thực và phần ảo)
    Y2: Mảng các giá trị Y_2 (độ phân cực)
    delta_eps (Δε), delta_0 (Δ0): Các hằng số dải năng lượng (lấy từ const)
    """
    # 1. Tính En(Y) và Omega_R(Y)
    E_n_val = En(Y1, n_index, N)
    Omega_R = OmegaRabi(Y2, n_index, t, N)
    
    # Lấy giá trị tại n hiện tại
    idx = n_index - 1
    Y1_n = Y1[idx]
    Y2_n = Y2[idx]
    
    # 2. Tính toán 3 thành phần của F_{2,n}
    
    # Thành phần 1: -(i / ħ) * [n*Δε - Δ0 - E_n(Y)] * Y_{2,n}
    term1 = -(1j / hbar) * (n_index * delta_eps - delta_0 - E_n_val) * Y2_n
    
    # Thành phần 2: + i * [1 - Re[Y_{1,n}] - Im[Y_{1,n}]] * Ω_n^R(Y)
    term2 = 1j * (1 - np.real(Y1_n) - np.imag(Y1_n)) * Omega_R
    
    # Thành phần 3: - Y_{2,n} / T2
    term3 = - (Y2_n / T2)
    
    # Tổng hợp
    return term1 + term2 + term3
