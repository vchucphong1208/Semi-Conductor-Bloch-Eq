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
