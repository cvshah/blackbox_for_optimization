import numpy as np
import time

def backtracking_line_search(func, x_k, p_k, g_x_k, alpha, rho, c1, f_x_k, norm_g_x_k):
    number_function_call_bls = 0
    nk = 1
    #checking the gradient of the function
    if norm_g_x_k >= 1:
        nk = 1
    else:
        nk = 1 - np.floor(np.log(norm_g_x_k))
    su = 0
    nk1 = nk.astype(np.int64)
    for i0 in range(nk1):
        f_x_kp1 = func.evaluate_function(x_k + p_k)
        su = su + f_x_kp1
        i0 += 1
    f_x_kplus1 = su / nk
    number_function_call_bls = number_function_call_bls + nk
    #print(f'alpha before = {alpha}')
    while (f_x_kplus1 > f_x_k + c1 * alpha * np.dot(g_x_k, p_k)):
        alpha = rho * alpha
        su1 = 0
        for i0 in range(nk):
            f_x_kp1 = func.evaluate_function(x_k + alpha*p_k)
            su1 = su1 + f_x_kp1
        f_x_kplus1 = su1/nk
        #print(f_x_kplus1)
        number_function_call_bls = number_function_call_bls + nk
        #print(f'rho = {rho}')
        time.sleep(0.5)
    #print(alpha)
    f_x_k = f_x_kplus1
    print(f_x_k)
    return alpha, number_function_call_bls, f_x_k
    

def BFGSVK_1(func, x_k, max_iteration = 5000, abs_tol = 10**(-5), line_search = False, alpha = 1, rho = 0.95, c1 = 0.01):
    # choosing an initial approximat hessian B_0
    v_k = np.identity(len(x_k))
    s_k = np.ones(len(x_k))
    y_k = np.ones(len(x_k))
    x_1 = []
    x_2 = []
    x_1.append(x_k[0])
    x_2.append(x_k[1])
    number_function_call = 0
    f_x_k = func.evaluate_function(x_k)
    # loop
    norm_g_x_k_matrix = []
    for n in range(max_iteration):
        # computing gradient G(x_k) = grad of f(x_k)
        g_x_k = func.gradient_function(x_k)
        norm_g_x_k = np.linalg.norm(g_x_k)
        norm_g_x_k_matrix = np.append(norm_g_x_k_matrix, norm_g_x_k)
        # stoping criteria mod of G(x_k) < e
        if np.linalg.norm(g_x_k) < abs_tol:
            break
        # computing the search direction p_k from B(x_k) * p_k = - G(x_k)
        p_k = np.matmul(v_k, (-1) * g_x_k)
        # computing the step size alpha 
        if line_search:
            #Backtracking line search
            alpha = 1
            alpha_ = backtracking_line_search(func, x_k, p_k, g_x_k, alpha, rho, c1, f_x_k, norm_g_x_k)
            alpha = alpha_[0]         
            number_function_call += alpha_[1]
            f_x_k = alpha_[2]
        else:
            alpha = 1
        # update the design variables x_k = x_k + alpha * p_k
        x_k = x_k + alpha * p_k
        # updating s_k = alpha * p_k and y_k = G(x_kplus1) - G(x_k)
        s_k = alpha * p_k
        #s_k_t = np.transpose(s_k)
        g_x_kplus1 = func.gradient_function(x_k)
        y_k = g_x_kplus1 - g_x_k
        #y_k_t = np.transpose(y_k)
        # updating the approximate hessian inverse v_kplus1
        sktyk = np.inner(s_k, y_k)
        skykt = np.outer(s_k, y_k)
        first_bracket = np.identity(len(x_k)) - (skykt / sktyk)
        ykskt = np.outer(y_k, s_k)
        second_bracket = np.identity(len(x_k)) - (ykskt / sktyk)
        firstmul = np.matmul(first_bracket, v_k)        
        firstterm = np.matmul(firstmul, second_bracket)
        skskt = np.outer(s_k, s_k)
        secondterm = skskt / sktyk
        v_k = firstterm + secondterm        
        # Record the best x values at the end of every cycle
        x_1.append(x_k[0])
        x_2.append(x_k[1])
    return x_k, n, norm_g_x_k_matrix, x_1, x_2, number_function_call
    # end loop
