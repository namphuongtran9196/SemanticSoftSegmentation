import numpy as np
from numpy.core.fromnumeric import shape
from scipy.sparse import eye, spdiags, diags, block_diag, csr_matrix, hstack
from scipy.sparse.linalg import lobpcg, cg


def get_U_and_V(comp, sp_pow):
    # Sparsity terms in the energy
    eps = 1e-2
    tmp_u = np.abs(comp.reshape(-1, 1))
    tmp_u[tmp_u < eps] = eps
    u = np.power(tmp_u, (sp_pow - 2))
    
    tmp_v = np.abs(1 - comp.reshape(-1, 1))
    tmp_v[tmp_v < eps] = eps
    v = np.power(tmp_v, (sp_pow - 2))
    return u, v


def sparsify_segments(soft_segments, laplacian, image_grad=None):

    sigma_s = 1
    sigma_f = 1
    delta = 100
    h, w, comp_cnt = soft_segments.shape
    N = h * w * comp_cnt
    
    if image_grad is None:
        # If image gradient is not provided, set the param to the default 0.9
        sp_pow = 0.90
    else:
        # Compute the per-pixel sparsity parameter from the gradient
        image_grad[image_grad > 0.1] = 0.1
        image_grad = image_grad + 0.9
        sp_pow = np.matlib.repmat(image_grad.T.reshape(-1, 1), comp_cnt, 1)
    

    # Iter count for pcg and main optimization
    iters_between_update = 100
    high_level_iters = 20
    
    # Get rid of very low/high alpha values and normalize
    soft_segments[soft_segments < 0.1] = 0
    soft_segments[soft_segments > 0.9] = 1
    soft_segments = soft_segments / np.sum(soft_segments, axis=2, keepdims=True)
    
        
    # Construct the linear system
    lap = laplacian.copy()
    
    for i in range(1, comp_cnt):
        laplacian = block_diag((laplacian, lap), format='csr')

    # The alpha constraint
    # C = np.matlib.repmat(eye(h*w, format='csr'), 1, comp_cnt)
    C = hstack([eye(h*w, format='csr') for _ in range(comp_cnt)])
    C = C.T @ C
    laplacian = laplacian + delta * C
    
    # The sparsification optimization
    soft_segments = soft_segments.transpose((2, 1, 0)).reshape(-1, 1)

    comp_init = soft_segments.copy()
    
    for iter in range(high_level_iters):
        if (iter + 1) % 5 == 0:
            print(f'               Iteration {iter + 1} of {high_level_iters}')
        
        u, v = get_U_and_V(soft_segments, sp_pow) # The sparsity energy    
        sp_u = spdiags(u.reshape([-1]), 0, N, N)
        sp_v = spdiags(v.reshape([-1]), 0, N, N)

        # A = Laplacian + sigmaS * (spdiags(u, 0, N, N) + spdiags(v, 0, N, N)) + sigmaF * speye(N);
        A = laplacian + sigma_s * (sp_u + sp_v) + sigma_f * eye(N);    
        b = sigma_s * v + sigma_f * comp_init + delta        
        # [softSegments, ~] = pcg(A, b, [], itersBetweenUpdate, [], [], softSegments);
        soft_segments, _ = cg(A, b, x0=soft_segments, maxiter=iters_between_update)
        soft_segments = soft_segments.reshape(-1, 1)
    
         
    # One final iter for good times (everything but sparsity)
    A = laplacian + sigma_f * eye(N)
    b = sigma_f * soft_segments + delta
    soft_segments, _ = cg(A, b, x0=soft_segments, maxiter=10*iters_between_update)
    soft_segments = soft_segments.reshape(-1, 1)
 
    # Ta-dah
    soft_segments = soft_segments.reshape([comp_cnt, w, h]).transpose((2, 1, 0))    
    return soft_segments
    
    
