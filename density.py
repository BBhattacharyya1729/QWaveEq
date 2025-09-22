import numpy as np
from numba import njit
from qiskit import QuantumCircuit

# -----------------------
# Pauli and basic gates
# -----------------------
# Predefined 2×2 unitary matrices for common gates
I2 = np.eye(2, dtype=np.complex128)                     # Identity
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)     # Pauli-X
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)  # Pauli-Y
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)    # Pauli-Z
H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)  # Hadamard

# -----------------------
# Density initialization
# -----------------------
@njit
def random_density(n_qubits):
    """
    Generate a random density matrix for an n-qubit system.

    Args:
        n_qubits (int): Number of qubits.

    Returns:
        rho (np.ndarray): A (2^n, 2^n) positive semidefinite density matrix
                          with trace = 1.
    """
    dim = 1 << n_qubits
    A = (np.random.randn(dim, dim) + 1j*np.random.randn(dim, dim))/np.sqrt(2)
    rho = A @ A.conj().T
    rho /= np.trace(rho)
    return rho

# -----------------------
# Single-qubit unitary in-place
# -----------------------
@njit(fastmath=True, cache=True)
def apply_unitary_to_qubit_dm_inplace(rho, U, qubit_index, n_qubits):
    """
    Apply a single-qubit unitary operator to a density matrix in place.

    Args:
        rho (np.ndarray): Density matrix (2^n x 2^n).
        U (np.ndarray): 2x2 unitary matrix.
        qubit_index (int): Target qubit index (0 = leftmost/MSB).
        n_qubits (int): Total number of qubits.

    Notes:
        This directly updates `rho` without allocating a new matrix.
    """
    dim = 1 << n_qubits
    mask = 1 << (n_qubits - 1 - qubit_index)
    half_dim = dim >> 1

    zero_indices = np.empty(half_dim, np.int64)
    one_indices  = np.empty(half_dim, np.int64)
    zi = oi = 0
    for i in range(dim):
        if (i & mask) == 0:
            zero_indices[zi] = i; zi += 1
        else:
            one_indices[oi] = i; oi += 1

    U00, U01, U10, U11 = U[0,0], U[0,1], U[1,0], U[1,1]
    Uc00, Uc01, Uc10, Uc11 = np.conj(U[0,0]), np.conj(U[0,1]), np.conj(U[1,0]), np.conj(U[1,1])

    tmp = np.empty((2,2), dtype=np.complex128)

    for a in range(half_dim):
        i0 = zero_indices[a]
        i1 = one_indices[a]
        for b in range(half_dim):
            j0 = zero_indices[b]
            j1 = one_indices[b]

            r00, r01 = rho[i0,j0], rho[i0,j1]
            r10, r11 = rho[i1,j0], rho[i1,j1]

            tmp[0,0] = U00*r00 + U01*r10
            tmp[0,1] = U00*r01 + U01*r11
            tmp[1,0] = U10*r00 + U11*r10
            tmp[1,1] = U10*r01 + U11*r11

            rho[i0,j0] = tmp[0,0]*Uc00 + tmp[0,1]*Uc01
            rho[i0,j1] = tmp[0,0]*Uc10 + tmp[0,1]*Uc11
            rho[i1,j0] = tmp[1,0]*Uc00 + tmp[1,1]*Uc01
            rho[i1,j1] = tmp[1,0]*Uc10 + tmp[1,1]*Uc11


# -----------------------
# Single-qubit depolarizing (in-place, fast & exact)
# -----------------------
@njit(fastmath=True, cache=True)
def apply_single_qubit_depolarizing_inplace(rho, lam, qubit_index, n_qubits):
    """
    Apply exact single-qubit depolarizing noise to a density matrix.

    Implements:
        rho <- (1 - lam) * rho + lam * ( I/2 ⊗ Tr_q(rho) )

    Args:
        rho (np.ndarray): Density matrix (modified in-place).
        lam (float): Noise parameter, 0 ≤ lam ≤ 1.
        qubit_index (int): Target qubit index.
        n_qubits (int): Total qubits.
    """
    if lam == 0.0:
        return
    dim = 1 << n_qubits
    mask = 1 << (n_qubits - 1 - qubit_index)
    tmp = rho.copy()
    one_minus = 1.0 - lam
    inv2 = 0.5

    for i in range(dim):
        i_other = i & (~mask)
        i_bit = (i & mask) != 0
        for j in range(dim):
            j_other = j & (~mask)
            j_bit = (j & mask) != 0

            # mixed term only contributes when target-qubit basis indices match
            if i_bit == j_bit:
                mixed = inv2 * (tmp[i_other, j_other] + tmp[i_other | mask, j_other | mask])
            else:
                mixed = 0.0 + 0.0j

            rho[i, j] = one_minus * tmp[i, j] + lam * mixed

# -----------------------
# Two-qubit depolarizing (in-place, fast & exact)
# -----------------------
@njit(fastmath=True, cache=True)
def apply_two_qubit_depolarizing_inplace(rho, lam, q1, q2, n_qubits):
    """
    Apply exact two-qubit depolarizing noise.

    Implements:
        rho <- (1 - lam) * rho + lam * ( I/4 ⊗ Tr_{q1,q2}(rho) )

    Args:
        rho (np.ndarray): Density matrix.
        lam (float): Noise parameter.
        q1, q2 (int): Qubit indices.
        n_qubits (int): Total number of qubits.
    """
    if lam == 0.0:
        return
    if q1 == q2:
        # fall back to single-qubit formula
        apply_single_qubit_depolarizing_inplace(rho, lam, q1, n_qubits)
        return

    dim = 1 << n_qubits
    mask1 = 1 << (n_qubits - 1 - q1)
    mask2 = 1 << (n_qubits - 1 - q2)
    both_masks = mask1 | mask2
    tmp = rho.copy()
    one_minus = 1.0 - lam
    inv4 = 0.25

    for i in range(dim):
        others_i = i & (~both_masks)
        for j in range(dim):
            others_j = j & (~both_masks)
            # mixed term only when the two-qubit basis indices for row and col match
            if (i & both_masks) == (j & both_masks):
                mixed = (tmp[others_i, others_j]
                         + tmp[others_i | mask2, others_j | mask2]
                         + tmp[others_i | mask1, others_j | mask1]
                         + tmp[others_i | mask1 | mask2, others_j | mask1 | mask2]) * inv4
            else:
                mixed = 0.0 + 0.0j

            rho[i, j] = one_minus * tmp[i, j] + lam * mixed


# -----------------------
# CX gate (CNOT) in-place
# -----------------------
@njit(fastmath=True, cache=True)
def apply_cx_dm_inplace(rho, control, target, n_qubits):
    """
    Apply a controlled-X (CNOT) gate to a density matrix in place.

    Args:
        rho (np.ndarray): Density matrix (2^n x 2^n).
        control (int): Control qubit index.
        target (int): Target qubit index.
        n_qubits (int): Total number of qubits.
    """
    dim = 1 << n_qubits
    c_mask = 1 << (n_qubits - 1 - control)
    t_mask = 1 << (n_qubits - 1 - target)
    tmp = rho.copy()
    for i in range(dim):
        for j in range(dim):
            i2 = i ^ (t_mask if (i & c_mask) else 0)
            j2 = j ^ (t_mask if (j & c_mask) else 0)
            rho[i,j] = tmp[i2,j2]

# -----------------------
# CP gate (controlled phase) in-place
# -----------------------
@njit(fastmath=True, cache=True)
def apply_cp_dm_inplace(rho, control, target, n_qubits, theta):
    """
    Apply a controlled-phase gate in place.

    Args:
        rho (np.ndarray): Density matrix.
        control (int): Control qubit index.
        target (int): Target qubit index.
        n_qubits (int): Number of qubits.
        theta (float): Phase angle in radians.
    """
    if theta == 0: return
    dim = 1 << n_qubits
    c_mask = 1 << (n_qubits - 1 - control)
    t_mask = 1 << (n_qubits - 1 - target)
    phase = np.exp(1j*theta)
    phase_conj = np.exp(-1j*theta)
    for i in range(dim):
        i_ct = (i & (c_mask|t_mask)) == (c_mask|t_mask)
        for j in range(dim):
            j_ct = (j & (c_mask|t_mask)) == (c_mask|t_mask)
            if i_ct and not j_ct: rho[i,j] *= phase
            elif j_ct and not i_ct: rho[i,j] *= phase_conj

# -----------------------
# SWAP in-place
# -----------------------
@njit(fastmath=True, cache=True)
def apply_swap_dm_inplace(rho, q1, q2, n_qubits):
    """
    Apply a SWAP gate between two qubits.

    Args:
        rho (np.ndarray): Density matrix.
        q1, q2 (int): Qubit indices to swap.
        n_qubits (int): Total number of qubits.
    """
    if q1 == q2:
        return
    dim = 1 << n_qubits
    q1_mask = 1 << (n_qubits - 1 - q1)
    q2_mask = 1 << (n_qubits - 1 - q2)

    tmp = rho.copy()
    for i in range(dim):
        # permute row index
        b1 = (i & q1_mask) >> (n_qubits - 1 - q1)
        b2 = (i & q2_mask) >> (n_qubits - 1 - q2)
        i2 = i
        if b1 != b2:
            i2 ^= q1_mask | q2_mask

        for j in range(dim):
            # permute column index
            b1j = (j & q1_mask) >> (n_qubits - 1 - q1)
            b2j = (j & q2_mask) >> (n_qubits - 1 - q2)
            j2 = j
            if b1j != b2j:
                j2 ^= q1_mask | q2_mask

            rho[i, j] = tmp[i2, j2]

# -----------------------
# Single-qubit gate wrappers
# -----------------------
# These are convenience functions that directly call the
# general-purpose unitary application function with fixed unitaries.

def apply_s_gate(rho, idx, n): apply_unitary_to_qubit_dm_inplace(rho, np.array([[1,0],[0,1j]], np.complex128), idx, n)
def apply_sdg_gate(rho, idx, n): apply_unitary_to_qubit_dm_inplace(rho, np.array([[1,0],[0,-1j]], np.complex128), idx, n)
def apply_x_gate(rho, idx, n): apply_unitary_to_qubit_dm_inplace(rho, X, idx, n)
def apply_z_gate(rho, idx, n): apply_unitary_to_qubit_dm_inplace(rho, Z, idx, n)
def apply_h_gate(rho, idx, n): apply_unitary_to_qubit_dm_inplace(rho, H, idx, n)
def apply_rz_gate(angle, rho, idx, n): apply_unitary_to_qubit_dm_inplace(rho, np.array([[np.exp(-1j*angle/2),0],[0,np.exp(1j*angle/2)]], np.complex128), idx, n)
def apply_ry_gate(angle, rho, idx, n):
    c,s = np.cos(angle/2), np.sin(angle/2)
    apply_unitary_to_qubit_dm_inplace(rho, np.array([[c,-s],[s,c]], np.complex128), idx, n)
def apply_p_gate(angle, rho, idx, n): apply_unitary_to_qubit_dm_inplace(rho, np.array([[1,0],[0,np.exp(1j*angle)]], np.complex128), idx, n)

# -----------------------
# Density matrix from circuit
# -----------------------
def get_density_matrix(circ, lam):
    """
    Simulate a Qiskit QuantumCircuit as a noisy density matrix evolution.

    Args:
        circ (QuantumCircuit): The circuit to simulate.
        lam (float): Depolarizing noise parameter (applied after each gate).

    Returns:
        rho (np.ndarray): Final density matrix (2^n x 2^n).
    """
    ops = get_operations(circ)
    n = circ.num_qubits
    rho = np.zeros((2**n,2**n), np.complex128)
    rho[0,0] = 1
    for name, params, idx_list in ops:
        if name=='s': 
            apply_s_gate(rho, idx_list[0], n)
            apply_single_qubit_depolarizing_inplace(rho, lam, idx_list[0], n)
        elif name=='h': 
            apply_h_gate(rho, idx_list[0], n)
            apply_single_qubit_depolarizing_inplace(rho, lam, idx_list[0], n)
        elif name=='cx': 
            apply_cx_dm_inplace(rho, idx_list[0], idx_list[1], n)
            apply_two_qubit_depolarizing_inplace(rho, lam, idx_list[0], idx_list[1], n)
        elif name=='rz':
            apply_rz_gate(params[0], rho, idx_list[0], n)
            apply_single_qubit_depolarizing_inplace(rho, lam, idx_list[0], n)
        elif name=='ry':
            apply_ry_gate(params[0], rho, idx_list[0], n)
            apply_single_qubit_depolarizing_inplace(rho, lam, idx_list[0], n)
        elif name=='sdg':
            apply_sdg_gate(rho, idx_list[0], n)
            apply_single_qubit_depolarizing_inplace(rho, lam, idx_list[0], n)
        elif name=='z':
            apply_z_gate(rho, idx_list[0], n)
            apply_single_qubit_depolarizing_inplace(rho, lam, idx_list[0], n)
        elif name=='x':
            apply_x_gate(rho, idx_list[0], n)
            apply_single_qubit_depolarizing_inplace(rho, lam, idx_list[0], n)
        elif name=='cp':
            apply_cp_dm_inplace(rho, idx_list[0], idx_list[1], n, params[0])
            apply_two_qubit_depolarizing_inplace(rho, lam, idx_list[0], idx_list[1], n)
        elif name=='p':
            apply_p_gate(params[0], rho, idx_list[0], n)
            apply_single_qubit_depolarizing_inplace(rho, lam, idx_list[0], n)
        elif name == 'swap':
            apply_swap_dm_inplace(rho, idx_list[0], idx_list[1], n)
            apply_two_qubit_depolarizing_inplace(rho, lam, idx_list[0], idx_list[1], n)
    return rho

# -------------------------------
# Circuit Tools
# -------------------------------
def get_operations(circ):
    """
    Extract a list of operations from a Qiskit circuit.

    Args:
        circ (QuantumCircuit): Circuit.

    Returns:
        list: Each element is [name, params, [qubit indices]].
    """
    ops = []
    for inst in circ.data:
        ops.append([inst.operation.name, inst.operation.params, [_._index for _ in inst.qubits]])
    return ops

def fold_circ(circ, fold_prob):
    """
    Stochastically "fold" a circuit by inserting gate-inverse pairs.

    Args:
        circ (QuantumCircuit): Input circuit.
        fold_prob (float): Probability of folding each gate.

    Returns:
        QuantumCircuit: New circuit with random folds applied.

    Notes:
        This is used for error mitigation via "probabilistic folding."
    """
    ops = get_operations(circ)
    new_circ = QuantumCircuit(circ.num_qubits)
    for name, params, idx_list in ops:
        if name == 's':
            new_circ.s(idx_list[0])
            if(np.random.random() < fold_prob):
                new_circ.sdg(idx_list[0])
                new_circ.s(idx_list[0])
        elif name == 'h':
            new_circ.h(idx_list[0])
            
            if(np.random.random() < fold_prob):
                new_circ.h(idx_list[0])
                new_circ.h(idx_list[0])
        elif name == 'cx':
            new_circ.cx(idx_list[0], idx_list[1])
            if(np.random.random() < fold_prob):
                new_circ.cx(idx_list[0], idx_list[1])
                new_circ.cx(idx_list[0], idx_list[1])
        elif name == 'rz':
            new_circ.rz(params[0], idx_list[0]) 
            if(np.random.random() < fold_prob):     
                new_circ.rz(-params[0], idx_list[0]) 
                new_circ.rz(params[0], idx_list[0])   
        elif name == 'ry':
            new_circ.ry(params[0], idx_list[0])
            if(np.random.random() < fold_prob): 
                new_circ.ry(-params[0], idx_list[0]) 
                new_circ.ry(params[0], idx_list[0]) 
        elif name == 'sdg':
            new_circ.sdg(idx_list[0])
            if(np.random.random() < fold_prob): 
                new_circ.s(idx_list[0])
                new_circ.sdg(idx_list[0])
        elif name == 'z':
            new_circ.z(idx_list[0])
            if(np.random.random() < fold_prob): 
                new_circ.z(idx_list[0])
                new_circ.z(idx_list[0])
        elif name == 'x':
            new_circ.x(idx_list[0])
            if(np.random.random() < fold_prob): 
                new_circ.x(idx_list[0])
                new_circ.x(idx_list[0])
        elif name == 'cp':
            new_circ.cp(params[0], idx_list[0], idx_list[1])        
            if(np.random.random() < fold_prob):
                new_circ.cp(-params[0], idx_list[0], idx_list[1])    
                new_circ.cp(params[0], idx_list[0], idx_list[1])    
        elif name == 'p':
            new_circ.p(params[0], idx_list[0])    
            if(np.random.random() < fold_prob):
                new_circ.p(-params[0], idx_list[0])
                new_circ.p(params[0], idx_list[0])
    return new_circ


#### ZNE ####
def exponential_fit(x_data, y_data, N):
    """
    Extrapolate data with an exponential fit given the behavior at infinity

    Args:
        x_data (list[Float]): x data.
        y_data (list[Float]): y data.
        N (float or Int): x-> infinity limit

    Returns:
        Fitted exponential fit function

    """
    transformed_y_data = np.log(N - np.array(y_data))
    A = np.vstack([np.ones(len(x_data)), x_data]).T

    c1,c2 =np.linalg.inv(A.T @ A) @ A.T @ transformed_y_data
    
    return lambda x : N - np.exp(c1 + c2 * x)