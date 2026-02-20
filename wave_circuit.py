from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import numpy as np
from functools import reduce

X = np.array([[0,1],[1,0]],dtype=complex)
Y = np.array([[0,-1j],[1j,0]],dtype=complex)
Z = np.array([[1,0],[0,-1]],dtype=complex)
H = np.array([[1,1],[1,-1]],dtype=complex)/np.sqrt(2)

#### Swap Operators ###
def int_to_bin(i, n):
    return [(i//2**_) % 2 for _ in range(n)][::-1]

def bin_to_int(b):
    return np.sum([v * 2 ** (len(b) - i - 1) for i,v in enumerate(b)])

def swap_circ(n):
    qc = QuantumCircuit(n)
    for i in range(n//2):
         qc.swap(i,n-i-1)
    return qc

def swap_matrix(n):
    N = 2**n 
    P = np.zeros((N, N))
    for i in range(N):
        P[bin_to_int(int_to_bin(i, n)[::-1]), i ] = 1
    return P

### QFT Circuits ###
def SQFT_circuit(n, swap = False):
    qc = QuantumCircuit(n)
    qc.x(0)
    for i in range(n):
        qc.h(i)
        for j in range(i+1,n):
            qc.cp(np.pi/(2**(j-i)),j,i)
    if swap: 
        for i in range(n//2):
            qc.swap(i,n-i-1)
        qc.x(0)
    return qc

### Discrete Laplacian (divided by n^2)
def L_P(n):
    out = np.zeros((n,n))
    for i in range(0,n):
        out[i,(i-1)%n ] = 1
        out[i,i ] = -2
        out[i,(i+1)%n ] = 1
    return out

### QFT Matrices ###
def SQFT_matrix(n):
    l = range(-n//2,n//2)
    Q = np.zeros((n,n),dtype=complex)
    omega =  np.exp(1j * 2 * np.pi/n)
    for i in range(n):
        for j in range(n):
            Q[i,j] = omega**(l[i]*l[j])/np.sqrt(n)
    return Q

### Hamiltonians and Spectra
def spectrum_P(n):
    return -4*np.cos(np.linspace(0,n-1,n)*np.pi/(n))**2

def linear_spectrum_P(n):
    return 2*(np.linspace(0,n-1,n)*np.pi/(n)) - np.pi

def linear_Ham_P(n):
    out = np.kron(H, SQFT_matrix(n))  @ np.kron(Z, np.diag(n * linear_spectrum_P(n))) @ np.kron(H, SQFT_matrix(n).conjugate().T)
    return out

def Ham_P(n):
    out = np.kron(H, SQFT_matrix(n))  @ np.kron(Z, np.diag(n * np.sqrt(-spectrum_P(n)))) @ np.kron(H, SQFT_matrix(n).conjugate().T)
    return out

def evolve_op(n, t):
    diag = np.kron([1,-1], n * spectrum_P(n))
    return np.kron(H, SQFT_matrix(n)) @ np.diag(np.exp(-1j * t * diag)) @ np.kron(H, SQFT_matrix(n).conjugate().T)

def linear_evolve_op(n, t):
    diag = np.kron([1,-1], n * linear_spectrum_P(n))
    return np.kron(H, SQFT_matrix(n)) @ np.diag(np.exp(-1j * t * diag)) @ np.kron(H, SQFT_matrix(n).conjugate().T)

##################################################################################################################

def linear_evo_cir(n, t, swap = False):
    qc = QuantumCircuit(n+1)
    for i in range(-1,n):
        qc.p(-np.pi*t*(2**(n-i)),i+1)
    for i in range(0,n):
        qc.cp(np.pi*t*(2**(n-i+1)),0,i+1)
    return qc

######## Full Circuits ###########
def simulation_circuit(n, swap = False):
    t = Parameter('t')
    qc = QuantumCircuit(n+1)
    if(swap):
        qc.h(0)
        qc = qc.compose(SQFT_circuit(n, swap=True).inverse(), range(1, n+1))
        qc = qc.compose(linear_evo_cir(n, t), range(0, n+1))
        qc = qc.compose(SQFT_circuit(n, swap=True), range(1, n+1))
        qc.h(0)
    else:
        qc.h(0)
        qc.x(1)
        qc = qc.compose(SQFT_circuit(n).inverse(), range(1, n+1)[::-1])
        qc = qc.compose(linear_evo_cir(n, t), [0] + list(range(1, n+1)[::-1]))
        qc = qc.compose(SQFT_circuit(n), range(1, n+1)[::-1])
        qc.x(1)
        qc.h(0)
    return qc

def energy_circuit(n, swap = False):
    t = Parameter('t')
    qc = QuantumCircuit(n+1)
    if(swap):
        qc.h(0)
        qc = qc.compose(SQFT_circuit(n, swap=True).inverse(), range(1, n+1))
        qc = qc.compose(linear_evo_cir(n, t), range(0, n+1))
        qc.h(0)
    else:
        qc.h(0)
        qc.x(1)
        qc = qc.compose(SQFT_circuit(n).inverse(), range(1, n+1)[::-1])
        qc = qc.compose(linear_evo_cir(n, t), [0] + list(range(1, n+1)[::-1]))
        qc.h(0)
    return qc

###### Energy Sampling #############
def energy_expectation(state, swap = False):
    N = len(state)//2 
    n = int(np.log2(N))
    U = 0 
    T = 0
    d = spectrum_P(N)
    
    for idx in range(N):
        if not swap:
            u_i, v_i = state[idx], state[idx+N]
            U += u_i.conjugate() * d[idx] * u_i
            T += v_i.conjugate() * d[idx] * v_i
        else:
            new_idx = bin_to_int(int_to_bin(idx, n)[::-1])
            u_i, v_i = state[new_idx], state[new_idx+N]
            U += u_i.conjugate() * d[idx] * u_i
            T += v_i.conjugate() * d[idx] * v_i

    return U * (-N), T * (-N)

###################################################################################################################
#### Layers -> Circuits ###
M = 1/np.sqrt(2) * np.array([
    [1,0,0,1j],
    [0,1j,1,0],
    [0,1j,-1,0],
    [1,0,0,-1j]
])
Gamma =  1/4 * np.array([
        [1,1,1,1],
        [1,1,-1,-1],
        [-1,1,-1,1],
        [1,-1,-1,1],
    ])
CCZ = np.diag([1,1,1,-1])

PAULIS = {
    'I':np.array(np.eye(2),dtype=complex),
    'X':np.array([[0,1],[1,0]],dtype=complex),
    'Y':np.array([[0,-1j],[1j,0]],dtype=complex),
    'Z':np.array([[1,0],[0,-1]],dtype=complex),
}

II = np.kron(PAULIS['I'],PAULIS['I'])
IZ = np.kron(PAULIS['I'],PAULIS['Z'])
ZI = np.kron(PAULIS['Z'],PAULIS['I'])
ZZ = np.kron(PAULIS['Z'],PAULIS['Z'])

XX = np.kron(PAULIS['X'],PAULIS['X'])
YY = np.kron(PAULIS['Y'],PAULIS['Y'])
ZZ = np.kron(PAULIS['Z'],PAULIS['Z'])


def decompose_U4(O):

    O_new =  M.conjugate().T @ O @ M

    R,I = O_new.real, O_new.imag

    QL, _, QR = np.linalg.svd(R)
    QR = QR.T 

    D = QL.T @ O_new @ QR

    if np.linalg.det(QL) <0:
        QL = QL @ CCZ
        D = CCZ @ D 

    if np.linalg.det(QR) <0:
        QR = QR @ CCZ
        D = D @ CCZ  

    theta = np.angle(np.diag(D)) % (2 * np.pi)
    k = (Gamma @ theta) % (2 * np.pi)

    A =  M @ QL @ M.conjugate().T
    B =  M@ QR.T @ M.conjugate().T

    A0, _, A1  = np.linalg.svd(np.reshape(np.reshape(A,(2,2,2,2)).transpose((0,2,1,3)), (4,4)), full_matrices=False)
    A0 = np.reshape(A0[:,0],(2,2))*np.sqrt(2)
    A1 = np.reshape(A1[0],(2,2))*np.sqrt(2)

    B0, _, B1  = np.linalg.svd(np.reshape(np.reshape(B,(2,2,2,2)).transpose((0,2,1,3)), (4,4)), full_matrices=False)
    B0 = np.reshape(B0[:,0],(2,2))*np.sqrt(2)
    B1 = np.reshape(B1[0],(2,2))*np.sqrt(2)

    return (A0,A1, B0, B1,k)

def R_XYZ_circ(h):
    alpha, beta, gamma = h
    qc = QuantumCircuit(2)
    qc.rz(-np.pi/2,1)
    qc.cx(1,0)
    qc.rz(-2 * gamma + np.pi/2,0)
    qc.ry(-np.pi/2 + 2 * alpha,1)
    qc.cx(0,1)
    qc.ry(-2 * beta+ np.pi/2,1)
    qc.cx(1,0)
    qc.rz(np.pi/2,0)
    return qc

def SU2_ZYZ(op):
    alpha = 0
    beta = 0 
    theta = 0
    if np.isclose(abs(op[1,0]),0):
        beta = -np.angle(op[0,0])*2
    if np.isclose(abs(op[0,0]),0):
        theta = np.pi 
        alpha = 2 * np.angle(op[1,0])
    else:
        alpha = (np.angle(op[1,1]/op[0,0]) + np.angle(-op[1,0]/op[0,1]))/2
        beta = (np.angle(op[1,1]/op[0,0]) - np.angle(-op[1,0]/op[0,1]))/2
        phases  = np.array([
            [np.exp(-1j * (alpha + beta)/2), -np.exp(-1j * (alpha - beta)/2)],
            [np.exp(1j * (alpha - beta)/2), np.exp(1j * (alpha + beta)/2)]
        ])

        op_phased = op/phases 
        phi = np.angle(op_phased[0,0])
        theta = 2 * np.atan2((op_phased/np.exp(1j * phi))[1,0].real,(op_phased/np.exp(1j * phi))[0,0].real)
        
    qc = QuantumCircuit(1)
    qc.rz(beta,0)
    qc.ry(theta,0)
    qc.rz(alpha,0)
    return qc

def U4_circ(O):
    (A0,A1, B0, B1,k) = decompose_U4(O)
    qc = QuantumCircuit(2)
    qc.append(SU2_ZYZ(B0),[0])
    qc.append(SU2_ZYZ(B1),[1])
    qc.append(R_XYZ_circ(k[1:]),[0,1])
    qc.append(SU2_ZYZ(A0),[0])
    qc.append(SU2_ZYZ(A1),[1])
    return qc.decompose()

def get_circ_from_layer(L):
    n = len(L)
    qc = QuantumCircuit(n)
    for i,op in enumerate(L[:-1]):
        qc.append(U4_circ(op),[n-i-2, n-i-1])
    qc.append(SU2_ZYZ(L[-1]),[0])
    return qc.decompose()

def convert_layers_to_circuit(L):
    n = len(L[0])
    qc = QuantumCircuit(n)
    for l in L:
        qc.append(get_circ_from_layer(l), range(n))
    return qc.decompose()

##################################################################################################################
from qiskit.converters import circuit_to_dag, dag_to_circuit
def collect_single_qubit_runs(dag, qubit):
    runs = []
    current = []

    for node in dag.nodes_on_wire(qubit, only_ops=True):
        if node.op.num_qubits == 1:
            current.append(node)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs

def reduce_circ(qc):
    dag = circuit_to_dag(qc)
    for q in dag.qubits:
        runs = collect_single_qubit_runs(dag, q)
        
        for run in runs:
            if len(run) <= 3:
                continue
        
            U = reduce(lambda x,y: x@ y, [node.op.to_matrix() for node in run][::-1])
            U_su2 = U/np.sqrt(np.linalg.det(U))
            qc = SU2_ZYZ(U_su2)
            subdag = circuit_to_dag(qc)

            # 4. Substitute
            anchor = run[0]
            dag.substitute_node_with_dag(
                anchor,
                subdag,
                wires={subdag.qubits[0]: q}
            )

            # 5. Remove the rest
            for node in run[1:]:
                dag.remove_op_node(node)
    return dag_to_circuit(dag)

def replace_phase(circ):
    dag = circuit_to_dag(circ)
    for op in dag.op_nodes():
        if op.name == 'cp':            
            t = op.params[0]
            qc_new = QuantumCircuit(2)
            qc_new.rz(t/2, 1)
            qc_new.cx(0,1)
            qc_new.rz(t/2, 0)
            qc_new.rz(-t/2, 1)
            qc_new.cx(0,1)
            
            subdag = circuit_to_dag(qc_new)
            dag.substitute_node_with_dag(
                    op,
                    subdag,
                    wires={subdag.qubits[0]: op.qargs[0],subdag.qubits[1]: op.qargs[1]}
                )
        elif op.name == 'p':
            t = op.params[0]
            qc_new = QuantumCircuit(1)
            qc_new.rz(t, 0)
            subdag = circuit_to_dag(qc_new)
            dag.substitute_node_with_dag(
                    op,
                    subdag,
                    wires={subdag.qubits[0]: op.qargs[0]}
                )

    return dag_to_circuit(dag)

def collect_rzs(dag, qubit):
    runs = []
    current = []

    for node in dag.nodes_on_wire(qubit, only_ops=True):
        if node.name == 'rz':
            current.append(node)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs

def reduce_rzs(qc):
    dag = circuit_to_dag(qc)
    for q in dag.qubits:
        runs = collect_rzs(dag, q)
        
        for run in runs:
            t = [node.params[0] for node in run]
            t = np.sum(t)

            qc = QuantumCircuit(1)
            if not (isinstance(t, float) and np.isclose(t,0)):
                qc.rz(t, 0)
            subdag =  circuit_to_dag(qc)
            anchor = run[0]
            dag.substitute_node_with_dag(
                anchor,
                subdag,
                wires={subdag.qubits[0]: q}
            )

            # 5. Remove the rest
            for node in run[1:]:
                dag.remove_op_node(node)
    return dag_to_circuit(dag)

from qiskit import transpile

def full_reduce(circ):
    tqc = transpile(
        reduce_rzs(reduce_circ(circ)),
        basis_gates=["rz", "ry", "rx", "cx"],
        optimization_level=3
    )
    return tqc

# def decompose_O2(op):
#     det = np.linalg.det(op)
#     theta = 2 * np.atan2(op[1,0], op[0,0])
#     qc = QuantumCircuit(1)
#     if(det < 0):
#         qc.z(0)
#     qc.ry(theta, 0)
#     return qc

# magic_circ = QuantumCircuit(2)
# magic_circ.s(0)
# magic_circ.s(1)
# magic_circ.h(1)
# magic_circ.cx(1,0)
# magic_circ.draw()

# CNOT = np.array([
#  [1,0,0,0],
#  [0,1,0,0],
#  [0,0,0,1],
#  [0,0,1,0]   
# ])

# M = 1/np.sqrt(2) * np.array([
#     [1, 1j, 0, 0], 
#     [0, 0, 1j, 1],
#     [0, 0, 1j, -1],
#     [1, -1j, 0, 0], 
# ])

# def SU2_ZYZ(op):
#     alpha = (np.angle(op[1,1]/op[0,0]) + np.angle(-op[1,0]/op[0,1]))/2
#     beta = (np.angle(op[1,1]/op[0,0]) - np.angle(-op[1,0]/op[0,1]))/2
#     phases  = np.array([
#         [np.exp(-1j * (alpha + beta)/2), -np.exp(-1j * (alpha - beta)/2)],
#         [np.exp(1j * (alpha - beta)/2), np.exp(1j * (alpha + beta)/2)]
#     ])

#     op_phased = op/phases 
#     phi = np.angle(op_phased[0,0])
#     theta = 2 * np.atan2((op_phased/np.exp(1j * phi))[1,0].real,(op_phased/np.exp(1j * phi))[0,0].real)
#     return alpha, theta, beta

# def decompose_O4(op):
#     det = np.linalg.det(op)
#     new_op = op
#     if(det < 0):
#         new_op = CNOT @ op
#     basis_op = M @ new_op @ M.conjugate().T

#     B = (basis_op[:2, :2]).copy()
#     A = np.array(
#         [
#             [basis_op[0,0]/B[0,0], basis_op[0,2]/B[0,0]],
#             [basis_op[2,0]/B[0,0], basis_op[2,2]/B[0,0]]   
#         ]
#     )
#     f = np.linalg.det(A)
#     A = A/np.sqrt(f)
#     B = B * np.sqrt(f)
    
#     alpha_A, theta_A, beta_A = SU2_ZYZ(A)
#     alpha_B, theta_B, beta_B = SU2_ZYZ(B)
    
#     qc = QuantumCircuit(2)

#     qc = qc.compose(magic_circ, [0,1])

#     qc.rz(beta_A,0)
#     qc.ry(theta_A,0)
#     qc.rz(alpha_A,0)

#     qc.rz(beta_B,1)
#     qc.ry(theta_B,1)
#     qc.rz(alpha_B,1)

#     qc = qc.compose(magic_circ.inverse(), [0,1])
    
#     if(det < 0):
#         qc.cx(0,1)
#     return qc

# def convert_layer_to_circuit(L):
#     n = len(L)
#     qc = QuantumCircuit(n)
    
#     for i,g in enumerate(L[:-1]):
#         qc = qc.compose(decompose_O4(g), [n-i-2, n-i-1])
    
#     qc = qc.compose(decompose_O2(L[-1]), [0])
#     return qc

# def convert_layers_to_circuit(L):
#     qc = convert_layer_to_circuit(L[0])
#     for l in L[1:]:
#         qc = qc.compose(convert_layer_to_circuit(l))
#     return qc


# def QFT_circuit(n):
#     qc = QuantumCircuit(n)
#     for i in range(n):
#         qc.h(i)
#         for j in range(i+1,n):
#             qc.cp(np.pi/(2**(j-i)),j,i)
#     for i in range(n//2):
#         qc.swap(i,n-i-1)
#     return qc

# def IQFT_circuit(n):
#     qc = QuantumCircuit(n)
#     for i in range(n):
#         qc.h(i)
#         for j in range(i+1,n):
#             qc.cp(-np.pi/(2**(j-i)),j,i)
#     for i in range(n//2):
#         qc.swap(i,n-i-1)
#     return qc


# def P_circ(n):
#     qc = QuantumCircuit(n)
#     qc.z(0)

#     for i in range(n):
#         qc.p(-np.pi/(2**(i+1)) ,i)


#     for i in range(1,n):
#         qc.cx(0,i)
        
#     for i in range(1,n)[::-1]:
#         qc = qc.compose(mcx_decomp(i+1),[0]+list(range(n-i,n))[::-1])

#     qc.h(0)
#     qc.x(range(1,n))
#     qc.ry(np.pi/4,0)
#     qc = qc.compose(mcx_decomp(n),list(range(1,n))+[0])
#     qc.ry(-np.pi/4,0)
#     qc.x(range(1,n))
#     return qc

# def Q_circ(n):
#     qc = QuantumCircuit(n)
#     qc.h(0)
#     for i in range(1,n):
#         qc.cx(0,i)
#     return qc


# ###C.T
# def PFQ_circ(n):
#     qc = QuantumCircuit(n+1)
#     qc = qc.compose(Q_circ(n+1),range(n+1))
#     qc = qc.compose(IQFT_circuit(n+1),range(n+1))
#     qc = qc.compose(P_circ(n+1),range(n+1))
#     return qc

# ###C
# def IPFQ_circ(n):
#     qc = QuantumCircuit(n+1)
#     qc = qc.compose(P_circ(n+1).inverse(),range(n+1))
#     qc = qc.compose(QFT_circuit(n+1),range(n+1))
#     qc = qc.compose(Q_circ(n+1).inverse(),range(n+1))
#     return qc

# def evolve_circ(n,t):
#     qc = QuantumCircuit(n+2)
#     qc = qc.compose(PFQ_circ(n),[0] + list(range(2,n+2)))
#     qc.h(1)
#     for i in range(0,n):
#         qc.p(-np.pi*t*(2**(n-i-1)),i+2)
#     for i in range(0,n):
#         qc.cp(np.pi*t*(2**(n-i)),1,i+2)
#     qc.h(1)
#     qc = qc.compose(IPFQ_circ(n),[0] + list(range(2,n+2)))
#     return qc
# ############################### Matrix Methods ##################################


# def QFT_matrix(n):
#     Q = np.zeros((n,n),dtype=complex)
#     omega =  np.exp(1j * 2 * np.pi/n)
#     for i in range(n):
#         for j in range(n):
#             Q[i,j] = omega**(i*j)/np.sqrt(n)
#     return Q

# def IQFT_matrix(n):
#     Q = np.zeros((n,n),dtype=complex)
#     omega =  np.exp(-1j * 2 * np.pi/n)
#     for i in range(n):
#         for j in range(n):
#             Q[i,j] = omega**(i*j)/np.sqrt(n)
#     return Q

# def P(n):
#     w = np.exp(np.pi * 1j/(2*n) )
#     out = np.zeros((2*n,n),dtype=complex)
#     out[0,0] = 1
#     for i in range(1, n):
#         out[i,i] = w**i/np.sqrt(2)
#         out[2*n-i,i] = w.conjugate()**i/np.sqrt(2)
#     return out

# def Q(n):
#     out = np.zeros((2*n,n),dtype=complex)
#     for i in range(n):
#         out[i,i] = 1/np.sqrt(2)
#         out[2*n-i-1,i] = 1/np.sqrt(2)
#     return out

# def C(n):
#     out = np.zeros((n,n))
#     for i in range(n):
#         for j in range(n):
#             out[i,j] =  np.cos((i+1/2) * (j) * np.pi/(n)) * np.sqrt(2/n)
#             if(j==0):
#                 out[i,j] *=np.sqrt(1/2)
#     return out


# def spectrum_N(n):
#     return -4*np.sin(np.linspace(0,n-1,n)*np.pi/(2*n))**2


# def linear_spectrum_N(n):
#     return (np.linspace(0,n-1,n)*np.pi/(n))

# def linear_Ham_N(n,L=1):
#     out = np.kron(H, C(n))  @ np.kron(Z, np.diag(n/L * linear_spectrum_N(n))) @ np.kron(H, C(n).T)
#     return out


# def Ham_N(n,L=1):
#     out = np.kron(H, C(n))  @ np.kron(Z, np.diag(n/L * np.sqrt(-spectrum_N(n)))) @ np.kron(H, C(n).T)
#     return out

# def L_N(n):
#     out = np.zeros((n,n))
#     out[0,0] = -1
#     out[0,1] = 1
#     for i in range(1,n-1):
#         out[i,i-1 ] = 1
#         out[i,i ] = -2
#         out[i,i+1 ] = 1
    
#     out[n-1,n-1] = -1
#     out[n-1,n-2] = 1
#     return out

# def triag_circ(n):
#     qc = QuantumCircuit(n)
#     for i in range(1,n):
#         for j in range(i):
#             if(-i+j != -1):
#                 qc.crx(np.pi/2**(j+1-int(i+1 == n)),n-i-1,n-i+j)
#             else:
#                 qc.cp(np.pi/2**(j+1-int(i+1 == n)),n-i-1,n-i+j)
#     for i in range(1,n-1)[::-1]:
#         for j in range(i):
#             if(-i+j != -1):
#                 qc.crx(-np.pi/2**(j+1),n-i-1,n-i+j)
#             else:
#                 qc.cp(-np.pi/2**(j+1),n-i-1,n-i+j)

#     return qc

# def minus_triag_circ(n):
#     qc = QuantumCircuit(n)
#     for i in range(1,n):
#         for j in range(i):
#             qc.crx(np.pi/2**(j+1-int(i+1 == n)) * (-1)**int(i+1 == n),n-i-1,n-i+j)
#     for i in range(1,n-1)[::-1]:
#         for j in range(i):
#             qc.crx(-np.pi/2**(j+1),n-i-1,n-i+j)
#     return qc


# def mcx_decomp(n):
#     qc = QuantumCircuit(n)
#     qc.h(-1)
#     qc = qc.compose(triag_circ(n),range(n))
#     qc = qc.compose(minus_triag_circ(n-1),range(n-1))
#     qc.h(-1)
#     return qc

#### Periodic Circuit###
# def SQFT_circuit(n):
#     qc = QuantumCircuit(n)
#     qc.x(0)
#     for i in range(n):
#         qc.h(i)
#         for j in range(i+1,n):
#             qc.cp(np.pi/(2**(j-i)),j,i)
#     for i in range(n//2):
#         qc.swap(i,n-i-1)
#     qc.x(0)
#     return qc

# def SIQFT_circuit(n):
#     qc = QuantumCircuit(n)
#     qc.x(0)
#     for i in range(n):
#         qc.h(i)
#         for j in range(i+1,n):
#             qc.cp(-np.pi/(2**(j-i)),j,i)
#     for i in range(n//2):
#         qc.swap(i,n-i-1)
#     qc.x(0)
#     return qc

# def SQFT_matrix(n):
#     l = range(-n//2,n//2)
#     Q = np.zeros((n,n),dtype=complex)
#     omega =  np.exp(1j * 2 * np.pi/n)
#     for i in range(n):
#         for j in range(n):
#             Q[i,j] = omega**(l[i]*l[j])/np.sqrt(n)
#     return Q

# def SIQFT_matrix(n):
#     l = range(-n//2,n//2)
#     Q = np.zeros((n,n),dtype=complex)
#     omega =  np.exp(-1j * 2 * np.pi/n)
#     for i in range(n):
#         for j in range(n):
#             Q[i,j] = omega**(l[i]*l[j])/np.sqrt(n)
#     return Q

# def L_P(n):
#     out = np.zeros((n,n))
#     for i in range(0,n):
#         out[i,(i-1)%n ] = 1
#         out[i,i ] = -2
#         out[i,(i+1)%n ] = 1
#     return out

# def spectrum_P(n):
#     return -4*np.cos(np.linspace(0,n-1,n)*np.pi/(n))**2

# def linear_spectrum_P(n):
#     return 2*(np.linspace(0,n-1,n)*np.pi/(n)) - np.pi

# def linear_Ham_P(n,L=1):
#     out = np.kron(H, SQFT_matrix(n))  @ np.kron(Z, np.diag(n/L * linear_spectrum_P(n))) @ np.kron(H, SIQFT_matrix(n).T)
#     return out


# def Ham_P(n,L=1):
#     out = np.kron(H, SQFT_matrix(n))  @ np.kron(Z, np.diag(n/L * np.sqrt(-spectrum_P(n)))) @ np.kron(H, SIQFT_matrix(n).T)
#     return out

# def evolve_circ_P(n,t):
#     qc = QuantumCircuit(n+1)
#     qc.h(0)
#     qc = qc.compose(SIQFT_circuit(n),range(1,n+1))
#     for i in range(-1,n):
#         qc.p(-np.pi*t*(2**(n-i)),i+1)
#     for i in range(0,n):
#         qc.cp(np.pi*t*(2**(n-i+1)),0,i+1)
#     qc.h(0)
#     qc = qc.compose(SQFT_circuit(n),range(1,n+1))
#     return qc

# #### Periodic Methods
# def SQFT_circuit(n):
#     qc = QuantumCircuit(n)
#     qc.x(0)
#     for i in range(n):
#         qc.h(i)
#         for j in range(i+1,n):
#             qc.cp(np.pi/(2**(j-i)),j,i)
#     for i in range(n//2):
#         qc.swap(i,n-i-1)
#     qc.x(0)
#     return qc

# def SIQFT_circuit(n):
#     qc = QuantumCircuit(n)
#     qc.x(0)
#     for i in range(n):
#         qc.h(i)
#         for j in range(i+1,n):
#             qc.cp(-np.pi/(2**(j-i)),j,i)
#     for i in range(n//2):
#         qc.swap(i,n-i-1)
#     qc.x(0)
#     return qc

# def SQFT_matrix(n):
#     l = range(-n//2,n//2)
#     Q = np.zeros((n,n),dtype=complex)
#     omega =  np.exp(1j * 2 * np.pi/n)
#     for i in range(n):
#         for j in range(n):
#             Q[i,j] = omega**(l[i]*l[j])/np.sqrt(n)
#     return Q

# def SIQFT_matrix(n):
#     l = range(-n//2,n//2)
#     Q = np.zeros((n,n),dtype=complex)
#     omega =  np.exp(-1j * 2 * np.pi/n)
#     for i in range(n):
#         for j in range(n):
#             Q[i,j] = omega**(l[i]*l[j])/np.sqrt(n)
#     return Q

# def L_P(n):
#     out = np.zeros((n,n))
#     for i in range(0,n):
#         out[i,(i-1)%n ] = 1
#         out[i,i ] = -2
#         out[i,(i+1)%n ] = 1
#     return out

# def spectrum_P(n):
#     return -4*np.cos(np.linspace(0,n-1,n)*np.pi/(n))**2

# def linear_spectrum_P(n):
#     return 2*(np.linspace(0,n-1,n)*np.pi/(n)) - np.pi

# def linear_Ham_P(n,L=1):
#     out = np.kron(H, SQFT_matrix(n))  @ np.kron(Z, np.diag(n/L * linear_spectrum_P(n))) @ np.kron(H, SIQFT_matrix(n).T)
#     return out


# def Ham_P(n,L=1):
#     out = np.kron(H, SQFT_matrix(n))  @ np.kron(Z, np.sign(linear_spectrum_P(n))*np.diag(n/L * np.sqrt(-spectrum_P(n)))) @ np.kron(H, SIQFT_matrix(n).T)
#     return out

# def evolve_circ_P(n,t):
#     qc = QuantumCircuit(n+1)
#     qc.h(0)
#     qc = qc.compose(SIQFT_circuit(n),range(1,n+1))
#     for i in range(-1,n):
#         qc.p(-np.pi*t*(2**(n-i)),i+1)
#     for i in range(0,n):
#         qc.cp(np.pi*t*(2**(n-i+1)),0,i+1)
#     qc.h(0)
#     qc = qc.compose(SQFT_circuit(n),range(1,n+1))
#     return qc