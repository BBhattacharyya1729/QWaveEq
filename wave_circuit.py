from qiskit import QuantumCircuit
#from qiskit.quantum_info import Operator
#from qiskit.circuit.library import MCXGate
import numpy as np

def QFT_circuit(n):
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.h(i)
        for j in range(i+1,n):
            qc.cp(np.pi/(2**(j-i)),j,i)
    for i in range(n//2):
        qc.swap(i,n-i-1)
    return qc

def IQFT_circuit(n):
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.h(i)
        for j in range(i+1,n):
            qc.cp(-np.pi/(2**(j-i)),j,i)
    for i in range(n//2):
        qc.swap(i,n-i-1)
    return qc


def P_circ(n):
    qc = QuantumCircuit(n)
    qc.z(0)

    for i in range(n):
        qc.p(-np.pi/(2**(i+1)) ,i)


    for i in range(1,n):
        qc.cx(0,i)
        
    for i in range(1,n)[::-1]:
        qc = qc.compose(mcx_decomp(i+1),[0]+list(range(n-i,n))[::-1])

    qc.h(0)
    qc.x(range(1,n))
    qc.ry(np.pi/4,0)
    qc = qc.compose(mcx_decomp(n),list(range(1,n))+[0])
    qc.ry(-np.pi/4,0)
    qc.x(range(1,n))
    return qc

def Q_circ(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(1,n):
        qc.cx(0,i)
    return qc


###C.T
def PFQ_circ(n):
    qc = QuantumCircuit(n+1)
    qc = qc.compose(Q_circ(n+1),range(n+1))
    qc = qc.compose(IQFT_circuit(n+1),range(n+1))
    qc = qc.compose(P_circ(n+1),range(n+1))
    return qc

###C
def IPFQ_circ(n):
    qc = QuantumCircuit(n+1)
    qc = qc.compose(P_circ(n+1).inverse(),range(n+1))
    qc = qc.compose(QFT_circuit(n+1),range(n+1))
    qc = qc.compose(Q_circ(n+1).inverse(),range(n+1))
    return qc

def evolve_circ(n,t):
    qc = QuantumCircuit(n+2)
    qc = qc.compose(PFQ_circ(n),[0] + list(range(2,n+2)))
    qc.h(1)
    for i in range(0,n):
        qc.p(-np.pi*t*(2**(n-i-1)),i+2)
    for i in range(0,n):
        qc.cp(np.pi*t*(2**(n-i)),1,i+2)
    qc.h(1)
    qc = qc.compose(IPFQ_circ(n),[0] + list(range(2,n+2)))
    return qc
############################### Matrix Methods ##################################


def QFT_matrix(n):
    Q = np.zeros((n,n),dtype=complex)
    omega =  np.exp(1j * 2 * np.pi/n)
    for i in range(n):
        for j in range(n):
            Q[i,j] = omega**(i*j)/np.sqrt(n)
    return Q

def IQFT_matrix(n):
    Q = np.zeros((n,n),dtype=complex)
    omega =  np.exp(-1j * 2 * np.pi/n)
    for i in range(n):
        for j in range(n):
            Q[i,j] = omega**(i*j)/np.sqrt(n)
    return Q

def P(n):
    w = np.exp(np.pi * 1j/(2*n) )
    out = np.zeros((2*n,n),dtype=complex)
    out[0,0] = 1
    for i in range(1, n):
        out[i,i] = w**i/np.sqrt(2)
        out[2*n-i,i] = w.conjugate()**i/np.sqrt(2)
    return out

def Q(n):
    out = np.zeros((2*n,n),dtype=complex)
    for i in range(n):
        out[i,i] = 1/np.sqrt(2)
        out[2*n-i-1,i] = 1/np.sqrt(2)
    return out

def C(n):
    out = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            out[i,j] =  np.cos((i+1/2) * (j) * np.pi/(n)) * np.sqrt(2/n)
            if(j==0):
                out[i,j] *=np.sqrt(1/2)
    return out


def spectrum_N(n):
    return -4*np.sin(np.linspace(0,n-1,n)*np.pi/(2*n))**2



X= np.array([[0,1],[1,0]],dtype=complex)
Z = np.array([[1,0],[0,-1]],dtype=complex)
H = np.array([[1,1],[1,-1]],dtype=complex)/np.sqrt(2)


def linear_spectrum_N(n):
    return (np.linspace(0,n-1,n)*np.pi/(n))

def linear_Ham_N(n,L=1):
    out = np.kron(H, C(n))  @ np.kron(Z, np.diag(n/L * linear_spectrum_N(n))) @ np.kron(H, C(n).T)
    return out


def Ham_N(n,L=1):
    out = np.kron(H, C(n))  @ np.kron(Z, np.diag(n/L * np.sqrt(-spectrum_N(n)))) @ np.kron(H, C(n).T)
    return out

def L_N(n):
    out = np.zeros((n,n))
    out[0,0] = -1
    out[0,1] = 1
    for i in range(1,n-1):
        out[i,i-1 ] = 1
        out[i,i ] = -2
        out[i,i+1 ] = 1
    
    out[n-1,n-1] = -1
    out[n-1,n-2] = 1
    return out

def triag_circ(n):
    qc = QuantumCircuit(n)
    for i in range(1,n):
        for j in range(i):
            if(-i+j != -1):
                qc.crx(np.pi/2**(j+1-int(i+1 == n)),n-i-1,n-i+j)
            else:
                qc.cp(np.pi/2**(j+1-int(i+1 == n)),n-i-1,n-i+j)
    for i in range(1,n-1)[::-1]:
        for j in range(i):
            if(-i+j != -1):
                qc.crx(-np.pi/2**(j+1),n-i-1,n-i+j)
            else:
                qc.cp(-np.pi/2**(j+1),n-i-1,n-i+j)

    return qc

def minus_triag_circ(n):
    qc = QuantumCircuit(n)
    for i in range(1,n):
        for j in range(i):
            qc.crx(np.pi/2**(j+1-int(i+1 == n)) * (-1)**int(i+1 == n),n-i-1,n-i+j)
    for i in range(1,n-1)[::-1]:
        for j in range(i):
            qc.crx(-np.pi/2**(j+1),n-i-1,n-i+j)
    return qc


def mcx_decomp(n):
    qc = QuantumCircuit(n)
    qc.h(-1)
    qc = qc.compose(triag_circ(n),range(n))
    qc = qc.compose(minus_triag_circ(n-1),range(n-1))
    qc.h(-1)
    return qc


#### Periodic Circuit###
def SQFT_circuit(n):
    qc = QuantumCircuit(n)
    qc.x(0)
    for i in range(n):
        qc.h(i)
        for j in range(i+1,n):
            qc.cp(np.pi/(2**(j-i)),j,i)
    for i in range(n//2):
        qc.swap(i,n-i-1)
    qc.x(0)
    return qc

def SIQFT_circuit(n):
    qc = QuantumCircuit(n)
    qc.x(0)
    for i in range(n):
        qc.h(i)
        for j in range(i+1,n):
            qc.cp(-np.pi/(2**(j-i)),j,i)
    for i in range(n//2):
        qc.swap(i,n-i-1)
    qc.x(0)
    return qc

def SQFT_matrix(n):
    l = range(-n//2,n//2)
    Q = np.zeros((n,n),dtype=complex)
    omega =  np.exp(1j * 2 * np.pi/n)
    for i in range(n):
        for j in range(n):
            Q[i,j] = omega**(l[i]*l[j])/np.sqrt(n)
    return Q

def SIQFT_matrix(n):
    l = range(-n//2,n//2)
    Q = np.zeros((n,n),dtype=complex)
    omega =  np.exp(-1j * 2 * np.pi/n)
    for i in range(n):
        for j in range(n):
            Q[i,j] = omega**(l[i]*l[j])/np.sqrt(n)
    return Q

def L_P(n):
    out = np.zeros((n,n))
    for i in range(0,n):
        out[i,(i-1)%n ] = 1
        out[i,i ] = -2
        out[i,(i+1)%n ] = 1
    return out

def spectrum_P(n):
    return -4*np.cos(np.linspace(0,n-1,n)*np.pi/(n))**2

def linear_spectrum_P(n):
    return 2*(np.linspace(0,n-1,n)*np.pi/(n)) - np.pi

def linear_Ham_P(n,L=1):
    out = np.kron(H, SQFT_matrix(n))  @ np.kron(Z, np.diag(n/L * linear_spectrum_P(n))) @ np.kron(H, SIQFT_matrix(n).T)
    return out


def Ham_P(n,L=1):
    out = np.kron(H, SQFT_matrix(n))  @ np.kron(Z, np.diag(n/L * np.sqrt(-spectrum_P(n)))) @ np.kron(H, SIQFT_matrix(n).T)
    return out

def evolve_circ_P(n,t):
    qc = QuantumCircuit(n+1)
    qc.h(0)
    qc = qc.compose(SIQFT_circuit(n),range(1,n+1))
    for i in range(-1,n):
        qc.p(-np.pi*t*(2**(n-i)),i+1)
    for i in range(0,n):
        qc.cp(np.pi*t*(2**(n-i+1)),0,i+1)
    qc.h(0)
    qc = qc.compose(SQFT_circuit(n),range(1,n+1))
    return qc


#### Periodic Methods
def SQFT_circuit(n):
    qc = QuantumCircuit(n)
    qc.x(0)
    for i in range(n):
        qc.h(i)
        for j in range(i+1,n):
            qc.cp(np.pi/(2**(j-i)),j,i)
    for i in range(n//2):
        qc.swap(i,n-i-1)
    qc.x(0)
    return qc

def SIQFT_circuit(n):
    qc = QuantumCircuit(n)
    qc.x(0)
    for i in range(n):
        qc.h(i)
        for j in range(i+1,n):
            qc.cp(-np.pi/(2**(j-i)),j,i)
    for i in range(n//2):
        qc.swap(i,n-i-1)
    qc.x(0)
    return qc

def SQFT_matrix(n):
    l = range(-n//2,n//2)
    Q = np.zeros((n,n),dtype=complex)
    omega =  np.exp(1j * 2 * np.pi/n)
    for i in range(n):
        for j in range(n):
            Q[i,j] = omega**(l[i]*l[j])/np.sqrt(n)
    return Q

def SIQFT_matrix(n):
    l = range(-n//2,n//2)
    Q = np.zeros((n,n),dtype=complex)
    omega =  np.exp(-1j * 2 * np.pi/n)
    for i in range(n):
        for j in range(n):
            Q[i,j] = omega**(l[i]*l[j])/np.sqrt(n)
    return Q

def L_P(n):
    out = np.zeros((n,n))
    for i in range(0,n):
        out[i,(i-1)%n ] = 1
        out[i,i ] = -2
        out[i,(i+1)%n ] = 1
    return out

def spectrum_P(n):
    return -4*np.cos(np.linspace(0,n-1,n)*np.pi/(n))**2

def linear_spectrum_P(n):
    return 2*(np.linspace(0,n-1,n)*np.pi/(n)) - np.pi

def linear_Ham_P(n,L=1):
    out = np.kron(H, SQFT_matrix(n))  @ np.kron(Z, np.diag(n/L * linear_spectrum_P(n))) @ np.kron(H, SIQFT_matrix(n).T)
    return out


def Ham_P(n,L=1):
    out = np.kron(H, SQFT_matrix(n))  @ np.kron(Z, np.sign(linear_spectrum_P(n))*np.diag(n/L * np.sqrt(-spectrum_P(n)))) @ np.kron(H, SIQFT_matrix(n).T)
    return out

def evolve_circ_P(n,t):
    qc = QuantumCircuit(n+1)
    qc.h(0)
    qc = qc.compose(SIQFT_circuit(n),range(1,n+1))
    for i in range(-1,n):
        qc.p(-np.pi*t*(2**(n-i)),i+1)
    for i in range(0,n):
        qc.cp(np.pi*t*(2**(n-i+1)),0,i+1)
    qc.h(0)
    qc = qc.compose(SQFT_circuit(n),range(1,n+1))
    return qc