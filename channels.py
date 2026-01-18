# channels.py
import numpy as np

# --- Helpers ---
def _bloch_state(theta, phi):
    # |psi> = cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>
    c, s = np.cos(theta/2.0), np.sin(theta/2.0)
    ket = np.array([c, np.exp(1j*phi)*s], dtype=complex)
    rho = np.outer(ket, ket.conj())
    return rho

def von_neumann_entropy(rho):
    evals = np.linalg.eigvalsh((rho + rho.conj().T)/2)
    evals = np.clip(np.real(evals), 0, 1)
    evals = evals/np.sum(evals)
    nz = evals[evals > 1e-15]
    return float(-np.sum(nz * np.log2(nz)))

def partial_trace(rho_AB, dims=(2,2), keep='A'):
    da, db = dims
    rho = rho_AB.reshape(da, db, da, db)
    if keep == 'A':
        return np.einsum('ijik->jk', rho)
    elif keep == 'B':
        return np.einsum('iijk->jk', rho)
    else:
        raise ValueError("keep must be 'A' or 'B'")

# --- Channels ---
def depolarizing_Kraus(p):
    # N_p(rho) = (1-p) rho + p/3 sum_{i=1}^3 sigma_i rho sigma_i
    K0 = np.sqrt(1 - p) * np.eye(2)
    Kx = np.sqrt(p/3) * np.array([[0,1],[1,0]], complex)
    Ky = np.sqrt(p/3) * np.array([[0,-1j],[1j,0]], complex)
    Kz = np.sqrt(p/3) * np.array([[1,0],[0,-1]], complex)
    return [K0, Kx, Ky, Kz]

def amplitude_damping_Kraus(gamma):
    # Standard AD channel (T=0 bath)
    E0 = np.array([[1,0],[0,np.sqrt(1-gamma)]], complex)
    E1 = np.array([[0, np.sqrt(gamma)],[0,0]], complex)
    return [E0, E1]

def gadc_Kraus(gamma, Nth):
    # Generalized amplitude damping (finite temperature)
    # Nielsen & Chuang (GADC)
    g = gamma
    p = Nth
    E0 = np.sqrt(p)*np.array([[1,0],[0,np.sqrt(1-g)]], complex)
    E1 = np.sqrt(p)*np.array([[0,np.sqrt(g)],[0,0]], complex)
    E2 = np.sqrt(1-p)*np.array([[np.sqrt(1-g),0],[0,1]], complex)
    E3 = np.sqrt(1-p)*np.array([[0,0],[np.sqrt(g),0]], complex)
    return [E0,E1,E2,E3]

def apply_channel(Ks, rho):
    out = np.zeros((2,2), dtype=complex)
    for K in Ks:
        out += K @ rho @ K.conj().T
    return out

def isometric_extension(Ks, rho):
    # Build Stinespring isometry from Kraus ops: V = sum_k |k>_E \otimes K_k
    dE = len(Ks)
    V = np.vstack([np.kron(np.eye(1, dE, k), K) for k, K in enumerate(Ks)])  # (dE*2) x 2
    psi = rho_to_purification(rho)
    phi_AE = (np.kron(np.eye(dE*2), np.eye(1)) @ (np.kron(np.eye(1), np.eye(2)) @ psi))  # not used
    # Instead: construct channel output and environment state from Kraus rule on purification
    # Purify rho as |psi>_AA' and apply channel to A'. Then trace as needed.
    psi_AA = state_purification(rho)  # |psi> in A(sys)-A'(ref)
    # Apply channel on A' via Kraus
    out = 0
    for k, K in enumerate(Ks):
        ket = np.kron(np.eye(2), K) @ psi_AA  # (A sys) ⊗ (B out)
        ek = np.zeros((len(Ks), 1), complex); ek[k,0] = 1.0
        ket = np.kron(ek, ket)  # E ⊗ A ⊗ B
        out += ket @ ket.conj().T
    # Return rho_B, rho_E, and the tripartite state
    rho_B = partial_trace(out, dims=(len(Ks)*2, 2, 2), keep='B')  # tricky — we won't use this helper further
    return out

# Simple purification utilities (qubit)
def state_purification(rho):
    evals, vecs = np.linalg.eigh(rho)
    evals = np.clip(evals, 0, 1)
    evals = evals/np.sum(evals)
    # |psi> = sum_i sqrt(lambda_i) |i>_A ⊗ |i>_A'
    psi = np.zeros((4,1), complex)
    for i,(lam,vi) in enumerate(zip(evals, vecs.T)):
        psi += np.sqrt(lam) * np.kron(vi.reshape(2,1), vi.reshape(2,1))
    return psi

def channel_output_and_env(Ks, rho):
    # From a purification, apply channel on system and then trace
    psi = state_purification(rho)       # |psi>_{RA}
    Rdim, Adim = 2, 2
    out_RBE = 0
    for k,K in enumerate(Ks):
        ek = np.zeros((len(Ks),1), complex); ek[k,0]=1
        U = np.kron(np.eye(Rdim), np.kron(ek, K))  # R ⊗ E ⊗ B <- apply on A
        ket = U @ psi
        out_RBE += ket @ ket.conj().T
    # trace E or B to get marginals
    # We'll avoid heavy reshaping: compute rho_B = sum_k K rho K^\dagger; rho_R = Tr_A rho = same as input marginal
    rho_B = apply_channel(Ks, rho)
    # The complementary output entropy equals S(output of complementary channel).
    # For qubits we can use S(complement) = S(env), computed from Stinespring with Kraus probabilities:
    # diagonal env state with p_k = Tr(K_k rho K_k^\dagger) only if Kraus are orthogonal; safer to build rho_E explicitly:
    dE = len(Ks)
    rho_E = np.zeros((dE,dE), complex)
    for i,Ki in enumerate(Ks):
        for j,Kj in enumerate(Ks):
            rho_E[i,j] = np.trace(Ki @ rho @ Kj.conj().T)
    return rho_B, rho_E

# Coherent information and EA capacity
def coherent_information(Ks, rho):
    rho_B, rho_E = channel_output_and_env(Ks, rho)
    return von_neumann_entropy(rho_B) - von_neumann_entropy(rho_E)

def mutual_information_output(Ks, rho):
    # I(A;B) for purification |psi>_{RA}, channel on A -> B. Equivalent to S(rho_R)+S(rho_B)-S(rho_RB).
    # For qubit input, S(rho_R)=S(rho) (same eigenvalues).
    rho_B, _ = channel_output_and_env(Ks, rho)
    S_R = von_neumann_entropy(rho)
    S_B = von_neumann_entropy(rho_B)
    # Build the joint rho_RB by acting channel on half of purification:
    psi = state_purification(rho)          # |psi>_{RA}
    rho_RB = 0
    for K in Ks:
        U = np.kron(np.eye(2), K)          # R untouched, A -> B
        ket = U @ psi
        rho_RB += ket @ ket.conj().T
    S_RB = von_neumann_entropy(rho_RB)
    return S_R + S_B - S_RB

def grid_optimize_on_bloch(Ks, fine=721):
    thetas = np.linspace(0, np.pi, fine)
    phis   = np.linspace(0, 2*np.pi, fine//2)
    Ic_best, CE_best = -1e9, -1e9
    for th in thetas:
        for ph in phis:
            rho = _bloch_state(th, ph)
            ic  = coherent_information(Ks, rho)
            ce  = mutual_information_output(Ks, rho)
            if ic > Ic_best: Ic_best = ic
            if ce > CE_best: CE_best = ce
    return Ic_best, CE_best
