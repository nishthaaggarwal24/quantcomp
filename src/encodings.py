"""Quantum feature maps; all returned circuits act on an initialized zero state."""
import numpy as np
from qiskit import QuantumCircuit


def angle_encoding(x, bandwidth=1.0):
    x = np.asarray(x, dtype=float)
    qc = QuantumCircuit(len(x))
    for i, value in enumerate(x): qc.ry(float(bandwidth * value), i)
    return qc


def zz_encoding(x, bandwidth=1.0, reps=1):
    """Havlicek-style ZZ feature map with explicit pairwise interactions."""
    x = np.asarray(x, dtype=float); n = len(x)
    qc = QuantumCircuit(n)
    for _ in range(reps):
        for i in range(n): qc.h(i)
        for i in range(n): qc.p(float(2 * bandwidth * x[i]), i)
        for i in range(n):
            for j in range(i + 1, n):
                phase = 2 * (np.pi - bandwidth * x[i]) * (np.pi - bandwidth * x[j])
                qc.cx(i, j); qc.p(float(phase), j); qc.cx(i, j)
    return qc


def amplitude_encoding(x):
    """Prepare normalized real vector in Qiskit's little-endian basis ordering."""
    x = np.asarray(x, dtype=float)
    if x.shape != (8,): raise ValueError("Amplitude encoding expects exactly 8 features")
    norm = np.linalg.norm(x)
    if norm == 0: raise ValueError("Cannot amplitude-encode a zero vector")
    a = x / norm
    qc = QuantumCircuit(3)
    # First split on the most significant qubit (q2), then q1, then q0.
    # Each prefix denotes fixed values of already prepared higher qubits.
    for level, target in enumerate((2, 1, 0)):
        ncontrols = level
        for prefix in range(2**ncontrols):
            # Gather amplitudes whose higher-order bits match this prefix.
            indices = [i for i in range(8) if (i >> (3-level)) == prefix] if level else list(range(8))
            if level == 0:
                left = np.linalg.norm(a[:4]); right = np.linalg.norm(a[4:])
            else:
                mask = (1 << (3-level)) - 1
                left = np.linalg.norm([a[i] for i in indices if not ((i >> target) & 1)])
                right = np.linalg.norm([a[i] for i in indices if (i >> target) & 1])
            theta = 2 * np.arctan2(right, left)
            if not ncontrols:
                qc.ry(float(theta), target)
            else:
                controls = list(range(target+1, 3))
                bits = [(prefix >> (bit-(target+1))) & 1 for bit in controls]
                for q, bit in zip(controls, bits):
                    if not bit: qc.x(q)
                qc.mcry(float(theta), controls, target, None, mode="noancilla")
                for q, bit in zip(controls, bits):
                    if not bit: qc.x(q)
    return qc


def build_encoding(name, x, bandwidth=1.0, reps=1):
    if name == "angle": return angle_encoding(x, bandwidth)
    if name == "zz": return zz_encoding(x, bandwidth, reps)
    if name == "amplitude": return amplitude_encoding(x)
    raise ValueError(f"Unknown encoding: {name}")
