"""Trainable RY/RZ layers with a nearest-neighbor CNOT ring."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def weight_count(n_qubits, depth): return 2 * n_qubits * (depth + 1)


def make_ansatz(n_qubits, depth=1, params=None):
    if depth < 0: raise ValueError("depth must be nonnegative")
    params = params or ParameterVector("w", weight_count(n_qubits, depth))
    if len(params) != weight_count(n_qubits, depth): raise ValueError("incorrect parameter count")
    qc = QuantumCircuit(n_qubits); p = 0
    for layer in range(depth + 1):
        for q in range(n_qubits):
            qc.ry(params[p], q); p += 1
            qc.rz(params[p], q); p += 1
        if layer < depth:
            for q in range(n_qubits): qc.cx(q, (q + 1) % n_qubits)
    return qc, params
