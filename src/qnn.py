"""Exact-probability variational quantum classifier using Qiskit Aer."""
from dataclasses import dataclass
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from .ansatz import make_ansatz, weight_count
from .encodings import build_encoding

@dataclass
class QNNConfig:
    encoding: str = "angle"
    depth: int = 1
    bandwidth: float = 1.0
    reupload: bool = False
    encoding_reps: int = 1

class VariationalClassifier:
    def __init__(self, config=None, seed=0):
        self.config = config or QNNConfig()
        self.n_qubits = 3 if self.config.encoding == "amplitude" else 8
        self.n_weights = weight_count(self.n_qubits, self.config.depth)
        self.parameters = ParameterVector("w", self.n_weights)
        self.weights = np.random.default_rng(seed).normal(0, .15, self.n_weights)
        self.simulator = AerSimulator(method="statevector", max_parallel_experiments=4, max_parallel_threads=2)
        self._compiled_cache = {}

    def _circuit(self, x):
        n = self.n_qubits
        qc = QuantumCircuit(n)
        enc = build_encoding(self.config.encoding, x, self.config.bandwidth, self.config.encoding_reps)
        if not self.config.reupload or self.config.encoding == "amplitude":
            qc.compose(enc, inplace=True)
            ansatz, _ = make_ansatz(n, self.config.depth, self.parameters)
            qc.compose(ansatz, inplace=True)
        else:
            # Re-encode the same classical feature vector before every rotation layer.
            p = 0
            for layer in range(self.config.depth + 1):
                qc.compose(enc, inplace=True)
                for q in range(n):
                    qc.ry(self.parameters[p], q); p += 1
                    qc.rz(self.parameters[p], q); p += 1
                if layer < self.config.depth:
                    for q in range(n): qc.cx(q, (q + 1) % n)
        return qc

    def _compiled_circuits(self, X):
        rows = [np.asarray(x, dtype=float) for x in X]
        keys = [tuple(row.tolist()) for row in rows]
        missing = list(dict.fromkeys(key for key in keys if key not in self._compiled_cache))
        if missing:
            raw = [self._circuit(np.asarray(key)) for key in missing]
            compiled = transpile(raw, basis_gates=["u", "cx"], optimization_level=0)
            for key, circuit in zip(missing, compiled):
                circuit.save_probabilities(list(range(3)))
                self._compiled_cache[key] = circuit
        return [self._compiled_cache[key] for key in keys]

    def _run_with_weights(self, circuits, weights):
        mapping = {p: float(w) for p,w in zip(self.parameters,weights)}
        bound = [circuit.assign_parameters(mapping) for circuit in circuits]
        result = self.simulator.run(bound).result()
        out = np.asarray([result.data(i)["probabilities"] for i in range(len(bound))], dtype=float)
        out = np.clip(out, 1e-12, 1)
        return out / out.sum(axis=1, keepdims=True)

    def probabilities(self, X, weights=None):
        X = np.atleast_2d(np.asarray(X, dtype=float)); weights = self.weights if weights is None else np.asarray(weights)
        if weights.shape != (self.n_weights,): raise ValueError("weight vector has wrong shape")
        return self._run_with_weights(self._compiled_circuits(X), weights)

    def loss(self, X, y, weights=None):
        p = self.probabilities(X, weights)
        return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())

    def gradient(self, X, y, weights=None):
        """Parameter-shift gradient; shifted sample circuits are batched in Aer."""
        w = self.weights.copy() if weights is None else np.asarray(weights, dtype=float).copy()
        X = np.atleast_2d(np.asarray(X, dtype=float)); y = np.asarray(y)
        circuits = self._compiled_circuits(X)
        p0 = self._run_with_weights(circuits, w)
        shifted_circuits=[]
        for j in range(len(w)):
            for sign in (1,-1):
                shifted = w.copy(); shifted[j] += sign*np.pi/2
                mapping = {p: float(value) for p,value in zip(self.parameters,shifted)}
                shifted_circuits.extend(c.assign_parameters(mapping) for c in circuits)
        result = self.simulator.run(shifted_circuits).result()
        shifted_probs=np.asarray([result.data(i)["probabilities"] for i in range(len(shifted_circuits))])
        shifted_probs=shifted_probs.reshape(len(w),2,len(X),8)
        grad=np.empty_like(w)
        rows=np.arange(len(y))
        p_correct=np.clip(p0[rows,y],1e-12,None)
        for j in range(len(w)):
            dp=(shifted_probs[j,0,rows,y]-shifted_probs[j,1,rows,y])/2
            grad[j]=(-dp/p_correct).mean()
        return grad

    def fit(self, X, y, X_val=None, y_val=None, epochs=20, batch_size=32, lr=.03, seed=0, patience=5):
        X = np.asarray(X); y = np.asarray(y); rng = np.random.default_rng(seed)
        m = np.zeros_like(self.weights); v = np.zeros_like(self.weights); step = 0
        best = self.weights.copy(); best_acc = -1; stale = 0; history=[]
        for epoch in range(epochs):
            order = rng.permutation(len(y))
            for start in range(0, len(y), batch_size):
                ix = order[start:start+batch_size]; g = self.gradient(X[ix], y[ix])
                step += 1; m=.9*m+.1*g; v=.999*v+.001*g*g
                self.weights -= lr*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
            train_loss = self.loss(X, y)
            if X_val is not None:
                pred = self.predict(X_val); acc = float(np.mean(pred == y_val))
            else: acc = -train_loss
            history.append({"epoch": epoch+1, "loss": train_loss, "validation_accuracy": acc})
            if acc > best_acc:
                best_acc=acc; best=self.weights.copy(); stale=0
            else:
                stale += 1
                if stale >= patience: break
        self.weights = best
        return history

    def predict_proba(self, X): return self.probabilities(X)
    def predict(self, X): return self.predict_proba(X).argmax(axis=1)
