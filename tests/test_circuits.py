import numpy as np
from qiskit.quantum_info import Statevector
from src.ansatz import make_ansatz, weight_count
from src.encodings import amplitude_encoding, angle_encoding, zz_encoding
from src.qnn import QNNConfig, VariationalClassifier

def test_amplitude_state_matches_normalized_vector():
    x=np.array([1.,2.,3.,4.,5.,6.,7.,8.]); qc=amplitude_encoding(x)
    actual=Statevector.from_instruction(qc).data
    assert np.allclose(actual,x/np.linalg.norm(x),atol=1e-8)

def test_ansatz_weight_count_and_unique_use():
    for n,d in ((8,1),(3,2)):
        qc,p=make_ansatz(n,d)
        assert len(p)==2*n*(d+1)==weight_count(n,d)
        assert len(qc.parameters)==len(p)

def test_readout_probabilities_match_statevector():
    for encoding, x in (("angle",np.linspace(.1,2.1,8)),("amplitude",np.linspace(-1,1,8))):
        model=VariationalClassifier(QNNConfig(encoding,1),seed=3); observed=model.probabilities([x])[0]
        qc=model._circuit(x).assign_parameters(dict(zip(model.parameters,model.weights)))
        qc.data = [item for item in qc.data if item.operation.name != "save_probabilities"]
        expected=Statevector.from_instruction(qc).probabilities(qargs=[0,1,2])
        assert np.allclose(observed,expected,atol=1e-8)
        assert np.isclose(observed.sum(),1)

def test_parameter_shift_gradient_matches_finite_difference():
    model=VariationalClassifier(QNNConfig("angle",0),seed=4)
    X=np.array([[.1]*8,[.7]*8]); y=np.array([0,1]); w=model.weights.copy()
    grad=model.gradient(X,y,w)
    eps=1e-5; fd=[]
    for j in range(len(w)):
        plus=w.copy(); minus=w.copy(); plus[j]+=eps; minus[j]-=eps
        fd.append((model.loss(X,y,plus)-model.loss(X,y,minus))/(2*eps))
    assert np.allclose(grad,fd,atol=2e-4,rtol=2e-3)
