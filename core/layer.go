package core

import c "github.com/Mehul-Kumar-27/go-neural/constants"

type Layer struct {
	Neurons []*Neuron
}

func NewLayer(input, n_rows, n_cols int) *Layer {
	neurons := make([]*Neuron, 0)
	for i := 0; i < input; i++ {
		neurons = append(neurons, NewNeuron(c.TANH, nil, input, n_rows, n_cols))
	}
	return &Layer{
		Neurons: neurons,
	}
}
