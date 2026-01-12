package core

import (
	"fmt"
	"math/rand/v2"

	c "github.com/Mehul-Kumar-27/go-neural/constants"
	t "github.com/Mehul-Kumar-27/go-neural/tensor"
)

type Neuron struct {
	Weights            []*t.Tensor
	Bias               *t.Tensor
	ActivationFunction string
}

func NewNeuron(activationFunction string, bias *t.Tensor, inputs, n_rows, n_cols int) *Neuron {
	min := -1.0
	max := 1.0
	weights := make([]*t.Tensor, inputs)
	for i := range weights {
		data := make([][]float64, n_rows)
		for row := range data {
			data[row] = make([]float64, n_cols)
			for col := range data[row] {
				data[row][col] = rand.Float64()*(max-min) + min
			}
		}
		label := fmt.Sprintf("w%d", i)
		weights[i] = t.NewTensor(label, data, nil, nil, c.None)
	}

	if bias == nil {
		bias = t.NewTensorWithShape(n_rows, n_cols, nil, nil, c.None)
		for row := range bias.Data {
			for col := range bias.Data[row] {
				bias.Data[row][col] = rand.Float64()*(max-min) + min
			}
		}
		label := fmt.Sprintf("bias")
		bias = t.NewTensor(label, bias.Data, nil, nil, c.None)
	}

	return &Neuron{
		Weights:            weights,
		Bias:               bias,
		ActivationFunction: activationFunction,
	}
}

func (n *Neuron) Print() {
	fmt.Println("Weights:")
	for i, weight := range n.Weights {
		fmt.Printf("Weight %d: %v\n", i, weight.Data)
	}
	fmt.Printf("Bias: %v\n", n.Bias.Data)
	fmt.Printf("Activation Function: %s\n", n.ActivationFunction)
}

/*
Activates the neuron and implements the forward pass.
*/
func (n *Neuron) ForwardPass(inputs []*t.Tensor) (*t.Tensor, error) {
	summationAccumulator := make([]t.Tensor, 0)
	summationAccumulator = append(summationAccumulator, *n.Bias)

	for i, input := range inputs {
		// corresponding weight is n.Weights[i]
		weight := n.Weights[i]
		// multiply the input by the weight
		product, err := t.NewTensorFromOp(c.MUL, *input, *weight)
		if err != nil {
			return nil, err
		}
		summationAccumulator = append(summationAccumulator, *product)

	}

	summation, err := t.NewTensorFromOp(c.ADD, summationAccumulator...)
	if err != nil {
		return nil, err
	}

	activation, err := t.NewTensorFromOp(n.ActivationFunction, *summation)
	if err != nil {
		return nil, err
	}

	return activation, nil
}
