package main

import (
	"time"

	c "github.com/Mehul-Kumar-27/go-neural/constants"
	"github.com/Mehul-Kumar-27/go-neural/core"
	"github.com/Mehul-Kumar-27/go-neural/tensor"
	"go.uber.org/zap"
)

func main() {
	logger := zap.NewExample()
	start_time := time.Now()
	defer logger.Sync()

	t1 := tensor.NewTensor("t1", [][]float64{
		{2},
	}, nil, nil, c.None)

	// w1 := tensor.NewTensor("w1", [][]float64{
	// 	{-3.0},
	// }, nil, nil, c.None)

	t2 := tensor.NewTensor("t2", [][]float64{
		{0.0},
	}, nil, nil, c.None)

	// w2 := tensor.NewTensor("w2", [][]float64{
	// 	{1},
	// }, nil, nil, c.None)

	b1 := tensor.NewTensor("bias", [][]float64{
		{6.88137},
	}, nil, nil, c.None)

	neuron := core.NewNeuron(c.TANH, b1, 2, 1, 1)

	inputs := make([]*tensor.Tensor, 0)

	inputs = append(inputs, t1)
	inputs = append(inputs, t2)

	activation, err := neuron.ForwardPass(inputs)
	if err != nil {
		logger.Error("Error in forward pass: ", zap.Error(err))
	}
	activation.Print()

	end_time := time.Now()
	logger.Info("Time taken: ", zap.Duration("time", end_time.Sub(start_time)))
}
