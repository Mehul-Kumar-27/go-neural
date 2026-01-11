package main

import (
	"github.com/Mehul-Kumar-27/go-neural/tensor"
	"go.uber.org/zap"
	"time"
)

func main() {
	logger := zap.NewExample()
	defer logger.Sync()

	t1 := tensor.NewTensor([][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}, nil, nil, "")

	t2 := tensor.NewTensor([][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}, nil, nil, "")

	t3 := tensor.NewTensor([][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}, nil, nil, "")
	start_time := time.Now()
	t4, err := tensor.NewTensorFromOp(tensor.MUL, t1, t2, t3)
	if err != nil {
		logger.Error(err.Error())
		return
	}
	end_time := time.Now()
	logger.Info("Time taken: ", zap.Duration("time", end_time.Sub(start_time)))
	t4.GraphPrint()

	// Generate SVG visualization
	err = t4.GraphPrintSVG("graph.svg")
	if err != nil {
		logger.Error("Failed to generate SVG", zap.Error(err))
		return
	}
	logger.Info("SVG graph generated successfully: graph.svg")

}
