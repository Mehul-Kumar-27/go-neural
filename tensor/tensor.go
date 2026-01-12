package tensor

import (
	"fmt"
	"math"
	"strings"

	c "github.com/Mehul-Kumar-27/go-neural/constants"
	"github.com/Mehul-Kumar-27/go-neural/logger"
	"go.uber.org/zap"
)

type Tensor struct {
	Label      string
	Data       [][]float64
	LeftChild  *Tensor
	RightChild *Tensor
	Op         string
	Gradient   *Tensor
}

func NewTensor(label string, data [][]float64, leftChild *Tensor, rightChild *Tensor, op string) *Tensor {
	return &Tensor{
		Label:      label,
		Data:       data,
		LeftChild:  leftChild,
		RightChild: rightChild,
		Op:         op,
	}
}

func NewTensorWithShape(row, col int, leftChild *Tensor, rightChild *Tensor, op string) *Tensor {
	data := make([][]float64, row)
	for i := range data {
		data[i] = make([]float64, col)
	}
	leftChildLabel := ""
	rightChildLabel := ""
	if leftChild != nil {
		leftChildLabel = leftChild.Label
	}
	if rightChild != nil {
		rightChildLabel = rightChild.Label
	}
	label := fmt.Sprintf("%s %s %s", leftChildLabel, op, rightChildLabel)
	return NewTensor(label, data, nil, nil, "")
}

func NewTensorFromOp(op string, tensors ...Tensor) (*Tensor, error) {
	switch op {
	case c.ADD:
		return addMany(tensors...)
	case c.MUL:
		return mulMany(tensors...)
	case c.TANH:
		return tanh(tensors...)
	case c.DIV:
		return nil, ErrNotImplemented
	default:
		return nil, ErrInvalidOp
	}
}

/*
Shape returns the shape of the tensor as a tuple of (rows, columns).
*/
func (t *Tensor) Shape() (int, int) {
	return len(t.Data), len(t.Data[0])
}

func (t *Tensor) Print() {
	rows, cols := t.Shape()

	logger := logger.NewLogger()
	defer logger.Sync()

	logger.Info("")

	logger.Info("--------------------------------")
	logger.Info("Tensor Shape:", zap.Int("rows", rows), zap.Int("cols", cols))
	logger.Info("Tensor Label:", zap.String("label", t.Label))
	gradientStr := ""
	if t.Gradient != nil {
		var parts []string
		for _, row := range t.Gradient.Data {
			var rowParts []string
			for _, v := range row {
				rowParts = append(rowParts, fmt.Sprintf("%.2f", v))
			}
			parts = append(parts, "["+strings.Join(rowParts, ", ")+"]")
		}
		gradientStr = "[" + strings.Join(parts, ", ") + "]"
	}
	logger.Info("Tensor Gradient:", zap.String("gradient", gradientStr))

	for row := 0; row < len(t.Data); row++ {
		var rowStr []string
		for col := 0; col < len(t.Data[row]); col++ {
			rowStr = append(rowStr, fmt.Sprintf("%.2f", t.Data[row][col]))
		}
		fmt.Printf("| %s |\n", strings.Join(rowStr, " | "))
	}

	logger.Info("--------------------------------")
	logger.Info("")
}

func (t *Tensor) Children() []*Tensor {
	children := make([]*Tensor, 0)

	if t.LeftChild != nil {
		children = append(children, t.LeftChild)
	}
	if t.RightChild != nil {
		children = append(children, t.RightChild)
	}
	return children
}

func (t *Tensor) ChildrenCount() int {
	return len(t.Children())
}

func (t *Tensor) FromOp() string {
	return t.Op
}

/*
Copy returns a copy of the tensor.
*/
func (t *Tensor) Copy() Tensor {
	return Tensor{
		Data: append([][]float64{}, t.Data...),
	}
}

/*
InitializeGradientForRootNode initializes the gradient for the root node.
*/
func (t *Tensor) InitializeGradientForRootNode() error {
	row, col := t.Shape()
	t.Gradient = NewTensorWithShape(row, col, nil, nil, "")

	for row := range t.Gradient.Data {
		for col := range t.Gradient.Data[row] {
			t.Gradient.Data[row][col] = 1.0
		}
	}

	return t.CalculateGradientFromRoot()
}

func (t *Tensor) CalculateGradientFromRoot() error {
	t.Print()
	if t.ChildrenCount() == 0 {
		return nil
	}

	for i, child := range t.Children() {

		row, col := child.Shape()
		dl_dx := NewTensorWithShape(row, col, nil, nil, "")

		switch t.Op {
		case c.ADD:
			/// L = X + Y
			/// dl/dx = 1
			for row := range dl_dx.Data {
				for col := range dl_dx.Data[row] {
					dl_dx.Data[row][col] = 1.0
				}
			}

		case c.MUL:
			/// L = X * Y
			/// dl/dx = Y
			sibling_index := (i + 1) % 2
			sibling := t.Children()[sibling_index]
			for row := range dl_dx.Data {
				for col := range dl_dx.Data[row] {
					dl_dx.Data[row][col] = sibling.Data[row][col]
				}
			}

		case c.TANH:
			/// L = tanh(X)
			/// dl/dx = 1 - tanh(X)^2
			for row := range dl_dx.Data {
				for col := range dl_dx.Data[row] {
					dl_dx.Data[row][col] = 1.0 - math.Pow(t.Data[row][col], 2)
				}
			}
		}

		gradient_tensor, err := NewTensorFromOp(c.MUL, *dl_dx, *t.Gradient)
		if err != nil {
			return err
		}
		child.Gradient = gradient_tensor
		err = child.CalculateGradientFromRoot()
		if err != nil {
			return err
		}
	}

	return nil
}
