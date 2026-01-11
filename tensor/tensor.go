package tensor

import (
	"fmt"
	"os"
	"strings"

	"github.com/Mehul-Kumar-27/go-neural/logger"
	svg "github.com/ajstarks/svgo"
	"go.uber.org/zap"
)

type Tensor struct {
	Data       [][]float64
	LeftChild  *Tensor
	RightChild *Tensor
	Op         string
}

func NewTensor(data [][]float64, leftChild *Tensor, rightChild *Tensor, op string) Tensor {
	return Tensor{
		Data:       data,
		LeftChild:  leftChild,
		RightChild: rightChild,
		Op:         op,
	}
}

func NewTensorWithShape(row, col int, leftChild *Tensor, rightChild *Tensor, op string) Tensor {
	data := make([][]float64, row)
	for i := range data {
		data[i] = make([]float64, col)
	}
	return NewTensor(data, nil, nil, "")
}

func NewTensorFromOp(op string, tensors ...Tensor) (*Tensor, error) {
	switch op {
	case ADD:
		return addMany(tensors...)
	case MUL:
		return mulMany(tensors...)
	case DIV:
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

/*
Copy returns a copy of the tensor.
*/
func (t *Tensor) Copy() Tensor {
	return Tensor{
		Data: append([][]float64{}, t.Data...),
	}
}

func (t *Tensor) GraphPrint() {
	logger := logger.NewLogger()
	defer logger.Sync()

	fmt.Println("\n=== Computational Graph ===")
	t.graphPrintRecursive(0)
	fmt.Println("===========================\n")
}

func (t *Tensor) graphPrintRecursive(depth int) {
	indent := strings.Repeat("  ", depth)

	// Print current tensor
	fmt.Printf("%sTensor [%dx%d]", indent, len(t.Data), len(t.Data[0]))
	if t.Op != "" {
		fmt.Printf(" (Op: %s)", t.Op)
	}
	fmt.Println()

	// Print tensor data in grid format
	for row := 0; row < len(t.Data); row++ {
		var rowStr []string
		for col := 0; col < len(t.Data[row]); col++ {
			rowStr = append(rowStr, fmt.Sprintf("%.2f", t.Data[row][col]))
		}
		fmt.Printf("%s  | %s |\n", indent, strings.Join(rowStr, " | "))
	}

	// Print graph connections
	if t.LeftChild != nil || t.RightChild != nil {
		fmt.Printf("%s  ↓\n", indent)

		if t.LeftChild != nil && t.RightChild != nil {
			// Both children exist
			fmt.Printf("%s  [Left Child] ──┐\n", indent)
			t.LeftChild.graphPrintRecursive(depth + 1)
			fmt.Printf("%s               │\n", indent)
			fmt.Printf("%s               ├──→ [%s] ──→ Current Tensor\n", indent, t.Op)
			fmt.Printf("%s               │\n", indent)
			fmt.Printf("%s  [Right Child] ─┘\n", indent)
			t.RightChild.graphPrintRecursive(depth + 1)
		} else if t.LeftChild != nil {
			// Only left child exists
			fmt.Printf("%s  [Left Child] ──→ [%s] ──→ Current Tensor\n", indent, t.Op)
			t.LeftChild.graphPrintRecursive(depth + 1)
		} else if t.RightChild != nil {
			// Only right child exists
			fmt.Printf("%s  [Right Child] ──→ [%s] ──→ Current Tensor\n", indent, t.Op)
			t.RightChild.graphPrintRecursive(depth + 1)
		}
		fmt.Println()
	}
}

// GraphPrintSVG generates an SVG visualization of the computational graph
func (t *Tensor) GraphPrintSVG(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Calculate graph dimensions
	width := 800
	height := t.calculateGraphHeight() * 150
	if height < 400 {
		height = 400
	}

	canvas := svg.New(file)
	canvas.Start(width, height)

	// Add styles
	canvas.Def()
	canvas.DefEnd()

	// Draw the graph starting from position
	t.drawNodeSVG(canvas, width/2, 50, 0)

	canvas.End()
	return nil
}

// Helper struct to track node positions
type nodePosition struct {
	x, y   int
	width  int
	height int
}

// calculateGraphHeight calculates the total height needed for the graph
func (t *Tensor) calculateGraphHeight() int {
	if t.LeftChild == nil && t.RightChild == nil {
		return 1
	}

	leftHeight := 0
	rightHeight := 0

	if t.LeftChild != nil {
		leftHeight = t.LeftChild.calculateGraphHeight()
	}
	if t.RightChild != nil {
		rightHeight = t.RightChild.calculateGraphHeight()
	}

	maxHeight := leftHeight
	if rightHeight > maxHeight {
		maxHeight = rightHeight
	}

	return maxHeight + 1
}

// drawNodeSVG recursively draws the computational graph as SVG
func (t *Tensor) drawNodeSVG(canvas *svg.SVG, x, y, depth int) nodePosition {
	nodeWidth := 160
	nodeHeight := 80

	// Draw current node box
	canvas.Rect(x-nodeWidth/2, y, nodeWidth, nodeHeight, "fill:lightblue;stroke:black;stroke-width:2")

	// Draw tensor info
	shapeText := fmt.Sprintf("Tensor [%dx%d]", len(t.Data), len(t.Data[0]))
	canvas.Text(x, y+25, shapeText, "text-anchor:middle;font-size:12px;font-weight:bold")

	// Draw tensor data (first row only to keep it compact)
	if len(t.Data) > 0 && len(t.Data[0]) > 0 {
		dataText := "["
		for i := 0; i < len(t.Data[0]) && i < 3; i++ {
			if i > 0 {
				dataText += ", "
			}
			dataText += fmt.Sprintf("%.1f", t.Data[0][i])
		}
		if len(t.Data[0]) > 3 {
			dataText += "..."
		}
		dataText += "]"
		canvas.Text(x, y+45, dataText, "text-anchor:middle;font-size:10px")
	}

	// Draw operation if exists
	if t.Op != "" {
		canvas.Text(x, y+65, fmt.Sprintf("Op: %s", t.Op), "text-anchor:middle;font-size:10px;font-style:italic")
	}

	currentPos := nodePosition{x: x, y: y, width: nodeWidth, height: nodeHeight}

	// Draw children if they exist
	if t.LeftChild != nil || t.RightChild != nil {
		childY := y + nodeHeight + 100

		if t.LeftChild != nil && t.RightChild != nil {
			// Both children exist
			leftX := x - 200
			rightX := x + 200

			// Draw left child
			leftPos := t.LeftChild.drawNodeSVG(canvas, leftX, childY, depth+1)

			// Draw right child
			rightPos := t.RightChild.drawNodeSVG(canvas, rightX, childY, depth+1)

			// Draw arrows from children to current node
			// Left arrow
			canvas.Line(leftPos.x, leftPos.y, x-50, y+nodeHeight,
				"stroke:black;stroke-width:2;marker-end:url(#arrow)")

			// Right arrow
			canvas.Line(rightPos.x, rightPos.y, x+50, y+nodeHeight,
				"stroke:black;stroke-width:2;marker-end:url(#arrow)")

			// Draw operation label between arrows
			if t.Op != "" {
				opY := y + nodeHeight + 20
				canvas.Rect(x-25, opY-15, 50, 20, "fill:yellow;stroke:black;stroke-width:1;rx:5")
				canvas.Text(x, opY, t.Op, "text-anchor:middle;font-size:12px;font-weight:bold")
			}

		} else if t.LeftChild != nil {
			// Only left child
			leftPos := t.LeftChild.drawNodeSVG(canvas, x, childY, depth+1)

			// Draw arrow from left child to current
			canvas.Line(leftPos.x, leftPos.y, x, y+nodeHeight,
				"stroke:black;stroke-width:2;marker-end:url(#arrow)")

			// Draw operation label
			if t.Op != "" {
				opY := (leftPos.y + y + nodeHeight) / 2
				canvas.Rect(x-25, opY-15, 50, 20, "fill:yellow;stroke:black;stroke-width:1;rx:5")
				canvas.Text(x, opY, t.Op, "text-anchor:middle;font-size:12px;font-weight:bold")
			}

		} else if t.RightChild != nil {
			// Only right child
			rightPos := t.RightChild.drawNodeSVG(canvas, x, childY, depth+1)

			// Draw arrow from right child to current
			canvas.Line(rightPos.x, rightPos.y, x, y+nodeHeight,
				"stroke:black;stroke-width:2;marker-end:url(#arrow)")

			// Draw operation label
			if t.Op != "" {
				opY := (rightPos.y + y + nodeHeight) / 2
				canvas.Rect(x-25, opY-15, 50, 20, "fill:yellow;stroke:black;stroke-width:1;rx:5")
				canvas.Text(x, opY, t.Op, "text-anchor:middle;font-size:12px;font-weight:bold")
			}
		}
	}

	return currentPos
}
