package tensor

const (
	ADD = "+"
	SUB = "-"
	MUL = "*"
	DIV = "/"
)

func ValidateShape(op string, a ...Tensor) error {
	switch op {
	case ADD:
		return ValidateShapeForAdd(a...)
	case MUL:
		return ValidateShapeForMul(a...)
	default:
		return ErrInvalidOp
	}
}

func ValidateShapeForAdd(a ...Tensor) error {
	row, col := a[0].Shape()
	for _, t := range a[1:] {
		r, c := t.Shape()
		if r != row || c != col {
			return ErrShapeMismatch
		}
	}
	return nil
}

func ValidateShapeForMul(a ...Tensor) error {
	if len(a) < 2 {
		return nil
	}
	_, col := a[0].Shape()

	for _, t := range a[1:] {
		rt, ct := t.Shape()
		if col != rt {
			return ErrShapeMismatch
		}

		col = ct
	}

	return nil

}

/*
Add returns the sum of the tensors.
Divide the tensors into as many groups of 2 as possible and create a new tensor for each group and add the two tensors as the children.
Add the tensors in each group and return the result.
The result is a new tensor with the shape of the first tensor.
*/
func addMany(a ...Tensor) (*Tensor, error) {
	if len(a) == 0 {
		return nil, nil
	}
	if err := ValidateShape(ADD, a...); err != nil {
		return nil, err
	}

	for len(a) > 1 {
		var nextTensor []Tensor
		for i := 0; i < len(a); i += 2 {
			if i+1 < len(a) {
				// i and i + 1 indexes are valid
				t, err := Add(a[i], a[i+1])
				if err != nil {
					return nil, err
				}
				nextTensor = append(nextTensor, *t)
			} else {
				nextTensor = append(nextTensor, a[i])
			}
		}
		a = nextTensor
	}

	return &a[0], nil
}

func Add(a, b Tensor) (*Tensor, error) {
	row, col := a.Shape()
	result := NewTensorWithShape(row, col, &a, &b, ADD)

	for row := range result.Data {
		for col := range result.Data[row] {
			result.Data[row][col] = a.Data[row][col] + b.Data[row][col]
			result.LeftChild = &a
			result.RightChild = &b
		}
	}

	return &result, nil
}

func mulMany(a ...Tensor) (*Tensor, error) {
	if len(a) == 0 {
		return nil, nil
	}

	if err := ValidateShape(MUL, a...); err != nil {
		return nil, err
	}

	result := a[0]

	for i := 1; i < len(a); i++ {
		r, err := Mul(result, a[i])
		if err != nil {
			return nil, err
		}
		result = *r
	}

	return &result, nil
}

func Mul(a, b Tensor) (*Tensor, error) {
	ra, ca := a.Shape()
	_, cb := b.Shape()
	result := NewTensorWithShape(ra, cb, &a, &b, MUL)

	for row := range result.Data {
		for col := range result.Data[row] {
			result.Data[row][col] = 0
			for k := 0; k < ca; k++ {
				result.Data[row][col] += a.Data[row][k] * b.Data[k][col]
			}
		}
	}

	return &result, nil
}
