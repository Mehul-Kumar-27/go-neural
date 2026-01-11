package tensor

import "errors"

var (
	ErrShapeMismatch = errors.New("dimension mismatch")
	ErrInvalidOp = errors.New("invalid operation")
	ErrNotImplemented = errors.New("not implemented method")
)
