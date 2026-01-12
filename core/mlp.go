package core

type MLP struct {
	Layers []*Layer
}

func NewMLP(input, n_rows, n_cols int) *MLP {
	layers := make([]*Layer, 0)
	for i := 0; i < len(layers); i++ {
		layers = append(layers, NewLayer(input, n_rows, n_cols))
	}
	return &MLP{
		Layers: layers,
	}
}
