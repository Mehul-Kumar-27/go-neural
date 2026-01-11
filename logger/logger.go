package logger

import "go.uber.org/zap"

func NewLogger() *zap.Logger {
	config := zap.NewProductionConfig()
	config.EncoderConfig.TimeKey = ""
	config.EncoderConfig.LevelKey = ""
	config.EncoderConfig.CallerKey = ""
	config.Encoding = "console"

	logger, err := config.Build()
	if err != nil {
		panic(err)
	}
	return logger
}
