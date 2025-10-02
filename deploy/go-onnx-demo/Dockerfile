FROM golang:1.21-alpine AS builder

WORKDIR /app
COPY . .

RUN go mod init go-onnx-demo || true
RUN go mod tidy
RUN go build -o onnx-service .

FROM alpine:3.18
WORKDIR /app
COPY --from=builder /app/onnx-service .
COPY model.onnx .

EXPOSE 8080
CMD ["./onnx-service"]
