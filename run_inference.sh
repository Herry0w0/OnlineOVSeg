#!/bin/bash
# Inference script runner

# Set default values
CONFIG_FILE="configs/inference_config.yaml"
INPUT_PATH=""
OUTPUT_DIR=""
SCENE_ID=""
LOG_LEVEL="INFO"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --input)
            INPUT_PATH="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --scene-id)
            SCENE_ID="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --visualize)
            VISUALIZE="--visualize"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --input INPUT_PATH [OPTIONS]"
            echo "  --config      Configuration file (default: configs/inference_config.yaml)"
            echo "  --input       Input data path (required)"
            echo "  --output      Output directory"
            echo "  --scene-id    Scene ID"
            echo "  --log-level   Logging level (default: INFO)"
            echo "  --visualize   Enable visualization"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$INPUT_PATH" ]; then
    echo "Error: --input argument is required!"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file $CONFIG_FILE not found!"
    exit 1
fi

echo "Starting inference with configuration: $CONFIG_FILE"

# Build command
CMD="python scripts/inference.py --config $CONFIG_FILE --input $INPUT_PATH --log-level $LOG_LEVEL"

if [ ! -z "$OUTPUT_DIR" ]; then
    CMD="$CMD --output $OUTPUT_DIR"
fi

if [ ! -z "$SCENE_ID" ]; then
    CMD="$CMD --scene-id $SCENE_ID"
fi

if [ ! -z "$VISUALIZE" ]; then
    CMD="$CMD $VISUALIZE"
fi

echo "Running: $CMD"
exec $CMD
