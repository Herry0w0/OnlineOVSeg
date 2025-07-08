#!/bin/bash
# Training script runner

# Set default values
CONFIG_FILE="configs/train_config.yaml"
RESUME_CHECKPOINT=""
LOG_LEVEL="INFO"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --resume)
            RESUME_CHECKPOINT="$2" 
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--config CONFIG_FILE] [--resume CHECKPOINT] [--log-level LEVEL]"
            echo "  --config      Configuration file (default: configs/train_config.yaml)"
            echo "  --resume      Checkpoint to resume from"
            echo "  --log-level   Logging level (default: INFO)"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file $CONFIG_FILE not found!"
    exit 1
fi

echo "Starting training with configuration: $CONFIG_FILE"

# Build command
CMD="python scripts/train.py --config $CONFIG_FILE --log-level $LOG_LEVEL"

if [ ! -z "$RESUME_CHECKPOINT" ]; then
    CMD="$CMD --resume $RESUME_CHECKPOINT"
fi

echo "Running: $CMD"
exec $CMD
