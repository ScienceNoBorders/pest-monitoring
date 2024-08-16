#!/bin/bash

PYTHON_SCRIPT="./YoloRecognition.py"
PID_FILE="/tmp/YoloRecognition.pid"

start_script() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 $PID > /dev/null 2>&1; then
            echo "Script is already running with PID $PID."
            return
        else
            echo "Found PID file but no process is running. Cleaning up..."
            rm -f "$PID_FILE"
        fi
    fi

    echo "Starting script..."
    nohup python3 "$PYTHON_SCRIPT" > /dev/null 2>&1 &
    echo $! > "$PID_FILE"
    sleep 1  # 确保进程已经启动
    PID=$(cat "$PID_FILE")

    if kill -0 $PID > /dev/null 2>&1; then
        echo "Script started with PID $PID."
    else
        echo "Failed to start script."
        rm -f "$PID_FILE"
    fi
}

stop_script() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 $PID > /dev/null 2>&1; then
            echo "Stopping script with PID $PID..."
            kill -9 $PID
            rm -f "$PID_FILE"
            echo "Script stopped."
        else
            echo "No process found with PID $PID. Cleaning up..."
            rm -f "$PID_FILE"
        fi
    else
        echo "No PID file found. Script is not running."
    fi
}

status_script() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 $PID > /dev/null 2>&1; then
            echo "Script is running with PID $PID."
        else
            echo "Found PID file but no process is running with PID $PID."
        fi
    else
        echo "Script is not running."
    fi
}

case "$1" in
    start)
        start_script
        ;;
    stop)
        stop_script
        ;;
    restart)
        stop_script
        start_script
        ;;
    status)
        status_script
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
