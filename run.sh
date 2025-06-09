#!/bin/bash

# Student Score Prediction Pipeline Execution Script
# This script runs the complete end-to-end pipeline for student score prediction

set -e  # Exit on any error

echo "========================================"
echo "Student Score Prediction Pipeline"
echo "========================================"
echo "Starting pipeline execution at $(date)"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Check if required files exist
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found"
    exit 1
fi

if [ ! -f "main.py" ]; then
    echo "Error: main.py not found"
    exit 1
fi

echo "✅ Prerequisites check passed"
echo ""

# Run the main pipeline
echo "🚀 Executing main pipeline..."
echo "Command: python main.py"
echo ""

python main.py

echo ""
echo "========================================"
echo "Pipeline Execution Summary"
echo "========================================"

# Check if the primary output file exists
if [ -f "data/modeling_outputs/student_score_predictions.csv" ]; then
    echo "✅ Primary output file created: data/modeling_outputs/student_score_predictions.csv"
    echo "📊 File size: $(du -h data/modeling_outputs/student_score_predictions.csv | cut -f1)"
    echo "📊 Number of predictions: $(($(wc -l < data/modeling_outputs/student_score_predictions.csv) - 1))"
else
    echo "❌ Primary output file not found: student_score_predictions.csv"
    exit 1
fi



if [ -f "pipeline_execution.log" ]; then
    echo "✅ Execution log created: pipeline_execution.log"
fi

echo ""
echo "🎉 Pipeline execution completed successfully at $(date)"
echo "========================================"