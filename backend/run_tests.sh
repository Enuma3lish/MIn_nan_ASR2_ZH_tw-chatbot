#!/bin/bash

# Run Tests Script for Min Nan & Chinese Voice Chatbot Backend

echo "======================================"
echo "Min Nan & Chinese Voice Chatbot Tests"
echo "======================================"
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "❌ pytest is not installed"
    echo "Installing test dependencies..."
    pip install -r requirements.txt
    echo ""
fi

echo "Select test option:"
echo "1. Run all tests (with coverage)"
echo "2. Run unit tests only"
echo "3. Run integration tests only"
echo "4. Run fast tests (exclude model loading)"
echo "5. Run specific test file"
echo "6. View coverage report"
echo ""

read -p "Enter option (1-6): " option

case $option in
    1)
        echo ""
        echo "Running all tests with coverage..."
        pytest --cov=app --cov-report=html --cov-report=term-missing -v
        ;;
    2)
        echo ""
        echo "Running unit tests only..."
        pytest -m unit -v
        ;;
    3)
        echo ""
        echo "Running integration tests only..."
        pytest -m integration -v
        ;;
    4)
        echo ""
        echo "Running fast tests (excluding model loading)..."
        pytest -m "not slow" -v
        ;;
    5)
        echo ""
        echo "Available test files:"
        echo "  1. test_asr_service.py"
        echo "  2. test_tts_service.py"
        echo "  3. test_translation_service.py"
        echo "  4. test_api_endpoints.py"
        echo ""
        read -p "Enter file number (1-4): " file_num

        case $file_num in
            1)
                pytest tests/unit/test_asr_service.py -v
                ;;
            2)
                pytest tests/unit/test_tts_service.py -v
                ;;
            3)
                pytest tests/unit/test_translation_service.py -v
                ;;
            4)
                pytest tests/integration/test_api_endpoints.py -v
                ;;
            *)
                echo "Invalid option"
                ;;
        esac
        ;;
    6)
        echo ""
        echo "Opening coverage report..."
        if [ -d "htmlcov" ]; then
            if [[ "$OSTYPE" == "darwin"* ]]; then
                open htmlcov/index.html
            elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
                xdg-open htmlcov/index.html
            else
                echo "Coverage report available at: htmlcov/index.html"
            fi
        else
            echo "❌ No coverage report found. Run tests with coverage first (option 1)"
        fi
        ;;
    *)
        echo "Invalid option"
        ;;
esac

echo ""
echo "======================================"
echo "Test run complete!"
echo "======================================"
