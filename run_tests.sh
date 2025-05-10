#!/bin/bash

# Set up colored output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Running nuchad package tests...${NC}"
echo

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest is not installed. Please install it with 'uv add pytest'${NC}"
    exit 1
fi

# Check if the data directory exists
if [ ! -d "data" ]; then
    echo -e "${YELLOW}Warning: 'data' directory does not exist. Creating it...${NC}"
    mkdir -p data
fi

# Check if the random_nuchad.csv file exists
if [ ! -f "data/random_nuchad.csv" ]; then
    echo -e "${YELLOW}Warning: 'data/random_nuchad.csv' does not exist.${NC}"
    echo -e "${YELLOW}Tests will be skipped unless this file is available.${NC}"
    echo -e "${YELLOW}Please add the data file to the data directory.${NC}"
fi

# Create results directory if it doesn't exist
if [ ! -d "results" ]; then
    echo -e "${YELLOW}Creating 'results' directory...${NC}"
    mkdir -p results
fi

# Run the tests
echo -e "${YELLOW}Starting tests...${NC}"
pytest -xvs src/tests/test_analysis_scripts.py

# Check if the tests passed
if [ $? -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
else
    echo -e "${RED}Some tests failed. Please check the output above for details.${NC}"
fi 