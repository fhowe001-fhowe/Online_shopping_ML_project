#!/bin/bash

# 1. Stop the script if any command fails
set -e

# 2. Print what we are doing so we can track progress in the terminal
echo "---Running...---" 

# 3. Execute Script 
python src/main_.py

# 4. Success message
echo "--Run Successful!---"