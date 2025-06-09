#!/bin/bash

# Script to set up the Python virtual environment and install dependencies

VENV_NAME=".venv"

# Check if Python3 is available
if ! command -v python3 &> /dev/null
then
    echo "Python3 could not be found. Please install Python 3."
    exit 1
fi

# Create the virtual environment
if [ ! -d "$VENV_NAME" ]
then
    echo "Creating virtual environment '$VENV_NAME'..."
    python3 -m venv $VENV_NAME
else
    echo "Virtual environment '$VENV_NAME' already exists."
fi

# Activate the virtual environment
# Note: Activation is shell-specific. This script assumes bash/zsh.
# The effects of 'source' are temporary and apply only to this script's execution
# and any child processes it might run.
# Users will need to run 'source .venv/bin/activate' manually in their own shell.
echo "Attempting to activate virtual environment..."
source $VENV_NAME/bin/activate

# Check if activation was successful (by checking if VIRTUAL_ENV is set)
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Failed to activate virtual environment automatically."
    echo "Please activate it manually by running: source $VENV_NAME/bin/activate"
    # exit 1 # Optionally exit if activation is critical for the script's next steps
fi

# Install dependencies
if [ -f "requirements.txt" ]
then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        echo "Dependencies installed successfully."
        echo ""
        echo "Environment setup complete!"
        echo "If you plan to use OpenAI or Hugging Face Hub, remember to set your API keys."
        echo "For example, create a .env file with:"
        echo "OPENAI_API_KEY='your_openai_api_key'"
        echo "HUGGINGFACEHUB_API_TOKEN='your_huggingface_api_token'"
        echo ""
        echo "Activate the environment in your current shell by running: source $VENV_NAME/bin/activate"
    else
        echo "Failed to install dependencies. Please check requirements.txt and your internet connection."
        exit 1
    fi
else
    echo "requirements.txt not found. Cannot install dependencies."
    exit 1
fi

# Deactivating is generally not needed within the script itself,
# as the activation is local to the script's execution context.
# However, if you wanted to show it:
# echo "Deactivating virtual environment (within script scope)..."
# deactivate
# echo "To work on the project, activate the environment in your terminal: source $VENV_NAME/bin/activate"

exit 0
