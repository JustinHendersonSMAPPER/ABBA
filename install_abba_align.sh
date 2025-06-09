#!/bin/bash
# Install ABBA-Align as a command-line tool

echo "Installing ABBA-Align..."

# Install in development mode so changes are reflected immediately
pip install -e . -f setup_abba_align.py

echo "Installation complete!"
echo "You can now use 'abba-align' command"