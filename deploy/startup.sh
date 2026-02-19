#!/bin/bash
export PYTHONPATH=/home/site/wwwroot/antenv/lib/python3.10/site-packages
cd /home/site/wwwroot
echo "Starting uvicorn on port ${PORT:-8000}"
echo "PYTHONPATH is: $PYTHONPATH"
/home/site/wwwroot/antenv/bin/python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}