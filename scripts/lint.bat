@echo off

:: move to project root
cd ..

echo [0;33m "Formatting (RUFF)..." [0m
ruff format ./deinterlace ./tests

echo [0;33m "Linting (RUFF)..." [0m
ruff check ./deinterlace ./tests -o .ruff.json --output-format json --fix --no-cache

:: too opinionated for linting tests
echo [0;33m "Linting (FLAKE8 PLUGINS)..." [0m
flake8 ./deinterlace

echo [0;33m "Finished Formatting & Linting
