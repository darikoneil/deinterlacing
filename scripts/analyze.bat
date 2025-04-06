@echo off

echo [0;33m "Formatting & Linting --> deinterlacing" [0m

:: move to project root
cd ..

echo [0;33m "Formatting (RUFF)..." [0m
:: run ruff formatter
ruff format

echo [0;33m "Linting (RUFF)..." [0m
:: run ruff linter
ruff check ./deinterlacing ./tests -o .ruff.json --output-format json --fix --no-cache

echo [0;33m "Linting (FLAKE8 PLUGINS)..." [0m
:: run flake8 first to check dunder all and class attributes order
flake8 ./deinterlacing

echo [0;33m "Finished Formatting & Linting --> deinterlacing" [0m
