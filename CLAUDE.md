# AI Hedge Fund - Agent Guidelines

## Commands
- **Install**: `poetry install`
- **Run**: `poetry run python src/main.py --ticker AAPL,MSFT,NVDA`
- **Backtest**: `poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA`
- **Format**: `poetry run black . && poetry run isort .`
- **Lint**: `poetry run flake8 src/`
- **Type Check**: `poetry run mypy src/`
- **Test**: `poetry run pytest`
- **Single Test**: `poetry run pytest tests/path_to_test.py::test_function`

## Code Style
- **Formatting**: Black with 420 character line length
- **Imports**: Use isort, standard library first, then third-party, then local
- **Type Hints**: Required for all function arguments and return values
- **Models**: Use Pydantic BaseModel for structured data
- **Naming**: snake_case for functions/variables, CamelCase for classes
- **Documentation**: Google-style docstrings with Args/Returns sections
- **Error Handling**: Use try/except with specific exceptions and fallbacks
- **Variables**: Use descriptive names that explain intent
- **Functions**: Small, single-purpose functions with clear docstrings