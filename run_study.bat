@echo off
cd /d "%~dp0"
uv run --locked --extra neural python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || goto fail
uv run --locked --extra neural pytest -q --basetemp=.pytest-tmp || goto fail
uv run --locked --extra neural python scripts/download_data.py || goto fail
uv run --locked --extra neural python scripts/make_split.py || goto fail
uv run --locked --extra neural python scripts/run_classical_study.py || goto fail
uv run --locked --extra neural python scripts/run_learned_study.py || goto fail
set MPLBACKEND=Agg
uv run --locked --extra neural python scripts/make_figure.py || goto fail
echo.
echo Finished. Results are in results\classical, results\learned and results\figures.
pause
exit /b 0
:fail
echo.
echo A step failed. Read the message above.
pause
exit /b 1
