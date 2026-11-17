@echo off
cd /d "%~dp0"
echo === Criando ambiente virtual ===
python -m venv venv || (echo Instale o Python 3.11 ou 3.12 de python.org marcando "Add to PATH" & pause & exit /b 1)
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo === Testando GPU ===
python -c "import onnxruntime as o; print('Provedores:', o.get_available_providers())"
echo Se aparecer CUDAExecutionProvider, a GPU esta pronta. Rode iniciar.bat
pause
