VENV = .venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip3

UVICORN = $(VENV)/bin/uvicorn


run: $(VENV)/bin/activate
	$(UVICORN) app:app --reload --port 39001

diffusion: $(VENV)/bin/activate
	$(PYTHON) diffusion.py

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	# https://stackoverflow.com/questions/66992585/how-does-one-use-pytorch-cuda-with-an-a100-gpu
	$(PIP) install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

	$(PIP) install -r requirements.txt


clean:
	rm -rf __pycache__
	rm -rf $(VENV)