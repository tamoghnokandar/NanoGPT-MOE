# Training a MoE Model from scratch
This is an implementation of a 4 expert Mixture of experts model based on the [NanoGPT project](https://github.com/KellerJordan/modded-nanogpt) by Keller Jordan. Each expert has the same architecture as the dense model, so the overall MoE is larger than the dense baseline, but has the same number of activated parameters. 
# Running the MoE model
```bash
git clone https://github.com/KellerJordan/modded-nanogpt.git && cd modded-nanogpt
uv venv
uv pip install -r requirements.txt
python fineweb10B.py 24
./train.sh
```
