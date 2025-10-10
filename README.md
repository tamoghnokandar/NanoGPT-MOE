# Training a MoE Model from scratch
This is an implementation of a 4 expert Mixture of experts model based on the [NanoGPT project](https://github.com/KellerJordan/modded-nanogpt) by Keller Jordan. All experts share the same architecture as the dense model, resulting in a larger overall MoE compared to the dense baseline, though only an equivalent number of parameters are activated per forward pass. I have integrated auxiliary load-balancing loss, and a Z-loss for stability in the code.
# Running the MoE model
```bash
git clone https://github.com/KellerJordan/modded-nanogpt.git && cd modded-nanogpt
uv venv
uv pip install -r requirements.txt
python fineweb10B.py 24
./train.sh
```
