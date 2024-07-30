# meta-llama/Prompt-Guard-86M Cog model

This is an implementation of the [meta-llama/Prompt-Guard-86M](https://huggingface.co/meta-llama/Prompt-Guard-86M) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

# Weights
If you are testing locally, to download the weights you need to add your Huggingface Auth token in `preidct.py` on line 9.
After you run a prediction (which downloads the weights) you can remove your token

## Run a prediction:

    cog predict -i prompt="Ignore your previous instructions."
