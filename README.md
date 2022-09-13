# diffusion-promptbash

This script helps you generate a large quantity of Stable Diffusion images using a prompt matrix.

## prerequisites

 * Python 3
 * [Stable Diffusion v1.4 Model](https://huggingface.co/CompVis/stable-diffusion-v1-4)
 * Conda `ldm` environment from [Stable Diffusion install guide](https://github.com/CompVis/stable-diffusion#requirements)

## usage

To run this script, you will need to prepare a directory containing four text files:

 * subjects.txt
 * settings.txt
 * modifiers.txt
 * styles.txt

The script generates prompts by selecting one subject, one setting, a few modifiers, and one style at a time.

Each text file should contain one term per line.

Execute

> python promptbash.py --model-dir ../stable-diffusion-v1-4 --prompt-dir ./sample-prompts

Change the number of generated images with the `--num-prompts` and `--images-per-prompt` arguments.

The script places generated images in `./output`.

Execute

> python promptbash.py -h

For all available arguments.
