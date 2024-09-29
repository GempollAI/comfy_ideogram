# Comfy Ideogram

ComfyUI nodes for using Ideogram APIs, including:
* Text to Image
  * with color palette utility
* Remix - Text with Reference Image to Image
* Describe: Image to Text
* Upscale: Upscale Image 

## Usage
Prerequisite: [Ideogram](https://ideogram.ai/) API Key

1. Set the API key in environment variables `IDEOGRAM_KEY` before ComfyUI starts or set it on the node.
2. Use the node as you like
   * If you want to pick a color instead of inputting a color hex value, use `ColorSelect` node.

Example: You can use `example_all_nodes.json` to test all nodes.

## LICENSE
[MIT License](LICENSE)

## Acknowledgements

* `ColorSelect` is modified from [LayerStyle](https://github.com/chflame163/ComfyUI_LayerStyle)'s `ColorPicker`
  * LayerStyle is licensed under MIT License.