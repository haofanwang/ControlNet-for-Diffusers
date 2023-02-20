# ControlNet-for-Any-Basemodel

This repository provides the simplest tutorial code for using ControlNet with basemodel in the diffuser framework. Our work builds highly on other excellent works. Although theses works have made some attemptes, there is no tutorial for supporting diverse ControlNet in diffusers.


# Instruction
Our goal is to replace the basemodel of ControlNet and infer in diffusers framework. The original [ControlNet](https://github.com/lllyasviel/ControlNet) is trained in pytorch_lightning, and the [released weights](https://huggingface.co/lllyasviel/ControlNet/tree/main/models) with only [stable-diffusion-1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) as basemodel. However, it is more flexible for users to adopt their own basemodel instead of sd-1.5. Now, let's take [anything-v3](https://huggingface.co/Linaqruf/anything-v3.0/tree/main) as an example. We will show you how to achieve this (ControlNet-AnythingV3) step by step.

### (1) The first step is to replace basemodel. 

Fortunately, ControlNet has already provided a [guideline](https://github.com/lllyasviel/ControlNet/discussions/12) to transfer the ControlNet to any other community model. The logic behind is as below, where we keep the added control weights and only replace the basemodel. Note that this may not work always, as ControlNet may has some trainble weights in basemodel.
 
 ```bash
 NewBaseModel-ControlHint = NewBaseModel + OriginalBaseModel-ControlHint - OriginalBaseModel
 ```

First, we clone this repo from ControlNet.
 ```bash
 git clone https://github.com/lllyasviel/ControlNet.git
 cd ControlNet
 ```

Then, we have to prepared required weights for [OriginalBaseModel](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main) (path_sd15), [OriginalBaseModel-ControlHint](https://huggingface.co/lllyasviel/ControlNet/tree/main/models) (path_sd15_with_control), [NewBaseModel](https://huggingface.co/Linaqruf/anything-v3.0/tree/main) (path_input). You only need to download following weights, and we use pose as ControlHint and anything-v3 as our new basemodel for instance. We put all weights inside ./models.

 ```bash
 path_sd15 = './models/v1-5-pruned.ckpt'
 path_sd15_with_control = './models/control_sd15_openpose.pth'
 path_input = './models/anything-v3-full.safetensors'
 path_output = './models/control_any3_openpose.pth'
 ```
 
 Finally, we can directly run
 ```bash
 python tool_transfer_control.py
 ```

If successful, you will get the new model. This model can already be used in ControlNet codebase.

```bash
models/control_any3_openpose.pth
 ```

If you want to try with other models, you can just define your own path_sd15_with_control and path_input.

### (2) The second step is to convert into diffusers

Gratefully, [Takuma Mori](https://github.com/takuma104) has supported it in this recent [PR](https://github.com/huggingface/diffusers/pull/2407), so that we can easily achieve this. As it is still under-devlopement, so it may be unstable.

```bash
git clone https://github.com/takuma104/diffusers.git
cd diffusers
git checkout controlnet
```

Given the path of the generated model in step (1), run
```bash
python ./scripts/convert_controlnet_to_diffusers.py --checkpoint_path control_any3_openpose.pth  --dump_path control_any3_openpose --device cpu
```

We have the saved model in control_any3_openpose. Now we can test it as regularly.

```bash
from diffusers import StableDiffusionControlNetPipeline
from diffusers.utils import load_image

pose_image = load_image('https://huggingface.co/takuma104/controlnet_dev/resolve/main/pose.png')
pipe = StableDiffusionControlNetPipeline.from_pretrained("control_any3_openpose").to("cuda")

pipe.safety_checker = lambda images, clip_input: (images, False)

image = pipe(prompt="1gril,masterpiece,graden", controlnet_hint=pose_image).images[0]
image.save("generated.png")
```

The generated result may not be good enough as the pose is kind of hard. So to make sure everything goes well, we suggest to generate a normal pose via [PoseMaker](https://huggingface.co/spaces/jonigata/PoseMaker) or use our provided pose image in ./images/pose.png.

<img src="https://github.com/haofanwang/ControlNet-for-Diffusers/blob/main/images/pose.png" width="25%" height="25%"> <img src="https://github.com/haofanwang/ControlNet-for-Diffusers/blob/main/images/generated.png" width="25%" height="25%">


# Acknowledgement
We first thanks the author of [ControlNet](https://github.com/lllyasviel/ControlNet) for such a great work, our converting code is borrowed from [here](https://github.com/lllyasviel/ControlNet/discussions/12). We are also appreciated the contributions from this [pull request](https://github.com/huggingface/diffusers/pull/2407) in diffusers, so that we can load ControlNet into diffusers.
