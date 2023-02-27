# ControlNet-for-Any-Basemodel [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BI0TobTdjTI1VBSTjLXKOfh6Ps7uj6Ye?usp=sharing)

This repository provides the simplest tutorial code for developers using ControlNet with basemodel in the diffuser framework instead of WebUI. Our work builds highly on other excellent works. Although theses works have made some attemptes, there is no tutorial for supporting diverse ControlNet in diffusers.

<center><img src="https://github.com/lllyasviel/ControlNet/raw/main/github_page/he.png" width="100%" height="100%"></center> 

We have also supported [T2I-Adapter-for-Diffusers](https://github.com/haofanwang/T2I-Adapter-for-Diffusers), [Lora-for-Diffusers](https://github.com/haofanwang/Lora-for-Diffusers). Don't be mean to give us a star if it is helful to you.

# ControlNet + Anything-v3
Our goal is to replace the basemodel of ControlNet and infer in diffusers framework. The original [ControlNet](https://github.com/lllyasviel/ControlNet) is trained in pytorch_lightning, and the [released weights](https://huggingface.co/lllyasviel/ControlNet/tree/main/models) with only [stable-diffusion-1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) as basemodel. However, it is more flexible for users to adopt their own basemodel instead of sd-1.5. Now, let's take [anything-v3](https://huggingface.co/Linaqruf/anything-v3.0/tree/main) as an example. We will show you how to achieve this (ControlNet-AnythingV3) step by step. We do provide a Colab demo [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BI0TobTdjTI1VBSTjLXKOfh6Ps7uj6Ye?usp=sharing), but it only works for Colab Pro users with larger RAM.

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

Gratefully, [Takuma Mori](https://github.com/takuma104) has supported it in this recent [PR](https://github.com/huggingface/diffusers/pull/2407), so that we can easily achieve this. As it is still under-devlopement, so it may be unstable, thus we have to use a specific commit version. I will reformat this section once the PR is mergered into diffusers.

```bash
git clone https://github.com/takuma104/diffusers.git
cd diffusers
git checkout 9a37409663a53f775fa380db332d37d7ea75c915
pip install .
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


# ControlNet + Inpainting

This is to support ControlNet with the ability to only modify a target region instead of full image just like [stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting). For now, we provide the condition (pose, segmentation map) beforehands, but you can use adopt pre-trained detector used in ControlNet.

We have provided the [required pipeline](https://github.com/haofanwang/ControlNet-for-Diffusers/blob/main/pipeline_stable_diffusion_controlnet_inpaint.py) for usage. But please note that this file is fragile without complete testing, we will consider support it in diffusers framework formally later. Also, we find that ControlNet (sd1.5 based) is not compatible to [stable-diffusion-2-inpainting](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting), as some layers have different modules and dimension, if you forcibly load the weights and skip those unmatching layers, the result will be bad

```bash
# assume you already know the absolute path of installed diffusers
cp pipeline_stable_diffusion_controlnet_inpaint.py  PATH/pipelines/stable_diffusion
```

Then, you need to import this new added pipeline in corresponding files
```
PATH/pipelines/__init__.py
PATH/__init__.py
```

Now, we can run

```
import torch
from diffusers.utils import load_image
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionControlNetInpaintPipeline

# we have downloaded models locally, you can also load from huggingface
# control_sd15_seg is converted from control_sd15_seg.safetensors using instructions above
pipe_control = StableDiffusionControlNetInpaintPipeline.from_pretrained("./diffusers/control_sd15_seg",torch_dtype=torch.float16).to('cuda')
pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained("./diffusers/stable-diffusion-inpainting",torch_dtype=torch.float16).to('cuda')

# yes, we can directly replace the UNet
pipe_control.unet = pipe_inpaint.unet
pipe_control.unet.in_channels = 4

# we also the same example as stable-diffusion-inpainting
image = load_image("https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png")
mask = load_image("https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png")

# the segmentation result is generated from https://huggingface.co/spaces/hysts/ControlNet
control_image = load_image('tmptvkkr0tg.png')

image = pipe_control(prompt="Face of a yellow cat, high resolution, sitting on a park bench", 
                     negative_prompt="lowres, bad anatomy, worst quality, low quality",
                     controlnet_hint=control_image, 
                     image=image,
                     mask_image=mask,
                     num_inference_steps=100).images[0]

image.save("inpaint_seg.jpg")
```

The following images are original image, mask image, segmentation (control hint) and generated new image.

<img src="https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png" width="20%" height="20%"> <img src="https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png" width="20%" height="20%"> <img src="https://github.com/haofanwang/ControlNet-for-Diffusers/blob/main/images/tmptvkkr0tg.png" width="20%" height="20%"> <img src="https://github.com/haofanwang/ControlNet-for-Diffusers/blob/main/images/inpaint_seg.jpg" width="20%" height="20%">

You can also use pose as control hint. But please note that it is suggested to use OpenPose format, which is consistent to the training process. If you just want to test a few images without install OpenPose locally, you can directly use [online demo of ControlNet](https://huggingface.co/spaces/hysts/ControlNet) to generate pose image given the resized 512x512 input.

```
image = load_image("./images/pose_image.jpg")
mask = load_image("./images/pose_mask.jpg")
pose_image = load_image('./images/pose_hint.png')

image = pipe_control(prompt="Face of a young boy smiling", 
                     negative_prompt="lowres, bad anatomy, worst quality, low quality",
                     controlnet_hint=pose_image, 
                     image=image,
                     mask_image=mask,
                     num_inference_steps=100).images[0]

image.save("inpaint_pos.jpg")
```

<img src="https://github.com/haofanwang/ControlNet-for-Diffusers/blob/main/images/pose_image.jpg" width="20%" height="20%"> <img src="https://github.com/haofanwang/ControlNet-for-Diffusers/blob/main/images/pose_mask.jpg" width="20%" height="20%"> <img src="https://github.com/haofanwang/ControlNet-for-Diffusers/blob/main/images/pose_hint.png" width="20%" height="20%"> <img src="https://github.com/haofanwang/ControlNet-for-Diffusers/blob/main/images/inpaint_pos.jpg" width="20%" height="20%">

# ControlNet + Inpainting + Img2Img
We have uploaded [pipeline_stable_diffusion_controlnet_inpaint_img2img.py](https://github.com/haofanwang/ControlNet-for-Diffusers/blob/main/pipeline_stable_diffusion_controlnet_inpaint_img2img.py) to support img2img. You can follow the same instruction as [this section](https://github.com/haofanwang/ControlNet-for-Diffusers#controlnet--inpainting).

# Multi-ControlNet (experimental)
Similar to [T2I-Adapter](https://github.com/TencentARC/T2I-Adapter), ControlNet also supports multiple control images as input. The idea behind is simple, as the base model is frozen, we can combine the outputs from ControlNet1 and ControlNet2, and use it as input to UNet. Here, we provide pseudocode for reference. You need to modify the pipeline as below.

```
control1 = controlnet1(latent_model_input, t, encoder_hidden_states=prompt_embeds, controlnet_hint=controlnet_hint1)
control2 = controlnet2(latent_model_input, t, encoder_hidden_states=prompt_embeds, controlnet_hint=controlnet_hint2)

# please note that the weights should be adjusted accordingly
control1_weight = 1.00 # control_any3_openpose
control2_weight = 0.50 # control_sd15_depth

merged_control = []
for i in range(len(control1)):
    merged_control.append(control1_weight*control[i]+control2_weight*control_1[i])
control = merged_control

noise_pred = unet(latent_model_input, t, encoder_hidden_states=prompt_embeds, cross_attention_kwargs=cross_attention_kwargs, control=control).sample
```

Here is an example of Multi-ControlNet, where we use pose and depth map are control hints. The test images are both credited to [T2I-Adapter](https://github.com/TencentARC/T2I-Adapter).

<img src="https://github.com/haofanwang/ControlNet-for-Diffusers/blob/main/images/person_keypose.png" width="20%" height="20%"> <img src="https://github.com/haofanwang/ControlNet-for-Diffusers/blob/main/images/desk_depth.png" width="20%" height="20%"> <img src="https://github.com/haofanwang/ControlNet-for-Diffusers/blob/main/images/controlnet_test_pose_multi1.jpeg" width="20%" height="20%">

# Acknowledgement
We first thanks the author of [ControlNet](https://github.com/lllyasviel/ControlNet) for such a great work, our converting code is borrowed from [here](https://github.com/lllyasviel/ControlNet/discussions/12). We are also appreciated the contributions from this [pull request](https://github.com/huggingface/diffusers/pull/2407) in diffusers, so that we can load ControlNet into diffusers.

# Contact
The repo is still under active development, if you have any issue when using it, feel free to open an issue.
