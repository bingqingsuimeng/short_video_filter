---
tags:
- clip
- mobileclip2
pipeline_tag: zero-shot-image-classification
license: apple-amlr
---

# ONNX export of timm/MobileCLIP2-S0-OpenCLIP

This model is an export of [timm/MobileCLIP2-S0-OpenCLIP](https://huggingface.co/timm/MobileCLIP2-S0-OpenCLIP). It can be used with the [`open_clip_inference`](https://crates.io/crates/open_clip_inference) rust crate, or any other ONNX Runtime based implementation.

## Usage with `open_clip_inference` in Rust: 

```rust
use open_clip_inference::Clip;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_id = "RuteNL/MobileCLIP2-S0-OpenCLIP-ONNX";
    let clip = Clip::from_hf(model_id).build().await?;

    let img = image::open(Path::new("assets/img/cat_face.jpg")).expect("img load err");
    let texts = &[
        "A photo of a cat",
        "A photo of a dog",
        "A photo of a beignet",
    ];

    let results = clip.classify(&img, texts)?;

    for (text, prob) in results {
        println!("{}: {:.2}%", text, prob * 100.0);
    }

    Ok(())
}
```

---

# Model card for MobileCLIP2-S0-OpenCLIP

These weights and model card are adapted from the original Apple model at https://huggingface.co/apple/MobileCLIP2-S0. This version uses canonical OpenCLIP configs and weight naming.

MobileCLIP2 was introduced in [MobileCLIP2: Improving Multi-Modal Reinforced Training](http://arxiv.org/abs/2508.20691) (TMLR August 2025 <mark>Featured</mark>), by Fartash Faghri, Pavan Kumar Anasosalu Vasu, Cem Koc, Vaishaal Shankar, Alexander T Toshev, Oncel Tuzel, Hadi Pouransari.

This repository contains the **MobileCLIP2-S0** checkpoint.

### Highlights

* `MobileCLIP2-S4` matches the accuracy of SigLIP-SO400M/14 with 2x fewer parameters and surpasses DFN ViT-L/14 at 2.5x lower latency measured on iPhone12 Pro Max.
* `MobileCLIP-S3/S4` are our new architectures trained on MobileCLIP’s training dataset, DataCompDR-1B (dashed lines).
* Our smallest variant `MobileCLIP-S0` obtains similar zero-shot performance as [OpenAI](https://arxiv.org/abs/2103.00020)'s ViT-B/16 model while being 4.8x faster and 2.8x smaller.
* `MobileCLIP-S2` obtains better avg zero-shot performance than [SigLIP](https://arxiv.org/abs/2303.15343)'s ViT-B/16 model while being 2.3x faster and 2.1x smaller, and trained with 3x less seen samples.
* `MobileCLIP-B (LT)` attains zero-shot ImageNet performance of **77.2%** which is significantly better than recent works like [DFN](https://arxiv.org/abs/2309.17425) and [SigLIP](https://arxiv.org/abs/2303.15343) with similar architectures or even [OpenAI's ViT-L/14@336](https://arxiv.org/abs/2103.00020).


## Checkpoints and Results (Original Apple links)

| Model                                                     | # Seen <BR>Samples (B) | # Params (M) <BR> (img + txt) | Latency (ms) <BR> (img + txt) | IN-1k Zero-Shot <BR> Top-1 Acc. (%) | Avg. Perf. (%) <BR> on 38 datasets |
|:----------------------------------------------------------|:----------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------------:|:----------------------------------:|
| [MobileCLIP2-S0](https://hf.co/apple/MobileCLIP2-S0)      |           13           |          11.4 + 42.4          |           1.5 + 1.6           |               71.5                 |                59.7                 |
| [MobileCLIP2-S2](https://hf.co/apple/MobileCLIP2-S2)      |           13           |          35.7 + 63.4          |           3.6 + 3.3           |               77.2                 |                64.1                 |
| [MobileCLIP2-B](https://hf.co/apple/MobileCLIP2-B)        |           13           |          86.3 + 63.4          |          10.4 + 3.3           |               79.4                 |                65.8                 |
| [MobileCLIP2-S3](https://hf.co/apple/MobileCLIP2-S3)      |           13           |         125.1 + 123.6         |           8.0 + 6.6           |               80.7                 |                66.8                 |
| [MobileCLIP2-L/14](https://hf.co/apple/MobileCLIP2-L-14)  |           13           |         304.3 + 123.6         |          57.9 + 6.6           |               81.9                 |                67.8                 |
| [MobileCLIP2-S4](https://hf.co/apple/MobileCLIP2-S4)      |           13           |         321.6 + 123.6         |          19.6 + 6.6           |               81.9                 |                67.5                 |
| [MobileCLIP-S0](https://hf.co/apple/MobileCLIP-S0)        |           13           |          11.4 + 42.4          |           1.5 + 1.6           |               67.8                 |                58.1                 |
| [MobileCLIP-S1](https://hf.co/apple/MobileCLIP-S1)        |           13           |          21.5 + 63.4          |           2.5 + 3.3           |               72.6                 |                61.3                 |
| [MobileCLIP-S2](https://hf.co/apple/MobileCLIP-S2)        |           13           |          35.7 + 63.4          |           3.6 + 3.3           |               74.4                 |                63.7                 |
| [MobileCLIP-B](https://hf.co/apple/MobileCLIP-B)          |           13           |          86.3 + 63.4          |          10.4 + 3.3           |               76.8                 |                65.2                 |
| [MobileCLIP-B (LT)](https://hf.co/apple/MobileCLIP-B-LT)  |           36           |          86.3 + 63.4          |          10.4 + 3.3           |               77.2                 |                65.8                 |
| [MobileCLIP-S3](https://hf.co/apple/MobileCLIP-S3)        |           13           |         125.1 + 123.6         |           8.0 + 6.6           |               78.3                 |                66.3                 |
| [MobileCLIP-L/14](https://hf.co/apple/MobileCLIP-L-14)    |           13           |         304.3 + 123.6         |          57.9 + 6.6           |               79.5                 |                66.9                 |
| [MobileCLIP-S4](https://hf.co/apple/MobileCLIP-S4)        |           13           |         321.6 + 123.6         |          19.6 + 6.6           |               79.4                 |                68.1                 |


## How to Use

```py
import torch
import open_clip
from PIL import Image
from urllib.request import urlopen
from timm.utils import reparameterize_model

model, _, preprocess = open_clip.create_model_and_transforms('MobileCLIP2-S0', pretrained='dfndr2b')
model.eval()
tokenizer = open_clip.get_tokenizer('MobileCLIP2-S0')

# For inference/model exporting purposes, optionally reparameterize for better performance
model = reparameterize_model(model)

image = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))
image = preprocess(image).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat", "a doughnut"])

with torch.no_grad(), torch.amp.autocast(image.device.type):
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)
```
