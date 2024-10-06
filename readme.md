<div align="center">
<h2>[CVPR 2024] From-Ground-To-Objects: Coarse-to-Fine Self-supervised Monocular Depth Estimation of Dynamic Objects with Ground Contact Prior
</h2>

<div>    
    <a href='https://sites.google.com/view/jaehomoon/' target='_blank'>Jaeho Moon</a>&nbsp;
    <a href='https://sites.google.com/view/juan-luis-gb' target='_blank'>Juan Luis Gonzalez Bello</a>&nbsp;
    <a href='https://www.viclab.kaist.ac.kr/' target='_blank'>Byeongjun Kwon</a>&nbsp;
    <a href='https://www.viclab.kaist.ac.kr/' target='_blank'>Munchurl Kim</a><sup>†</sup>
</div>
<div>
    <sup>†</sup>Co-corresponding author</span>
</div>
<div>
    Korea Advanced Institute of Science and Technology, South Korea
</div>

<div>
    <h4 align="center">
        <a href="https://kaist-viclab.github.io/From_Ground_To_Objects_site/" target='_blank'>
        <img src="https://img.shields.io/badge/🐳-Project%20Page-blue">
        </a>
        <a href="https://arxiv.org/abs/2312.10118" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2312.10118-b31b1b.svg">
        </a>
        <a href="https://youtu.be/-pOJ1g01G6o?si=De4mXRqFK-ClzaWR" target='_blank'>
        <img src="https://img.shields.io/badge/Demo%20Video-%23FF0000.svg?logo=YouTube&logoColor=white">
        </a>
    </h4>
</div>

---

<div align="center">
    <h4>
        This repository is the official PyTorch implementation of "From-Ground-To-Objects: Coarse-to-Fine Self-supervised Monocular Depth Estimation of Dynamic Objects with Ground Contact Prior".
    </h4>
</div>
</div>


## Data Prepare

Download dynamic object masks for Cityscapes dataset from [DynamicDepth github](https://github.com/AutoAILab/DynamicDepth)


## Pretraind Model

Pretrained models can be downloaded from [here](https://drive.google.com/drive/folders/1C9SHo3_sRe1OYBREKhxXsDCosuXGZNK6?usp=drive_link)

Put model checkpoints (`mono_encoder.pth` & `mono_depth.pth`) in `/checkpoints/MonoViT/`


### Depth estimation results on Cityscapes.

`WIR`: Whole Image Region / `DOR`: Dynamic Object Region

| Method       | Input size |     | abs rel | a1    | 
|--------------|------------|-----|---------|-------|
| Ours-MonoViT | 192 x 640  | WIR | 0.087   | 0.921 | 
|              |            | DOR | 0.099   | 0.910 | 

Precomputed results (`disparity_map` & `error_map`) can be downloaded from [here](https://drive.google.com/drive/folders/1hlEcE_AcRWhREth0tTj8a06u0XXS7J_g?usp=drive_link)


## Test

```
# Test pretrained MonoViT with our proposed method on Cityscapes dataset
python test.py --config ./configs/test_monovit_cs.yaml
```






## Citation

If you find this work useful, please consider citing:
```BibTex
@inproceedings{moon2024ground,
  title={From-Ground-To-Objects: Coarse-to-Fine Self-supervised Monocular Depth Estimation of Dynamic Objects with Ground Contact Prior},
  author={Moon, Jaeho and Bello, Juan Luis Gonzalez and Kwon, Byeongjun and Kim, Munchurl},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10519--10529},
  year={2024}
}
```


## Reference

Monodepth: https://github.com/nianticlabs/monodepth2

MonoViT: https://github.com/zxcqlf/MonoViT

DynamicDepth: https://github.com/AutoAILab/DynamicDepth

## Acknowledgement

This work was supported by the Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT): No. 2021-0-00087, Development of high-quality conversion technology for SD/HD low-quality media and No. RS2022-00144444, Deep Learning Based Visual Representational Learning and Rendering of Static and Dynamic Scenes.