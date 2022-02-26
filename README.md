# FusionDN_pytorch
unofficial implementation FusionDN_pytorch



This repository is reimplement the code of the following [repo](https://github.com/hanna-xu/FusionDN)<br>
```
@inproceedings{xu2020aaai,
  title={FusionDN: A Unified Densely Connected Network for Image Fusion},
  author={Xu, Han and Ma, Jiayi and Le, Zhuliang and Jiang, Junjun and Guo, Xiaojie},
  booktitle={Proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI)},
  pages={12484--12491},
  year={2020}
}
```


It is a unified model for multiple image fusion tasks, including:<br>
1) visible and infrared image fusion<br>
2) multi-exposure image fusion<br>
3) multi-focus image fusion<br>



## Framework:<br>
 Overall procedure:<br>
<div align=center><img src="https://github.com/LEE-SEON-WOO/FusionDN_pytorch/blob/main/imgs/procedure.jpg" width="440" height="290"/></div><br>

Intuitive description of data flow and the process of EWC:<br>
<div align=center><img src="https://github.com/LEE-SEON-WOO/FusionDN_pytorch/blob/main/imgs/MultiTask.jpg" width="510" height="200"/></div><br>

## Fused results:<br>
<div align=center><img src="https://github.com/LEE-SEON-WOO/FusionDN_pytorch/blob/main/imgs/res1.jpg" width="900" height="490"/></div>
<div align=center><img src="https://github.com/LEE-SEON-WOO/FusionDN_pytorch/blob/main/imgs/res2.jpg" width="900" height="400"/></div>