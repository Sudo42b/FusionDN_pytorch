# FusionDN_pytorch
unofficial implementation FusionDN_pytorch

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FLEE-SEON-WOO%2FFusionDN_pytorch&count_bg=%231A0FD9&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)


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
