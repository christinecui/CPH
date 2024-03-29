# CPH
The codes of paper Effective Comparative Prototype Hashing for Unsupervised Domain Adaptation, AAAI 2024.
https://ojs.aaai.org/index.php/AAAI/article/view/28674

**Prerequisites**
Requirements for pytorch
   
    
**Usage**

We integrate train step and test step in a bash file T00X_{dataset}.sh, please run it as follows:

    ./T001_office_home.sh
    ./T002_office_31.sh
    ./T003_digits.sh

If you find our approach useful in your research, please consider citing:

@inproceedings{CPH2024,

  author       = {Hui Cui and Lihai Zhao and Fengling Li and Lei Zhu and Xiaohui Han and Jingjing Li},
  
  title        = {Effective Comparative Prototype Hashing for Unsupervised Domain Adaptation},
  
  booktitle    = {AAAI},
  
  pages        = {8329--8337},
  
  year         = {2024}
}
