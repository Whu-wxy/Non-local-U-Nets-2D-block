# Non-local U-Net for Biomedical Image Segmentation
Implement 2D global aggregation block of Non local U-Nets in pytorch.



尝试了一下，非常占显存。



修正一个错误：并不能任意形状，忽视了求导的问题。

![image1](https://github.com/Whu-wxy/Non-local-U-Nets-2D-block/blob/master/1.png)

![image2](https://github.com/Whu-wxy/Non-local-U-Nets-2D-block/blob/master/2.png)

# Origin tensorflow codes
https://github.com/divelab/Non-local-U-Nets
