# PINet_PG
Code for our PG paper [Human Pose Transfer by Adaptive Hierarchical Deformation](http://cic.tju.edu.cn/faculty/likun/pg2020.pdf)

This is Pytorch implementation for pose transfer on DeepFashion dataset. The code is extremely borrowed from [Pose Transfer](https://github.com/tengteng95/Pose-Transfer). Thanks for their work!

# Requirement
```
conda create -n tip python=3.6
conda install pytorch=1.2 cudatoolkit=10.0 torchvision
pip install scikit-image pillow pandas tqdm dominate 
```



# Data

Data preparation for images and keypoints can follow [Pose Transfer](https://github.com/tengteng95/Pose-Transfer)
Parsing data can be found from [baidu](https://pan.baidu.com/s/1Ic8sIY-eYhGnIoZdDlhgxA) (fetch code:abcd) or [Google drive](https://drive.google.com/file/d/1xwm5cOrj2LSAp8H1wA4_YqK32pCtnXir/view?usp=sharing)

# Test
You can directly download our test results from [baidu](https://pan.baidu.com/s/15tcKgRV12NGByIrr4qoqDw) (fetch code: abcd) or [Google drive](https://drive.google.com/file/d/1r8ebcw3IW7-3AKGkJtcW03ckMVWeAYGZ/view?usp=sharing).<br>
Pre-trained checkpoint can be found from [baidu](https://pan.baidu.com/s/1Orvpt42lV-R2uzI-10q3_A) (fetch code: abcd) or [Google drive](https://drive.google.com/file/d/1xwm5cOrj2LSAp8H1wA4_YqK32pCtnXir/view?usp=sharing) and put it in the folder (-->checkpoints-->fashion_PInet_PG).

**Test by yourself** <br>

```python
python test.py --dataroot ./fashion_data/ --name fashion_PInet_PG --model PInet --phase test --dataset_mode keypoint --norm instance --batchSize 1 --resize_or_crop no --gpu_ids 0 --BP_input_nc 18 --no_flip --which_model_netG PInet --checkpoints_dir ./checkpoints --pairLst ./fashion_data/fasion-resize-pairs-test.csv --which_epoch latest --results_dir ./results
```



# Citation

If you use this code, please cite our paper.

```
@article{pinet,
	author = {Zhang, Jinsong and Liu, Xingzi and Li, Kun},
	title = {Human Pose Transfer by Adaptive Hierarchical Deformation},
	journal = {Computer Graphics Forum},
	volume = {39},
	number = {7},
	pages = {325-337},
	year = {2020}
}
```



# Acknowledgments

Our code is based on [Pose Transfer](https://github.com/tengteng95/Pose-Transfer).
