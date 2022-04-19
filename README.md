# vfm
step1 下载数据

git clone https://github.com/cj13051859722/virtual-field-method.git


step2 运行脚本

cd virtual-field-method

step2.1 ansys巴西劈裂均质 0.32 3.75e9

python vfm.py --formate=ansys --data_file=data/bxpl_hom.txt --width=0.1 --height=0.1 --force=-1470 --thickness=0.008 --homogenous=1 --vfs=4,5

step2.2 ansys巴西劈裂非均质 石英泊松比0.08 其余物质泊松比0.26 石英弹模9.64e10 其余物质弹模7.0e10

python vfm.py --formate=ansys --data_file=data/bxpl_non.txt --width=0.05 --height=0.05 --force=-20434 --thickness=0.01 --vfs=16,17,18,19

step2.3 ansys三点弯曲均质 0.32 3.75e9

python vfm.py --formate=ansys --data_file=data/sdw_hom.txt --width=0.2 --height=0.05 --force=-1382 --thickness=0.008 --homogenous=1 --vfs=1,7

step2.4 ansys三点弯曲非均质 石英泊松比0.08 其余物质泊松比0.26 石英弹模9.64e10 其余物质弹模7.0e10

python vfm.py --formate=ansys --data_file=data/sdw_non.txt --width=0.2 --height=0.05 --force=-1705 --thickness=0.01 --vfs=0,1,6,7

step2.5 dic巴西劈裂均质 0.32 3.75e9

python vfm.py --formate=dic --data_file=data/bxpl_hom.mat --width=0.1 --height=0.1 --force=-1470 --thickness=0.008 --homogenous=1 --vfs=4,5

step2.6 dic巴西劈裂非均质 石英泊松比0.08 其余物质泊松比0.26 石英弹模9.64e10 其余物质弹模7.0e10

python vfm.py --formate=dic  --image_file=data/bxpl_non.jpg --data_file=data/bxpl_non.mat --width=0.05 --height=0.05 --force=-20434 --thickness=0.01 --vfs=16,17,18,19

step2.7 dic三点弯曲均质 0.32 3.75e9

python vfm.py --formate=dic --data_file=data/sdw_hom.mat --width=0.2 --height=0.05 --force=-1382 --thickness=0.008 --homogenous=1 --vfs=1,7

step2.8 dic三点弯曲非均质 石英泊松比0.08 其余物质泊松比0.26 石英弹模9.64e10 其余物质弹模7.0e10

python vfm.py --formate=dic --image_file=data/sdw_non.jpg --data_file=data/sdw_non.mat --width=0.2 --height=0.05 --force=-1705 --thickness=0.01 --vfs=0,1,6,7


question:

1.data数据获取方式

ansys数据：运行ansys apdl命令，文件位于ansys_apdl。运行ansys_apdl/bxpl_hom.txt可获得data/bxpl_hom.txt。

DIC数据：使用ncorr开源软件进行图像分析后，截取strain数据保存。

2.apdl代码建模命令

python ansys_apdl/ansys_apdl.py