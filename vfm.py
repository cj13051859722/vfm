# -*- coding: UTF-8 -*-
"""
ansys巴西劈裂均质 0.32 3.75e9
python vfm.py --formate=ansys --data_file=data/bxpl_hom.txt --width=0.1 --height=0.1 --force=-1470 --thickness=0.008 --homogenous=1 --vfs=4,5
ansys巴西劈裂非均质 石英泊松比0.08 其余物质泊松比0.26 石英弹模9.64e10 其余物质弹模7.0e10
python vfm.py --formate=ansys --data_file=data/bxpl_non.txt --width=0.05 --height=0.05 --force=-20434 --thickness=0.01 --vfs=16,17,18,19
ansys三点弯曲均质 0.32 3.75e9
python vfm.py --formate=ansys --data_file=data/sdw_hom.txt --width=0.2 --height=0.05 --force=-1382 --thickness=0.008 --homogenous=1 --vfs=1,7
ansys三点弯曲非均质 石英泊松比0.08 其余物质泊松比0.26 石英弹模9.64e10 其余物质弹模7.0e10
python vfm.py --formate=ansys --data_file=data/sdw_non.txt --width=0.2 --height=0.05 --force=-1705 --thickness=0.01 --vfs=0,1,6,7

dic巴西劈裂均质 0.32 3.75e9
python vfm.py --formate=dic --data_file=data/bxpl_hom.mat --width=0.1 --height=0.1 --force=-1470 --thickness=0.008 --homogenous=1 --vfs=4,5
dic巴西劈裂非均质 石英泊松比0.08 其余物质泊松比0.26 石英弹模9.64e10 其余物质弹模7.0e10
python vfm.py --formate=dic  --image_file=data/bxpl_non.jpg --data_file=data/bxpl_non.mat --width=0.05 --height=0.05 --force=-20434 --thickness=0.01 --vfs=16,17,18,19
dic三点弯曲均质 0.32 3.75e9
python vfm.py --formate=dic --data_file=data/sdw_hom.mat --width=0.2 --height=0.05 --force=-1382 --thickness=0.008 --homogenous=1 --vfs=1,7
dic三点弯曲非均质 石英泊松比0.08 其余物质泊松比0.26 石英弹模9.64e10 其余物质弹模7.0e10
python vfm.py --formate=dic --image_file=data/sdw_non.jpg --data_file=data/sdw_non.mat --width=0.2 --height=0.05 --force=-1705 --thickness=0.01 --vfs=0,1,6,7
"""
import numpy as np
import argparse
from vfm_utils import valfromtxt, valfrommat, getab

def vfm(formate, data_file, image_file, width, height, force, thickness, homogenous, vfs, noise=0):
    # step1 处理数据
    if formate=='ansys':
        values = valfromtxt(data_file,noise)
    else:
        values,_,_,_ = valfrommat(data_file,image_file,width=width,height=height)
    # step2 获得a，b矩阵
    a, b = getab(values, [width, height, force, thickness], vfs=vfs)
    # step3 求解
    if homogenous:
        a_tmp = [[a[vfs[0]][0], a[vfs[0]][2]], [a[vfs[1]][0], a[vfs[1]][2]]]
        b_tmp = [b[vfs[0]], b[vfs[1]]]
        q = np.linalg.solve(a_tmp, b_tmp)
        v = q[1] / q[0]
        e = q[0] * (1 - v ** 2)
    else:
        a_tmp = [a[vfs[0]], a[vfs[1]], a[vfs[2]], a[vfs[3]]]
        b_tmp = [b[vfs[0]], b[vfs[1]], b[vfs[2]], b[vfs[3]]]
        q = np.linalg.solve(a_tmp, b_tmp)
        v = q[2] / q[0], q[3] / q[1]
        e = q[0] * (1 - v[0] ** 2), q[1] * (1 - v[1] ** 2)
    return v,e

if __name__=="__main__":
    # step1 输入参数
    parser = argparse.ArgumentParser()
    # ansys/dic变形场数据
    parser.add_argument('--formate', type=str, help="dic or ansys", default="ansys")
    parser.add_argument('--data_file', type=str, help="strain field data")
    parser.add_argument('--image_file', type=str, help="dic experimental binarization picture, valid when dic and non-homogeneous", default="")
    # 边界条件
    parser.add_argument('--width', type=float)
    parser.add_argument('--height', type=float)
    parser.add_argument('--force', type=float)
    parser.add_argument('--thickness', type=float)
    # 均质或非均质
    parser.add_argument('--homogenous', type=bool, default=False)
    # 虚场编号
    parser.add_argument('--vfs', type=str)
    # 其他
    parser.add_argument('--noise', type=float,default=0.0)
    args = parser.parse_args()
    print(args)

    # step2 运行
    v,e = vfm(
        formate=args.formate,
        data_file=args.data_file, 
        image_file=args.image_file,
        width=args.width, 
        height=args.height, 
        force=args.force, 
        thickness=args.thickness, 
        homogenous=args.homogenous, 
        vfs=[int(num) for num in args.vfs.split(",")], 
        noise=0)
    print(v,e)

    print("ok")