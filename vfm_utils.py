# -*- coding: UTF-8 -*-
import numpy as np
from collections import defaultdict
import math
import cv2
import scipy.io as scio

# 虚场法
class VIRTUAL_FIELD:
    def __init__(self, vals, visual, span):
        """
        :param vals: 材料的基本参数格式：[width,height]
        :param visual: DIC摄像区域 对三点弯曲虚场有效
        :param span: 跨距 对三点弯曲虚场有效
        """
        self.width = vals[0]
        self.height = vals[1]
        self.visual = visual
        self.span = span
        self.num = 20  # 当前虚场个数

    def Getvals(self, num, x1, x2):
        if num == 0:
            ans = self.Virtual0(x1, x2)  # 三点弯、巴西劈裂
        elif num == 1:
            ans = self.Virtual1(x1, x2)  # 三点弯、巴西劈裂
        elif num == 2:
            ans = self.Virtual2(x1, x2)  # 三点弯
        elif num == 3:
            ans = self.Virtual3(x1, x2)  # 三点弯、巴西劈裂
        elif num == 4:
            ans = self.Virtual4(x1, x2)  # 三点弯、巴西劈裂等同于0
        elif num == 5:
            ans = self.Virtual5(x1, x2)  # 三点弯、巴西劈裂
        elif num == 6:
            ans = self.Virtual6(x1, x2)  # 三点弯
        elif num == 7:
            ans = self.Virtual7(x1, x2)  # 三点弯
        elif num == 8:
            ans = self.Virtual8(x1, x2)  # 巴西劈裂
        elif num == 9:
            ans = self.Virtual9(x1, x2)  # 带visual x2为一次函数 xy
        elif num == 10:
            ans = self.Virtual10(x1, x2)  # 带visual x1为二次函数 xx
        elif num == 11:
            ans = self.Virtual11(x1, x2)  # 带visual x1为三次函数 xx xy
        elif num == 12:
            ans = self.Virtual12(x1, x2)  # 带visual  x2为二次函数 xy
        elif num == 13:
            ans = self.Virtual13(x1, x2)  # 带visual x1为线性函数 xx
        elif num == 14:
            ans = self.Virtual14(x1, x2)  # 带visual x2为正弦函数 xy
        elif num == 15:
            ans = self.Virtual15(x1, x2)  # 带visual x2为二次函数 yy、xy
        elif num == 16:
            ans = self.Virtual16(x1, x2)  # 巴西劈裂、三点弯
        elif num == 17:
            ans = self.Virtual17(x1, x2)  # 巴西劈裂
        elif num == 18:
            ans = self.Virtual18(x1, x2)  # 巴西劈裂、三点弯
        elif num == 19:
            ans = self.Virtual19(x1, x2)  # 巴西劈裂
        else:
            print("this virtual field is error!")
            ans = [0, 0, 0, 0]  # 留用
        return ans

    def Virtual0(self, x1, x2):
        """
        u=0
        v=−y
        :param x1:x1坐标
        :param x2:x2坐标
        :return: 虚场结果 三个的虚应变和顶部在y方向的位移 下同
        """
        sigma1 = 0
        sigma2 = -1
        sigma6 = 0
        u = -self.height
        return [sigma1, sigma2, sigma6, u]

    def Virtual1(self, x1, x2):
        """
        u= xy
        v=y(y-h)
        """
        sigma1 = x2
        sigma2 = 2 * x2 - self.height
        sigma6 = x1
        u = 0
        return [sigma1, sigma2, sigma6, u]

    def Virtual2(self, x1, x2):
        """
        u=x**2-span**2
        v=0
        """
        sigma1 = 2 * x1
        sigma2 = 0
        sigma6 = 0
        u = 0
        return [sigma1, sigma2, sigma6, u]

    def Virtual3(self, x1, x2):
        """
        u= y**2
        v= y(x-h)
        """
        sigma1 = 0
        sigma2 = x1 - self.height
        sigma6 = x2
        u = - self.height ** 2
        return [sigma1, sigma2, sigma6, u]

    def Virtual4(self, x1, x2):
        """
        u=0
        v=y
        """
        sigma1 = 0
        sigma2 = 1
        sigma6 = 0
        u = self.height
        return [sigma1, sigma2, sigma6, u]

    def Virtual5(self, x1, x2):
        """
        u=xy
        v=0
        """
        sigma1 = x2
        sigma2 = 0
        sigma6 = x1
        u = 0
        return [sigma1, sigma2, sigma6, u]

    def Virtual6(self, x1, x2):
        """
        u=-xy
        v=(x**2-span**2)/2
        """
        sigma1 = -x2
        sigma2 = 0
        sigma6 = 0
        u = -self.span ** 2/2
        return [sigma1, sigma2, sigma6, u]

    def Virtual7(self, x1, x2):
        """
        u=0
        v=(x**2-span**2)/2
        """
        sigma1 = 0
        sigma2 = 0
        sigma6 = x1
        u = -self.span ** 2/2
        return [sigma1, sigma2, sigma6, u]

    def Virtual8(self, x1, x2):
        """
        u= x
        v= 0
        """
        sigma1 = 1
        sigma2 = 0
        sigma6 = 0
        u = 0
        return [sigma1, sigma2, sigma6, u]

    def Virtual9(self, x1, x2):
        """
        u= 0
        v= x1+b when -span<x1<0
        v= -x1+b when span<x1<0
        """
        sigma1 = 0
        sigma2 = 0
        if -self.visual <= x1 < 0:
            sigma6 = 1
        elif 0 < x1 <= self.visual:
            sigma6 = -1
        else:
            sigma6 = 0
        u = self.visual
        return [sigma1, sigma2, sigma6, u]

    def Virtual10(self, x1, x2):
        """
        u= x1**2-visual**2   -visual<x1<visual
        v= 0
        """
        if abs(x1) < self.visual:
            sigma1 = 2 * x1
        else:
            sigma1 = 0
        sigma2 = 0
        sigma6 = 0
        u = 0
        return [sigma1, sigma2, sigma6, u]

    def Virtual11(self, x1, x2):
        """
        u= (x1**2-visual**2)*x2   -visual<x1<visual
        v= 0
        """
        if abs(x1) < self.visual:
            sigma1 = 2 * x1 * x2
        else:
            sigma1 = 0
        sigma2 = 0
        if abs(x1) < self.visual:
            sigma6 = x1 ** 2 - self.visual ** 2
        else:
            sigma6 = 0
        u = 0
        return [sigma1, sigma2, sigma6, u]

    def Virtual12(self, x1, x2):
        """
        u= 0
        v= x1**2-visual**2 -visual<x1<visual
        """
        sigma1 = 0
        sigma2 = 0
        if abs(x1) < self.visual:
            sigma6 = 2 * x1
        else:
            sigma6 = 0
        u = -self.visual ** 2
        return [sigma1, sigma2, sigma6, u]

    def Virtual13(self, x1, x2):
        """
        u= x1+b when -span<x1<0
        u= -x1+b when span<x1<0
        v= 0
        """
        if -self.visual <= x1 < 0:
            sigma1 = 1
        elif 0 <= x1 <= self.visual:
            sigma1 = -1
        else:
            sigma1 = 0
        sigma2 = 0
        sigma6 = 0
        u = 0
        return [sigma1, sigma2, sigma6, u]

    def Virtual14(self, x1, x2):
        """
        u= 0
        v= 2*visual*sin(pi*x1/(2*visual)+pi/2)
        """
        sigma1 = 0
        sigma2 = 0
        if abs(x1) < self.visual:
            sigma6 = math.cos((math.pi * x1) / (2 * self.visual) + math.pi / 2)
        else:
            sigma6 = 0
        u = 2 * self.visual / math.pi
        return [sigma1, sigma2, sigma6, u]

    def Virtual15(self, x1, x2):
        """
        u= 0
        v= (x1+self.visual)*x2、(-x1+self.visual)*x2
        """
        sigma1 = 0
        if -self.visual <= x1 < 0:
            sigma2 = x1 + self.visual
        elif 0 <= x1 <= self.visual:
            sigma2 = -x1 + self.visual
        else:
            sigma2 = 0
        if -self.visual <= x1 < 0:
            sigma6 = x2
        elif 0 <= x1 <= self.visual:
            sigma6 = -x2
        else:
            sigma6 = 0
        u = self.height * self.visual
        return [sigma1, sigma2, sigma6, u]

    def Virtual16(self, x1, x2):
        """
        u=0
        v=−sin(pi*y/h/2)
        """
        sigma1 = 0
        sigma2 = -math.pi*math.cos(math.pi*x2/self.height*0.5)/self.height*0.5
        sigma6 = 0
        u = -1
        return [sigma1, sigma2, sigma6, u]

    def Virtual17(self, x1, x2):
        """
        u=e**(x/height)
        v=0
        """
        sigma1 = math.exp(x1/self.height)/self.height
        sigma2 = 0
        sigma6 = 0
        u = 0
        return [sigma1, sigma2, sigma6, u]

    def Virtual18(self, x1, x2):
        """
        u=0
        v=-y**3
        """
        sigma1 = 0
        sigma2 = -3*x2**2
        sigma6 = 0
        u = -self.height**3
        return [sigma1, sigma2, sigma6, u]

    def Virtual19(self, x1, x2):
        """
        u=e**(x**2/height)
        v=0
        """
        sigma1 = 2*x1*math.exp(x1**2/self.height)/self.height
        sigma2 = 0
        sigma6 = 0
        u = 0
        return [sigma1, sigma2, sigma6, u]


def valfromtxt(data_file,noise=None):
    """
    读取ansys分析的应变场，对比真实图片，提取坐标点、材料属性、面积及应力。
    :param data_file: 带解析的ansys数据
    :return: vals参数表，格式为{node:[x1,x2,mat,area,strain_x,strain_y,strain_xy]}
    """
    with open(data_file, 'r') as f:
        values = f.read().split("\n")

    ans = defaultdict(list)
    for value in values:
        if value == '':
            continue
        tmp = [i.strip(' ').strip('.') for i in value.split(' ')] # 以空格间隔

        value_after = []
        for t in tmp:
            if t == "":
                continue
            elif 'E' in t: # 带E浮点数
                t = t.split('E')
                val = float(t[0]) * 10 ** int(t[1])
            else: # 正常浮点数
                val = float(t)
            if noise and len(value_after)>4:
                val+=np.random.normal(loc =0.0 , scale= noise)
            value_after.append(val)
        ans[value_after[0]] += value_after[1:]
    return ans


def valfrommat(data_file, image_file, height, width, index=-1):
    """
    读取dic分析的应变场，对比真实图片，提取坐标点、材料属性、面积及应力。
    :param data_file：应变场数据位置
    :param image_file：图片位置，当均质时数据为""
    :param index: 第几个数据，dic数据为最后一项
    :param height：试件高
    :param width：试件宽
    :return: vals参数表，格式为{node:[x1,x2,mat,area,strain_x,strain_y,strain_xy]}
    """
    # step1 读取mat文件和图片
    data = scio.loadmat(data_file)
    strain_x = data['Strains']['plot_exx_ref_formatted'][0][index].tolist()
    strain_y = data['Strains']['plot_eyy_ref_formatted'][0][index].tolist()
    strain_xy = data['Strains']['plot_exy_ref_formatted'][0][index].tolist()
    if image_file=="": # 均质
        pic_bin=np.ones(shape=(1000,1000))
    else: # 非均质
        pic_gray=cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
        _,pic_bin = cv2.threshold(pic_gray, 170, 255, cv2.THRESH_BINARY)
        pic_bin[pic_bin==255]=2
        pic_bin[pic_bin==0]=1    
    # step2 处理边界区域 去零
    rows, cols = len(strain_x), len(strain_x[0])
    up,left,down,right = 10000000,10000000,-1,-1
    # step2.1 左上
    for i in range(rows):
        for j in range(cols):
            if strain_x[i][j] != 0 or strain_y[i][j] != 0 or strain_xy[i][j] != 0:
                up=min(up,i)
                down=max(down,i)
                left=min(left,j)
                right=max(right,j)
    # step2.3 组合
    strain_x = [strain[left:right + 1] for strain in strain_x][up:down+1]
    strain_y = [strain[left:right + 1] for strain in strain_y][up:down+1]
    strain_xy = [strain[left:right + 1] for strain in strain_xy][up:down+1]
    # step 3 转化为预定格式：{1: [0, 0, 0, 0, strain_x, strain_y, strain_xy]} {node:[x1,x2,mat,area,strain_x,strain_y,strain_xy]}
    rows, cols = len(strain_x), len(strain_x[0])
    print(rows,cols)
    # step 4 面积
    area = (height / rows) * (width / cols)
    # step 5 总和
    x_subset,y_subset=pic_bin.shape[0]/rows,pic_bin.shape[1]/cols
    vals = {}
    node = 0
    for i in range(rows):
        for j in range(cols):
            if strain_x[i][j]==0 and strain_y[i][j]==0 and strain_xy[i][j]==0:
                continue
            node = node + 1
            x1 = width/cols*(j+0.5)-width/2  # x方向
            x2 = height - height/rows*(i+0.5)  # y方向
            vals[node] = [x1, x2, pic_bin[int((i+0.5)*x_subset+0.5)][int((j+0.5)*y_subset+0.5)], area, strain_x[i][j], strain_y[i][j], -strain_xy[i][j]] # dic和ansys的剪应变为负号关系
    return vals,strain_x,strain_y,strain_xy


def getab(vals, att, visual=40 * 10 ** -3, span=90 * 10 ** -3, vfs=None):
    """
    :param vals: 参数表,格式为{node:[x1,x2,mat,area,strain_x,strain_y,strain_xy]}
    :param att: [width, height, F, t]
    :param visual: DIC摄像区域 对三点弯曲虚场有效
    :param span: 跨距 对三点弯曲虚场有效
    :param vfs：虚场编号，若无，则计算所有虚场
    :return: a,b的numpy矩阵
    """
    width, height, F, t = att
    virtual_fields = VIRTUAL_FIELD([width, height], visual, span)
    field_num = virtual_fields.num
    a = [[0] * 4 for _ in range(field_num)]
    b = [0] * field_num
    for row in range(field_num):
        if vfs is None or row in vfs:
            for col in range(4):
                for [x1, x2, mat, area, sigma1, sigma2, sigma6] in vals.values():
                    if col % 2 + 1 == int(mat):  # 属性属于该行 0、2：mat=1 1、3：mat=2
                        sigma1_vir, sigma2_vir, sigma6_vir, u = virtual_fields.Getvals(row, x1, x2)  # row代表使用哪个虚场
                        if col == 0 or col == 1:  # ∫1▒〖𝜀_1 〖𝜀_1^∗〗^((1) )+𝜀_2 〖𝜀_2^∗〗^((1) )+〖1/2 𝜀〗_6 〖𝜀_6^∗〗^((1) ) 𝑑S_1 〗
                            a[row][col] += (sigma1 * sigma1_vir + sigma2 * sigma2_vir + sigma6 * sigma6_vir * 0.5) * area
                        else:  # loc[0] == 2 or loc[0] == 3 ∫1▒〖𝜀_2 〖𝜀_1^∗〗^((1) )+𝜀_2 〖𝜀_1^∗〗^((1) )−〖1/2 𝜀〗_6 〖𝜀_6^∗〗^((1) ) 𝑑S_1 〗
                            a[row][col] += (sigma2 * sigma1_vir + sigma1 * sigma2_vir - sigma6 * sigma6_vir * 0.5) * area
            b[row] = F * u / t
    return np.array(a), np.array(b)


def bias(x, x_ori):
    return abs((x - x_ori) / x_ori)


# 根据ansys数据生成dic数据
def neartest_strain(x1,x2,vals_ansys:dict):
    """
    通过坐标值x1和x2，找到ansys结果中离该坐标最近的一个点并复制其参数
    """
    val=[] #[x1,x2,mat,area,strain_x,strain_y,strain_xy]
    min_dis=float("inf")
    for val_tmp in vals_ansys.values():
        if (val_tmp[0]-x1)**2+(val_tmp[1]-x2)**2<min_dis:
            val=val_tmp[:]
            min_dis=(val_tmp[0]-x1)**2+(val_tmp[1]-x2)**2
    return val


def mat_from_ansys(rows, cols,height,width,file_ansys,circle):
    """
    :param rows:dic数据rows
    :param cols:dic数据cols
    :param height:dic数据高度
    :param width:dic数据宽度
    :param file_ansys:ansys文件地址
    :return vals_dic:dic数据个数的参数表,格式为{node:[x1,x2,mat,area,strain_x,strain_y,strain_xy]}
    :return strain_x,strain_y,strain_xy:应变值
    从ansys结果数据(ansys网格需更细)中生成DIC数据
    """
    area = (height / rows) * (width / cols)
    vals_ansys=valfromtxt(file_ansys)
    vals_dic = {}
    strain_x,strain_y,strain_xy=[[0]*cols for i in range(rows)],[[0]*cols for i in range(rows)],[[0]*cols for i in range(rows)]
    node = 0
    for i in range(rows):
        print(i,rows)
        for j in range(cols):
            node = node + 1
            # step1 生成坐标值
            x1 = width/cols*(j+0.5)-width/2  # x方向
            x2 = height - height/rows*(i+0.5)  # y方向
            # step2.0 如果是圆盘，则点不在圆盘内跳过：
            if circle and x1**2+(x2-height/2)**2>height**2/4:
                continue
            # step2 获取最近的点的坐标及其属性
            neartest_strains=neartest_strain(x1,x2,vals_ansys) #[x1,x2,mat,area,strain_x,strain_y,strain_xy]
            vals_dic[node] = [x1, x2, neartest_strains[2], area, neartest_strains[4], neartest_strains[5], neartest_strains[6]]
            strain_x[i][j],strain_y[i][j],strain_xy[i][j]=neartest_strains[4], neartest_strains[5], -neartest_strains[6]  # dic和ansys的剪应变为负号关系
    return vals_dic,strain_x,strain_y,strain_xy


if __name__ == "__main__":
    # # bxpl_hom
    # rows, cols,height,width=360,358,100 * 10 ** -3, 100 * 10 ** -3
    # file_ansys='../ansys/bxpl_hom/strain_elem.txt'
    # # 生成
    # vals_ansysdic,strain_x_ansysdic,strain_y_ansysdic,strain_xy_ansysdic = mat_from_ansys(rows, cols,height,width,file_ansys,circle=True)
    # # 保存至mat文件
    # data = {"Strains":{'plot_exx_ref_formatted':np.array(strain_x_ansysdic),\
    #         'plot_eyy_ref_formatted':np.array(strain_y_ansysdic),\
    #         'plot_exy_ref_formatted':np.array(strain_xy_ansysdic),\
    #         }}
    # scio.savemat('./data/bxpl_hom_copy.mat',data)

    # # bxpl_non
    # rows, cols,height,width=207,207,50 * 10 ** -3, 50 * 10 ** -3
    # file_ansys='../ansys/bxpl_non/strain_elem.txt'
    # # 生成
    # vals_ansysdic,strain_x_ansysdic,strain_y_ansysdic,strain_xy_ansysdic = mat_from_ansys(rows, cols,height,width,file_ansys,circle=False)
    # # 保存至mat文件
    # data = {"Strains":{'plot_exx_ref_formatted':np.array(strain_x_ansysdic),\
    #         'plot_eyy_ref_formatted':np.array(strain_y_ansysdic),\
    #         'plot_exy_ref_formatted':np.array(strain_xy_ansysdic),\
    #         }}
    # scio.savemat('./data/bxpl_non_copy.mat',data)

    # # sdw_hom
    # rows, cols,height,width=124,510,50 * 10 ** -3, 200 * 10 ** -3
    # file_ansys='../ansys/sdw_hom/strain_elem.txt'
    # # 生成
    # vals_ansysdic,strain_x_ansysdic,strain_y_ansysdic,strain_xy_ansysdic = mat_from_ansys(rows, cols,height,width,file_ansys,circle=False)
    # # 保存至mat文件
    # data = {"Strains":{'plot_exx_ref_formatted':np.array(strain_x_ansysdic),\
    #         'plot_eyy_ref_formatted':np.array(strain_y_ansysdic),\
    #         'plot_exy_ref_formatted':np.array(strain_xy_ansysdic),\
    #         }}
    # scio.savemat('./data/sdw_hom_copy.mat',data)

    # # sdw_non
    # rows, cols,height,width=119,478,50 * 10 ** -3, 200 * 10 ** -3
    # file_ansys='../ansys/sdw_non/strain_elem.txt'
    # # 生成
    # vals_ansysdic,strain_x_ansysdic,strain_y_ansysdic,strain_xy_ansysdic = mat_from_ansys(rows, cols,height,width,file_ansys,circle=False)
    # # 保存至mat文件
    # data = {"Strains":{'plot_exx_ref_formatted':np.array(strain_x_ansysdic),\
    #         'plot_eyy_ref_formatted':np.array(strain_y_ansysdic),\
    #         'plot_exy_ref_formatted':np.array(strain_xy_ansysdic),\
    #         }}
    # scio.savemat('./data/sdw_non_copy.mat',data)

    # # 其他
    # data = scio.loadmat('data/bxpl_non_copy.mat')
    # strain_x = data['Strains']['plot_exx_ref_formatted'][0][-1].tolist()
    # strain_y = data['Strains']['plot_eyy_ref_formatted'][0][-1].tolist()
    # strain_xy = data['Strains']['plot_exy_ref_formatted'][0][-1].tolist()
    # data = {"Strains":{'plot_exx_ref_formatted':np.array(strain_x),\
    #         'plot_eyy_ref_formatted':np.array(strain_y),\
    #         'plot_exy_ref_formatted':-np.array(strain_xy),\
    #         }}
    # scio.savemat('./data/bxpl_non_copy.mat',data)
    print("ok")
