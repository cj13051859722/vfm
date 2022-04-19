import cv2


def imshow(window_name, pic):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, pic)
    cv2.waitKey(0)


if __name__ == '__main__':
    # step1 读取图片
    pic_bgr = cv2.imread('data/bxpl_non.jpg', cv2.IMREAD_COLOR) # TODO 更改文件位置
    for i in range(pic_bgr.shape[0]):
        for j in range(pic_bgr.shape[1]):
            if (i-440)**2+(j-444)**2>435**2:
                for k in range(3):
                    pic_bgr[i][j][k]=225
    imshow("pic_gray", pic_bgr)
    # step2 使用大津法进行二值化（python）
    pic_gray = cv2.imread('data/bxpl_non.jpg') # TODO 更改文件位置
    pic_gray = cv2.cvtColor(pic_gray, cv2.COLOR_BGR2GRAY)
    ret, pic_bin = cv2.threshold(pic_gray, 170, 255, cv2.THRESH_BINARY)
    #
    # imshow("pic_gray", pic_gray)
    # imshow("pic", pic_bin)
    # step3 删除小块面积区域
    # step3.1 寻找轮廓
    contours, hierarchy = cv2.findContours(pic_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # step3.2 按照轮廓面积从大到小排序
    def area(contour):
        return cv2.contourArea(contour)


    contours.sort(key=area, reverse=True)

    # step3.3 寻找面积
    # 面积大于1500的
    for thresh in range(100,101,100): # TODO 改变面积大小
        areas = []
        for contour in contours:
            areas.append(cv2.contourArea(contour))
        ind = len(areas)
        for i in range(len(areas)):
            if areas[i] < thresh:
                ind = i
                break
        # step3.4 只取前面
        contours_fin = contours[:ind]
        cv2.drawContours(pic_bgr, contours_fin, -1, (0, 255, 0), 1)
        imshow("pic_bgr"+str(thresh), pic_bgr)
    cv2.imwrite("./pic/pic_bgr"+str(thresh)+".jpg", pic_bgr)

    # step4 将轮廓打印出来
    def isval(xds, cur):
        """
        :param xd: 已保存线段 [[[c_old, r_old], [c, r]]]
        :param cur: 当前线段  [[c_old, r_old], [c, r]]
        :return: 是否有重合
        """
        vals = True
        for xd in xds:
            # 单个线段是否相交 A=xd[0] B=xd[1] C=cur[0] D=cur[1]
            # 是否有相同点
            samenode=False
            if xd[0]==cur[0] or xd[0]==cur[1] or  xd[1]==cur[0] or xd[1]==cur[1]:
                samenode=True
            # AB*AD与AB*AC异号
            AB = [xd[1][0] - xd[0][0], xd[1][1] - xd[0][1]]
            AD = [cur[0][0] - xd[0][0], cur[0][1] - xd[0][1]]
            AC = [cur[1][0] - xd[0][0], cur[1][1] - xd[0][1]]
            ans0,ans1=AB[0] * AD[1] - AB[1] * AD[0],AB[0] * AC[1] - AB[1] * AC[0]
            val = (ans0!=0 or ans1!=0) and samenode or ans0*ans1 > 0
            # DC*DA与DC*DB异号
            DC = [cur[1][0] - cur[0][0], cur[1][1] - cur[0][1]]
            DA = [xd[0][0] - cur[0][0], xd[0][1] - cur[0][1]]
            DB = [xd[1][0] - cur[0][0], xd[1][1] - cur[0][1]]
            ans0,ans1 = DC[0] * DA[1] - DC[1] * DA[0],DC[0] * DB[1] - DC[1] * DB[0]
            val = (ans0!=0 or ans1!=0) and samenode or ans0*ans1 > 0 or val
            vals = vals and val
        return vals


    # print(isval([[[-1, 0], [0, 2]]], [[-2, 0], [2, 0]]))
    with open("model.txt", 'w') as f:
        h, w, c = pic_bgr.shape
        k = 6
        for index, contour in enumerate(contours_fin):
            # step4.1 打印单个点坐标
            k_new = k
            # all_node = 5 + int(25 * (1 - index / ind))  # TODO 更改点的个数 4 + int(36 * (1 - index / ind)) 4-36个点
            all_node = 250 if areas[index]>2000 else 5 + int(25 * (1 - index / ind))
            step = contour.shape[0] // all_node if contour.shape[0] > all_node else 1  # 随面积小而点个数减小
            k_list = []
            xds = []
            sets = set()
            for i in range(0, contour.shape[0], step):
                # 去重：
                c, r = round(contour[i][0][0] / w * 50 - 25,7)/1000, round(50 - contour[i][0][1] / h * 50,7)/1000
                # 直线去相交：
                if i==0 or (c,r) not in sets and isval(xds, [[c_old, r_old], [c, r]]): # 非最后一个点
                    sets.add((c,r))
                    k_list.append([k_new,c,r])
                    k_new += 1
                    if i:
                        xds.append([[c_old, r_old], [c, r]])
                    c_old, r_old = c, r

            # 最后一个点与首点,若无效，则最后一个点弹出
            while not isval(xds, [[k_list[0][1], k_list[0][2]], [k_list[-1][1], k_list[-1][2]]]):
                k_list.pop()
                xds.pop()

            # step4.1 打印连线
            for k_new,c,r in k_list:
                 f.write("K," + str(k_new) + "," + str(c) + "," + str(r) + ",," + '\r\n')
            # step4.2 打印连线
            for lstr in range(k, k_new):
                f.write("LSTR," + str(lstr) + "," + str(lstr + 1) + '\r\n')
            f.write("LSTR," + str(k_new) + "," + str(k) + '\r\n')
            # step4.3 打印连面
            f.write("LSEL, , , ," + str(k-1) + ',' + str(k_new-1) + '\r\n')
            f.write("AL,ALL" + '\r\n')
            k = k_new+1
    print('ok')
    """
    K,5,50,50,, 
    K,6,50,75,,   
    K,7,-50,75,,
    K,8,-50,50,,
    LSTR,       5,       6  
    LSTR,       6,       7  
    LSTR,       7,       8 
    LSTR,       8,       5  
    AL,5,6,7,8
    """
