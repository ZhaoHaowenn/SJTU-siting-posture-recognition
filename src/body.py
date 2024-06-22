import cv2
import numpy as np
import math
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.utils.prune as prune
from src import util
from src.model import bodypose_model

class Body(object):
    def __init__(self, model_path):
        self.model = bodypose_model()                  #创建了一个self.model对象，该对象是由bodypose_model()函数返回的人体姿势估计模型
        if torch.cuda.is_available():                  #判断GPU是否可用。如果GPU可用，则将模型移动到GPU上运行
            self.model = self.model.cuda()
        model_dict = util.transfer(self.model, torch.load(model_path))    # #torch.load(model_path)会从指定的路径model_path加载模型参数，然后将参数转移到模型对象中
        self.model.load_state_dict(model_dict)         #将模型参数加载到模型对象中，以便模型可以使用这些参数进行预测
        self.model.eval()                              #将模型设置为评估模式，这意味着模型将停用梯度计算，并且可以更快地进行预测

    def __call__(self, oriImg):
        scale_search = [0.5]                           #定义缩放比例
        boxsize = 368                                  #设置输入图像的大小
        stride = 8                                     #设置步长，用于对图像进行分块时的步长
        padValue = 128                                 #对图像进行pad操作时用于填充的值
        thre1 = 0.1                                    #定义阈值，用于姿态检测的结果筛选
        thre2 = 0.05
        multiplier = [x * boxsize / oriImg.shape[0] for x in scale_search]    #计算缩放后的图像大小，并根据缩放比例调整boxsize
        heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))        #创建两个用于存储结果的数组，heatmap_avg用于存储关键点热力图，paf_avg用于存储骨骼连线图
        paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

        for m in range(len(multiplier)):               #循环遍历输入的不同倍率
            scale = multiplier[m]                      #获取当前循环中的倍率
            imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)      #对原图进行resize操作，将其缩小或放大至当前倍率的大小
            imageToTest_padded, pad = util.padRightDownCorner(imageToTest, stride, padValue)                 #对resize后的图像进行填充操作，并获取填充的像素数
            im = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5      #将填充后的图像转为4维张量，并进行归一化
            im = np.ascontiguousarray(im)              #对转换后的张量进行拷贝操作

            data = torch.from_numpy(im).float()        #将numpy数组转为PyTorch的Tensor对象，并将数据类型设为float型
            if torch.cuda.is_available():              #判断当前环境是否支持CUDA，如果支持，则将Tensor对象传送到GPU上进行计算
                data = data.cuda()
            with torch.no_grad():                      #使用预训练好的模型进行计算，得到关键点heatmap和PAF向量
                Mconv7_stage6_L1, Mconv7_stage6_L2 = self.model(data)
            Mconv7_stage6_L1 = Mconv7_stage6_L1.cpu().numpy()    #将PyTorch的Tensor对象转为NumPy数组
            Mconv7_stage6_L2 = Mconv7_stage6_L2.cpu().numpy()


            heatmap = np.transpose(np.squeeze(Mconv7_stage6_L2), (1, 2, 0))  # output 1 is heatmaps
            heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)      #对heatmap进行resize操作，将其放大至原图大小
            heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]    #根据填充像素数，截取heatmap图像
            heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)   #对heatmap进行resize操作，将其缩小至原图大小。

            paf = np.transpose(np.squeeze(Mconv7_stage6_L1), (1, 2, 0))
            paf = cv2.resize(paf, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)              #将特征图上的值放大到原始输入图像的大小
            paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]      #对特征图进行裁剪，去除填充的边缘区域，使其大小与输入图像的大小相同
            paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)        #将特征图缩小到输入图像的大小

            heatmap_avg += heatmap_avg + heatmap / len(multiplier)       #计算多个缩放因子下的平均热图和平均PAF
            paf_avg += + paf / len(multiplier)

        all_peaks = []        #定义peak_counter计数器，初始化为0
        peak_counter = 0

        for part in range(18):           #通过for循环遍历18个人体关键点
            map_ori = heatmap_avg[:, :, part]           #获取heatmap_avg张量中对应人体关键点类型的二维热图
            one_heatmap = gaussian_filter(map_ori, sigma=3)       #使用高斯滤波器对热图进行平滑化处理，生成one_heatmap张量

            map_left = np.zeros(one_heatmap.shape)      #基于one_heatmap生成map_left、map_right、map_up和map_down四个张量(全零数组用于初始化）
            map_left[1:, :] = one_heatmap[:-1, :]       #分别表示每个像素点左侧、右侧、上方和下方像素点的热值
            map_right = np.zeros(one_heatmap.shape)
            map_right[:-1, :] = one_heatmap[1:, :]
            map_up = np.zeros(one_heatmap.shape)
            map_up[:, 1:] = one_heatmap[:, :-1]
            map_down = np.zeros(one_heatmap.shape)
            map_down[:, :-1] = one_heatmap[:, 1:]

            peaks_binary = np.logical_and.reduce(       #计算 one_heatmap 与其周围像素值的最大值，并与阈值 thre1 进行比较
                (one_heatmap >= map_left, one_heatmap >= map_right, one_heatmap >= map_up, one_heatmap >= map_down, one_heatmap > thre1))
            peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  #获取 peaks_binary 中值为 True 的位置
            #使用 map_ori[x[1], x[0]] 获取这个位置对应的置信度分数，并将其与二元组合并成一个三元组 (x, y, score)，得到一个包含所有关键点位置及其对应置信度分数的列表
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
            #创建一个长度为关键点数量的整数列表 peak_id，其中每个元素表示一个关键点的唯一标识符。
            peak_id = range(peak_counter, peak_counter + len(peaks))
            #将 peaks_with_score 列表中的每个三元组 (x, y, score) 与其对应的标识符 peak_id 合并成一个四元组 (x, y, score, id)
            peaks_with_score_and_id = [peaks_with_score[i] + (peak_id[i],) for i in range(len(peak_id))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)

        #表示骨架中相邻的两个关键点在all_peaks中的索引
        limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                   [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                   [1, 16], [16, 18], [3, 17], [6, 18]]
        #表示中间关键点对应的热图在paf_avg中的通道索引,用于计算 PAF 的分数
        mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], \
                  [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], \
                  [55, 56], [37, 38], [45, 46]]

        connection_all = []    #用于保存检测出的每对关键点之间的连接
        special_k = []         #用于保存某些特殊的连接
        mid_num = 10           #表示将两个关键点之间的连接分成多少段，以便在中间计算得分

        for k in range(len(mapIdx)):        #对每个连接进行遍历
            score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]      #取的是 paf_avg 中与连接 part 有关的 PAF
            candA = all_peaks[limbSeq[k][0] - 1]       #分别是骨架中相邻的两个关键点在all_peaks中的索引所对应的关键点坐标及其得分
            candB = all_peaks[limbSeq[k][1] - 1]
            nA = len(candA)                #分别是candA和candB的长度
            nB = len(candB)
            indexA, indexB = limbSeq[k]     #骨架中相邻的两个关键点在all_peaks中的索引
            if (nA != 0 and nB != 0):
                connection_candidate = []   #用于存储可能的连接候选项
                for i in range(nA):
                    for j in range(nB):
                        vec = np.subtract(candB[j][:2], candA[i][:2])   #计算两个关键点之间的向量
                        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])     #向量的模
                        norm = max(0.001, norm)
                        vec = np.divide(vec, norm)   #将向量归一化，得到单位向量vec

                        startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                            np.linspace(candA[i][1], candB[j][1], num=mid_num)))    #表示从第 i 个节点到第 j 个节点之间的等间距点

                        #对于每个中间点，代码会从score_mid中获取两个坐标分别对应到x和y分量的得分，分别存放在vec_x和vec_y中
                        vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                          for I in range(len(startend))])
                        vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                          for I in range(len(startend))])

                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])      #将vec_x和vec_y与向量vec做点积，得到每个中间点的得分
                        score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                            0.5 * oriImg.shape[0] / norm - 1, 0)                                    #将所有中间点的得分求平均，并加上一个距离先验得分
                        criterion1 = len(np.nonzero(score_midpts > thre2)[0]) > 0.8 * len(score_midpts)  #关键点之间的得分必须高于阈值thre2的中间点数目的80%
                        criterion2 = score_with_dist_prior > 0           #两个关键点之间的得分必须大于0。
                        #如果两个条件都满足，则将关键点之间的连接信息添加到connection_candidate中
                        if criterion1 and criterion2:
                            connection_candidate.append(
                                [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)   #按照得分进行排序，得分高的排在前面
                connection = np.zeros((0, 5))    #使用一个numpy数组connection来存储已经确定的连接
                # 依次遍历connection_candidate中的连接信息，将其添加到connection中
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if (i not in connection[:, 3] and j not in connection[:, 4]):
                        connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                        if (len(connection) >= min(nA, nB)):
                            break

                connection_all.append(connection)
            else:         #如果两组关键点中有一组为空，则将其对应的索引添加到special_k中，并将一个空数组添加到connection_all中
                special_k.append(k)
                connection_all.append([])


        subset = -1 * np.ones((0, 20))    #创建一个空的 0 行 20 列的数组，存储关键点组合的情况
        candidate = np.array([item for sublist in all_peaks for item in sublist])    #将二维数组 all_peaks 展开为一维，存储每个关键点的坐标及其置信度

        for k in range(len(mapIdx)):      #遍历所有limbSeq中的元素
            if k not in special_k:        #如果这个limbSeq是由两个同侧的关键点组成的则跳过
                partAs = connection_all[k][:, 0]       #获取这个limbSeq中起始点A的编号，一共有len(connection_all[k])个组合
                partBs = connection_all[k][:, 1]       #获取这个limbSeq中终止点B的编号，一共有len(connection_all[k])个组合
                indexA, indexB = np.array(limbSeq[k]) - 1    #获取limbSeq中两个点的编号

                for i in range(len(connection_all[k])):  #遍历这个limbSeq中所有的组合
                    found = 0            #用以记录subset的组合中出现的关键点A或B的个数
                    subset_idx = [-1, -1]     #用以记录subset中与这个组合匹配的元素的行
                    for j in range(len(subset)):  #遍历所有subset中的行，找到是否有与这个组合匹配的情况
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:   #如果subset中有关键点A或B在这个组合中出现了
                            subset_idx[found] = j    #记录下subset中这一行的编号
                            found += 1      #匹配数+1

                    if found == 1:       #如果subset中只有一行与这个组合匹配上了
                        j = subset_idx[0]     #找到这一行的编号
                        if subset[j][indexB] != partBs[i]:   #如果subset中这一行中与B对应的元素不是这个组合中的关键点B
                            subset[j][indexB] = partBs[i]    #则将subset中这一行中对应的B元素设置为这个组合中的关键点B
                            subset[j][-1] += 1    #这一行中记录的关键点数量+1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]   #这一行中记录的总置信度+该关键点的置信度+该组合的置信度
                    elif found == 2:  #如果subset中有两行与这个组合匹配上了，即这两行中有A和B都在这个组合中的情况
                        j1, j2 = subset_idx    #记录下这两行在subset中的编号
                        #membership是一个长度为18的二进制数组（对应于每个身体部位），值为1表示相应的身体部位同时存在于j1和j2中。[：-2]的切片用于排除数组的最后两个元素，它们用于存储分数和subset中部件的数量
                        membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                        #如果j1和j2中没有身体部位同时存在，则将两个subset合并
                        if len(np.nonzero(membership == 2)[0]) == 0:
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:  #如果j1和j2中存在身体部位，则将当前的身体部位partBs[i]添加到j1中
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                    # 如果subset中没有与当前组合匹配的身体部位,则在subset中创建一个新行表示当前组合
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])

        deleteIdx = []
        for i in range(len(subset)):  #遍历subset中的每一行
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:  #如果这一行中的总体得分低于4或者平均每个部位得分低于0.4
                deleteIdx.append(i)   #将这一行的索引添加到deleteIdx列表中
        subset = np.delete(subset, deleteIdx, axis=0)   #使用np.delete函数删除subset中对应的行，删除的行的索引是deleteIdx列表中的值

        # subset是一个n20的数组，其中0-17是对应候选框的索引，第18个元素是这个subset的总分数，第19个元素是这个subset中包含的部位的数量
        # candidate是一个n4的数组，其中每一行对应一个候选框，包括候选框的左上角坐标(x,y)，候选框的得分(score)和候选框的id
        return candidate, subset    #返回处理后的candidate和subset


if __name__ == "__main__":

    body_estimation = Body('../model/body_pose_model.pth')
    test_image = '../image/1.png'
    oriImg = cv2.imread(test_image)  # B,G,R order
    candidate, subset = body_estimation(oriImg)
    canvas = util.draw_bodypose(oriImg, candidate, subset)
    cv2.imshow('estimated_image', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

