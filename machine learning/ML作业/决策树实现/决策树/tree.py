# 决策树生成 version：3.0
# latest edition: 2020/4/18
# author:韩轶凡
# 本程序使用西瓜数据集2.0，生成西瓜决策树，最终结果返回字典格式
# 可对data_set()函数中对dataset，label两项根据需要自行更改
# 本程序可配合graph.py一同使用，生成可视化决策树结果
# 具体使用方法为将本程序返回的字典结果复制进graph.py的主函数中的对应位置，运行graph.py即可获得可视化结果
# 注：由于时间限制，可视化操作借鉴《机器学习实战》

from math import log

"""
特征及其分类：
色泽：青绿-0 乌黑-1 浅白-2
根蒂：蜷缩-0 稍蜷-1 硬挺-2
敲声：浊响-0 沉闷-1 清脆-2
纹理：清晰-0 稍糊-1 模糊-2
脐部：凹陷-0 稍凹-1 平坦-2
触感：硬滑-0 软粘-1
好瓜：是-0   否-1
"""
def data_set():
    dataset = [ [0,0,0,0,0,0,"好瓜"],
                [1,0,1,0,0,0,"好瓜"],
                [1,0,0,0,0,0,"好瓜"],
                [0,0,1,0,0,0,"好瓜"],
                [2,0,0,0,0,0,"好瓜"],
                [0,1,0,0,1,1,"好瓜"],
                [1,1,0,1,1,1,"好瓜"],
                [1,1,0,0,1,0,"好瓜"],
                [1,1,1,1,1,0,"坏瓜"],
                [0,2,2,0,2,1,"坏瓜"],
                [2,2,2,2,2,0,"坏瓜"],
                [2,0,0,2,2,1,"坏瓜"],
                [0,1,0,1,0,0,"坏瓜"],
                [2,1,1,1,0,0,"坏瓜"],
                [1,1,0,0,1,1,"坏瓜"],
                [2,0,0,2,2,0,"坏瓜"],
                [0,0,1,1,1,0,"坏瓜"] ]
    label = ["色泽","根蒂","敲声","纹理","脐部","触感"]
    return dataset,label

def shannon_ent(dataset):
    num_of_entries = len(dataset)
    label_set = {}
    for vec in dataset:
        label = vec[-1]
        if label not in label_set.keys():
            label_set[label] = 1
        else:
            label_set[label] += 1
    ent = 0
    for key in label_set:
        prob = float( label_set[key]/num_of_entries )
        ent -= prob * log(prob, 2)
    return ent

def split_data_set(dataset, axis, value):
    result = []
    for vec in dataset:
        if vec[axis] == value:
            new_vec = vec[:axis]
            new_vec.extend(vec[axis+1:])
            result.append(new_vec)
    return result

def best_split(dataset):
    num_of_fea = len(dataset[0])-1
    base_entropy = shannon_ent(dataset)
    best_info_gain = 0.0
    best_fea = -1
    for i in range(num_of_fea):
        kinds = set([ vec[i] for vec in dataset])
        new_entropy = 0
        for kind in kinds:
            sub_data_set = split_data_set(dataset,i,kind)
            prob = len(sub_data_set)/float(len(dataset))
            new_entropy += prob*shannon_ent(sub_data_set)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_fea = i
    return best_fea

def vote(classlist):
    class_set = set(classlist)
    dict = {}
    for i in class_set:
        dict[i] = classlist.count(i)
    return max(dict,key=dict.get())
    # k = 0
    # v = 0
    # for key,value in dict:
    #     if value > v:
    #         v = value
    #         k = key
    #     else:continue
    # return k

def creat_tree(dataset,label):
    kinds = [ vec[-1] for vec in dataset ]    # 类别标签
    if kinds.count(kinds[0]) == len(kinds):   # 全部属于一类的情况
        return kinds[0]
    if len(dataset[0]) == 1:                  # 所有特征遍历完毕
        return vote(kinds)
    best_fea = best_split(dataset)
    best_fea_label = label[best_fea]
    j_tree = {best_fea_label:{}}
    del(label[best_fea])
    fea_kind = set( [ vec[best_fea] for vec in dataset])
    for fea in fea_kind:
        sub_label = label[:]
        j_tree[best_fea_label][fea] = creat_tree( split_data_set(dataset,best_fea,fea), sub_label )
    return j_tree



if __name__ == "__main__":
    dataset,label = data_set()[0],data_set()[1]
    print(creat_tree(dataset,label))