from cProfile import label
from calendar import c
from matplotlib import ticker
import numpy as np
import matplotlib.pyplot as plt


question_type_dict = np.load("question_type_top1.npy", allow_pickle=True).item()
print(question_type_dict)
answer_type_dict = np.load("nolstm_answer_type_top1.npy", allow_pickle=True).item()
print(answer_type_dict)

answer_type_list = []
answer_type_total = []
answer_type_right = []

for answer_type in answer_type_dict:
    answer_type_list.append(answer_type)
    answer_type_total.append(answer_type_dict[answer_type]['total'])
    answer_type_right.append(answer_type_dict[answer_type]['right'])

question_type_list = []
question_type_total = []
question_type_right = []

for question_type in question_type_dict:
    if question_type_dict[question_type]['total'] > 100:
        question_type_list.append(question_type)
        question_type_total.append(question_type_dict[question_type]['total'])
        question_type_right.append(question_type_dict[question_type]['right'])




def bar_with_percentage_plot(answer_type, answer_total, answer_right):

    # 绘图参数, 第一个参数是x轴的数据, 第二个参数是y轴的数据,
    # 第三个参数是柱子的大小, 默认值是1(值在0到1之间), color是柱子的颜色, alpha是柱子的透明度
    plt.bar(answer_type, answer_total, 0.25, color='b', alpha=0.3, label='total')
    plt.bar(answer_type, answer_right, 0.25, color='b', alpha=0.8, label='right')
    
    plt.tick_params(axis='x', labelsize=6)
    plt.xticks(rotation=-20)
    

    # 添加轴标签
    plt.ylabel('total questions')

    # 标题
    plt.title('accuracy in answer type')

    # 设置Y轴的刻度范围
    y_max = max(answer_total)
    plt.ylim([0, y_max *1.5])

    accuracy = list(map(lambda x: int(x[1] / x[0] * 100), zip(answer_total, answer_right)))

    # 为每个条形图添加数值标签
    for x, y in enumerate(answer_total):
        plt.text(x, y + 20, str(accuracy[x]) + "%", ha='center')

    plt.legend()
    # 显示图形
    plt.savefig("nolstm_answer_accuracy_top1.png")

bar_with_percentage_plot(answer_type_list, answer_type_total, answer_type_right)