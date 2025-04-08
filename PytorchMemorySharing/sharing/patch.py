import matplotlib.pyplot as plt
import numpy as np

labels = ['512', '1024', '2048']
pytorch = [9626, 19534, 39640]
share = [3754, 7610, 14726]
improve = [61,61,63]
pytorch = [x/1024 for x in pytorch]
share = [x/1024 for x in share]

x = np.arange(len(labels))  
width = 0.3  
plt.rcParams['font.size'] = 20

fig, ax = plt.subplots()
rects1 = ax.bar(x + width/2, pytorch, width, label='Pytorch')
rects2 = ax.bar(x + 1.5*width, share, width, label='Share')

# 在每个柱子上面加数据
for rect in rects2:
    yval = rect.get_height()
    text = "-"+str(improve[rects2.index(rect)])+"%"
    ax.text(rect.get_x() + rect.get_width()/2, yval + 0.05, text, ha='center', va='bottom',fontsize=13)

ax.set_xlabel('Batch Size')  # 设定x轴的单位为Batch Size
ax.set_ylabel('Memory Usage (GiB)')  # 设定y轴的单位为Memory Usage (MiB)
ax.set_title('SEP+PatchTST')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.savefig('patch.png')  # 放在plt.show()之前
plt.show()