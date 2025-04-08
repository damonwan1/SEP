import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

labels = ['512', '1024', '2048']
pac_pytorch = [1614, 2712, 5134]
pac_share = [1324, 2174, 3732]
pac_improve = [18,20,27]
pac_pytorch = [x/1024 for x in pac_pytorch]
pac_share = [x/1024 for x in pac_share]

patch_pytorch = [9626, 19534, 39640]
patch_share = [3754, 7610, 14726]
patch_improve = [61,61,63]
patch_pytorch = [x/1024 for x in patch_pytorch]
patch_share = [x/1024 for x in patch_share]

trace_pytorch = [5146, 9438, 18928]
trace_share = [3582, 7156, 14238]
trace_improve = [30,24,25]
trace_pytorch = [x/1024 for x in trace_pytorch]
trace_share = [x/1024 for x in trace_share]

x = np.arange(len(labels))  
width = 0.3 
#plt.rcParams['font.size'] = 20

#第一张图
fig, axs = plt.subplots(1,3)
rects1 = axs[0].bar(x + width/2, pac_pytorch, width, label='Pytorch')
rects2 = axs[0].bar(x + 1.5*width, pac_share, width, label='Share')

# 在每个柱子上面加数据
for rect in rects2:
    yval = rect.get_height()
    text = "-"+str(pac_improve[rects2.index(rect)])+"%"
    axs[0].text(rect.get_x(), yval + 0.05, text, va='bottom',fontsize=10)

axs[0].set_xlabel('Batch Size')  # 设定x轴的单位为Batch Size
axs[0].set_ylabel('Memory Usage (GiB)')  # 设定y轴的单位为Memory Usage (MiB)
axs[0].set_title('SEP+PAC')
axs[0].set_xticks(x)
axs[0].set_xticklabels(labels)
axs[0].legend()

fig.tight_layout()

#第二张图
#fig, axs = plt.subplots(2)
rects1 = axs[1].bar(x + width/2, patch_pytorch, width, label='Pytorch')
rects2 = axs[1].bar(x + 1.5*width, patch_share, width, label='Share')

# 在每个柱子上面加数据
for rect in rects2:
    yval = rect.get_height()
    text = "-"+str(patch_improve[rects2.index(rect)])+"%"
    axs[1].text(rect.get_x(), yval + 0.05, text, va='bottom',fontsize=10)

axs[1].set_xlabel('Batch Size')  # 设定x轴的单位为Batch Size
#axs[1].set_ylabel('Memory Usage (GiB)')  # 设定y轴的单位为Memory Usage (MiB)
axs[1].set_title('SEP+PatchTST')
axs[1].set_xticks(x)
axs[1].set_xticklabels(labels)
#axs[1].legend()

fig.tight_layout()

#第三张图
#fig, axs = plt.subplots(3)
rects1 = axs[2].bar(x + width/2, trace_pytorch, width, label='Pytorch')
rects2 = axs[2].bar(x + 1.5*width, trace_share, width, label='Share')

# 在每个柱子上面加数据
for rect in rects2:
    yval = rect.get_height()
    text = "-"+str(trace_improve[rects2.index(rect)])+"%"
    axs[2].text(rect.get_x(), yval + 0.05, text, va='bottom',fontsize=10)



axs[2].set_xlabel('Batch Size')  # 设定x轴的单位为Batch Size
#axs[2].set_ylabel('Memory Usage (GiB)')  # 设定y轴的单位为Memory Usage (MiB)
axs[2].set_title('SEP+TRACE')
axs[2].set_xticks(x)
axs[2].set_xticklabels(labels)
#axs[2].legend()
formatter = ticker.FuncFormatter(lambda x, pos: '%.0f' % x)
axs[2].yaxis.set_major_formatter(formatter)

fig.tight_layout()


plt.savefig('memory.png',dpi=400)  # 放在plt.show()之前
plt.show()