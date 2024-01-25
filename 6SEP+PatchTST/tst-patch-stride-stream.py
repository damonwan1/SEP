import collections
import gzip
import os
import time
import utils
import struct
import shutil
from absl import app
from absl import flags
from absl import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.cuda as cuda
#import compress_model
import arithmeticcoding_fast
import utils
from torch.profiler import profile, record_function, ProfilerActivity
import threading
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import copy
from mixModel import PatchTST_backbone
## 原始方法，未加同步操作

# max_workers表示工人数量,也就是jincheng池里面的数量
#processPooling = multiprocessing.Pool(processes=4)
# 任务列表

def write_to_file(content):
  with open("test.txt", "w") as file:
    file.write(content)
gpudevice = torch.device("cuda")


    
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
torch.set_printoptions(profile="full") 
FLAGS = flags.FLAGS

# Model parameters
flags.DEFINE_integer('batch_size', 512, 'Batch size for training.')
flags.DEFINE_float('learning_rate', 1e-3, 'Adam Optimizer learning rate.')
flags.DEFINE_integer('hidden_dim', 256, 'Feature dimension.')
flags.DEFINE_integer('vocab_dim', 64, 'Feature dimension.')
flags.DEFINE_integer('n_layers', 1, 'Number of Attention layers.')
flags.DEFINE_integer('ffn_dim', 4096, 'MLP dimension in model.')
flags.DEFINE_integer('n_heads',8, 'Number of heads for attention.')
flags.DEFINE_string(
    'feature_type', 'sqr',
    'Nonlinearity function for feature. Can be relu, elu+1, sqr, favor+, or favor+{int}.'
)
flags.DEFINE_enum(
    'compute_type', 'iter', ['iter', 'ps', 'parallel_ps'],
    'Which type of method to compute: iter = iterative algorithm, ps = implementation using torch.cumsum, parallel_ps = implementation using custom log prefix sum implementation.'
)
flags.DEFINE_float('weight_decay', 0.0, 'Weight decay for regularization.')

# Training parameters
flags.DEFINE_string('gpu_id', '0', 'ID of GPU.')
flags.DEFINE_integer('random_seed', 0, 'Random seed for both Numpy and Torch.')
flags.DEFINE_integer('print_step', 1000, 'Interval to print metrics.')
# Dataset parameters
flags.DEFINE_integer('seq_len', 64, 'Maximum sequence length (L).')
flags.DEFINE_integer('patch_len', 4, 'Maximum sequence length (L).')
flags.DEFINE_integer('stride',1, 'Maximum sequence length (L).')
flags.DEFINE_integer('vocab_size', 256, 'Vocabulary size of data.')
flags.DEFINE_string('input_dir', '../../../data/image', 'input data dir')
flags.DEFINE_string('prefix', '_image', 'output dir')


def coder_write(queue,temp_dir,compressed_file,bs1,cumul1,series1,ind1):
  print(f"start to right")
  # Define enc ****************************
  f = [open(temp_dir+"/"+compressed_file+'.'+str(i),'wb') for i in range(bs1)]
  bitout = [arithmeticcoding_fast.BitOutputStream(f[i]) for i in range(bs1)]
  enc = [arithmeticcoding_fast.ArithmeticEncoder(32, bitout[i]) for i in range(bs1)]
  # ********************************
  # Init. enc
  for i in range(bs1):
    for j in range(FLAGS.seq_len):
      enc[i].write(cumul1, series1[ind1[i]+j])
  # ****************************************
  # start write for each train
  coder_num = 1
  while True:
    if not queue.empty():
      #print("子进程：队列中结果个数：")
      #print(queue.qsize())
      data = queue.get()
      if data is None:
        break
      bs,cumul_batch,y=data
      #print("取回队列的次数:"+ str(coder_num))
      coder_num = coder_num +1
      #print("Child:"+str(coder_num))
      #print(cumul_batch)
      for i in range(bs):
        enc[i].write(cumul_batch[i,:], y[i])
    else:
      time.sleep(1)
  # *******************************************
  for i in range(bs1):
    enc[i].finish()
    bitout[i].close()
    f[i].close()
  print("write end!") 

def encode(temp_dir, compressed_file, FLAGS, series, train_data, last_train_data,queue,enc):
  
  stream_num = 4
  bs = FLAGS.batch_size
  prob = np.ones(FLAGS.vocab_size)/FLAGS.vocab_size
  cumul = np.zeros(FLAGS.vocab_size+1, dtype=np.uint64)
  cumul[1:] = np.cumsum(prob*10000000 + 1)
  
  iter_num = len(train_data) // FLAGS.batch_size
  ind = np.array(range(bs))*iter_num
  iter_num -= FLAGS.seq_len
  
  write_process = multiprocessing.Process(target=coder_write, args=(queue,temp_dir,compressed_file,bs,cumul,series,ind))
  write_process.start()
  
  cumul_batch = np.zeros((bs, FLAGS.vocab_size+1), dtype = np.uint64)

  #model = PatchTST_backbone(FLAGS.seq_len,FLAGS.patch_len,FLAGS.stride,FLAGS.n_heads).cuda()
  model = PatchTST_backbone(FLAGS.seq_len,FLAGS.patch_len,FLAGS.stride,FLAGS.n_heads,FLAGS.vocab_dim,
                            FLAGS.batch_size).cuda()
  
  optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay, betas=(.9, .999))
  print(iter_num)
  # 定义三个时间的初始值
  time_gpu_to_gpu = 0.0
  time_training = 0.0
  time_gpu_to_cpu = 0.0
  time_write = 0.0
  T1 = time.time()
  
   # Define Pooling
  streamPooling=[]
  prob=""
  #eventPooling=[]
  # 4 Init.
  for i in range(0,stream_num):
    stream = torch.cuda.Stream()
    #event = torch.cuda.Event()
    streamPooling.append(stream)
    #eventPooling.append(event)
  # step = 4
  # 需要修改，能被4整除
  record_num = 1
  for train_index in range(0,iter_num,stream_num):
    # 被4不能整除的后几轮，就用之前的方法吧
    if train_index+stream_num > iter_num:
      for train_index in range(train_index+1,iter_num):
        model.train()
        train_batch = train_data[ind, :]

        y = train_batch[:, -1]

        # GPU与GPU之间的操作
        start = time.time()
        train_batch = torch.from_numpy(train_batch).cuda().long()
        end = time.time()
        time_gpu_to_gpu += end - start

        # 训练过程
        start = time.time()
        train_loss, logits = model.full_loss(train_batch, with_grad=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        end = time.time()
        time_training += end - start

        logits = logits.transpose(1, 2)

        # 从预测结果中获取概率。
        prob = logits[:, -1, :]

        # GPU回到CPU的操作
        start = time.time()
        prob = F.softmax(prob, dim=1).detach().cpu().numpy()

        start = time.time()
        # 计算累积概率，并将结果保存在cumul_batch数组中。
        cumul_batch[:,1:] = np.cumsum(prob*10000000 + 1, axis = 1)
        #print("Main:"+str(train_index))
        #print(cumul_batch)
        # 将累积概率和目标标签写入enc对象中。
        queue.put((bs,copy.deepcopy(cumul_batch),y))
        ind += 1
      continue
    # where is train data from????
    model.train()
    train_batch_list = []
    y_list = []
    # ????没变化？？？？
    # 分批先取出 4 组训练数据
    for i in range(0,stream_num):
      train_batch = train_data[ind, :]
      y = train_batch[:, -1]
      y_list.append(y)
      train_batch = torch.from_numpy(train_batch)
      train_batch_list.append(train_batch)
      ind += 1
    # 4个stream执行4组训练数据
    for i in range(0,stream_num):
      with torch.cuda.stream(streamPooling[i]):
        #print("stream:"+str(i)+"start!!!!!")
        # *** H2D ***
        # stream1的H2D现在执行，其他的都放到stream1训练的时候，并行执行，因CPU那时候会有很长的同步等待
        if i == 0:
          train_batch_list[i] = train_batch_list[i].to(gpudevice,non_blocking=True).long()
        #train_batch_tmp = train_batch_list[i].to(gpudevice,non_blocking=True).long()
        # Wait for last Train computing
        # if i != 0 :
        #   streamPooling[i].wait_event(eventPooling[i-1])
        # GPU COMPUTING
        train_loss, logits = model.full_loss(train_batch_list[i], with_grad=True,inputs_list=train_batch_list,streamPooling=streamPooling)
        #GPU
        optimizer.step()
        #CPU
        optimizer.zero_grad(set_to_none=True)
        logits = logits.transpose(1, 2)
        # 从预测结果中获取概率。
        #CPU
        prob = logits[:, -1, :]
        # Notify the Next Stream to Execute GPU Computing
        # if i == 3:
        #   streamPooling[0].record()
        # else:
        #eventPooling[i].record()
        
        #softmax 在GPU，detach在GPU
        probGPU = F.softmax(prob, dim=1).detach()
        # GPU回到CPU的操作
        prob = probGPU.to("cpu", non_blocking=True).numpy()
        #print("stream:"+str(i)+"end!!!!!")
        streamPooling[i].synchronize()
        # 计算累积概率，并将结果保存在cumul_batch数组中。
        cumul_batch[:,1:] = np.cumsum(prob*10000000 + 1, axis = 1)
        #print("Main:"+str(train_index+i))
        #print(cumul_batch)
        # 将累积概率和目标标签写入enc对象中。
        # 进程池 异步
        queue.put((bs,copy.deepcopy(cumul_batch),y_list[i]))
        #print("放入queue的次数:"+str(record_num))
        record_num = record_num + 1
        #print("主进程：队列中结果个数：")
        #print(queue.qsize())
        #processPooling.apply_async(coder_write, args=(bs,enc,cumul_batch,y,i))
        #threadPooling.submit(coder_write, (bs,enc,cumul_batch,y))
      # 每1000轮输出值
    if train_index % FLAGS.print_step == 0:
      size = 0
      for cf in os.listdir(temp_dir):
        size += os.path.getsize(temp_dir+"/"+cf)
      print(train_index, ":", train_loss.item()/np.log(2), "size:", size/(1024*1024))
    # if train_index == 12:
    #   break
  # 防止后续提交
  #processPooling.close()
  # 等待所有任务完成
  #processPooling.join()
#   # 训练过程结束后，获取注意力权重，并将其转换为numpy数组
#   attention_scores = model.backbone.encoder.layers[0].attn.cpu().detach().numpy()
#   print(attention_scores.shape)

#   # 使用numpy的save函数将注意力权重保存到.npy文件中
    
#   np.save('{}.npy'.format(FLAGS.prefix), attention_scores)
  
  print("waiting for write process!")
  queue.put(None)
  write_process.join()
  print("end join!")

  if last_train_data is not None:
    print("last series")
    f = open(temp_dir+"/"+compressed_file+'.last','wb')
    bitout = arithmeticcoding_fast.BitOutputStream(f)
    enc = arithmeticcoding_fast.ArithmeticEncoder(32, bitout)
    prob = np.ones(FLAGS.vocab_size)/FLAGS.vocab_size
    cumul = np.zeros(FLAGS.vocab_size+1, dtype=np.uint64)
    cumul[1:] = np.cumsum(prob*10000000 + 1)
  
    for j in range(len(last_train_data)):
      enc.write(cumul, last_train_data[j])
    print("Last encode part don't need inference.")
  
    enc.finish()
    bitout.close()
    f.close()
  
  return
    
def var_int_encode(byte_str_len, f):
  while True:
    this_byte = byte_str_len&127
    byte_str_len >>= 7
    if byte_str_len == 0:
      f.write(struct.pack('B',this_byte))
      break
    f.write(struct.pack('B',this_byte|128))
    byte_str_len -= 1

def var_int_decode(f):
    byte_str_len = 0
    shift = 1
    while True:
        this_byte = struct.unpack('B', f.read(1))[0]
        byte_str_len += (this_byte & 127) * shift
        if this_byte & 128 == 0:
                break
        shift <<= 7
        byte_str_len += shift
    return byte_str_len

def main(_):
#   prof = torch.profiler.profile(
#             on_trace_ready=torch.profiler.tensorboard_trace_handler('./'),
#             record_shapes=True,
#             with_stack=True)

#   prof.start()
  print("start!")
  start_time = time.time()
  queue = multiprocessing.Queue()
  enc = ""
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_id
  np.random.seed(FLAGS.random_seed)
  torch.manual_seed(FLAGS.random_seed)

  temp_dir = "{}_{}_{}_{}_bs{}_{}_seq{}_temp".format(FLAGS.prefix, FLAGS.vocab_dim, FLAGS.hidden_dim, FLAGS.ffn_dim, FLAGS.batch_size, FLAGS.n_layers, FLAGS.seq_len)
  compressed_file = temp_dir.replace("_temp", ".compressed")
  
  if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)
  os.mkdir(temp_dir)
  
  def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))
  
  
  with open(FLAGS.input_dir, 'rb') as fp:#, encoding='latin-1') as fp:
    series = np.frombuffer(fp.read(), dtype=np.uint8)
  train_data = strided_app(series, FLAGS.seq_len+1, 1)

  total_length = len(train_data)
  if total_length % FLAGS.batch_size == 0:
    encode(temp_dir, compressed_file, FLAGS, series, train_data, None,queue,enc)
  else:
    l = total_length // FLAGS.batch_size * FLAGS.batch_size
    encode(temp_dir, compressed_file, FLAGS, series[:l+FLAGS.seq_len], train_data[:l], series[l:],queue,enc)
  #queue.put(None)
  
  #Combined compressed results
  f = open(compressed_file+'.combined','wb')
  for i in range(FLAGS.batch_size):
    f_in = open(temp_dir+'/'+compressed_file+'.'+str(i),'rb')
    byte_str = f_in.read()
    byte_str_len = len(byte_str)
    var_int_encode(byte_str_len, f)
    f.write(byte_str)
    f_in.close()
  
  if total_length % FLAGS.batch_size != 0:
    f_in = open(temp_dir+'/'+compressed_file+'.last','rb')
    byte_str = f_in.read()
    byte_str_len = len(byte_str)
    var_int_encode(byte_str_len, f)
    f.write(byte_str)
    f_in.close()
  f.close()
  
  total = 0
  for ff in os.listdir(temp_dir):
    total += os.path.getsize(temp_dir+'/'+ff)
  
  print(total/(1024*1024))
  
  elapsed_time = time.time() - start_time
  print("压缩完成，总共用时：{} 秒".format(elapsed_time))
  print("@@@@@@@@@@@@@")
  print("end！")
  #prof.step()
  # prof.stop()
  #threadPooling.shutdown()
  print("main执行完毕")

if __name__ == '__main__':

  app.run(main)

  
    
