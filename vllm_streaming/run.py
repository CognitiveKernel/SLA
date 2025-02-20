import torch
from multiprocessing import Pool
import time

def run_on_single_gpu(device):
    a = torch.randn(2000,2000).cuda(device)
    b = torch.randn(2000,2000).cuda(device)
    ta = a
    tb = b
    while True:
        a = ta
        b = tb
        a = torch.sin(a)
        b = torch.sin(b)
        a = torch.cos(a)
        b = torch.cos(b)
        a = torch.tan(a)
        b = torch.tan(b)
        a = torch.exp(a)
        b = torch.exp(b)
        a = torch.log(a)
        b = torch.log(b)
        b = torch.matmul(a, b)
        time.sleep(0.001)

if __name__ == '__main__':
    # set_start_method('spawn')
    print('start running')
    num_gpus = torch.cuda.device_count()
    pool = Pool(processes=num_gpus)
    pool.map(run_on_single_gpu, range(num_gpus))
    pool.close()
    pool.join()