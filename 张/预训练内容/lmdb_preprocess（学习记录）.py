#主要用于单细胞RNA测序(scRNA-seq)数据的预处理和转换为LMDB格式，以便高效存储和读取大规模数据集。
import scanpy as sc, numpy as np, pandas as pd, anndata as ad
from scipy import sparse
import os
import pickle
from scipy.sparse import issparse
from memory_profiler import profile
import lmdb  #注意 这是一个python包 有用的
import pyarrow as pa

#预处理函数 ：将原始数据与参考数据集(PanglaoDB)对齐
def preprocess_old():
    panglao = sc.read_h5ad('./data/panglao_10000.h5ad') #读取参考数据集
    data = sc.read_h5ad('./data/your_raw_data.h5ad') #读取原始数据
    counts = sparse.lil_matrix((data.X.shape[0], panglao.X.shape[1]), dtype=np.float32) #创建稀疏矩阵，
    ref = panglao.var_names.tolist()
    obj = data.var_names.tolist()
    #将原始数据基因与参考基因对齐
    for i in range(len(ref)):
        if ref[i] in obj:
            loc = obj.index(ref[i])
            counts[:, i] = data.X[:, loc]

    counts = counts.tocsr()
    new = ad.AnnData(X=counts)
    new.var_names = ref
    new.obs_names = data.obs_names
    new.obs = data.obs
    new.uns = panglao.uns
    #基础过滤和归一化处理
    sc.pp.filter_cells(new, min_genes=200)
    sc.pp.normalize_total(new, target_sum=1e4)
    sc.pp.log1p(new, base=2)
    new.write('./data/preprocessed_data.h5ad') #保存预处理后的数据

#批量预处理函数 批量处理多个h5ad文件
def pretrain_h5ad(data_dir, gene_file, file_list=None, output_dir=None):
    files = os.listdir(data_dir) #支持通过文件列表筛选要处理的数据集
    if file_list is not None:
        with open(file_list, 'r') as f:
            dataset_ids = [i.strip() for i in f.readlines()]
        use_file = [i for i in files if i.strip('.h5ad').split('_')[-1] in dataset_ids] #自动跳过已处理文件
    else:
        use_file = files
    with open(gene_file, 'rb') as gf:
        gene_list = np.array(pickle.load(gf))
    deal_list = [i.split('h5ad')[0]+'h5ad' for i in os.listdir(output_dir)]
    print(deal_list)
    for i in use_file:
        if i in deal_list:
            print('deal :', i)
            continue
        print(i)
        try: #对于大数据集(>100,000细胞)自动分块处理
            adata = sc.read_h5ad(os.path.join(data_dir, i))
            if adata.shape[0] > 100000:
                for index in range(0, adata.shape[0], 100000):
                    sub_data = adata[index: index+100000, :]
                    sub_data = make_data(sub_data, gene_list) #调用make_data()进行基因对齐 这个函数很重要啊 在哪里呢
                    print(sub_data)
                    if sub_data is not None:
                        sc.pp.filter_cells(sub_data, min_genes=200)
                        sc.pp.normalize_total(sub_data, target_sum=1e4)
                        sc.pp.log1p(sub_data, base=2)
                        sub_data.write(os.path.join(output_dir, 'PRO_'+i+'_'+str(index)))
            else:
                adata = make_data(adata, gene_list) #调用make_data()进行基因对齐
                print(adata)
                if adata is not None: #基础过滤和归一化处理
                    sc.pp.filter_cells(adata, min_genes=200)
                    sc.pp.normalize_total(adata, target_sum=1e4)
                    sc.pp.log1p(adata, base=2)
                    adata.write(os.path.join(output_dir, 'PRO_'+i))
        except Exception as e:
            print('error:', i)
            print(e)

#数据转换函数 make_data()：将输入数据与参考基因集对齐
@profile
def make_data(adata, ref_genes):
    if sparse.issparse(adata.X): #检查数据是否为稀疏矩阵    
        adata.X = adata.X.toarray() #必要时转换为密集矩阵
    print(adata)
    if np.min(adata.X) != 0:
        return None
    adata.var_names = adata.var['Symbol']
    new_data = np.zeros((adata.X.shape[0], len(ref_genes))) #创建全零矩阵，尺寸为(细胞数×参考基因数)
    useful_gene_index = np.where(adata.var_names.isin(ref_genes)) #筛选有效基因
    useful_gene = adata.var_names[useful_gene_index] 
    if not sparse.issparse(adata.X):
        new_data[:, np.where(np.isin(ref_genes, useful_gene))] = adata.X[:, useful_gene_index] #将原始数据中存在的基因填充到对应位置
    else:
        new_data[:, np.where(np.isin(ref_genes, useful_gene))] = adata.X.toarray()[:, useful_gene_index]
    new_data = sparse.csr_matrix(new_data)
    new = ad.AnnData(X=new_data)
    new.var_names = ref_genes
    new.obs = adata.obs
    return new #返回新的AnnData对象

#重点来了：LMDB转换函数 folder2lmdb() 将预处理后的h5ad文件转换为LMDB格式
def folder2lmdb(dpath, write_frequency=100000):
    print("Generate LMDB to %s" % dpath) #创建训练集和验证集两个LMDB数据库
    train_db = lmdb.open(dpath + '/train.db', map_size=536870912000*4, readonly=False, meminit=False, map_async=True) #这个模块是外置的
    val_db = lmdb.open(dpath + '/val.db', map_size=536870912000, readonly=False, meminit=False, map_async=True) #自动分配5%数据作为验证集

    txn = train_db.begin(write=True)
    val_txn = val_db.begin(write=True)
    length = 0
    val_len = 0
    for f in os.listdir(dpath):
        if f.startswith('PRO'):
            adata = sc.read_h5ad(os.path.join(dpath, f))
            data = adata.X
            print(f)
            val_index = np.random.randint(0, data.shape[0], np.ceil(data.shape[0]*0.05).astype(np.int32))
            # print(val_index)
            for i in range(data.shape[0]):
                x = data[i].toarray()
                if i in val_index:
                    val_txn.put(u'{}'.format(val_len).encode('ascii'), x)
                    val_len += 1
                    if (val_len + 1) % write_frequency == 0:
                        print('val write: ', val_len)
                        val_txn.commit()
                        val_txn = val_db.begin(write=True)
                else:
                    txn.put(u'{}'.format(length).encode('ascii'), x)
                    length += 1
                if (length + 1) % write_frequency == 0: #支持分批写入(默认每100,000条提交一次)
                    print('write: ', length)
                    txn.commit()
                    txn = train_db.begin(write=True)

            print(length, val_len)
    # finish iterating through dataset 使用异步映射提高性能
    txn.commit()
    val_txn.commit()
    with train_db.begin(write=True) as txn, val_db.begin(write=True) as val_txn:
        txn.put(b'__len__', str(length).encode()) #记录数据总条数到__len__键
        val_txn.put(b'__len__', str(val_len).encode())
    print("Flushing database ...")
    train_db.sync()
    train_db.close()
    val_db.sync()
    val_db.close()

#LMDB读取函数 get_lmdb_data() 测试读取LMDB数据 
def get_lmdb_data(dpath):
    db = lmdb.open(dpath) #还是这个模块
    with db.begin() as txn:
        print(txn.get(b'__len__')) #获取数据总条数
        print(txn.get(b'__len__').decode("utf-8"))
        print(int(txn.get(b'__len__').decode("utf-8")))
        for i in range(1, 11): #读取前10条数据并打印统计信息
            value = txn.get(u'{}'.format(i).encode('ascii'))

            if value:
                value = np.frombuffer(value)
                print(np.min(value), np.max(value))


if __name__ == '__main__':
    #pretrain_h5ad('/home/share/huada/home/qiuping1/workspace/llm/scbert/data/stereo_miner/stomics_dataset',
     #             '/home/share/huada/home/qiuping1/workspace/llm/scbert/data/gene2vec_names_list.pkl',
     #             '/home/share/huada/home/qiuping1/workspace/llm/scbert/data/stereo_miner/human_data.txt',
     #             '/home/share/huada/home/qiuping1/workspace/llm/scbert/data/stereo_miner/human_data')
    folder2lmdb('/home/share/huada/home/qiuping1/workspace/llm/scbert/data/stereo_miner/human_data')
    # get_lmdb_data('/home/share/huada/home/qiuping1/workspace/llm/scbert/data/stereo_miner/human_brain/train.lmdb')

