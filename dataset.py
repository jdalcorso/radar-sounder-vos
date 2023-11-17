import os
import torch
import scipy
from torch.utils.data import Dataset 
from sklearn.cluster import KMeans

# Better version
class VideoDataset(Dataset):
    def __init__(self, filepath = '/data/videos/class_0'):
        self.filepath = filepath
        self.videos = []
        self.pairs = []

        filelist = os.listdir(filepath)
        filelist = [filepath+'/'+fn for fn in filelist if fn.endswith('.pt') and fn.startswith('v')]
        print('Total number of videos:', len(filelist))
        for f in filelist:
            video = torch.load(f)
            self.videos.append(video)
        
        for i in range(len(self.videos)):
            video = self.videos[i]
            for j in range(video.shape[1]-1):
                self.pairs.append(video[0,j:j+2,:,:])

        print('Total number of pairs', len(self.pairs))

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self,index):
        item = self.pairs[index].unsqueeze(0).float()
        return item, torch.tensor(0)


class SingleVideo(Dataset):
    def __init__(self, filepath = '/data/videos/class_0'):
        self.filepath = filepath
        self.video = torch.load(filepath+'/video111.pt')
        self.label = torch.load(filepath+'/lbl111.pt')      

    def __len__(self):
        return 1
    
    def __getitem__(self,index):
        return self.video, self.label


class MCORDS1Dataset(Dataset):
    def __init__(self, filepath ='/data/MCoRDS1_2010_DC8/RG_MCoRDS1_2010_DC8.pt', dim = (400,48), factor = 1):
        self.filepath = filepath
        self.pairs = []
        t = torch.load(filepath)
        th,tw = t.shape
        n_patches = tw//dim[1]

        offsets = []
        for i in range(factor):
            offsets.append(dim[1]//factor*i)
            
        for j in range(factor):
            for i in range(n_patches-1):
                t1 = t[:dim[0],j+ i*dim[1]:j+ i*dim[1]+dim[1]]
                t2 = t[:dim[0],j+ i*dim[1]+dim[1]:j+ i*dim[1]+2*dim[1]]
                self.pairs.append(torch.stack([t1,t2]))
        print('Total pairs:', len(self.pairs))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self,index):
        return self.pairs[index].float().unsqueeze(0), torch.tensor(0)

class SingleVideoMCORDS1(Dataset):
    def __init__(self, filepath = '/data/MCoRDS1_2010_DC8/', dim = (400,48), nframes = 35):
        self.filepath = filepath
        self.dim = dim
        self.nframes = 35
        self.rg = torch.load(filepath+'/RG_MCoRDS1_2010_DC8.pt')[:dim[0],:]
        self.label = torch.load(filepath+'/lbl_cresis_2010_002.pt')
        self.video = self.rg.unsqueeze(0).unsqueeze(0)
        self.video = torch.permute(self.video.unfold(3,dim[1],dim[1]),[0,1,3,2,4]).squeeze(0)[:,:self.nframes,:,:]     

    def __len__(self):
        return 1
    
    def __getitem__(self,index):
        return self.video, self.label[:self.dim[0],:self.dim[1]]

class TestDataset(Dataset):
    # return rg, sg with dimension (dim[0],npatches*dim[1])
    def __init__(self, filepath = '/data/MCoRDS1_2010_DC8/', dim = (400,48), npatches = 28):
        self.filepath = filepath
        self.dim = dim
        self.npatches = npatches
        self.rg = torch.load(filepath+'/RG2_MCoRDS1_2010_DC8.pt')[:dim[0],:]
        self.sg = torch.load(filepath+'/SG2_MCoRDS1_2010_DC8.pt')[:dim[0],:] # seg where uncertain class is noise or air (for propagation)
        self.sr = torch.load(filepath+'/SG3_MCoRDS1_2010_DC8.pt')[:dim[0],:] # seg with uncertain class (4) (only for report)
        self.nrg = self.rg.shape[1]//(self.dim[1]*self.npatches)

        # Trim exceeding dataset
        self.rg = self.rg[:,:self.nrg*self.npatches*self.dim[1]]
        self.sg = self.sg[:,:self.nrg*self.npatches*self.dim[1]]
        self.sr = self.sr[:,:self.nrg*self.npatches*self.dim[1]]

        print('Tot test dataset dimension:',self.rg.shape)
        print('Single radargram length:', self.dim[1]*self.npatches)
        print('Number of radargrams:', self.nrg)

    def __len__(self):
        return self.nrg

    def __getitem__(self, index):
        dim = self.dim
        np = self.npatches
        rg = self.rg[:,index*(np*dim[1]) : index*(np*dim[1]) + (np*dim[1])]
        sg = self.sg[:,index*(np*dim[1]) : index*(np*dim[1]) + (np*dim[1])]
        return rg, sg
    
class MCORDS1Miguel(Dataset):
    def __init__(self, filepath ='/data2/MCORDS1_Miguel/', dim = (1248,24), from_scratch = False):
        self.filepath = filepath
        self.H = dim[0]
        self.W = dim[1]
        self.pairs = []
        if from_scratch:
            self.data = self.merge_ds(filepath, dim)
        else:
            self.data = torch.load(filepath+'full_rg.pt')
        print('Pairs:', self.data.shape[1]//self.W-1)
 
    def merge_ds(self, filepath, dim):
        all_files = os.listdir(filepath)
        mat_files = sorted([file for file in all_files if file.endswith('.mat')])
        t0 = torch.zeros(dim[0],1)
        for file in mat_files:
            mat = scipy.io.loadmat(filepath + file)
            dat_name = list(mat.keys())[-1]
            offset = -(mat[dat_name].shape[1]%self.W)
            if offset == 0:
                t0 = torch.cat([t0, torch.tensor(mat[dat_name][:dim[0],:])], dim = 1)
            else:
                t0 = torch.cat([t0, torch.tensor(mat[dat_name][:dim[0],:-(mat[dat_name].shape[1]%self.W)])], dim = 1)
        return t0[:,1:]

    def __len__(self):
        return self.data.shape[1]//self.W-1

    def __getitem__(self, index):
        t1 = self.data[:,self.W*index:self.W*index+self.W]
        t2 = self.data[:,self.W*(index+1):self.W*(index+1)+self.W]
        return torch.stack([t1,t2]).float().unsqueeze(0), torch.tensor(0)