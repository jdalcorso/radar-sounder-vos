import os
import torch
from torch.utils.data import Dataset 
from sklearn.cluster import KMeans

class VideoDataset(Dataset):
    def __init__(self, filepath = '/data/videos/class_0'):
        self.filepath = filepath
        self.videos = []

        filelist = os.listdir(filepath)
        filelist = [filepath+'/'+fn for fn in filelist if fn.endswith('.pt') and fn.startswith('v')]
        print(filelist, 'Total number of videos:', len(filelist))
        for f in filelist:
            video = torch.load(f)
            self.videos.append(video)        

    def __len__(self):
        return 111
    
    def __getitem__(self,index):
        item = self.videos[index]
        rnd = torch.randint(item.shape[1]-1,(1,)).item()  
        return item[:,rnd:rnd+2,:,:].float(), torch.tensor(0)

# Better version
class VideoDataset2(Dataset):
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
    def __init__(self, filepath ='/data/MCoRDS1_2010_DC8/RG_MCoRDS1_2010_DC8.pt', dim = (400,48)):
        self.filepath = filepath
        self.pairs = []
        t = torch.load(filepath)
        th,tw = t.shape
        n_patches = tw//dim[1]

        for i in range(n_patches-1):
            t1 = t[:dim[0],i*dim[1]:i*dim[1]+dim[1]]
            t2 = t[:dim[0],i*dim[1]+dim[1]:i*dim[1]+2*dim[1]]
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