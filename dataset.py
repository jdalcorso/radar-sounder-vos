import os
import torch
from torch.utils.data import Dataset 

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
        print(filelist, 'Total number of videos:', len(filelist))
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

        #filelist = os.listdir(filepath)
        #filelist = [filepath+'/'+fn for fn in filelist if fn.endswith('.pt')]
        #print(filelist, 'Total number of videos:', len(filelist))
        #for f in filelist:
        #    video = torch.load(f)
        #    self.videos.append(video)        

    def __len__(self):
        return 1
    
    def __getitem__(self,index):
        return self.video, self.label