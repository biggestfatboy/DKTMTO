from torch.utils.data import Dataset
class lsgodata(Dataset):
    def __init__(self,archive,archive_fit):
        self.archive = archive
        self.archive_fit = archive_fit

    def __getitem__(self, idx):
        pop_batch = self.archive[idx]
        pop_fit = self.archive_fit[idx]
        return pop_batch,pop_fit

    def __len__(self):
        return self.archive.shape[0]

class MTO_data(Dataset):
    def __init__(self,archive_pop1,archive_pop2):
        self.archive_pop1 = archive_pop1
        self.archive_pop2 = archive_pop2

    def __getitem__(self, idx):
        pop_batch = self.archive_pop1[idx]
        pop_fit = self.archive_pop2[idx]
        return pop_batch,pop_fit

    def __len__(self):
        return self.archive_pop1.shape[0]