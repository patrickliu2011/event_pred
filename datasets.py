import numpy as np
import pandas as pd
import torch

import rasterio

class HousingDataset(torch.utils.data.Dataset):
    '''Housing Dataset, Seasonal Sentinel-2'''
    def __init__(self, 
                 csv_file = "/atlas/u/erikrozi/housing_event_pred/data/train_seasonal_eff.csv",
                 root_dir = "/atlas/u/pliu1/housing_event_pred/data/houses_new/", 
                 first_year = 2016,
                 last_year = 2019,
                 build_years = [2017, 2018], 
                 transform = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            first_year (int): Earliest year to allow images from.
            last_year (int): Last year to allow images from.
            build_years (list of ints): List of all build years to use.
            transform (callable, optional): Optional transform to be applied on tensor images.
        """
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df["year.built"].isin(build_years)]
        self.root_dir = root_dir
        self.transform = transform
        
        self.first_year = first_year
        self.last_year = last_year
        
        # Gets all unique image ids
        self.image_ids = self.df.iloc[:, 0].to_numpy()
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        '''Gets timeseries info for images in one area
        
        Args: 
            idx: Index of datapoint in dataset
            
        Returns dictionary containing:
            'id' (int): ID of image location
            'region' (str): Region of image location
            'year_built' (int): Year built as labeled in dataset
            'label' (int, 0 or 1): 0 if sampled image occurs before the build year, 1 if it occurs after
            'image_start' (torch.FloatTensor): Image at the start time
            'year_start' (int): Year of the start image 
            'month_start' (month): Month of the start date
            'image_end' (torch.FloatTensor): Image at the end time. Without transfermations, should have dimensions
                ChannelxHeightxWidth, with channel order R, G, B, B8 (NIR)
            'year_end' (int): Year of the end image 
            'month_end' (month): Month of the end date
            'image_sample' (torch.FloatTensor): Image at the sample time
            'year_sample' (int): Year of the sample image 
            'month_sample' (month): Month of the sample date
        '''
        row = self.df.iloc[idx]
        year_built = row["year.built"]
        
        label = 1 * (np.random.rand() < 0.5) # 0 for sampled image before construction year, 1 for after
        before_times = [(year, month) for year in range(self.first_year, year_built) for month in range(1, 12, 3)]
        after_times = [(year, month) for year in range(year_built+1, self.last_year+1) for month in range(1, 12, 3)]
        if label == 0: # Start -> Sampled -> Build -> End
            T_start, T_sample = sorted(np.random.choice(np.arange(len(before_times)), size=(2,), replace=False))
            T_end = np.random.choice(np.arange(len(after_times)), replace=False)
            T_start = before_times[T_start]
            T_end = after_times[T_end]
            T_sample = before_times[T_sample]
        else: # Start-> Build -> Sampled -> End
            T_start = np.random.choice(np.arange(len(before_times)), replace=False)
            T_sample, T_end = sorted(np.random.choice(np.arange(len(after_times)), size=(2,), replace=False))
            T_start = before_times[T_start]
            T_end = after_times[T_end]
            T_sample = after_times[T_sample]
            
        col_names = [f"sentinel_{y}_{(m - 1) // 3 + 1}" for y, m in [T_start, T_end, T_sample]]
        image_files = row[col_names].tolist()
        
        images = torch.FloatTensor(np.array([rasterio.open(name).read() for name in image_files])) 

        if self.transform:
            images = self.transform(images)
            
        img_start, img_end, img_sample = images
            
        return {
            'id': int(self.image_ids[idx]),
            'region': self.df.iloc[idx, :]['region'],
            'year_built': year_built,
            'label': label,
            'image_start': img_start,
            'year_start': T_start[0],
            'month_start': T_start[1],
            'image_end': img_end,
            'year_end': T_end[0],
            'month_end': T_end[1],
            'image_sample': img_sample,
            'year_sample': T_sample[0],
            'month_sample': T_sample[1],
        }