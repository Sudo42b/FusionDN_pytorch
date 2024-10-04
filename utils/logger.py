from typing import List
import os
import csv

class Logger(object):
    """
        Logger object for training process, supporting resume training
        Example:
            >>> batch_logger = Logger(os.path.join(args.log_folder, 'batch.log'),
                                        ['epoch', 'batch', 'loss', 'probs', 'lr'],
                                        args.log_resume)
            >>> batch_logger.log({'epoch': epoch, 
                                'batch': batch_idx, 
                                'loss': losses.val, 
                                'probs': prob_meter.val, 
                                'lr': optimizer.param_groups[0]['lr']})
    """
    def __init__(self, path:str, header:List[str], resume:bool=False):
        """
        Args:
            path (str): logging file path
            header (List[str]): a list of tags for values to track
            resume (bool, optional): a flag controling whether to create a new
                                    file or continue recording after the latest step. 
                                    Defaults to False.
        """
        self.log_file = None
        self.resume = resume
        self.header = header
        # Check to .txt file is used to store path
        if os.path.splitext(path)[1] != '.txt':
            path = os.path.join(path, 'log.txt')
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not self.resume:
            self.log_file = open(path, 'w')
            self.logger = csv.writer(self.log_file, delimiter='\t')
            self.logger.writerow(self.header)
        else:
            self.log_file = open(path, 'a+')
            self.log_file.seek(0, os.SEEK_SET)
            reader = csv.reader(self.log_file, delimiter='\t')
            self.header = next(reader)
            # move back to the end of file
            self.log_file.seek(0, os.SEEK_END)
            self.logger = csv.writer(self.log_file, delimiter='\t')

    def __del__(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for tag in self.header:
            assert tag in values, 'Please give the right value as defined'
            write_values.append(values[tag])
        self.logger.writerow(write_values)
        self.log_file.flush()