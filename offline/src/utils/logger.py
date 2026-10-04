import os
import csv

class Logger:
    def __init__(self, cfg):
        self.log_dir = os.getcwd() 
        self.checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.train_file = open(os.path.join(self.log_dir, "train.csv"), "w")
        self.train_writer = None
        
        self.eval_file = open(os.path.join(self.log_dir, "eval.csv"), "w")
        self.eval_writer = None

    def log_train(self, log_dict, step):
        log_dict['step'] = step
        if self.train_writer is None:
            self.train_writer = csv.DictWriter(self.train_file, fieldnames=log_dict.keys())
            self.train_writer.writeheader()
        
        self.train_writer.writerow(log_dict)
        self.train_file.flush()

    def log_eval(self, log_dict, step):
        log_dict['step'] = step
        if self.eval_writer is None:
            self.eval_writer = csv.DictWriter(self.eval_file, fieldnames=log_dict.keys())
            self.eval_writer.writeheader()
            
        self.eval_writer.writerow(log_dict)
        self.eval_file.flush()

    def close(self):
        self.train_file.close()
        self.eval_file.close()