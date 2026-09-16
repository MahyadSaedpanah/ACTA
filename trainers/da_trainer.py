import torch
import os
import sys
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
import collections

from sklearn.metrics import accuracy_score
from dataloader.dataloader import data_generator
from configs.data_model_configs import get_dataset_class
from algorithms.utils import fix_randomness, starting_logs
from algorithms import get_algorithm_class


from algorithms.utils import get_time
from algorithms.utils import AverageMeter
from sklearn.metrics import f1_score

torch.backends.cudnn.benchmark = True  
   

class da_trainer(object):
    """
   This class contain the main training functions for our AdAtime
    """
    def __init__(self, args):
        self.args = args
        self.da_method = args.da_method  # Selected  DA Method
        self.dataset = args.dataset  # Selected  Dataset
        self.backbone = args.backbone
        self.device = torch.device(args.device)  # device

        self.run_description = args.run_description
        self.experiment_description = args.experiment_description

        self.best_f1 = 0
        # paths
        self.home_path = os.getcwd()
        self.save_dir = args.save_dir
        self.data_path = os.path.join(args.data_path, self.dataset)
        self.create_save_dir()

        # Specify runs
        self.num_runs = args.num_runs

        # get dataset and base model configs
        self.dataset_configs = self.get_configs()
       
        # to fix dimension of features in classifier and discriminator networks.
        self.dataset_configs.t_feat_dim = self.dataset_configs.tcn_final_out_channles if args.backbone == "TCN" else self.dataset_configs.t_feat_dim
        self.args.seq_len = self.dataset_configs.sequence_len
        self.args.enc_in = self.dataset_configs.input_channels
        
    def test(self):
    
        run_name = f"{self.run_description}"
        # Logging
        self.avg_res_dir = os.path.join(self.save_dir, self.experiment_description, run_name, get_time())
        os.makedirs(self.avg_res_dir, exist_ok=True)

        self.exp_log_dir = os.path.join(self.avg_res_dir, 'res')
        os.makedirs(self.exp_log_dir, exist_ok=True)

        command = ' '.join(sys.argv)
        with open(os.path.join(self.avg_res_dir, 'command.txt'), "a") as file:
            command_list = command.split('--')
            for arg in command_list:
                file.write('--'+arg+'\n')


        scenarios = self.dataset_configs.scenarios  # return the scenarios given a specific dataset.
        df_a = pd.DataFrame(columns=['scenario','run_id','accuracy','f1'])
        df_s = pd.DataFrame(columns=['scenario','run_id','accuracy','f1'])
        self.trg_acc_list = []
        for i in scenarios[self.args.start:self.args.end]:
            src_id = i[0]
            trg_id = i[1]


            for run_id in range(self.num_runs):  # specify number of consecutive runs
                # fixing random seed
                fix_randomness(run_id)

                # Logging
                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir,
                                                                   src_id, trg_id, run_id)
                # Load data
                self.load_data(src_id, trg_id)
                # get algorithm
                print(self.da_method)
                algorithm_class = get_algorithm_class(self.da_method)
                algorithm = algorithm_class(self.dataset_configs, self.device, self.args)
                
                algorithm.to(self.device)
                self.algorithm = algorithm       
                self.logger.debug('Source Test Dataset {}  Target Test Dataset {}'.format(len(self.src_test_dl), len(self.trg_test_dl)))

                self.algorithm.load_model(os.path.join(self.args.test_model_prefix, 'res', f'{src_id}_to_{trg_id}_run_{run_id}', 'model.pth'))
                # test target
                acc, f1 = self.evaluate()
                log = {'scenario':i,'run_id':run_id,'accuracy':acc,'f1':f1}
                self.logger.debug('target acc {} f1 {}'.format(acc, f1))
                df_a = pd.concat([df_a, pd.DataFrame([log])], ignore_index=True)
                
                # test source
                acc, f1 = self.evaluate(data='s')
                log = {'scenario':i,'run_id':run_id,'accuracy':acc,'f1':f1}
                self.logger.debug('source acc {} f1 {}'.format(acc, f1))
                df_s = pd.concat([df_s, pd.DataFrame([log])], ignore_index=True)
                
                path =  os.path.join(self.avg_res_dir, 'test_target_results.csv')
                df_a.to_csv(path,sep = ',')
                path_s =  os.path.join(self.avg_res_dir, 'test_source_results.csv')
                df_s.to_csv(path_s,sep = ',')

       
        df_a = self.avg_result(df_a)
        df_s = self.avg_result(df_s)

        path =  os.path.join(self.avg_res_dir, 'test_target_results.csv')
        df_a.to_csv(path,sep = ',')
        path_s =  os.path.join(self.avg_res_dir, 'test_source_results.csv')
        df_s.to_csv(path_s,sep = ',')
        


            
    def train(self):

        run_name = f"{self.run_description}"
        # Logging
        self.avg_res_dir = os.path.join(self.save_dir, self.experiment_description, run_name, get_time())
        os.makedirs(self.avg_res_dir, exist_ok=True)

        self.exp_log_dir = os.path.join(self.avg_res_dir, 'res')
        os.makedirs(self.exp_log_dir, exist_ok=True)

        command = ' '.join(sys.argv)
        with open(os.path.join(self.avg_res_dir, 'command.txt'), "a") as file:
            command_list = command.split('--')
            for arg in command_list:
                file.write('--'+arg+'\n')

        scenarios = self.dataset_configs.scenarios  # return the scenarios given a specific dataset.
        df_a = pd.DataFrame(columns=['scenario','run_id','accuracy','f1'])
        df_s = pd.DataFrame(columns=['scenario','run_id','accuracy','f1'])
        self.trg_acc_list = []
        for i in scenarios[self.args.start:self.args.end]:
            src_id = i[0]
            trg_id = i[1]


            for run_id in range(self.num_runs):  # specify number of consecutive runs
                # fixing random seed
                fix_randomness(run_id)

                # Logging
                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir,
                                                                   src_id, trg_id, run_id)
                self.model_path = os.path.join(self.home_path, self.scenario_log_dir, 'model.pth')
                self.best_f1 = 0
                # Load data
                self.load_data(src_id, trg_id)

    
                # get algorithm
                print(self.da_method)
               
                algorithm_class = get_algorithm_class(self.da_method)
                algorithm = algorithm_class(self.dataset_configs, self.device, self.args)
                
                algorithm.to(self.device)
                self.algorithm = algorithm
                source_loss_history = {}
                target_loss_history = {}


                # Average meters
                loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
                self.logger.debug('Source Train Dataset {}  Target Train Dataset {}'.format(len(self.src_train_dl), len(self.trg_train_dl)))
                # train
                for epoch in range(1, self.args.num_epochs + 1):
                    self.logger.debug('Epoch Training {}/{}'.format(epoch, self.args.num_epochs))
                    joint_loaders = enumerate(zip(self.src_train_dl, self.trg_train_dl))
                    algorithm.train()

                    for step, ((src_x, src_y), (trg_x, trg_y)) in joint_loaders:
                        src_x, src_y, trg_x, trg_y = src_x.float().to(self.device), src_y.long().to(self.device), \
                                              trg_x.float().to(self.device), trg_y.long().to(self.device)

                        
                        losses = algorithm.update(src_x, src_y, trg_x)


                        for key, val in losses.items():
                            loss_avg_meters[key].update(val, src_x.size(0))
                            # save for source
                            if "Src" in key or "Domain" in key:
                                if key not in source_loss_history:
                                    source_loss_history[key] = []
                                source_loss_history[key].append(val)

                            # save for target
                            if key in ["align target tf loss", "cond_ent_loss_t", "cond_ent_loss_f"]:
                                if key not in target_loss_history:
                                    target_loss_history[key] = []
                                target_loss_history[key].append(val)
                                

                        
                        if step % self.args.print_freq == 0:
                            keys = loss_avg_meters.keys()
                            train_log = 'epoch {}   '.format(epoch)
                            for key in keys:
                                train_log += '{}    {:.3f}({:.3f})    '.format(key,loss_avg_meters[key].val, loss_avg_meters[key].avg)

                            self.logger.debug(train_log)

                    self.logger.debug(
                        'Epoch completed {}/{}'.format(
                            epoch,
                            self.args.num_epochs
                        )
                    )

                # Build source-only TSA codebook after the source-task
                # training stage. Target data is never used here.
                if hasattr(algorithm, "build_tsa_codebook"):
                    self.logger.debug(
                        "Building source-only TSA codebook"
                    )
                    tsa_summary = algorithm.build_tsa_codebook(
                        self.src_train_dl
                    )
                    self.logger.debug(
                        "TSA codebook: {}".format(tsa_summary)
                    )

                    # Stage-4 diagnostic only:
                    # compute the target-specific TSA gate on one
                    # unlabeled target batch. No optimization uses it yet.
                    if hasattr(algorithm, "target_tsa_gate"):
                        target_batch = next(iter(self.trg_train_dl))
                        target_x = target_batch[0].float().to(
                            self.device
                        )

                        gate_out = algorithm.target_tsa_gate(
                            target_x
                        )

                        gate = gate_out["gate"]
                        reliability = gate_out["reliability"]

                        self.logger.debug(
                            "TSA gate diagnostic: {}".format(
                                {
                                    "reliability_mean": float(
                                        reliability.mean().item()
                                    ),
                                    "gate_nonidentity_mean": float(
                                        gate[:, 1:].mean().item()
                                    ),
                                    "gate_nonidentity_min": float(
                                        gate[:, 1:].min().item()
                                    ),
                                    "gate_nonidentity_max": float(
                                        gate[:, 1:].max().item()
                                    ),
                                    "identity_mean": float(
                                        gate[:, 0].mean().item()
                                    ),
                                }
                            )
                        )

                # Save the final-epoch model only.
                # This avoids target-label model selection during UDA training.
                self.logger.debug('Saving ACTA final-epoch model')
                algorithm.save_model(self.model_path)

                # test target
                acc, f1 = self.evaluate(final=True)
                log = {'scenario':i,'run_id':run_id,'accuracy':acc,'f1':f1}
                self.logger.debug('target acc {} f1 {}'.format(acc, f1))
                df_a = pd.concat([df_a, pd.DataFrame([log])], ignore_index=True)
                
                # test source
                acc, f1 = self.evaluate(final=True, data='s')
                log = {'scenario':i,'run_id':run_id,'accuracy':acc,'f1':f1}
                self.logger.debug('source acc {} f1 {}'.format(acc, f1))
                df_s = pd.concat([df_s, pd.DataFrame([log])], ignore_index=True)
                
                path =  os.path.join(self.avg_res_dir, 'target_results.csv')
                df_a.to_csv(path,sep = ',')
                path_s =  os.path.join(self.avg_res_dir, 'source_results.csv')
                df_s.to_csv(path_s,sep = ',')




        df_a = self.avg_result(df_a)
        df_s = self.avg_result(df_s)

        path =  os.path.join(self.avg_res_dir, 'target_results.csv')
        df_a.to_csv(path,sep = ',')
        path_s =  os.path.join(self.avg_res_dir, 'source_results.csv')
        df_s.to_csv(path_s,sep = ',')

    
    def evaluate(self, final=False, data='t'):
        assert data in ['t', 's']
        self.algorithm.eval()
        if final == True:
            self.algorithm.load_model(self.model_path)
    
        self.trg_pred_labels = np.array([])
        self.trg_true_labels = np.array([])

        with torch.no_grad():
            if data == 't':
                dataloader = self.trg_test_dl
            elif data == 's':
                dataloader = self.src_test_dl

            for data, labels in dataloader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                predictions = self.algorithm.predict(data)

                # compute loss
                pred = predictions.detach().argmax(dim=1)  # get the index of the max log-probability

                self.trg_pred_labels = np.append(self.trg_pred_labels, pred.cpu().numpy())
                self.trg_true_labels = np.append(self.trg_true_labels, labels.data.cpu().numpy())
        
        accuracy = accuracy_score(self.trg_true_labels, self.trg_pred_labels)
        f1 = f1_score(self.trg_pred_labels, self.trg_true_labels, pos_label=None, average="macro")
        return accuracy*100, f1


    def get_configs(self):
        dataset_class = get_dataset_class(self.dataset)
        return dataset_class()

    def load_data(self, src_id, trg_id):
        self.src_train_dl, self.src_test_dl = data_generator(self.data_path, src_id, self.args)
        self.trg_train_dl, self.trg_test_dl = data_generator(self.data_path, trg_id, self.args)
    

    def create_save_dir(self):
        if not os.path.exists(self.save_dir):       
            os.mkdir(self.save_dir)

    def avg_result(self, df):
        """
        Robust aggregation across runs/scenarios.

        Keeps accuracy/f1 numeric and avoids inserting a None row,
        which can coerce pandas columns to object dtype on newer
        pandas versions.
        """

        if df.empty:
            return df

        work = df.copy()

        work["accuracy"] = pd.to_numeric(
            work["accuracy"],
            errors="coerce",
        )
        work["f1"] = pd.to_numeric(
            work["f1"],
            errors="coerce",
        )

        valid = work.dropna(
            subset=["scenario", "accuracy", "f1"]
        ).copy()

        grouped = (
            valid
            .groupby(
                "scenario",
                as_index=False,
                sort=False,
            )
            .agg(
                accuracy_mean=("accuracy", "mean"),
                accuracy_std=("accuracy", "std"),
                f1_mean=("f1", "mean"),
                f1_std=("f1", "std"),
            )
        )

        print(
            grouped[
                ["scenario", "accuracy_mean"]
            ].rename(
                columns={"accuracy_mean": "accuracy"}
            )
        )

        print(
            grouped[
                ["scenario", "accuracy_std"]
            ].rename(
                columns={"accuracy_std": "accuracy"}
            )
        )

        summary_rows = []

        for _, row in grouped.iterrows():
            summary_rows.append(
                {
                    "scenario": row["scenario"],
                    "run_id": "mean",
                    "accuracy": row["accuracy_mean"],
                    "f1": row["f1_mean"],
                }
            )

            summary_rows.append(
                {
                    "scenario": row["scenario"],
                    "run_id": "std",
                    "accuracy": row["accuracy_std"],
                    "f1": row["f1_std"],
                }
            )

        if len(grouped) > 0:
            summary_rows.append(
                {
                    "scenario": "all_mean",
                    "run_id": "all",
                    "accuracy": grouped["accuracy_mean"].mean(),
                    "f1": grouped["f1_mean"].mean(),
                }
            )

            summary_rows.append(
                {
                    "scenario": "mean_scenario_std",
                    "run_id": "all",
                    "accuracy": grouped["accuracy_std"].mean(),
                    "f1": grouped["f1_std"].mean(),
                }
            )

        if summary_rows:
            work = pd.concat(
                [
                    work,
                    pd.DataFrame(summary_rows),
                ],
                ignore_index=True,
            )

        return work