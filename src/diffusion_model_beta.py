import os
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from easydict import EasyDict as edict
from tqdm import tqdm

from .models.transformer_model import GraphTransformer
from .utils import F_deg, to_dense, create_masks, PlaceHolder


class BetaDiffusionBinaryEdge(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, **kwargs):
        """
        , train_metrics, sampling_metrics, visualization_tools, extra_features=None,
                 domain_features=None
        """
        super().__init__()

        self.cfg = cfg
        self.ddp = True if cfg.general.gpus>1 else False
        self.save_hyperparameters()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims

        self.model = GraphTransformer(
            n_layers=cfg.model.n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=cfg.model.hidden_mlp_dims,
            hidden_dims=cfg.model.hidden_dims,
            output_dims=output_dims,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
            ddpm='beta')
        self.input_space = cfg.model.input_space
        self.degree_dist = dataset_infos.degree_dist

        # parameters used in initial data (edge) transformation
        self.Scale = cfg.model.Scale            # 0.9
        self.Shift = cfg.model.Shift            # 0.09

        # parameters used in preconditioning
        self.apply_precondition = self.cfg.model.apply_precondition
        self.edge_dist = dataset_infos.edge_types

        # parameters for beta linear schedule
        self.beta_max = cfg.model.beta_max      # 20
        self.beta_min = cfg.model.beta_min      # 0.1
        self.beta_d = self.beta_max - self.beta_min

        # parameters for beta sigmoid schedule
        self.sigmoid_start = cfg.model.sigmoid_start    # 9
        self.sigmoid_end = cfg.model.sigmoid_end        # -9
        self.sigmoid_power = cfg.model.sigmoid_power    # 0.5

        self.noise_schedule = cfg.model.diffusion_noise_schedule
        if self.noise_schedule == 'linear':
            self.logit_alpha = lambda t: (-0.5 * self.beta_d * t**2 - self.beta_min * t).to(torch.float64).exp().logit().to(torch.float32)
        elif self.noise_schedule == 'sigmoid':
            self.logit_alpha = lambda t: self.sigmoid_start + (self.sigmoid_end - self.sigmoid_start) * (t**self.sigmoid_power)
        else:
            raise NotImplementedError(f"noise schedule '{self.noise_schedule}' is not implemented")

        # [v1.1 unique] parameters for latent distribution
        if not self.ddp:
            self.eta_E = torch.tensor(cfg.model.eta_E, dtype=torch.float32)
            self.eta_X = torch.tensor(cfg.model.eta_X, dtype=torch.float32)
        else:
            self.register_buffer("eta_E", torch.tensor(cfg.model.eta_E, dtype=torch.float32))   # otherwise these tensors are placed on cpu
            self.register_buffer("eta_X", torch.tensor(cfg.model.eta_X, dtype=torch.float32))

        # [v1 unique] parameters for combining training losses
        self.lambda_train = cfg.train.lambda_train

        # parameters for reverse sampling
        self.T = cfg.model.diffusion_steps        # number of sampling steps, 500
        self.valid_time_segs = 10
        self.nodes_dist = dataset_infos.nodes_dist

        self.train_epoch_losses = []
        self.valid_epoch_losses = []

    
    def forward(self, noisy_data, node_mask):
        if self.input_space == 'logit':
            input_X, input_E, input_y = noisy_data.logit_X_t, noisy_data.logit_E_t, noisy_data.y_t
        else:
            input_X, input_E, input_y = noisy_data.X_t, noisy_data.E_t, noisy_data.y_t
        z0_hat = self.model(input_X, input_E, input_y, node_mask)
        return z0_hat
    

    def training_step(self, data, batch_idx):
        # [v1 unique] convert x (node degree) to its percentile
        # this operation can map x to unif[0,1]
        x = F_deg(data.x, self.degree_dist)

        dense_data, node_mask = to_dense(
            x=x, edge_index=data.edge_index, batch=data.batch
        )    
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E

        # Transforming and masking raw X and E
        transformed_data = self.scale_shift(X, E, data.y, node_mask)
        X, E = transformed_data.X, transformed_data.E

        # Apply noise
        (bs, n), device = node_mask.shape, node_mask.device
        t = (1. - torch.rand(bs, 1) * (1 - 1e-5)).to(device)
        noisy_data = self.apply_noise(X, E, t, node_mask)

        # make prediction for E0, X0
        if self.apply_precondition:
            noisy_data = self.precondition(noisy_data, node_mask)
        z0_hat = self.forward(noisy_data, node_mask)

        # compute loss on edge, node
        E0_pred = z0_hat.E * self.Scale + self.Shift        # caveat: masked area deviates from 0
        X0_pred = z0_hat.X * self.Scale + self.Shift
        loss_E = self.compute_loss(pred=E0_pred, target=E, t=t, node_mask=node_mask, loss_on='edge')
        loss_X = self.compute_loss(pred=X0_pred, target=X, t=t, node_mask=node_mask, loss_on='node')
        loss = self.lambda_train[0] * loss_E + self.lambda_train[1] * loss_X

        # Log the training loss for each batch
        self.train_epoch_losses.append(loss)

        return loss
    
    def configure_optimizers(self):
        if self.cfg.train.annealing:
            opt_adamw = torch.optim.AdamW(
                self.parameters(), lr=self.cfg.train.steplr_init_value, amsgrad=True, weight_decay=self.cfg.train.weight_decay
            )
            # scheduler = torch.optim.lr_scheduler.StepLR(
            #     optimizer=opt_adamw, step_size=self.cfg.train.steplr_step_size, gamma=self.cfg.train.steplr_gamma
            # )
            def lr_lambda(current_epoch):
                gamma = self.cfg.train.steplr_gamma
                multiplier = gamma ** (current_epoch // self.cfg.train.steplr_step_size)
                if self.cfg.train.steplr_min_multiplier is not None:
                    multiplier = max(multiplier, self.cfg.train.steplr_min_multiplier)
                return multiplier
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=opt_adamw, lr_lambda=lr_lambda)

            sch_stepLR = {
                'scheduler': scheduler,
                'interval': 'step'
            }
            return [opt_adamw], [sch_stepLR]
        else:
            opt_adamw = torch.optim.AdamW(
                self.parameters(), lr=self.cfg.train.lr, amsgrad=True, weight_decay=self.cfg.train.weight_decay
            )
            return opt_adamw

    
    def on_train_epoch_start(self):
       self.start_epoch_time = time.time()
       self.train_epoch_losses = []
    

    def on_train_epoch_end(self):
        # 'outputs' is a list containing batch losses for all batches in the epoch
        if len(self.train_epoch_losses) == 0:
            pass
        else:
            train_epoch_loss = torch.stack(self.train_epoch_losses).mean()
            self.log('train_epoch_loss', train_epoch_loss)  # Log the epoch-level training loss
            print(f"training loss at epoch {self.current_epoch}: {train_epoch_loss.detach().cpu().numpy():.4f}, "
                f"time elapsed: {time.time() - self.start_epoch_time:.2f}s."
                )


    def validation_step(self, data, batch_idx):
        # [v1 unique] convert x (node degree) to its percentile
        # this operation can map x to unif[0,1]
        x = F_deg(data.x, self.degree_dist)

        dense_data, node_mask = to_dense(
            x=x, edge_index=data.edge_index, batch=data.batch
        )

        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E

        # Transforming and masking raw X and E
        transformed_data = self.scale_shift(X, E, data.y, node_mask)
        X, E = transformed_data.X, transformed_data.E

        (bs, n), device = node_mask.shape, node_mask.device
        valid_times = torch.linspace(0, 1, 2+self.valid_time_segs)[1:-1].to(device)
        valid_times = valid_times.unsqueeze(1).expand(-1, bs)    # 10, bs

        batch_loss = 0.
        for t in valid_times:
            t = t.unsqueeze(-1)   # bs, 1
            # Apply noise
            noisy_data = self.apply_noise(X, E, t, node_mask)
            
            # make prediction for E0
            if self.apply_precondition:
                noisy_data = self.precondition(noisy_data, node_mask)

            z0_hat = self.forward(noisy_data, node_mask)

            # compute loss on edge
            E0_pred = z0_hat.E * self.Scale + self.Shift        # caveat: masked area deviates from 0
            X0_pred = z0_hat.X * self.Scale + self.Shift
            loss_E = self.compute_loss(pred=E0_pred, target=E, t=t, node_mask=node_mask, loss_on='edge')
            loss_X = self.compute_loss(pred=X0_pred, target=X, t=t, node_mask=node_mask, loss_on='node')
            batch_loss += self.lambda_train[0] * loss_E + self.lambda_train[1] * loss_X

        # record the validation loss for each batch
        valid_loss = batch_loss / self.valid_time_segs
        self.valid_epoch_losses.append(valid_loss)

        return valid_loss
    
    
    def on_validation_epoch_start(self):
        """ reset self.valid_epoch_losses """
        print("validation starts")
        self.valid_epoch_losses = []

    
    def on_validation_epoch_end(self):
        valid_epoch_loss = torch.stack(self.valid_epoch_losses).mean()
        self.log('valid_epoch_loss', valid_epoch_loss)
        print(f"validation loss at epoch {self.current_epoch}: {valid_epoch_loss.detach().cpu().numpy():.4f}")

    
    def on_test_epoch_end(self):
        self.print(f"Testing on epoch {self.current_epoch} ...")

        samples_left_to_generate = self.cfg.genral.final_model_samples_to_generate
        chains_left_to_save = self.cfg.general.final_model_chains_to_save

        generated = edict()
        generated.graphs = []
        while samples_left_to_generate > 0:
            print(f'Samples left to generate: {samples_left_to_generate} / {self.cfg.genral.final_model_samples_to_generate}')
            bs = 2 * self.cfg.train.batch_size
            to_generate = min(samples_left_to_generate, bs)
            pred, _ = self.sample_batch(bs)
            generated.graphs.append(pred.detach().cpu())
        generated.graphs = torch.cat(generated.graphs, dim=0).numpy()

        generated.trajectory = self.sample_trajectory(chains_left_to_save).detach().cpu().numpy()

        current_path = os.getcwd()
        savepath = os.path.join(current_path, f'graphs_numerical/{self.name}/')
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        with open(os.path.join(savepath, f'gen_at_ep-{self.current_epoch}.pkl'), 'wb') as f:
            pickle.dump(generated, f)

        print("Experiment completed!")


    def scale_shift(self, X, E, y, node_mask):
        X = self.Scale * X + self.Shift
        E = self.Scale * E + self.Shift
        diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
        E[diag] = 0
        return PlaceHolder(X=X, E=E, y=y).mask(node_mask)
        
    
    def apply_noise(self, X, E, t, node_mask):
        """ [v1 unique] Sample X_t and E_t from clean X and E """

        # prepare masks
        x_mask, e_mask, triu_mask, _ = create_masks(node_mask)

        # convert uniform time to specifically scheduled alphas
        logit_alpha_t = self.logit_alpha(t)
        alpha_t = torch.sigmoid(logit_alpha_t).unsqueeze(2)    # bs, 1, 1

        # initialize noisy_data
        noisy_data = edict({'t': t, 'y_t':t})

        """ 
        sample from p(z_t | z_0), z_t | z_0 ~ Beta(eta * alpha_t * z_0, eta * (1 - alpha_t * z_0))
        the sampling step utilizes the following relationship between Beta r.v. and Gamma r.v.
        Xa ~ Gamma(a,1), Xb ~ Gamma(b,1), then Xa / (Xa + Xb) ~ Beta(a,b)
        """

        for i, z_0 in enumerate([E.squeeze(-1), X]):

            eta = self.eta_E if i == 0 else self.eta_X
            log_u = log_gamma(eta * alpha_t * z_0)
            log_v = log_gamma(eta - eta * alpha_t * z_0)
            logit_z_t = log_u - log_v

            # if applying noise to E
            if i == 0:
                logit_z_t = logit_z_t.unsqueeze(-1) * triu_mask
                logit_z_t = (logit_z_t + logit_z_t.transpose(1,2)) * e_mask
                z_t = logit_z_t.sigmoid() * e_mask
                noisy_data.update({'E_t': z_t, 'logit_E_t': logit_z_t})

            # if applying noise to X
            elif i == 1:
                logit_z_t = logit_z_t * x_mask
                z_t = logit_z_t.sigmoid() * x_mask
                noisy_data.update({'X_t': z_t, 'logit_X_t': logit_z_t})
            
            else:
                pass

        return noisy_data
    

    def precondition(self, noisy_data, node_mask):
        """ [v1 unique] standardizing noisy data """
        # prepare masks
        x_mask, e_mask, _, _ = create_masks(node_mask)
        bs, n = node_mask.shape

        logit_alpha_t = self.logit_alpha(noisy_data['t'])
        # logit_alpha_t could have multiple sizes, we need to resize it accordingly to ensure alpha_t has size (bs, 1, 1)
        if len(logit_alpha_t.shape) == 0:       # t is a scalar tensor (in reverse sampling)
            logit_alpha_t = logit_alpha_t * torch.ones(bs, 1, 1).to(self.device)
        elif len(logit_alpha_t.shape) == 2:     # t is a tensor with shape (bs, 1) (in training and validation)
            logit_alpha_t = logit_alpha_t.unsqueeze(-1)
        alpha_t = torch.sigmoid(logit_alpha_t)  # bs, 1, 1
        
        bounds = torch.tensor([self.Shift, self.Scale+self.Shift]).to(self.device)
        probs_E0 = self.edge_dist.to(self.device)
        probs_X0 = torch.tensor([0., 1.]).to(self.device)

        # 1. preconditioning the inputs in the original space
        if 'E_t' in noisy_data:
            mean_Et, std_Et = get_beta_stats(bounds=bounds, eta=self.eta_E, alpha_t=alpha_t, z0_dist='discrete', probs=probs_E0)
            E_t = (noisy_data['E_t'] - mean_Et.unsqueeze(-1)) / std_Et.unsqueeze(-1)
            noisy_data['E_t'] = E_t * e_mask

        if 'X_t' in noisy_data:
            mean_Xt, std_Xt = get_beta_stats(bounds=bounds, eta=self.eta_X, alpha_t=alpha_t, z0_dist='uniform')
            X_t = (noisy_data['X_t'] - mean_Xt) / std_Xt
            noisy_data['X_t'] = X_t * x_mask

        # 2. preconditioning the inputs in the logit space
        if 'logit_E_t' in noisy_data:
            mean_logit_Et, std_logit_Et = get_logit_beta_stats(bounds, self.eta_E, alpha_t, z0_dist='discrete', probs=probs_E0)
            logit_E_t = (noisy_data['logit_E_t'] - mean_logit_Et.unsqueeze(-1)) / std_logit_Et.unsqueeze(-1)
            noisy_data['logit_E_t'] = logit_E_t * e_mask

        if 'logit_X_t' in noisy_data:
            mean_logit_Xt, std_logit_Xt = get_logit_beta_stats(bounds, self.eta_X, alpha_t, z0_dist='uniform')
            logit_X_t = (noisy_data['logit_X_t'] - mean_logit_Xt) / std_logit_Xt
            noisy_data['logit_X_t'] = logit_X_t * x_mask

        return noisy_data
    

    def compute_loss(self, pred, target, t, node_mask, loss_on='edge'):
        """
            Loss function for beta diffusion.
        """
        # prepare masks
        x_mask, e_mask, _, _ = create_masks(node_mask)
        if loss_on == 'edge':
            z_mask = e_mask.to(pred.dtype)     # bs, n, n, 1
        else:
            z_mask = x_mask.to(pred.dtype)     # bs, n, 1

        # apply noise schedule
        logit_alpha_t, logit_alpha_s = self.logit_alpha(t), self.logit_alpha(t * 0.95)
        alpha_t, alpha_s = torch.sigmoid(logit_alpha_t), torch.sigmoid(logit_alpha_s)
        delta = (logit_alpha_s.to(torch.float64).sigmoid() - logit_alpha_t.to(torch.float64).sigmoid()).to(torch.float32)
        alpha_t, alpha_s, delta = alpha_t.unsqueeze(2), alpha_s.unsqueeze(2), delta.unsqueeze(2)

        if loss_on == 'edge':
            alpha_t, alpha_s, delta = alpha_t.unsqueeze(3), alpha_s.unsqueeze(3), delta.unsqueeze(3)

        eta = self.eta_E if loss_on == 'edge' else self.eta_X

        alpha_p = (eta * delta * target) * z_mask + (1. - z_mask)
        alpha_q = (eta * delta * pred) * z_mask + (1. - z_mask)
        beta_p  = (eta - eta * alpha_s * target) * z_mask + (1. - z_mask)
        beta_q  = (eta - eta * alpha_s * pred) * z_mask + (1. - z_mask)

        _alpha_p = (eta * alpha_t * target) * z_mask + (1. - z_mask)
        _alpha_q = (eta * alpha_t * pred) * z_mask + (1. - z_mask)
        _beta_p  = (eta - eta * alpha_t * target) * z_mask + (1. - z_mask)
        _beta_q  = (eta - eta * alpha_t * pred) * z_mask + (1. - z_mask)

        KLUB_conditional = (KL_gamma(alpha_q,alpha_p).clamp(0)\
                            + KL_gamma(beta_q,beta_p).clamp(0)\
                            - KL_gamma(alpha_q+beta_q,alpha_p+beta_p).clamp(0)).clamp(0)
        KLUB_conditional = KLUB_conditional * z_mask

        KLUB_marginal = (KL_gamma(_alpha_q,_alpha_p).clamp(0)\
                            + KL_gamma(_beta_q,_beta_p).clamp(0)\
                            - KL_gamma(_alpha_q+_beta_q,_alpha_p+_beta_p).clamp(0)).clamp(0)
        KLUB_marginal = KLUB_marginal * z_mask
        KLUB = .99 * KLUB_conditional + .01 * KLUB_marginal
        
        # option 1. unifromly weigh each effective edge position
        # normalization = z_mask.numel() / z_mask.sum()
        # loss = KLUB.mean() * normalization

        # option 2. uniformly weigh each graph
        sum_dims = [i for i in range(1, len(z_mask.shape))]
        normalizeation = z_mask.sum(dim=sum_dims)
        loss = KLUB.sum(dim=sum_dims) / normalizeation

        return loss.mean() * z_mask.mean()
    

    def sample_p_zs_given_zt(self, s, t, logit_X_t, logit_E_t, node_mask):
        """
        p(zs | zt) = q(zs | zt, z0_hat)
        logit_X_t: noisy node features in logit space, shape = (bs, n, dx)
        logit_E_t: noisy adj in logit space, shape = (bs, n, n)
        ----------
        zs = zt + (1 - zt) * p_t2s, or
        logit(zs) = ln(exp(logit_zt) + exp(logit_p_t2s) + exp(logit_zt) * exp(p_t2s))
        in logit space
        """
        x_mask, e_mask, triu_mask, diag_mask = create_masks(node_mask)
        bs, n = node_mask.shape

        noisy_data = edict({'t': t, 'y_t': t * torch.ones(bs, 1).to(self.device)})
        noisy_data.logit_X_t = logit_X_t * x_mask
        noisy_data.logit_E_t = logit_E_t * triu_mask
        noisy_data.logit_E_t = (noisy_data.logit_E_t + noisy_data.logit_E_t.transpose(1,2)) * e_mask
        noisy_data.X_t = noisy_data.logit_X_t.sigmoid() * x_mask
        noisy_data.E_t = noisy_data.logit_E_t.sigmoid() * e_mask

        if self.apply_precondition:
            noisy_data = self.precondition(noisy_data, node_mask)

        # predict z0_hat
        with torch.no_grad():
            z0_hat = self.forward(noisy_data, node_mask)
        z0_hat_transformed = self.scale_shift(z0_hat.X, z0_hat.E, z0_hat.y, node_mask)

        # sample zt -> zs
        logit_alpha_t, logit_alpha_s = self.logit_alpha(t), self.logit_alpha(s)

        for i, pred in enumerate([z0_hat_transformed.E, z0_hat_transformed.X]):
            eta = self.eta_E if i == 0 else self.eta_X
            log_u = log_gamma((eta * (logit_alpha_s.sigmoid() - logit_alpha_t.sigmoid()) * pred).to(torch.float32))
            log_v = log_gamma((eta - eta * logit_alpha_s.sigmoid() * pred).to(torch.float32))
            logit_p_t2s = log_u - log_v

            # compute logit(zs)
            logit_z_t = logit_E_t if i == 0 else logit_X_t
            logit_z_s = torch.logsumexp(torch.stack([logit_z_t, logit_p_t2s, logit_p_t2s+logit_z_t], dim=-1), dim=-1)

            if i == 0:
                logit_E_s = logit_z_s
            else:
                logit_X_s = logit_z_s

        return logit_X_s, logit_E_s, z0_hat

    @torch.no_grad()
    def sample_batch(self, bs, random_state=None, reverse_steps=None, init_val_to_sample_zT=0.5):

        if random_state is not None:
            pl.seed_everything(random_state)

        # override reverse sampling steps
        if reverse_steps is None:
            reverse_steps = self.T

        # create node_mask
        n_nodes = self.nodes_dist.sample_n(bs, self.device)
        N = torch.max(n_nodes).item()
        arange = torch.arange(N, device=self.device).unsqueeze(0).expand(bs, -1)
        node_mask = arange < n_nodes.unsqueeze(1)

        # create other masks for future use
        x_mask, e_mask, triu_mask, _ = create_masks(node_mask)

        # unifrom sampling timestamps
        sample_steps = 1 - torch.arange(reverse_steps) / (reverse_steps - 1) * (1.  - 1e-5)
        sample_steps = torch.cat([sample_steps, torch.zeros(1)]).to(self.device)

        # initiate the reverse chain
        alpha_T = self.logit_alpha(sample_steps[0]).sigmoid()

        for i, shape in enumerate([(bs, N, N, 1), (bs, N, 1)]):
            init_val = (init_val_to_sample_zT * torch.ones(*shape) * self.Scale + self.Shift).to(self.device)
            eta = self.eta_E if i == 0 else self.eta_X
            log_u = log_gamma((eta * alpha_T * init_val).to(torch.float32))
            log_v = log_gamma((eta - eta * alpha_T * init_val).to(torch.float32))
            if i == 0:
                logit_E_s = log_u - log_v
            else:
                logit_X_s = log_u - log_v

        pbar = tqdm(total=len(sample_steps) - 1)
        for i, (t,s) in enumerate(zip(sample_steps[:-1], sample_steps[1:])):
            pbar.set_description(f"from {t:.3f} to {s:.3f}")
            logit_X_t, logit_E_t = logit_X_s, logit_E_s
            logit_X_s, logit_E_s, z0_hat = self.sample_p_zs_given_zt(s, t, logit_X_t, logit_E_t, node_mask)
            pbar.update(1)

        predict = z0_hat.E * e_mask

        sample = (logit_E_s.sigmoid() / self.logit_alpha(s).sigmoid()) * triu_mask
        sample = ((sample + sample.transpose(1,2)) - self.Shift)  / self.Scale
        sample = sample * e_mask

        return predict.clamp(0,1), sample.clamp(0,1), node_mask

    
    @torch.no_grad()
    def sample_batch_EX(self, bs, random_state=None, reverse_steps=None, init_val_to_sample_zT=0.5):

        if random_state is not None:
            pl.seed_everything(random_state)

        # override reverse sampling steps
        if reverse_steps is None:
            reverse_steps = self.T

        # create node_mask
        n_nodes = self.nodes_dist.sample_n(bs, self.device)
        N = torch.max(n_nodes).item()
        arange = torch.arange(N, device=self.device).unsqueeze(0).expand(bs, -1)
        node_mask = arange < n_nodes.unsqueeze(1)

        # create other masks for future use
        x_mask, e_mask, triu_mask, _ = create_masks(node_mask)

        # unifrom sampling timestamps
        sample_steps = 1 - torch.arange(reverse_steps) / (reverse_steps - 1) * (1.  - 1e-5)
        sample_steps = torch.cat([sample_steps, torch.zeros(1)]).to(self.device)

        # initiate the reverse chain
        alpha_T = self.logit_alpha(sample_steps[0]).sigmoid()

        for i, shape in enumerate([(bs, N, N, 1), (bs, N, 1)]):
            init_val = (init_val_to_sample_zT * torch.ones(*shape) * self.Scale + self.Shift).to(self.device)
            eta = self.eta_E if i == 0 else self.eta_X
            log_u = log_gamma((eta * alpha_T * init_val).to(torch.float32))
            log_v = log_gamma((eta - eta * alpha_T * init_val).to(torch.float32))
            if i == 0:
                logit_E_s = log_u - log_v
            else:
                logit_X_s = log_u - log_v

        pbar = tqdm(total=len(sample_steps) - 1)
        for i, (t,s) in enumerate(zip(sample_steps[:-1], sample_steps[1:])):
            pbar.set_description(f"from {t:.3f} to {s:.3f}")
            logit_X_t, logit_E_t = logit_X_s, logit_E_s
            logit_X_s, logit_E_s, z0_hat = self.sample_p_zs_given_zt(s, t, logit_X_t, logit_E_t, node_mask)
            pbar.update(1)

        predict_E = z0_hat.E * e_mask
        predict_X = z0_hat.X * x_mask

        return predict_E.clamp(0,1), predict_X.clamp(0,1), node_mask
    

    @torch.no_grad()
    def sample_trajectory(self, num_trajectories, chain_length=None, reverse_steps=None, random_state=None, init_val_to_sample_zT=0.5):

        bs = num_trajectories       # redefine batch size
        # override reverse sampling steps
        if reverse_steps is None:
            reverse_steps = self.T

        if chain_length is not None:
            if chain_length < 2:
                chain_length = 2
        else:
            chain_length = reverse_steps

        ind_seg = (reverse_steps - 1) / (chain_length - 1)
        take_inds = [round(ind_seg * k) for k in range(chain_length - 1)] + [reverse_steps - 1]

        # create node_mask
        n_nodes = self.nodes_dist.sample_n(bs, self.device)
        N = torch.max(n_nodes).item()
        arange = torch.arange(N, device=self.device).unsqueeze(0).expand(bs, -1)
        node_mask = arange < n_nodes.unsqueeze(1)

        # create other masks for future use
        x_mask, e_mask, triu_mask, _ = create_masks(node_mask)

        # unifrom sampling timestamps
        sample_steps = 1 - torch.arange(reverse_steps) / (reverse_steps - 1) * (1.  - 1e-5)
        sample_steps = torch.cat([sample_steps, torch.zeros(1)]).to(self.device)

        if random_state is not None:        
            pl.seed_everything(random_state)

        # initiate the reverse chain
        alpha_T = self.logit_alpha(sample_steps[0]).sigmoid()
        init_val = (init_val_to_sample_zT * torch.ones(bs, N, N+1) * self.Scale + self.Shift).to(self.device)
        log_u = log_gamma((self.eta * alpha_T * init_val).to(torch.float32))
        log_v = log_gamma((self.eta - self.eta * alpha_T * init_val).to(torch.float32))
        logit_E_s, logit_X_s = (log_u - log_v)[...,:N].unsqueeze(-1), (log_u - log_v)[...,N:]       # (bs, N, N, 1) and (bs, N, 1)

        traj = edict()
        traj.node_masks = node_mask
        traj.times = []
        traj.predict = []
        traj.original = []
        traj.scaled = []
        
        pbar = tqdm(total=len(sample_steps) - 1)

        for i, (t,s) in enumerate(zip(sample_steps[:-1], sample_steps[1:])):
            pbar.set_description(f"from {t:.3f} to {s:.3f}")
            logit_X_t, logit_E_t = logit_X_s, logit_E_s
            logit_X_s, logit_E_s, z0_hat = self.sample_p_zs_given_zt(s, t, logit_X_t, logit_E_t, node_mask)

            if i in take_inds:

                traj.times.append(t)

                E_0 = z0_hat.E * e_mask
                traj.predict.append(E_0.squeeze(-1).unsqueeze(1).clamp(0,1))    # bs, 1, n, n

                E_s = logit_E_s.sigmoid() * triu_mask
                E_s = (E_s + E_s.transpose(1,2)) * e_mask
                traj.original.append(E_s.squeeze(-1).unsqueeze(1).clamp(0,1))   # bs, 1, n, n
                
                E_s = (E_s / self.logit_alpha(s).sigmoid()) * e_mask
                traj.scaled.append(E_s.squeeze(-1).unsqueeze(1).clamp(0,1))     # bs, 1, n, n
            
            pbar.update(1)

        # organize output
        traj.predict = list(torch.cat(traj.predict, dim=1).clamp(0,1))
        traj.original = list(torch.cat(traj.original, dim=1).clamp(0,1))
        traj.scaled = list(torch.cat(traj.scaled, dim=1).clamp(0,1))

        return traj

        
def log_gamma(alpha):
    # alpha += torch.finfo(torch.float32).eps
    return torch.log(torch._standard_gamma(alpha.to(torch.float32)))


def KL_gamma(*args):
    """
    Calculates the KL divergence between two Gamma distributions.
    args[0]: alpha_p, the shape of the first Gamma distribution Gamma(alpha_p,beta_p).
    args[1]: alpha_q,the shape of the second Gamma distribution Gamma(alpha_q,beta_q).
    args[2]: beta_p, the rate (inverse scale) of the first Gamma distribution Gamma(alpha_p,beta_p).
    args[3]: beta_q, the rate (inverse scale) of the second Gamma distribution Gamma(alpha_q,beta_q).
    """    
    alpha_p = args[0]
    alpha_q = args[1]
    # alpha_p, alpha_q = alpha_p + torch.finfo(torch.float32).eps, alpha_q + torch.finfo(torch.float32).eps
    KL = (alpha_p-alpha_q)*torch.digamma(alpha_p)-torch.lgamma(alpha_p)+torch.lgamma(alpha_q)
    if len(args)>2:
        beta_p = args[2]
        beta_q = args[3]
        # beta_p, beta_q = beta_p + torch.finfo(torch.float32).eps, beta_q + torch.finfo(torch.float32).eps
        KL = KL + alpha_q*(torch.log(beta_p)-torch.log(beta_q))+alpha_p*(beta_q/beta_p-1.0)  
    return KL


def KL_beta(alpha_p,beta_p,alpha_q,beta_q):
    """
    Calculates the KL divergence between two Beta distributions
    KL(Beta(alpha_p,beta_p) || Beta(alpha_q,beta_q))
    """
    KL =KL_gamma(alpha_p,alpha_q)+KL_gamma(beta_p,beta_q)-KL_gamma(alpha_p+beta_p,alpha_q+beta_q)
    return KL


def get_beta_stats(bounds, eta, alpha_t, z0_dist='discrete', probs=None):
    """
    Calculate the mean and std of z_t ~ Beta(eta * alpha_t * z_0, eta * (1 - alpha_t * z_0))
    z_0 follows a categorical distribution, defined by the values of the probability of taking each value
    """
    
    if z0_dist == 'discrete':
        mean_z0 = (bounds * probs).sum()
        var_z0 = (bounds ** 2. * probs).sum() - mean_z0 ** 2.
    
    elif z0_dist == 'uniform':
        mean_z0 = bounds.mean()
        var_z0 = (bounds[1] - bounds[0]) ** 2.

    else:
        raise ValueError(f"z0_dist '{z0_dist}' is not defined (can only be 'discrete' or 'uniform').")
    
    mean_zt = alpha_t * mean_z0
    var_zt = (mean_z0 * (1 - mean_z0) - (alpha_t**2. * var_z0)) / (eta + 1.) + (alpha_t**2. * var_z0)

    return mean_zt, torch.sqrt(var_zt)


def get_logit_beta_stats(bounds, eta, alpha_t, z0_dist='discrete', probs=None):
    """
    Calculate the mean and std of logit(z_t), where z_t ~ Beta(eta * alpha_t * z_0, eta * (1 - alpha_t * z_0))
    z_0 follows a categorical distribution, defined by the values of the probability of taking each value
    """
    # bs, 1, 1
    a0 = eta * alpha_t * bounds[0]
    b0 = eta - a0
    a1 = eta * alpha_t * bounds[1]
    b1 = eta - a1

    device = bounds.device

    if z0_dist == 'discrete':
        # all are (bs, 1, 1)
        mean_dg_a = probs[0] * torch.digamma(a0) + probs[1] * torch.digamma(a1)
        mean_dg_b = probs[0] * torch.digamma(b0) + probs[1] * torch.digamma(b1)
        mean_tg_a = probs[0] * torch.polygamma(1, a0) + probs[1] * torch.polygamma(1, a1)
        mean_tg_b = probs[0] * torch.polygamma(1, b0) + probs[1] * torch.polygamma(1, b1)
        mean_dgsq_a = probs[0] * torch.digamma(a0)**2. + probs[1] * torch.digamma(a1)**2.
        mean_dgsq_b = probs[0] * torch.digamma(b0)**2. + probs[1] * torch.digamma(b1)**2.
    
    elif z0_dist == 'uniform':
        # all are (bs, 1, 1)
        mean_dg_a = (torch.lgamma(a1) - torch.lgamma(a0)) / (a1 - a0)
        mean_dg_b = (torch.lgamma(b0) - torch.lgamma(b1)) / (b0 - b1)
        mean_tg_a = (torch.digamma(a1) - torch.digamma(a0)) / (a1 - a0)
        mean_tg_b = (torch.digamma(b0) - torch.digamma(b1)) / (b0 - b1)
        
        u01_samples = torch.linspace(0, 1, 101).to(device)                   # (bs, )
        a_ = (a1 - a0).squeeze(-1) * u01_samples.unsqueeze(0) + a0.squeeze(-1)      # (bs, 101)
        b_ = (b0 - b1).squeeze(-1) * u01_samples.unsqueeze(0) + b1.squeeze(-1)      # (bs, 101)

        w_ = torch.ones(101, dtype=torch.float32).to(device) / 100
        w_[0] = w_[-1] = 1/200

        mean_dgsq_a = (torch.digamma(a_) ** 2. * w_.unsqueeze(0)).sum(dim=-1, keepdim=True)     # (bs, 1)
        mean_dgsq_a = mean_dgsq_a.unsqueeze(-1)
        mean_dgsq_b = (torch.digamma(b_) ** 2. * w_.unsqueeze(0)).sum(dim=-1, keepdim=True)     # (bs, 1)
        mean_dgsq_b = mean_dgsq_b.unsqueeze(-1)
    
    else:
        raise ValueError(f"z0_dist '{z0_dist}' is not defined (can only be 'discrete' or 'uniform').")
    
    var_dg_a = F.relu(mean_dgsq_a - mean_dg_a ** 2.)      # (bs, 1, 1)
    var_dg_b = F.relu(mean_dgsq_b - mean_dg_b ** 2.)      # (bs, 1, 1)

    mean_zt = mean_dg_a - mean_dg_b
    var_zt = mean_tg_a + mean_tg_b + var_dg_a + var_dg_b

    return mean_zt, torch.sqrt(var_zt)









        