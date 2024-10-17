import wandb, torch, tqdm, sys, os, json, math, gc
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.__str__())
from typing import List, Tuple, Union, Any
from dataset import CIFARDataset
from transformers import ViTForImageClassification
from torch.utils.data import Dataset

class ViTTrainer:
    def __init__(self, device: torch.device, config: dict):
        self.device = device
        self.config = config
        self.model_name_srt = config["model_name"].split("/")[-1]
        self.num_epochs = config["num_epochs"]
        self.wandb_log = config["wandb_log"]
        self.calc_norm = config["calc_norm"]
        self.project_name = f"vitB16-finetune-CIFAR100-final"
        self.checkpoint_dir = Path(Path.cwd(), f"Assignment1/outputs/ckpt/{self.project_name}/{config['ckpt_key']}")
        
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=100).to(self.device)
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.config["initial_learning_rate"], weight_decay=self.config["weight_decay"])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=1, eta_min=1e-5)
        self.train_ds = CIFARDataset(npz_file="/raid/speech/soumen/gnr/IITB-GNR650-ADLCV/Assignment1/outputs/denoised_image_labels_for_reshuffled_noisy_data.npz")
        self.test_ds = CIFARDataset(npz_file="/raid/speech/soumen/gnr/IITB-GNR650-ADLCV/Assignment1/outputs/test_image_labels.npz")
        self.train_dl = self.train_ds.prepare_dataloader(batch_size=config["batch_size"], frac=config["train_frac"])
        self.test_dl = self.test_ds.prepare_dataloader(batch_size=config["batch_size"], frac=config["test_frac"])
        
        if self.wandb_log:
            wandb.init(project=self.project_name, config=config)
            wandb.watch(self.model, log="all")
            wandb.define_metric("train/step")
            wandb.define_metric("test/step")
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("test/*", step_metric="test/step")

    def _find_norm(self, is_grad: bool) -> float:
        norm = 0
        for val in self.model.parameters():
            if val.requires_grad:
                if is_grad:
                    k = val
                else:
                    k = val.grad if val.grad is not None else torch.tensor(0.0, device=self.device)
                norm += (k ** 2).sum().item()
        norm = norm ** 0.5  
        return norm
    
    def _save_checkpoint(self, ep: int) -> None:
        checkpoint = {"epoch": ep, "model_state": self.model.state_dict(), "opt_state": self.optimizer.state_dict()}   
        if not Path.exists(self.checkpoint_dir):
            Path.mkdir(self.checkpoint_dir, parents=True, exist_ok=True)
        checkpoint_path = Path(self.checkpoint_dir, f"ckpt_ep_{ep}.pth")
        torch.save(checkpoint, checkpoint_path)
        print(f"[SAVE] ep: {ep}/{self.num_epochs-1}, checkpoint saved at: {checkpoint_path}")
    
    def _forward_batch(self, batch: dict, is_train: bool) -> torch.tensor:
        x, y = batch # (b, 3, 224, 224), (b,)
        x, y = x.to(self.device), y.to(self.device) 
        if is_train:
            self.model.train()
            out = self.model(x).logits
            out.requires_grad_(True)
            assert out.requires_grad == True
        else:
            self.model.eval()
            with torch.no_grad():
                out = self.model(x).logits
        return out # (b, c)
        
    def _calc_loss_batch(self, pred_outputs: torch.tensor, true_outputs: torch.tensor) -> torch.tensor:
        pred_outputs = pred_outputs.to(self.device)
        true_outputs = true_outputs.to(self.device)
        assert pred_outputs.dim() == 2, f"pred_outputs.shape = {pred_outputs.shape} must be (b, c)"
        assert true_outputs.dim() == 1, f"true_outputs.shape = {true_outputs.shape} must be (b,)"
        loss = torch.nn.functional.cross_entropy(input=pred_outputs, target=true_outputs)
        return loss # returns the computational graph also along with it
        
    def _calc_acc_batch(self, pred_outputs: torch.tensor, true_outputs: torch.tensor) -> torch.tensor:
        pred_outputs = pred_outputs.to(self.device)
        true_outputs = true_outputs.to(self.device)
        assert pred_outputs.dim() == 2, f"pred_outputs.shape = {pred_outputs.shape} must be (b, c)"
        assert true_outputs.dim() == 1, f"true_outputs.shape = {true_outputs.shape} must be (b,)"
        acc = (pred_outputs.argmax(dim=-1) == true_outputs).to(torch.float32).mean()
        return torch.tensor(acc.item()) # returns the tensor as a scalar number

    def _optimize_batch(self, pred_outputs: torch.tensor, true_outputs: torch.tensor, ep: int, batch_index: int) -> Tuple[float, float, float, float]:  
        loss = self._calc_loss_batch(pred_outputs=pred_outputs, true_outputs=true_outputs)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()       
        gn = self._find_norm(True) if self.calc_norm else -1
        pn = self._find_norm(False) if self.calc_norm else -1 
        lr = self.optimizer.param_groups[0]['lr'] 
        torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.config["clip_grad_norm_value"], norm_type=2.0)
        self.optimizer.step()
        self.scheduler.step(ep + batch_index/len(self.train_dl))
        return loss.item(), gn, pn, lr
    
    def _optimize_dataloader(self, ep: int) -> None:  
        with tqdm.tqdm(iterable=self.train_dl, desc=f"[TRAIN] ep: {ep}/{self.num_epochs-1}", total=len(self.train_dl), unit="step", colour="green") as pbar:
            for i, batch in enumerate(pbar):    
                pred_out = self._forward_batch(batch=batch, is_train=True) # (b, c)  
                true_out = batch[1] # (b,)   
                loss, gn, pn, lr = self._optimize_batch(pred_outputs=pred_out, true_outputs=true_out, ep=ep, batch_index=i)
                acc = self._calc_acc_batch(pred_outputs=pred_out, true_outputs=true_out)
                if self.wandb_log:
                    wandb.log({"train/loss": loss, "train/accuracy": acc, "train/learning_rate": lr, "train/grad_norm": gn, "train/param_norm": pn, "train/epoch": ep, "train/step": self.train_step})
                    self.train_step += 1
                pbar.set_postfix({"loss": f"{loss:.3f}", "acc": f"{acc:.3f}", "lr": f"{lr:.3e}", "gn": f"{gn:.3f}", "pn": f"{pn:.3f}"})
    
    def _test_dataloader(self, ep: int) -> None:  
        with tqdm.tqdm(iterable=self.test_dl, desc=f"[TEST] ep: {ep}/{self.num_epochs-1}", total=len(self.test_dl), unit="step", colour="green") as pbar:
            for i, batch in enumerate(pbar):    
                pred_out = self._forward_batch(batch=batch, is_train=False) # (b, c)  
                true_out = batch[1] # (b,)   
                acc = self._calc_acc_batch(pred_outputs=pred_out, true_outputs=true_out)
                loss = self._calc_loss_batch(pred_outputs=pred_out, true_outputs=true_out).item()
                if self.wandb_log:
                    wandb.log({"test/loss": loss, "test/accuracy": acc, "test/epoch": ep, "test/step": self.test_step})
                    self.test_step += 1
                pbar.set_postfix({"loss": f"{loss:.3f}", "acc": f"{acc:.3f}"})
            
    def _unfreeze_layers(self, layers_list: List[str]) -> None:
        """['cls_head', 'base.pooler', ...]"""
        if layers_list == ["all"]:
            for val in self.model.parameters():
                val.requires_grad_(True)
        else:
            for val in self.model.parameters():
                val.requires_grad_(False)
            for name, val in self.model.named_parameters():
                for layer in layers_list:
                    if name.startswith(layer):
                        val.requires_grad_(True)
                        break
        print(f"Gradients are computed for: ", [name for name, val in self.model.named_parameters() if val.requires_grad])
        total_params = sum([i.numel() for i in self.model.parameters()])
        total_trainable_params = sum([i.numel() for i in self.model.parameters() if i.requires_grad])
        print(f"Total number of parameters: {total_params}")
        print(f"Total number of trainable parameters: {total_trainable_params}")
        print(f"Finetuning efficiency: {total_trainable_params/total_params*100:.2f}%")
        
    def train(self) -> None:
        layers_list = ["all"]
        self._unfreeze_layers(layers_list=layers_list)
        self.train_step = 0
        self.test_step = 0
        for ep in range(self.num_epochs):
            self._optimize_dataloader(ep=ep)
            self._test_dataloader(ep=ep)
            self._save_checkpoint(ep=ep)
        if self.wandb_log:
            wandb.finish()
    
    def _load_checkpoint(self, name: str = "ckpt_ep_0.pth") -> None:
        checkpoint_path = Path(self.checkpoint_dir, name)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["opt_state"])
        print(f"[LOAD] ep: {checkpoint['epoch']}, checkpoint loaded from: {checkpoint_path}")

def main(model_name: str, device: torch.device) -> None:
    config = {
        "model_name": model_name,
        "num_epochs": 2, 
        "batch_size": 128,
        "train_frac": 1.0,
        "test_frac": 1.0,
        "num_class": 100,
        "clip_grad_norm_value": 10.0,
        "initial_learning_rate": 0.0001, 
        "weight_decay": 0.1,
        "ckpt_key": "denoise_all_alt",
        "calc_norm": True,
        "wandb_log": True
    }
    trainer = ViTTrainer(device=device, config=config)
    trainer.train()
    print("DONE")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    main('google/vit-base-patch16-224-in21k', device=device)
        
