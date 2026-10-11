from __future__ import annotations
import copy, json, random
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler

class SequencePreprocessor:
    """Training-only median imputation, scaling and explicit missing indicators."""
    def fit(self, sequence):
        flat=sequence.reshape(-1,sequence.shape[-1])
        self.median=np.nanmedian(flat,axis=0); self.median=np.nan_to_num(self.median,nan=0.)
        filled=np.where(np.isnan(flat),self.median,flat)
        self.scaler=StandardScaler().fit(filled)
        return self
    def transform(self,sequence):
        mask=np.isnan(sequence)
        filled=np.where(mask,self.median,sequence)
        scaled=self.scaler.transform(filled.reshape(-1,sequence.shape[-1])).reshape(sequence.shape)
        # Fixed bound stabilizes rare ratios; not fitted to test values. Raw targets are never clipped.
        return np.concatenate([np.clip(scaled,-8,8),mask.astype(float)],axis=2).astype(np.float32)
    def parameters(self):
        return {"median":self.median.tolist(),"mean":self.scaler.mean_.tolist(),"scale":self.scaler.scale_.tolist(),"clip":8}

def flat_features(sequence,sector):
    return np.column_stack([sequence.reshape(len(sequence),-1),sequence[:,-1,:]-sequence[:,-2,:],(np.asarray(sector)=="manufacturing").astype(float)])

def make_network(kind, input_size, length, hidden=32, text_size=0):
    import torch
    from torch import nn
    class Network(nn.Module):
        def __init__(self):
            super().__init__(); self.kind=kind
            if kind=="mlp":
                self.encoder=nn.Sequential(nn.Flatten(),nn.Linear(input_size*length,hidden*2),nn.GELU(),nn.Dropout(.1),nn.Linear(hidden*2,hidden),nn.GELU())
            elif kind=="gru":
                self.encoder=nn.GRU(input_size,hidden,batch_first=True)
            elif kind=="transformer":
                self.project=nn.Linear(input_size,hidden)
                self.positions=nn.Parameter(torch.randn(1,length,hidden)*.02)
                layer=nn.TransformerEncoderLayer(hidden,nhead=4,dim_feedforward=hidden*2,dropout=.1,batch_first=True,activation="gelu",norm_first=True)
                self.encoder=nn.TransformerEncoder(layer,num_layers=2,enable_nested_tensor=False)
            else: raise ValueError(kind)
            self.head=nn.Sequential(nn.LayerNorm(hidden+text_size),nn.Linear(hidden+text_size,hidden),nn.GELU(),nn.Linear(hidden,2))
        def forward(self,sequence,text=None):
            if self.kind=="mlp": h=self.encoder(sequence)
            elif self.kind=="gru": h=self.encoder(sequence)[0][:,-1]
            else: h=self.encoder(self.project(sequence)+self.positions)[:,-1]
            if text is not None: h=torch.cat([h,text],dim=1)
            return self.head(h)
    return Network()

def fit_neural(kind, sequence, frame, split, config, seed, save_path, text=None):
    import torch
    from torch import nn
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    model=make_network(kind,sequence.shape[-1],sequence.shape[1],config["hidden_size"],text.shape[1] if text is not None else 0)
    x=torch.tensor(sequence,dtype=torch.float32); extra=torch.tensor(text,dtype=torch.float32) if text is not None else None
    y=frame.y_margin.to_numpy(); center=float(y[split["train"]].mean()); scale=max(float(y[split["train"]].std()),1e-4)
    yr=torch.tensor((y-center)/scale,dtype=torch.float32); yc=torch.tensor(frame.y_stress.to_numpy(),dtype=torch.float32)
    opt=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=.01)
    generator=np.random.default_rng(seed); best=np.inf; stale=0; log=[]; state=None
    def forward(ids): return model(x[ids],extra[ids] if extra is not None else None)
    for epoch in range(config["epochs"]):
        model.train(); losses=[]
        ids=generator.permutation(split["train"])
        for start in range(0,len(ids),config["batch_size"]):
            batch=ids[start:start+config["batch_size"]]; p=forward(batch)
            loss=nn.functional.smooth_l1_loss(p[:,0],yr[batch])+.3*nn.functional.binary_cross_entropy_with_logits(p[:,1],yc[batch])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            p=forward(split["validation"])
            score=float((p[:,0]-yr[split["validation"]]).abs().mean()+.3*((torch.sigmoid(p[:,1])-yc[split["validation"]])**2).mean())
        log.append({"epoch":epoch+1,"training_loss":float(np.mean(losses)),"validation_score":score})
        if score<best-1e-5:
            best=score; state=copy.deepcopy(model.state_dict()); stale=0
        else: stale+=1
        if stale>=config["patience"]: break
    if state is None: raise RuntimeError("Neural training did not produce a finite validation score")
    model.load_state_dict(state); model.eval()
    point=[];prob=[]
    with torch.inference_mode():
        for start in range(0,len(frame),256):
            p=forward(np.arange(start,min(start+256,len(frame))))
            point.extend((p[:,0].numpy()*scale+center).tolist()); prob.extend(torch.sigmoid(p[:,1]).numpy().tolist())
    save_path=Path(save_path); save_path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":state,"kind":kind,"input_size":sequence.shape[-1],"length":sequence.shape[1],
                "hidden_size":config["hidden_size"],"text_size":text.shape[1] if text is not None else 0,
                "target_mean":center,"target_scale":scale},save_path)
    return np.asarray(point),np.asarray(prob),{"epochs":len(log),"best_validation_score":best,"training_log":log}

