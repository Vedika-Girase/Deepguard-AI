import argparse, json
from datetime import datetime
from pathlib import Path
import torch
from config import ExperimentConfig
from data_pipeline import build_loaders, dataset_summary
from evaluate import evaluate_model
from models.custom_cnn import CustomCNN
from models.resnet18_model import build_resnet18
from plot_results import generate_run_plots
from reproducibility import set_seed
from train import train_model


def make_run_id(model, tag):
    stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model.lower()}_{tag}_{stamp}"


def main():
    parser=argparse.ArgumentParser(description="DeepGuard reproducible image experiment runner")
    parser.add_argument("--model",choices=["cnn","resnet18"],default="cnn")
    parser.add_argument("--tag",default="2f_regularized")
    parser.add_argument("--epochs",type=int,default=25)
    parser.add_argument("--lr",type=float,default=3e-4)
    parser.add_argument("--weight-decay",type=float,default=1e-4)
    parser.add_argument("--dropout",type=float,default=0.5)
    parser.add_argument("--patience",type=int,default=5)
    parser.add_argument("--pretrained",action="store_true")
    parser.add_argument("--no-color-jitter",action="store_true")
    args=parser.parse_args()
    cfg=ExperimentConfig(epochs=args.epochs,learning_rate=args.lr,weight_decay=args.weight_decay,dropout=args.dropout,early_stopping_patience=args.patience,pretrained=args.pretrained,color_jitter=not args.no_color_jitter)
    set_seed(cfg.seed)
    device=torch.device(cfg.device); print("Device:",device)
    print("\nConfiguration:"); print(json.dumps({"model":args.model,"tag":args.tag,"epochs":cfg.epochs,"learning_rate":cfg.learning_rate,"weight_decay":cfg.weight_decay,"dropout":cfg.dropout,"early_stopping_patience":cfg.early_stopping_patience,"pretrained":cfg.pretrained,"color_jitter":cfg.color_jitter,"seed":cfg.seed},indent=4))
    loaders,datasets=build_loaders(cfg.data_dir,cfg.batch_size,cfg.num_workers,cfg.image_size,cfg.color_jitter)
    summary=dataset_summary(datasets); print("\nClass mapping:",datasets["train"].class_to_idx); print("\nDataset summary:"); print(json.dumps(summary,indent=4))
    if args.model=="cnn": model=CustomCNN(len(datasets["train"].classes),cfg.dropout); model_name="Custom CNN"
    else: model=build_resnet18(len(datasets["train"].classes),cfg.pretrained); model_name="ResNet18"
    run_id=make_run_id(args.model,args.tag); run_dir=Path(cfg.results_root)/run_id; run_dir.mkdir(parents=True,exist_ok=False); checkpoint=run_dir/"best_model.pth"
    train_info=train_model(model,loaders,device,epochs=cfg.epochs,learning_rate=cfg.learning_rate,weight_decay=cfg.weight_decay,scheduler_patience=cfg.scheduler_patience,scheduler_factor=cfg.scheduler_factor,min_lr=cfg.min_lr,early_stopping_patience=cfg.early_stopping_patience,checkpoint_path=checkpoint)
    model.load_state_dict(torch.load(checkpoint,map_location=device))
    metrics,_=evaluate_model(model,loaders["test"],device,datasets["test"].classes,run_dir)
    record={"run_id":run_id,"experiment":"Experiment 2","model":model_name,"dataset":"FaceForensics++ C23 mini subset","split":"source-video-independent","task":"image-based facial deepfake classification","image_size":cfg.image_size,"batch_size":cfg.batch_size,"epochs":cfg.epochs,"epochs_completed":train_info["epochs_completed"],"learning_rate":cfg.learning_rate,"optimizer":"AdamW","weight_decay":cfg.weight_decay,"dropout":cfg.dropout,"scheduler":"ReduceLROnPlateau","scheduler_patience":cfg.scheduler_patience,"scheduler_factor":cfg.scheduler_factor,"early_stopping_patience":cfg.early_stopping_patience,"pretrained":cfg.pretrained,"color_jitter":cfg.color_jitter,"seed":cfg.seed,"device":str(device),"augmentation":"RandomHorizontalFlip + RandomRotation(10) + optional ColorJitter","class_mapping":datasets["train"].class_to_idx,"dataset_summary":summary,**train_info,**metrics,"model_path":str(checkpoint),"notes":"Test set remains untouched during model selection. Source-video ID is used only for leakage-safe grouping."}
    record_path=run_dir/"experiment_record.json"; record_path.write_text(json.dumps(record,indent=4),encoding="utf-8"); generate_run_plots(record,run_dir)
    print("\nExperiment record saved:",record_path); print("Graphs saved in:",run_dir)

if __name__=="__main__": main()
