# DeepGuard completion checklist

## Research
- [ ] Run source-independent leakage audit and preserve PASS output
- [ ] Finalize Custom CNN experiment
- [ ] Run standardized ResNet18 experiment
- [ ] Generate model comparison plots
- [ ] Select final model using validation only
- [ ] Freeze final checkpoint
- [ ] External Celeb-DF test without retraining/fine-tuning
- [ ] Error analysis: false positives, false negatives, compression, identities, manipulation differences
- [ ] Record dataset versions, preprocessing, configs, metrics, confusion matrices, train/inference time

## Demo
- [ ] Put final checkpoint at `models/production/deepguard_model.pth`
- [ ] Start FastAPI backend
- [ ] Start React frontend
- [ ] Test image upload and prediction
- [ ] Verify API docs at `/docs`

## Documentation
- [ ] Update experiment log with final results
- [ ] Add final plots to report/PPT
- [ ] Add methodology and limitations
- [ ] Add reproducibility instructions
- [ ] Keep dataset and checkpoints out of Git if too large or restricted
