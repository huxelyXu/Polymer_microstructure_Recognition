## Polmer_Microstructure_Recognition
---
The microstructure of polymer materials is an important bridge between their molecular structure and macro-properties, which is of great significance to be effectively identified. With the increasing refinement of polymer material design, the microstructure of different polymer materials gradually converges, which is difficult to distinguish. In this study, the machine learning method is applied to recognize the microstructure. A highly accurate and interpretable model based on small experimental datasets has been completed by the methods of transfer learning and feature visualization. The result of the model can be explained from the perspective of physical chemistry. This work provides a new idea for the identification of microstructure and will help to further promote the process of intelligent polymer research and development.
### File Structure
---
```
.
├── addicted                       # image that used for illustration
├── Transfer_Learning              # transfer learning in different nerual neworks
|   └── src                        # training process
├── Feature_Visualization          # feature visualization to get interpretable results
|   └── src                        # raw algorithm 
├── Model                          # nerual neworks by PyTorch
├── Machine_Learning               # machine-learing models used for regression and classfication
├── LICENSE
├── pic_seg.py                     # to segment the raw images
└── README.md
```

---
All relevant data are available from the authors upon reasonable request 