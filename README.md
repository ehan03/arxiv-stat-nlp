# CPSC 477 Final Project

### *Predicting Primary Sub-Categories of Statistics ArXiv Papers*
Team Members: Eugene Han, Ali Aldous, Elder Veliz

## Setup
1. Clone the repository.
2. Create a new conda environment with the following command:
```
conda env create -f env.yml
```
3. Download the baseline and fine-tuned RoBERTa models from [this link](https://drive.google.com/drive/folders/1ryE379gpl3f91cRpio6zviJEHCU0RYn3?usp=drive_link).
4. Place the downloaded models in the `models` directory.
5. Run the cells of `full_pipeline.ipynb` inside the `notebooks` folder. We recommend skipping the training cells as the models are already trained and saved in the `models` directory and the training process is time-consuming and resource-intensive.
