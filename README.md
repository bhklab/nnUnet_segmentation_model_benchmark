# nnUnet_segmentation_model_benchmark
The data that was used for this project is stored at '/cluster/projects/radiomics/Gyn_Autosegmentation/Task80_gyn' in H4H radiomics folder.

<img src="workflow.png" alt="drawing" width="900" />


## How Testing Robusness of a Segmentation Model Works
### 1.Generate a Benchmarking Dataset
>- Change the input and output path in /segmentation_robustness/examples/dataset_generation.py and run this file.
>- The transformations for benchmarking dataset generation are stored in /segmentation_robustness/examples/roodmri/transforms/defaults.py, which can be modified if needed.

### 2.Testing the Model on the Benchmarking Dataset
>- Instructions for running nnUnet model inference can be found at https://github.com/MIC-DKFZ/nnUNet#run-inference.
>- Other models such as UNet/UNetR can be tested with the testing file segmentation_robustness/test.py.

### 3.Calculating Evaluation Metrics
>- Evaluation metrics can be calculated using /segmentation_robustness/generate_metrics.py, there are different functions in this file for multilabel/siglelabel segmentation.
