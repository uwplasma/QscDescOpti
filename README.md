# Project Title: Generating Good Stellarator Configurations

This project aims to generate good stellarator configurations from bad ones using deep generative models. By training a generator and a discriminator, we transform suboptimal stellarator configurations into optimal ones suitable for producing viable stellarator designs.

## File Descriptions

- **good_data.csv**: Contains 568 good stellarator configurations filtered from the original dataset. These configurations can successfully produce images of stellarators.
- **train_data.csv**: Comprises 568 entries selected from the 2,000 to 10,000 range of the original dataset, used as training data for the model.
- **test_data.csv**: Consists of 100 bad configurations randomly selected from the first 2,000 entries of the original dataset, used as test data.
- **model.py**: Defines the architecture of the generator and discriminator models.
- **train.py**: Script for training the generator model.
- **G_A2B_Base_55ac.pth**: Pre-trained model parameters for the generator, which can be directly used with test.py to simulate the generation of good configurations.
- **test.py**: Script for testing new trained model.
- **transform.py**: Script for transforming bad configurations in test_data.csv into good configurations. Currently, it can transform approximately 55% of the test data into good configurations.

## Requirements

To run the scripts, you need to have the following libraries installed:

```sh
pip install pandas numpy scikit-learn torch matplotlib seaborn
```

## How to use this model

1. **Prepare the Dataset**:
   - If you want to use this model to transform other bad configurations into good ones, replace the existing `test_data.csv` file with your dataset.
   - Ensure your new dataset is named `test_data.csv`.

2. **Run the Test Script**:
   - Execute `transform.py` to start the transformation process. The script will use the pre-trained model parameters to generate good configurations from the bad configurations in `test_data.csv`.

### Example Usage

```sh
# Replace the test_data.csv with your dataset
mv your_dataset.csv test_data.csv

# Run the test script
python transform.py
