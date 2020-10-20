# Gradual Fine-Tuning for Low-Resource Domain Adaptation
Gradually  fine-tuning  in  a  multi-step  process  can  yield  sub-stantial further gains and can be applied with-out modifying the model or learning objective. This method has been demonstrated to be effective in Event Extraction and Dialogue State Tracking.

## Event Extraction
We use [DYGIE++](https://github.com/dwadden/dygiepp) framwork to perform event extraction on the ACE 2005 corpus by considering Arabic as the target domain and English as the auxiliary domain.

Build virtual environment:
```
cd dygiepp
conda create --name dygiepp python=3.7
pip install -r requirements.txt
conda develop .   # Adds DyGIE to your PYTHONPATH
```
