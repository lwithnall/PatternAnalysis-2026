
# Pattern Analysis Assignment - ADNI ConvNeXt Model

## Algorithm Details
<u>Problem Statement</u>


<u>The ConvNeXt Algorithm</u>


## Feasability Review
**1. User Need, Scope & Acceptance Criteria** </br>
This model aims to be an automated triage application for clinicians. The prototype should be able to classify patient brain MRI slices as having Alzheimer’s Disease (AD) or being Cognitively Normal (CN).
Acceptance criteria; these are tested on held-out test / validation recordings:
-	Test accuracy ≥ 80% (project minimum) 
-	High confidence predictions (≥90%) should have an error rate ≤ 5%
-	Peak GPU VRAM usage shouldn’t go above 8GB
-	No overlap between train, test and validation datasets

**2. Model Choice & Course Concepts** </br>
Data is taken from the Alzheimer’s Disease Neuroimaging Initiative (ADNI) dataset. Images are 240x258 grayscale patient MRI scans, split into CN and AD classifications, with 20 slices per patient. The dataset partitions patients into training (556 NC, 520 AD) and test (227 NC, 223 AD) splits; details in Appendix 1. This project addresses the [Hard Difficulty] band of project 2.1 using a ConvNeXt model [1]. Baseline recordings will be taken using a standard CNN to establish an initial performance benchmark. Both models will begin with no pre-trained weights and draw upon class concepts such as convolution, loss quantification methods, and categorisation.

**3. Preliminary Feasibility Evidence** </br>
No patient overlap exists between partitions. NC and AD classes are balanced, being roughly 52% and 48% of the data, respectively. A patient-level, stratified portion (20%) of the training data was successfully taken as a validation set; details in Appendix 2. A 4-block CNN and torchivision’s ConvNext-tiny model were trained and tested on the validation set as a ‘smoke-test’.

<u>Validation Results</u>
<table>
<tr><th></th><th>CNN</th><th>ConvNeXt-tiny</th></tr>
<tr><td>Slice Acc</td><td>0.797</td><td>0.849</td></tr>
<tr><td>Slice Macro-F1</td><td>0.793</td><td>0.849</td></tr>
<tr><td>Patient Acc</td><td>0.843</td><td>0.912</td></tr>
<tr><td>Patient AUROC</td><td>0.943</td><td>0.967</td></tr>
</table>

Above validation recordings were taken from the model’s performance on its best epoch (#22). The ConvNeXt model performed better than the CNN in all categories, indicating possible applications for the given research dilemma. About 20% of validation set errors had confidence over 0.9, this should be further tested.

<u>Resource Usage</u></br>
The CNN implementation used around 2GB of VRAM with 0.39M parameters; epochs took around 7 seconds each. The ConvNeXt-tiny model used around 4GB of VRAM with 27.82M parameters; epochs took around 21 seconds each. Recordings reside within the predefined acceptance criteria.

**4. Risks, Budget & Fallback** </br>
Head shape and age variation could make it harder to extract anatomical biomarkers, leading to several ambiguous test cases. This is concerning given the high confidence on validation set errors. Confidence calibration and referral of uncertain cases for human review will be investigated.
Current model implementations reside within the 8GB VRAM requirement. The project will attempt to operate within 3 hours of A100 GPU usage time. A fallback in case of ConvNeXt failure will be further exploration of baseline and similar models.

**References** </br>
[1]	Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, “A ConvNet for the 2020s,” in Proc. CVPR, 2022, pp. 11976–11986.

<b>Appendix 1 – Data Overview (Before Validation Split)</b>
<table>
<tr><th>Split</th><th colspan="2">Patients</th><th colspan="2">Slices</th><th colspan="3">Slices per Patient</th></tr>
<tr><th></th><th>NC</th><th>AD</th><th>NC</th><th>AD</th><th>Min</th><th>Mean</th><th>Max</th></tr>
<tr><td>Train</td><td>556</td><td>520</td><td>11120</td><td>10400</td><td>20.0</td><td>20.0</td><td>20.0</td></tr>
<tr><td>Test</td><td>227</td><td>223</td><td>4540</td><td>4460</td><td>20.0</td><td>20.0</td><td>20.0</td></tr>
</table>

<b>Appendix 2 – Data Overview (After Validation Split)</b>
<table>
<tr><th>Split</th><th colspan="2">Patients</th><th colspan="2">Slices</th><th colspan="3">Slices per Patient</th></tr>
<tr><th></th><th>NC</th><th>AD</th><th>NC</th><th>AD</th><th>Min</th><th>Mean</th><th>Max</th></tr>
<tr><td>Train</td><td>444</td><td>416</td><td>8880</td><td>8320</td><td>20.0</td><td>20.0</td><td>20.0</td></tr>
<tr><td>Validation</td><td>112</td><td>104</td><td>2240</td><td>2080</td><td>20.0</td><td>20.0</td><td>20.0</td></tr>
<tr><td>Test</td><td>227</td><td>223</td><td>4540</td><td>4460</td><td>20.0</td><td>20.0</td><td>20.0</td></tr>
</table>

## Dependencies
Ripped from requirements.txt. Dependency list generated using the `pip freeze` command.
```
asttokens==3.0.2
cloudpickle==3.1.2
colorama==0.4.6
comm==0.2.3
contourpy==1.4.0
cycler==0.12.1
debugpy==1.8.22
executing==2.3.0
fonttools==4.67.0
ipykernel==7.4.0
ipython==9.17.1
ipython_pygments_lexers==1.1.1
jedi==0.20.1
joblib==1.6.0
jupyter_client==8.10.0
jupyter_core==5.9.1
kiwisolver==1.5.1
matplotlib==3.11.2
matplotlib-inline==0.2.2
narwhals==2.27.1
nest-asyncio2==1.7.4
nibabel==5.4.2
numpy==2.5.4
packaging==26.3
parso==0.8.7
pillow==13.0.0
platformdirs==4.13.5
prompt_toolkit==3.0.53
psutil==7.2.2
pure_eval==0.2.4
Pygments==2.21.0
pyparsing==3.3.3
python-dateutil==2.9.0.post0
pyzmq==27.2.0
scikit-learn==1.9.1
scipy==1.18.1
six==1.17.0
stack-data==0.6.3
threadpoolctl==3.7.0
tornado==6.5.10
traitlets==5.16.1
typing_extensions==4.16.0
wcwidth==0.9.3
```

## Result Reproducability
[Something something random state + same data partitions as in copm3710/ADNI folder]

## Algorithm Pipeline Demonstration
[Example inputs / outputs] -> include diagrams

## Preprocessing
[Preprocessing specifics w/ references]
[Justify training / validation / testing splits of data -> patient partitioning, val_split param, random_state]

## Open Research Dilemma Investigation

## AI Intelligence Usage Disclosure
<u>**Entry 1**</u></br>
**Model:** ChatGPT </br>
**Task:** Asked it to generate the `def print_split_stats(stats: dict)` function in `audit.py` </br>
**Other:**</br>
Helped make a function to print `split_stats` object to terminal. Verified by checking to make sure print statements were related to correct object parameters. Why use it? Figured it would make something look neat quickly. Tested by just running the `audit.py` file and checking the terminal output. Showed what I needed therefore W's. 

<u>**Entry 2**</u></br>
**Model:** ChatGPT </br>
**Task:** Asked it to convert my Validation Results table from word into a markdown-compatible format (table in feasability report section).</br>
**Other:**</br>
I gave it an unfinished table (I had written the first two rows myself), then got bored and asked GPT to format the rest for me. Validation was easy since 1. I just need to visually check the ouptut to ensure correctness, and 2. it's just finishing off the work I'd already done. 

<u>**Entry 3**</u></br>
**Model:** ChatGPT </br>
**Task:** Exact same as entry 2 except with the appendix tables for feasability review.</br>
**Other:**</br>
See enty 2. Although I just threw the data at it this time. I didn't start the tables at all. Verfieid output visually. 