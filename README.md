# Investigating Annotator Disagreement and Model Performance in Hate Speech Detection

## **Overview**
This project examines the role of **annotator positionality** in hate speech detection using **Natural Language Processing (NLP)**. It explores how annotators' backgrounds influence labeling decisions and evaluates the impact on the performance of automated models like **BERT**, **DeBERTa**, and **GPT-2**. By analyzing disagreements among annotators, the study aims to create more equitable and accurate hate speech detection systems.

## **Dataset**
The project uses the **Measuring Hate Speech** dataset from UCBerkeley DLab. The dataset contains:
- **39,565 comments** annotated by **7,912 annotators**.
- **135,556 rows** with labels representing the degree of hate speech.
- Demographic details of annotators and target groups.

The dataset is publicly available on [Hugging Face](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech).

## **Objectives**
1. **Analyze Annotator Disagreement:** Study patterns of disagreement among annotators and its correlation with their positionality.
2. **Evaluate Language Models:** Compare the performance of BERT, GPT-2, and DeBERTa for hate speech detection.
3. **Statistical Analysis:** Use metrics like Average Uncertainty-based Disagreement Rate (AUDR) and Average Agreement Deviation Rate (AADR) to quantify annotator disagreements.

## **Methodology**
- **Few-Shot Learning (BERT, GPT-2):** Fine-tuning on a limited dataset to evaluate adaptability with minimal training examples.
- **Zero-Shot Learning (DeBERTa):** Testing pre-trained models without task-specific fine-tuning.
- **Statistical Analysis:** Identifying trends in annotator disagreements based on demographic factors.

### **Preprocessing**
1. Handling null values and removing unnecessary rows.
2. Text cleaning (lowercasing, removing punctuation, numbers, and stop words).
3. Tokenization using pre-trained model tokenizers.

### **Training**
- **BERT:** Trained on 13,500 samples, achieving an accuracy of **78.39%**.
- **GPT-2:** Trained on 360 samples per class, achieving **56.71%** accuracy.
- **DeBERTa:** Zero-shot classification with accuracy of **47.24%**.

## **Results**
| Model      | Accuracy | Precision | Recall | F1 Score |
|------------|----------|-----------|--------|----------|
| **BERT**   | 78.39%   | 74.56%    | 78.39% | 75.89%   |
| **GPT-2**  | 56.71%   | 63.56%    | 56.71% | 51.19%   |
| **DeBERTa**| 47.24%   | 68.32%    | 47.24% | 51.40%   |

- All models struggled with ambiguous hate speech categories (Class 1).
- **BERT** showed the best performance due to its ability to generalize across diverse examples.

## **Statistical Findings**
- Annotator disagreements are influenced by demographics, such as gender, race, and ideology.
- Metrics like **AUDR** and **AADR** reveal significant disagreement trends in annotators from underrepresented or intersectional demographic groups.

## **Future Work**
1. Improve datasets with more diverse samples.
2. Enhance annotation guidelines to address biases and ambiguities.
3. Develop advanced NLP models for nuanced hate speech detection.
4. Explore ethical considerations between hate speech detection and free speech.

## **Requirements**
- Python 3.8+
- Libraries: `transformers`, `torch`, `pandas`, `nltk`

## **Usage**
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/your_repository.git
   cd your_repository
