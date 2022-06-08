from typing import Dict, Any, Callable, List, Tuple, Optional, Union
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels
from sklearn.base import BaseEstimator, TransformerMixin
import re


def calculate_classification_metrics(
    y_true: np.array,
    y_pred: np.array,
    average: Optional[str] = None,
    return_df: bool = True,
) -> Union[Dict[str, float], pd.DataFrame]:
    
    labels = unique_labels(y_true, y_pred)

    # get results
    precision, recall, f_score, support = metrics.precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=average
    )

    kappa = metrics.cohen_kappa_score(y_true, y_pred, labels=labels)
    accuracy = metrics.accuracy_score(y_true, y_pred)

    # create a pandas DataFrame
    if return_df:
        results = pd.DataFrame(
            {
                "class": labels,
                "f_score": f_score,
                "precision": precision,
                "recall": recall,
                "support": support,
                "kappa": kappa,
                "accuracy": accuracy,
            }
        )
    else:
        results = {
            "f1": f_score,
            "precision": precision,
            "recall": recall,
            "kappa": kappa,
            "accuracy": accuracy,
        }

    return results


def visualize_performance(
        df: pd.DataFrame,
        metrics: List[str],
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        ylim: Optional[Tuple[float, float]] = None,
        figsize: Optional[Tuple[int, int]] = None,
        use_class_names: bool = True
) -> None:
    
    unstacked_df = (
        df[metrics]
            .T.unstack()
            .reset_index()
            .rename(
            index=str, columns={"level_0": "class", "level_1": "metric", 0: "score"}
        )
    )

    if use_class_names:
        unstacked_df["class"] = unstacked_df["class"].apply(
            lambda x: df["class"].tolist()[x]
        )

    if figsize is None:
        figsize = (10, 7)

    # Diplay the graph
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    sns.barplot(x="class", y="score", hue="metric", data=unstacked_df, ax=ax)

    # Format the graph
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    if title is not None:
        ax.set_title(title, fontsize=20)

    if ylim is not None:
        ax.set_ylim(ylim)

    plt.tight_layout()


class BertTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(
            self,
            bert_tokenizer,
            bert_model,
            max_length: int = 60,
            embedding_func: Optional[Callable[[Tuple[torch.tensor]], torch.tensor]] = None,
    ):
        self.tokenizer = bert_tokenizer
        self.model = bert_model
        self.model.eval()
        self.max_length = max_length
        self.embedding_func = embedding_func

        if self.embedding_func is None:
            self.embedding_func = lambda x: x[0][:, 0, :]

        # TODO:: PADDING

    def _tokenize(self, text: str):
        tokenized_text = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_length
        )["input_ids"]
        attention_mask = [1] * len(tokenized_text)

        # bert takes in a batch so we need to unsqueeze the rows
        return (
            torch.tensor(tokenized_text).unsqueeze(0),
            torch.tensor(attention_mask).unsqueeze(0),
        )

    def _tokenize_and_predict(self, text: str):
        tokenized, attention_mask = self._tokenize(text)

        embeddings = self.model(tokenized, attention_mask)
        return self.embedding_func(embeddings)

    def transform(self, text: List[str]):
        if isinstance(text, pd.Series):
            text = text.tolist()

        with torch.no_grad():
            return torch.stack([self._tokenize_and_predict(string) for string in text])

    def fit(self, X, y=None):
        """No fitting necessary so we just return ourselves"""
        return self


def convert_df_to_conv_ai_dict(df: pd.DataFrame,
                               personality: List[str],
                               response_columns: List[str],
                               tokenizer: Callable[[str], List[str]],
                               max_tokens: Optional[int] = None,
                               n_candidates: int = 6
                               ) -> Dict[str, List[Any]]:
   
    # Add one because the index of the dataframe is the 0th position.
    tuple_map = {name: index + 1 for index, name in enumerate(df.columns.tolist())}

    train = []
    val = []
    # Step through every row in the dictionary
    for row in df.itertuples():

        # Get the question name and title
        # TODO:: MAKE THIS GENERAL YOU DUMB DUMB
        question_title = row[tuple_map["questionTitle"]]
        question_text = row[tuple_map["questionText"]]
        question_combined = question_title + " " + question_text

        # Step through every response column in the row
        for response_column in response_columns:

            # Get the true response
            true_response = row[tuple_map[response_column]]

            # We only want to add data if a good response exists
            if len(true_response) > 1:
                # Get candidate alternate sentances by sampling from all other questions
                candidates = sample_candidates(df, row[tuple_map["questionID"]], "questionID", "answerText",
                                               n_candidates)

                # Add the correct response to the end
                candidates.append(true_response)

                # We want to trim the size of the tokens
                if max_tokens is not None:
                    # Use the provided tokenizer to tokenize the input and truncate at max_tokens
                    question_combined = tokenizer.convert_tokens_to_string(
                        tokenizer.tokenize(question_combined)[:max_tokens])
                    candidates = [tokenizer.convert_tokens_to_string(tokenizer.tokenize(candidate)[:max_tokens]) for
                                  candidate in candidates]

                if len(candidates) != n_candidates + 1:
                    print(true_response)
                    assert False

                # Define the personality and the history
                d = {"personality": personality,
                     "utterances": [{"history": [question_combined],
                                     "candidates": candidates}]}
                if getattr(row, "split") == "train":
                    train.append(d)
                elif getattr(row, "split") == "val":
                    val.append(d)

    data = {"train": train, "valid": val}

    return data


def sample_candidates(df: pd.DataFrame, current_id: Any, id_column: str, text_column: str, n: int) -> List[str]:
    
    # We must only sample candidates from the correct data split to avoid information leakage across channels
    split = df[df[id_column] == current_id]["split"].tolist()[0]
    candidate_df = df[df["split"] == split]

    # Sample 3 random rows from the dataframe not matching the current id
    sampled_texts = candidate_df[candidate_df[id_column] != current_id].sample(n + 15)[text_column].tolist()

    # join them all
    text = " ".join(sampled_texts)

    # Replace all newlines with spaces...
    text_no_newline = re.sub("\n", " ", text).lower()

    # Split on punctuation
    split_text = re.split('[?.!]', text_no_newline)

    # Remove all empty lines
    filtered_text = [x.strip() for x in split_text if len(x.strip()) > 1]

    # Shuffle the list
    return np.random.choice(filtered_text, n).tolist()