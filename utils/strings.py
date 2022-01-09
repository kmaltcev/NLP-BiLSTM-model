button_label = "Analyse Authorship!"
page_title = "Plagiarism detection"
path_to_data_label = "Path to data"
author_under_test = "Author under test"
select_box_label = "Creation under test"
first_impostor_label = "First Impostors"
second_impostor_label = "Second Impostors"

embedding_progress_label = "Embedding in progress..."
preprocess_progress_label = "Preprocessing in progress..."
main_page_title = "# Plagiarism Detection Using Impostors Method"
prediction_by_creation_desc = "Predictions distribution by every creation"
summarized_pred_desc = "Predictions are summarized except creation under test. " \
                       "Creation under test is scaled up for easier analysis"
validation_barplot_desc = "Validation plot, training set is used for predictions. " \
                          "Therefore, distribution must be close to equal. If it's not, something is wrong"
length_err = "Impostors are not the same length. Please provide the same number of authors for each impostor"
lr_desc = "Learning rate controls how much to change the model in response to the estimated error each time the model " \
          "weights are updated."
empty_impostors_err = "Impostors are empty"


def test_set_len_err(test_set): return f"Author under test has only {len(test_set.data['book'].values)} " \
                                       f"creations, please provide more."


def same_impostors_err(name): return f"{name} paired with himself, predictions quality will not be valid. " \
                                     "Please provide different authors."


def test_set_title(author, creation): return f"## Test set: {author}, {creation}"


def training_title(nn_name): return f"{nn_name} Training in progress..."


def same_books_err(x): return f"These books looks are the same. The distance is: {x:.2f}"


def similar_books_err(x): return f"These books looks way too similar. The distance is: {x:.2f}"


def good_books_success(x): return f"Good choice for impostors! The distance is: {x:.2f}"


def diff_books_warning(x): return f"These books looks far too different. The distance is: {x:.2f}"


def plot_train_path(path): return f"{path}/preds_train_set_distribution.png"


def plot_by_book_preds_path(path): return f"{path}/preds_by_book_distribution.png"


def plot_summary_path(path): return f"{path}/preds_summary_distribution.png"


def experiment_title(idx, imp1, imp2): return f"### {idx + 1}. {imp1} vs. {imp2}"


def metric_result(pvalue):
    delta = "Original copy" if pvalue < 5 else "Suspicious"
    delta_color = "normal" if delta == "Original copy" else "inverse"
    return delta, delta_color


def computing_distance_label(names): return f"Computing distance for {names}"
