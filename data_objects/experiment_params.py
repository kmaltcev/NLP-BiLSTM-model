import os


class ExperimentsParams:
    def __init__(self, author_under_test, creation_under_test, first_impostor, second_impostor):
        self.author_under_test = author_under_test
        self.creation_under_test = creation_under_test
        self.first_impostor = first_impostor
        self.second_impostor = second_impostor
        self.path_to_plot = self.prepare_experiment()

    def prepare_experiment(self):
        path = f"./plots/{self.author_under_test}"
        if not os.path.isdir(path):
            os.mkdir(path)
        path = f"{path}/{self.first_impostor}_{self.second_impostor}"
        if not os.path.isdir(path):
            os.mkdir(path)
        return path

    def __str__(self):
        return f"Author under test: {self.author_under_test}\n" \
               f"Questionable creation: {self.creation_under_test}\n" \
               f"First impostor: {self.first_impostor}\n" \
               f"Second impostor: {self.second_impostor}"
